# scripts/serve-models.ps1 — поднимает три llama-server НА ХОСТЕ (не в docker).
#
# Почему это отдельный шаг, а не часть `docker-compose up -d`: решение заказчика —
# инференс-сервер в проде чужой, backend знает только его эндпоинт (plan.md §1, §11).
# `docker-compose.yml` собирает и запускает только `backend`+`gui`; без этого скрипта
# backend поднимется, но `/health` будет честно отвечать 503 (llama-эндпоинты
# недостижимы) — это ожидаемо, см. README.md «Порядок запуска».
#
# Единственный источник правды по флагам — этот файл (contracts/llm.md §8 хранит
# копию только для сверки). Три вызова проверены вживую на этой машине во время
# разработки T-11:
#   - LLM (Qwopus3.5-4B-Q4_K_M, -np 2 -c 32000): GET /health -> {"status":"ok"},
#     ~4.8 ГБ VRAM (n_ctx_seq округляется до 16128 на слот — тот же порядок, что
#     измерение "4307 МБ @ -c 16k" в plan.md §10; на -c 32000 полного пересчёта
#     под 8 ГБ не проводилось, см. предупреждение в README).
#   - Эмбеддер (bge-m3 Q8_0, --no-cache-prompt -cram 0 -ctxcp 0 -cpent -1):
#     "prompt cache is disabled" в логе, POST /v1/embeddings отвечает вектором.
#   - host.docker.internal ДОСТИЖИМ из контейнера при `--host 127.0.0.1` на этой
#     машине (Docker Desktop на Windows) — проверено `docker run curlimages/curl
#     ... http://host.docker.internal:<port>/health` -> 200. На Linux-хосте это
#     не гарантировано (host-gateway маршрутизирует на реальный IP хоста, не на
#     loopback) — там нужно биндить `--host 0.0.0.0`, см. README.
#
# Использование:
#   .\scripts\serve-models.ps1                 # поднять все три, дождаться /health, выйти
#   .\scripts\serve-models.ps1 -Stop            # остановить все три (по сохранённым PID)
#   .\scripts\serve-models.ps1 -LlamaServerBin 'C:\path\to\llama-server.exe'
#
# Переменные окружения (необязательные, приоритет выше поиска по умолчанию):
#   LLAMA_SERVER_BIN — путь к llama-server.exe (иначе: PATH, затем известный путь на
#                      этой машине из ecosystem.config.js)
#   MODELS_HOST_DIR  — корень весов НА ХОСТЕ (по умолчанию: .\models рядом с репо;
#                      ВНИМАНИЕ: это не то же самое, что MODELS_DIR в .env — та
#                      переменная это путь ВНУТРИ backend-контейнера, /models, для
#                      скрипта, который работает на хосте, она бесполезна)

[CmdletBinding()]
param(
    [switch]$Stop,
    [string]$LlamaServerBin,
    [string]$EnvFile,
    [int]$HealthTimeoutSec = 120
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot оказался пуст при некоторых способах запуска (`powershell -File`
# из-под обёртки другого процесса) — как минимум в среде, где готовился этот
# скрипт. $MyInvocation.MyCommand.Path устойчивее, поэтому основной источник
# каталога скрипта — он, $PSScriptRoot — запасной вариант.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) { $ScriptDir = $PSScriptRoot }
if (-not $ScriptDir) {
    Write-Error "Не удалось определить каталог скрипта (ни `$MyInvocation.MyCommand.Path, ни `$PSScriptRoot). Запусти скрипт напрямую: .\scripts\serve-models.ps1"
    exit 1
}
$RepoRoot = Resolve-Path (Join-Path $ScriptDir '..')
if (-not $EnvFile) { $EnvFile = Join-Path $RepoRoot '.env' }
$LogDir = Join-Path $RepoRoot 'scripts\logs'
$PidFile = Join-Path $LogDir 'serve-models.pids.json'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ─── .env: единственный источник эндпоинтов/имён файлов моделей (FR-32 касается
# python-кода backend; этот скрипт живёт на хосте вне docker и не подчиняется ей
# напрямую, но раз .env уже несёт эту информацию — вторая копия правды не нужна). ───
function Read-DotEnv([string]$Path) {
    $map = @{}
    if (-not (Test-Path $Path)) {
        return $map
    }
    foreach ($line in Get-Content $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }
        $idx = $trimmed.IndexOf('=')
        if ($idx -lt 1) { continue }
        $key = $trimmed.Substring(0, $idx).Trim()
        $val = $trimmed.Substring($idx + 1).Trim()
        # снять обрамляющие кавычки, если есть
        if ($val.Length -ge 2 -and (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'")))) {
            $val = $val.Substring(1, $val.Length - 2)
        }
        $map[$key] = $val
    }
    return $map
}

function Get-PortFromUrl([string]$Url, [string]$FallbackPort) {
    if ([string]::IsNullOrWhiteSpace($Url)) { return $FallbackPort }
    try {
        return ([Uri]$Url).Port.ToString()
    } catch {
        return $FallbackPort
    }
}

if (-not (Test-Path $EnvFile)) {
    Write-Error @"
Не найден $EnvFile

Этот скрипт читает LLM_ENDPOINT/EMBEDDING_ENDPOINT/RERANKER_ENDPOINT и имена
файлов моделей из .env (единственного источника, FR-32 — держим правило и здесь,
хоть скрипт и не часть backend). Сначала: cp .env.example .env
"@
    exit 1
}

$envMap = Read-DotEnv $EnvFile

$llmPort = Get-PortFromUrl $envMap['LLM_ENDPOINT'] '20099'
$embPort = Get-PortFromUrl $envMap['EMBEDDING_ENDPOINT'] '20100'
$rerPort = Get-PortFromUrl $envMap['RERANKER_ENDPOINT'] '20101'

$llmFile = $envMap['LLM_MODEL_FILE']
$embFile = $envMap['EMBEDDING_MODEL_FILE']
$rerFile = $envMap['RERANKER_MODEL_FILE']
$nParallel = $envMap['LLM_N_PARALLEL']; if ([string]::IsNullOrWhiteSpace($nParallel)) { $nParallel = '2' }
$ctxSize = $envMap['LLM_CONTEXT_SIZE']; if ([string]::IsNullOrWhiteSpace($ctxSize)) { $ctxSize = '32000' }

if (-not $llmFile -or -not $embFile -or -not $rerFile) {
    Write-Error "В $EnvFile отсутствуют LLM_MODEL_FILE / EMBEDDING_MODEL_FILE / RERANKER_MODEL_FILE — нечего запускать."
    exit 1
}

# ─── Остановка (по сохранённым PID) ───────────────────────────────────────────
if ($Stop) {
    if (-not (Test-Path $PidFile)) {
        Write-Host "Нет $PidFile — похоже, серверы этим скриптом не запускались."
        exit 0
    }
    $pids = Get-Content $PidFile -Raw | ConvertFrom-Json
    foreach ($p in @($pids.llm, $pids.embedding, $pids.reranker)) {
        if ($p -and (Get-Process -Id $p -ErrorAction SilentlyContinue)) {
            Write-Host "Останавливаю PID $p"
            Stop-Process -Id $p -Force
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    exit 0
}

# ─── Поиск бинарника llama-server ─────────────────────────────────────────────
# Порядок: -LlamaServerBin > $env:LLAMA_SERVER_BIN > PATH > известный путь на этой
# машине (тот же, что ecosystem.config.js использует для legacy-стека) — последний
# пункт полезен только на машине разработчика, override'ится первыми тремя на любой
# другой машине.
if (-not $LlamaServerBin) { $LlamaServerBin = $env:LLAMA_SERVER_BIN }
if (-not $LlamaServerBin) {
    $onPath = Get-Command 'llama-server.exe' -ErrorAction SilentlyContinue
    if ($onPath) { $LlamaServerBin = $onPath.Source }
}
if (-not $LlamaServerBin) {
    $knownDefault = 'C:\Users\bhunp\work-software\llama-cpp\llama-server.exe'
    if (Test-Path $knownDefault) { $LlamaServerBin = $knownDefault }
}
if (-not $LlamaServerBin -or -not (Test-Path $LlamaServerBin)) {
    Write-Error @'
llama-server.exe не найден.

Проверено (по порядку): параметр -LlamaServerBin, переменная окружения
LLAMA_SERVER_BIN, PATH, известный путь на машине разработчика.

Исправление: собери llama.cpp (или скачай релиз) и укажи путь явно:
  .\scripts\serve-models.ps1 -LlamaServerBin 'C:\путь\к\llama-server.exe'
или один раз:
  $env:LLAMA_SERVER_BIN = 'C:\путь\к\llama-server.exe'
'@
    exit 1
}

# ─── Поиск GGUF-файлов НА ХОСТЕ ────────────────────────────────────────────────
# MODELS_DIR из .env (по умолчанию "/models") — путь ВНУТРИ backend-контейнера,
# для этого скрипта бесполезен. На хосте веса лежат в ./models/llm/ и ./models/gguf/
# (см. README «Где взять веса»); ищем там, с плоским ./models/<файл> как fallback.
if (-not $env:MODELS_HOST_DIR) {
    $ModelsHostDir = Join-Path $RepoRoot 'models'
} else {
    $ModelsHostDir = $env:MODELS_HOST_DIR
}

function Find-ModelFile([string]$FileName, [string]$Label) {
    $candidates = @(
        (Join-Path $ModelsHostDir "llm\$FileName"),
        (Join-Path $ModelsHostDir "gguf\$FileName"),
        (Join-Path $ModelsHostDir $FileName)
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return (Resolve-Path $c).Path }
    }
    Write-Host ''
    Write-Host "ОШИБКА: не найден файл весов для $Label — '$FileName'" -ForegroundColor Red
    Write-Host 'Искал по путям:'
    foreach ($c in $candidates) { Write-Host "  - $c" }
    Write-Host ''
    Write-Host 'Это FR-31: сервис веса не скачивает, их кладёт Ивaн вручную.'
    Write-Host 'Ссылки и точное имя файла — в README.md «Где взять веса».'
    return $null
}

$llmPath = Find-ModelFile $llmFile 'LLM (LLM_MODEL_FILE)'
$embPath = Find-ModelFile $embFile 'эмбеддер (EMBEDDING_MODEL_FILE)'
$rerPath = Find-ModelFile $rerFile 'реранкер (RERANKER_MODEL_FILE)'

if (-not $llmPath -or -not $embPath -or -not $rerPath) {
    Write-Host ''
    Write-Error 'Отсутствуют веса — см. сообщения выше. Ни один сервер не запущен.'
    exit 1
}

# ─── Запуск ────────────────────────────────────────────────────────────────────
# Флаги — дословно contracts/llm.md §8, проверены вживую на этой машине (см.
# заголовок файла). --host 127.0.0.1: на Docker Desktop (Windows/Mac)
# host.docker.internal достаёт и loopback-порты хоста — проверено; на Linux
# host-gateway маршрутизирует на реальный IP, там нужен --host 0.0.0.0 (README).
function Start-LlamaServer([string]$Name, [string]$Port, [string[]]$ExtraArgs, [string]$ModelPath) {
    $logPath = Join-Path $LogDir "$Name.log"
    $args = @('-m', $ModelPath, '--host', '127.0.0.1', '--port', $Port) + $ExtraArgs
    Write-Host "Запускаю $Name : $LlamaServerBin $($args -join ' ')"
    $proc = Start-Process -FilePath $LlamaServerBin -ArgumentList $args `
        -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err" `
        -WindowStyle Hidden -PassThru
    return $proc
}

$llmArgs = @('-ngl', '99', '-c', $ctxSize, '-np', $nParallel, '-fa', 'on', '--jinja', '--no-webui')
# -b/-ub: llama-server defaults (2048/512) 500 on a single (query, doc) pair once it
# exceeds the physical batch size (-ub) -- RagFusedTop=50 candidates over the real
# 5856-chunk index (backend/rag/artifacts/chunks.jsonl) include table-heavy chunks
# that tokenize far denser than their character count suggests: chunk idx 790 (a
# medical BMI/height/weight table, source "Постановление Правительства РФ от
# 04_07_2013 N 565") is 1200 chars but 687 tokens on the reranker's tokenizer --
# combined with a realistic query it measured 826 tokens live on this machine, well
# past the default -ub 512 ("input (826 tokens) is too large to process. increase
# the physical batch size (current batch size: 512)"), and the `except Exception`
# in backend/ws/session.py's RAG_QUERY handling silently turns that into a fake
# "Простите, у меня сейчас сбой" instead of surfacing the failure. -ub 2048 (raised
# to match -b, since -b must be >= -ub) covers that with room to spare -- verified
# live against both the measured worst case (826 tokens) and an extreme one (the
# TRANSCRIPT_BUFFER_CHARS=5000 raw-transcript fallback query from
# dialogue/scenarios.yaml's "query (=intent.query | transcript)" paired with the
# same pathological chunk, ~1775 tokens, RAG_FUSED_TOP=50 candidates) -- both HTTP
# 200. VRAM cost: all three servers together measured 5786/8192 MiB at these
# settings on this machine, ~2.4 GB of headroom left.
$encoderArgs = @('-ngl', '99', '--no-cache-prompt', '-cram', '0', '-ctxcp', '0', '-cpent', '-1', '-b', '2048', '-ub', '2048')

$llmProc = Start-LlamaServer 'llm' $llmPort $llmArgs $llmPath
$embProc = Start-LlamaServer 'embedding' $embPort ($encoderArgs + @('--embedding')) $embPath
$rerProc = Start-LlamaServer 'reranker' $rerPort ($encoderArgs + @('--reranking')) $rerPath

@{ llm = $llmProc.Id; embedding = $embProc.Id; reranker = $rerProc.Id } | ConvertTo-Json | Set-Content $PidFile

# ─── Ждём /health каждого ──────────────────────────────────────────────────────
function Wait-Health([string]$Name, [string]$Port, [System.Diagnostics.Process]$Proc) {
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($Proc.HasExited) {
            $logPath = Join-Path $LogDir "$Name.log.err"
            Write-Host ''
            Write-Host "ОШИБКА: $Name (PID $($Proc.Id)) завершился при старте (код $($Proc.ExitCode))." -ForegroundColor Red
            if (Test-Path $logPath) {
                Write-Host "Последние строки $logPath :"
                Get-Content $logPath -Tail 20
            }
            return $false
        }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing
            if ($r.StatusCode -eq 200) {
                Write-Host "$Name : OK (127.0.0.1:$Port)" -ForegroundColor Green
                return $true
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
    Write-Host "ОШИБКА: $Name не ответил /health за $HealthTimeoutSec с." -ForegroundColor Red
    return $false
}

$ok = $true
$ok = (Wait-Health 'llm' $llmPort $llmProc) -and $ok
$ok = (Wait-Health 'embedding' $embPort $embProc) -and $ok
$ok = (Wait-Health 'reranker' $rerPort $rerProc) -and $ok

if ($ok) {
    Write-Host ''
    Write-Host 'Все три llama-server подняты и здоровы. Логи: scripts\logs\*.log' -ForegroundColor Green
    Write-Host 'Остановить: .\scripts\serve-models.ps1 -Stop'
    exit 0
} else {
    Write-Host ''
    Write-Host 'Не все серверы поднялись — см. ошибки выше. Работающие оставлены запущенными.' -ForegroundColor Red
    Write-Host 'Остановить всё: .\scripts\serve-models.ps1 -Stop'
    exit 1
}
