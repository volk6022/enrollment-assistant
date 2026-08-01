# scripts/run-acceptance.ps1 — T-12: прогон приёмочных A-01…A-13 ОДНОЙ командой
# (tasks.md T-12's own acceptance: "прогон приёмочных A-01…A-13 одной командой").
#
# Что запускает и что покрывает какой сценарий (полная разбивка и цифры — в
# отчёте задачи T-12):
#
#   tests/unit         — без внешней инфры, всегда.
#                         A-03/A-04/A-08 (dialogue.qnt автомат, T-07) +
#                         A-13 unit-слой (RagArtifactsMissingError, T-12).
#   tests/contract      — живой llama-server (три порта из .env). Уже умеет
#                         сам чисто скипаться (pytest.mark.skipif), если сервер
#                         недостижим — этот скрипт просто это не глушит.
#                         NFR-01 (TTFT), NFR-02 (инкрементальный prefill),
#                         llm.md §6 (cached_tokens, не факт cache_prompt).
#   tests/integration    — живой llama-server + CUDA (whisper) + собранный
#                         RAG-индекс. `tests/integration/conftest.py`
#                         НАРОЧНО не скипает сам себя при недостающей инфре
#                         (её собственный докстринг: "green run supposed to
#                         mean the acceptance scenarios were actually
#                         exercised end to end, not skipped") — поэтому ЭТОТ
#                         скрипт проверяет инфру САМ, до вызова pytest, и
#                         скипает уровень целиком с понятной причиной, а не
#                         даёт tests/integration упасть стек-трейсом
#                         "connection refused"/"CUDA error".
#                         A-01/A-02/A-05/A-07/A-09/A-10/A-11.
#   tests/load          — CUDA + omegaconf (silero_tts). Тоже сам скипается
#                         чисто (pytest.mark.skipif) при отсутствии GPU.
#                         NFR-03 (STT RTF), NFR-04 (event loop stall).
#
#   A-12 — GUI, вручную по docs/gui-spec-current.md §2/§3 (задача прямо
#          говорит: "автоматизировать не требую"). Не запускается отсюда.
#   A-13 — тест уровня "чистый клон → up -d → сервис не падает" был проверен
#          РЕАЛЬНЫМ запуском (`python -m backend.app` на временно
#          переименованном backend/rag/artifacts/, до и после) при разработке
#          T-12 — см. отчёт задачи; не автоматизирован здесь, потому что
#          требует полного холодного старта whisper/silero (GPU, десятки
#          секунд) ради проверки одной строки лога. `tests/unit/
#          test_rag_index_missing.py` покрывает слой, который РЕАЛЬНО бросает
#          исключение (`Indexes.__init__`), без повторения всего lifespan.
#
# Использование:
#   .\scripts\run-acceptance.ps1                    # всё, что доступно
#   .\scripts\run-acceptance.ps1 -SkipIntegration    # без tests/integration (быстрее)
#   .\scripts\run-acceptance.ps1 -SkipLoad           # без tests/load

[CmdletBinding()]
param(
    [switch]$SkipIntegration,
    [switch]$SkipLoad
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $Python)) {
    Write-Error "$Python не найден — сначала подними venv проекта (см. README.md), не глобальный python."
    exit 1
}

function Test-TcpPort {
    param([string]$HostName, [int]$Port)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $connectTask = $client.ConnectAsync($HostName, $Port)
        $ok = $connectTask.Wait(1500)
        $client.Close()
        return $ok
    } catch {
        return $false
    }
}

function Test-CudaVisible {
    $out = & $Python -c "import torch; print(torch.cuda.is_available())" 2>$null
    return ($LASTEXITCODE -eq 0 -and $out -match "True")
}

$results = [ordered]@{}

function Invoke-Tier {
    param([string]$Name, [string]$Path, [string[]]$ExtraArgs = @('-q'))
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Python -m pytest $Path @ExtraArgs
    $script:results[$Name] = $LASTEXITCODE
}

function Skip-Tier {
    param([string]$Name, [string]$Reason)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    Write-Host "SKIPPED: $Reason" -ForegroundColor DarkYellow
    $script:results[$Name] = "skipped"
}

# -- tier 1: unit (always runs, no external infra) --------------------------
Invoke-Tier -Name "tests/unit (A-03/A-04/A-08, A-13 unit layer)" -Path "tests/unit"

# -- infra probe --------------------------------------------------------------
$llmUp = Test-TcpPort -HostName "127.0.0.1" -Port 20099
$embUp = Test-TcpPort -HostName "127.0.0.1" -Port 20100
$rerUp = Test-TcpPort -HostName "127.0.0.1" -Port 20101
$llamaUp = $llmUp -and $embUp -and $rerUp
$cudaUp = Test-CudaVisible

Write-Host "`nInfra probe: llama-server(20099)=$llmUp embedding(20100)=$embUp reranker(20101)=$rerUp CUDA=$cudaUp" -ForegroundColor DarkGray

# -- tier 2: contract ---------------------------------------------------------
# The file self-skips (pytest.mark.skipif) if unreachable -- always safe to invoke.
Invoke-Tier -Name "tests/contract (NFR-01, NFR-02, llm.md cached_tokens)" -Path "tests/contract" -ExtraArgs @('-q', '-s')

# -- tier 3: integration -------------------------------------------------------
if ($SkipIntegration) {
    Skip-Tier -Name "tests/integration (A-01/A-02/A-05/A-07/A-09/A-10/A-11)" -Reason "-SkipIntegration passed"
} elseif (-not $llamaUp) {
    Skip-Tier -Name "tests/integration (A-01/A-02/A-05/A-07/A-09/A-10/A-11)" `
        -Reason "llama-server not reachable on 127.0.0.1:20099/20100/20101 -- start with .\scripts\serve-models.ps1"
} elseif (-not $cudaUp) {
    Skip-Tier -Name "tests/integration (A-01/A-02/A-05/A-07/A-09/A-10/A-11)" `
        -Reason "no CUDA device visible to torch -- whisper requires GPU (docs/streaming-research-findings.md §2)"
} else {
    Invoke-Tier -Name "tests/integration (A-01/A-02/A-05/A-07/A-09/A-10/A-11)" -Path "tests/integration" -ExtraArgs @('-q', '-s')
}

# -- tier 4: load ---------------------------------------------------------------
# The file self-skips (pytest.mark.skipif) if no CUDA/omegaconf -- always safe.
if ($SkipLoad) {
    Skip-Tier -Name "tests/load (NFR-03, NFR-04)" -Reason "-SkipLoad passed"
} else {
    Invoke-Tier -Name "tests/load (NFR-03, NFR-04)" -Path "tests/load" -ExtraArgs @('-q', '-s')
}

# -- summary ----------------------------------------------------------------
Write-Host "`n`n=== Summary ===" -ForegroundColor Cyan
foreach ($key in $results.Keys) {
    $v = $results[$key]
    if ($v -eq "skipped") {
        Write-Host ("  SKIP  " + $key) -ForegroundColor DarkYellow
    } elseif ($v -eq 0) {
        Write-Host ("  PASS  " + $key) -ForegroundColor Green
    } else {
        Write-Host ("  FAIL  " + $key + " (exit $v)") -ForegroundColor Red
    }
}
Write-Host "`nA-12 (GUI parity) -- manual only, docs/gui-spec-current.md §2/§3. Not run here." -ForegroundColor DarkGray
Write-Host "A-13 (clean-clone startup) -- verified by a real `python -m backend.app` run during T-12" -ForegroundColor DarkGray
Write-Host "               (rename backend/rag/artifacts/, confirm no crash + clear /health.rag message," -ForegroundColor DarkGray
Write-Host "               restore, confirm clean startup). See the T-12 delivery report for the transcript." -ForegroundColor DarkGray

$anyFail = $results.Values | Where-Object { $_ -ne "skipped" -and $_ -ne 0 }
if ($anyFail) { exit 1 } else { exit 0 }
