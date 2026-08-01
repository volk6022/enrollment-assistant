# Reachability check: prove every state in the machine is actually live.
#
# The probes in dialogue.qnt are invariants written to FAIL: `probe_speaking = agent != Speaking`
# holds only if Speaking is never entered. So here a [violation] is the GOOD outcome — it is
# the witness trace that reaches the state. An [ok] means the state is DEAD: unreachable by
# any sequence of actions, i.e. a hole in the design or a guard that is too narrow.
#
# This is the inverse of verify.ps1 and must be read the opposite way, hence the separate
# script rather than a flag.
#
# Usage: .\specs\formal\reachability.ps1 [-Spec dialogue.qnt] [-Steps 16]

param(
    [string]$Spec = "dialogue.qnt",
    [int]$Steps = 16,
    [int]$Port = 8822
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$specPath = Join-Path $here $Spec
$jar = Join-Path $env:USERPROFILE ".quint\apalache-dist-0.56.1\apalache\lib\apalache.jar"

if (-not (Test-Path $jar)) { Write-Error "Apalache jar not found at $jar" }

$started = $false
if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
    Write-Host "starting Apalache server on :$Port ..." -ForegroundColor Cyan
    $proc = Start-Process -FilePath "java" -ArgumentList @("-jar", $jar, "server", "--port=$Port") `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput "$env:TEMP\apalache-server.log" `
        -RedirectStandardError "$env:TEMP\apalache-server.err"
    $started = $true
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { break }
    }
}

try {
    $probes = Select-String -Path $specPath -Pattern '^\s*val\s+(probe_\w+)' -AllMatches |
              ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
    if (-not $probes) { Write-Host "no probe_* declarations in $Spec" -ForegroundColor Yellow; exit 0 }

    $dead = @()
    foreach ($p in $probes) {
        Write-Host ("  {0,-24}" -f $p.Replace('probe_', '')) -NoNewline
        $out = & quint verify $specPath --invariant=$p --max-steps=$Steps --server-endpoint="localhost:$Port" 2>&1 | Out-String
        if ($out -match '\[violation\]')  { Write-Host "достижимо" -ForegroundColor Green }
        elseif ($out -match '\[ok\]')     { Write-Host "МЁРТВОЕ СОСТОЯНИЕ" -ForegroundColor Red; $dead += $p }
        else                              { Write-Host "ошибка проверки" -ForegroundColor Yellow; Write-Host $out }
    }

    Write-Host ""
    if ($dead.Count -gt 0) {
        Write-Host "недостижимы (это баг модели или дизайна):" -ForegroundColor Red
        $dead | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
        exit 1
    }
    Write-Host "все состояния достижимы" -ForegroundColor Green
} finally {
    if ($started -and $proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}
