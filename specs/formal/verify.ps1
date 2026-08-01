# Run Quint verification (exhaustive, via Apalache) on this machine.
#
# Why this wrapper exists: `quint verify` shells out to Apalache's launcher, which on
# Windows is a .bat. Node 20+ refuses to spawn .bat/.cmd without a shell (the fix for
# CVE-2024-27980), so quint dies with `EINVAL spawn` before Apalache ever starts.
# Workaround: run the Apalache JAR ourselves in server mode and point quint at it with
# --server-endpoint. Everything else (typecheck, run) works with the plain CLI.
#
# Usage:
#   .\specs\formal\verify.ps1                       # verify every .qnt with its invariants
#   .\specs\formal\verify.ps1 -Spec dialogue.qnt    # just one
#   .\specs\formal\verify.ps1 -Steps 30

param(
    [string]$Spec = "",
    [int]$Steps = 20,
    [int]$Port = 8822
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$jar = Join-Path $env:USERPROFILE ".quint\apalache-dist-0.56.1\apalache\lib\apalache.jar"

if (-not (Test-Path $jar)) {
    Write-Error "Apalache jar not found at $jar. Run `quint verify` once to let it download, then re-run this script."
}

# Reuse an already-running server if there is one; otherwise start it.
$started = $false
$inUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $inUse) {
    Write-Host "starting Apalache server on :$Port ..." -ForegroundColor Cyan
    $proc = Start-Process -FilePath "java" -ArgumentList @("-jar", $jar, "server", "--port=$Port") `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput "$env:TEMP\apalache-server.log" `
        -RedirectStandardError "$env:TEMP\apalache-server.err"
    $started = $true
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { break }
    }
} else {
    Write-Host "reusing Apalache server already listening on :$Port" -ForegroundColor DarkGray
}

try {
    $specs = if ($Spec) { @(Join-Path $here $Spec) }
             else { Get-ChildItem -Path $here -Filter "*.qnt" | Where-Object { $_.Name -notlike "_*" } | ForEach-Object { $_.FullName } }

    $failed = @()
    foreach ($s in $specs) {
        $name = Split-Path -Leaf $s
        # Invariants are declared as `val inv_<something> = ...` by convention in this repo,
        # so the wrapper discovers them instead of hardcoding a list that silently rots.
        $invs = Select-String -Path $s -Pattern '^\s*val\s+(inv_\w+)' -AllMatches |
                ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        if (-not $invs) {
            Write-Host "$name : no `val inv_*` declarations, skipping" -ForegroundColor Yellow
            continue
        }
        Write-Host "`n=== $name ===" -ForegroundColor Cyan
        & quint typecheck $s
        if ($LASTEXITCODE -ne 0) { $failed += "$name (typecheck)"; continue }

        foreach ($inv in $invs) {
            Write-Host "  verify $inv ..." -NoNewline
            $out = & quint verify $s --invariant=$inv --max-steps=$Steps --server-endpoint="localhost:$Port" 2>&1 | Out-String
            if ($out -match '\[ok\]') {
                Write-Host " ok" -ForegroundColor Green
            } else {
                Write-Host " FAILED" -ForegroundColor Red
                Write-Host $out
                $failed += "$name :: $inv"
            }
        }
    }

    Write-Host ""
    if ($failed.Count -gt 0) {
        Write-Host "НЕ ПРОШЛИ:" -ForegroundColor Red
        $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
        exit 1
    }
    Write-Host "все инварианты проверены" -ForegroundColor Green
} finally {
    if ($started -and $proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}
