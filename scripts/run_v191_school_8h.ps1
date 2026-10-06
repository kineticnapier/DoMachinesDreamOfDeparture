param(
    [double]$Hours = 8.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not ("DmdodPowerState" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class DmdodPowerState
{
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern UInt32 SetThreadExecutionState(UInt32 esFlags);
}
"@
}

$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1

$logDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDir ("v191-school-" + $stamp + ".log")

Write-Host "=== DMDOD school run preflight ==="
Write-Host "repo: $RepoRoot"
Write-Host "budget: $Hours h"
Write-Host "log: $logPath"

try {
    [void][DmdodPowerState]::SetThreadExecutionState(
        $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
    )

    Write-Host "=== pytest preflight ==="
    & uv run --no-sync python -m pytest `
        tests/test_v190_actor_action_trust_dagger.py `
        tests/test_v191_budgeted_action_trust_dagger.py `
        tests/test_modern_cli_dagger.py `
        tests/test_fly_connectome_policy.py `
        tests/test_fly_connectome_cuda_no_grad_runtime.py `
        -q

    if ($LASTEXITCODE -ne 0) {
        throw "pytest preflight failed with exit code $LASTEXITCODE; training was not started"
    }

    Write-Host "=== starting unattended v1.9.1 budget run ==="
    $trainArgs = @(
        "run", "--no-sync", "python", "-u",
        "scripts/train_real_chart_v191_n_key_connectome_budgeted_action_trust_dagger.py",
        "data\DMDOD-v140-tuf-p11-p20",
        "checkpoints\real_chart_v172_fly4096_alpha_probe2.pt",
        "--output", "checkpoints\real_chart_v191_fly4096_school8h.pt",
        "--hours", ([string]$Hours),
        "--reserve-minutes", "8",
        "--max-trials", "10000",
        "--actor-steps", "8",
        "--anchor-limit", "20",
        "--validation-limit", "20",
        "--lr", "0.0003",
        "--stay-coef", "20",
        "--initial-action-rms", "0.01",
        "--min-action-rms", "0.00001",
        "--reject-shrink", "0.5",
        "--safe-grow", "1.25",
        "--chunk-steps", "192",
        "--device", "cuda"
    )

    # Windows PowerShell turns native stderr records into NativeCommandError
    # objects when stderr is merged in the PowerShell pipeline. PyTorch writes
    # harmless warnings to stderr, so merge stderr inside cmd.exe before
    # PowerShell sees the stream.
    $quotedArgs = $trainArgs | ForEach-Object {
        '"' + ([string]$_).Replace('"', '""') + '"'
    }
    $cmdLine = 'uv ' + ($quotedArgs -join ' ') + ' 2>&1'
    & cmd.exe /d /s /c $cmdLine | Tee-Object -FilePath $logPath
    $trainExit = $LASTEXITCODE
    if ($trainExit -ne 0) {
        throw "training failed with exit code $trainExit; see $logPath"
    }

    Write-Host "=== school run finished ==="
    Write-Host "checkpoint: checkpoints\real_chart_v191_fly4096_school8h.pt"
    Write-Host "log: $logPath"
}
finally {
    [void][DmdodPowerState]::SetThreadExecutionState($ES_CONTINUOUS)
}
