param(
    [string]$Config = "configs\training\school_8h.toml",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([System.IO.Path]::IsPathRooted($Config)) {
    $configPath = [System.IO.Path]::GetFullPath($Config)
}
else {
    $configPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Config))
}
if (-not (Test-Path $configPath)) {
    throw "training config not found: $configPath"
}

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
$configStem = [System.IO.Path]::GetFileNameWithoutExtension($configPath)
$logPath = Join-Path $logDir ("training-" + $configStem + "-" + $stamp + ".log")

Write-Host "=== DMDOD configured training ==="
Write-Host "repo: $RepoRoot"
Write-Host "config: $configPath"
Write-Host "log: $logPath"

try {
    [void][DmdodPowerState]::SetThreadExecutionState(
        $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
    )

    if (-not $SkipTests) {
        Write-Host "=== pytest preflight ==="
        $testArgs = @(
            "run", "--no-sync", "python", "-m", "pytest",
            "tests/test_training_config.py",
            "tests/test_training_budget.py",
            "tests/test_training_action_trust.py",
            "tests/test_training_real_chart.py",
            "tests/test_training_trajectory_trust.py",
            "tests/test_n_key_capacity.py",
            "tests/test_training_architecture.py",
            "tests/test_fly_connectome_policy.py",
            "tests/test_human_visible_policy.py",
            "tests/test_fly_connectome_cuda_no_grad_runtime.py",
            "-q"
        )
        & uv @testArgs
        if ($LASTEXITCODE -ne 0) {
            throw "pytest preflight failed with exit code $LASTEXITCODE; training was not started"
        }
    }

    $trainArgs = @(
        "run", "--no-sync", "python", "-u",
        "scripts/train_real_chart.py",
        "--config", $configPath
    )

    $quotedArgs = $trainArgs | ForEach-Object {
        '"' + ([string]$_).Replace('"', '""') + '"'
    }
    $cmdLine = 'uv ' + ($quotedArgs -join ' ') + ' 2>&1'

    Write-Host "=== starting training ==="
    & cmd.exe /d /s /c $cmdLine | Tee-Object -FilePath $logPath
    $trainExit = $LASTEXITCODE
    if ($trainExit -ne 0) {
        throw "training failed with exit code $trainExit; see $logPath"
    }

    Write-Host "=== training finished ==="
    Write-Host "log: $logPath"
}
finally {
    [void][DmdodPowerState]::SetThreadExecutionState($ES_CONTINUOUS)
}
