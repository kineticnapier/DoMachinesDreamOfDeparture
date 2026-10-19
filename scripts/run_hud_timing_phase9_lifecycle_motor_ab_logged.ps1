param(
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [Parameter(Mandatory = $true)]
    [string]$StdoutLog,
    [Parameter(Mandatory = $true)]
    [string]$StderrLog,
    [Parameter(Mandatory = $true)]
    [string]$StatusJson,
    [double]$WallTimeoutS = 180.0
)

$ErrorActionPreference = "Stop"
$started = Get-Date
$exitCode = -1
$state = "failed"
try {
    & "$PSScriptRoot\..\.venv\Scripts\python.exe" -u `
        "$PSScriptRoot\run_hud_timing_phase9_lifecycle_motor_ab_v01.py" `
        --output $Output --wall-timeout-s $WallTimeoutS `
        1> $StdoutLog 2> $StderrLog
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        $state = "completed"
    }
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $StderrLog
    $exitCode = 1
}
finally {
    $finished = Get-Date
    @{
        pid = $PID
        started_local = $started.ToString("o")
        finished_local = $finished.ToString("o")
        exit_code = $exitCode
        output = $Output
        stdout = $StdoutLog
        stderr = $StderrLog
        wall_timeout_s_per_arm = $WallTimeoutS
        status = $state
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusJson -Encoding utf8
}
exit $exitCode
