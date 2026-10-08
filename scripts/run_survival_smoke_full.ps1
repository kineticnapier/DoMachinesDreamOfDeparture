param(
    [switch]$SkipTests
)

$runner = Join-Path $PSScriptRoot "run_training.ps1"
& $runner -Config "configs\training\survival_smoke_full.toml" -SkipTests:$SkipTests
