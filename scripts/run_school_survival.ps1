param(
    [switch]$SkipTests
)

$runner = Join-Path $PSScriptRoot "run_training.ps1"
& $runner -Config "configs\training\school_survival_8h.toml" -SkipTests:$SkipTests
