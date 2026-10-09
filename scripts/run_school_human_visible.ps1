param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_training.ps1") `
    -Config "configs\training\school_human_visible_8h.toml" `
    -SkipTests:$SkipTests
