param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_training.ps1") `
    -Config "configs\training\human_visible_smoke.toml" `
    -SkipTests:$SkipTests
