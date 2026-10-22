param(
    [string]$GameRoot = 'C:\Program Files (x86)\Steam\steamapps\common\A Dance of Fire and Ice',
    [string]$OutputDirectory = 'artifacts\phase31-stock-runtime-probe-build-r1'
)

$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot '..\tools\stock-runtime-probe\StockRuntimeProbe.csproj'
$sourceDirectory = Split-Path -Parent $project
$managed = Join-Path $GameRoot 'A Dance of Fire and Ice_Data\Managed'
$assembly = Join-Path $managed 'Assembly-CSharp.dll'
$harmony = Join-Path $managed 'UnityModManager\0Harmony.dll'
$umm = Join-Path $managed 'UnityModManager\UnityModManager.dll'
$rdTools = Join-Path $managed 'RDTools.dll'
foreach ($required in @($assembly, $harmony, $umm, $rdTools)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing required reference: $required" }
}

$absoluteOutput = [IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDirectory))
if (Test-Path -LiteralPath $absoluteOutput) { throw "Refusing to overwrite: $absoluteOutput" }
New-Item -ItemType Directory -Path $absoluteOutput | Out-Null
dotnet build $project -c Release -p:GameManagedDir="$managed" -o $absoluteOutput
if ($LASTEXITCODE -ne 0) { throw "dotnet build failed: $LASTEXITCODE" }
Copy-Item -LiteralPath (Join-Path $sourceDirectory 'Info.json') -Destination $absoluteOutput
Copy-Item -LiteralPath (Join-Path $sourceDirectory 'probe-config.json') -Destination $absoluteOutput
$dll = Join-Path $absoluteOutput 'DmdodStockRuntimeProbe.dll'
$result = [ordered]@{
    format = 'dmdod-phase31-probe-build-v0.1'
    built_utc = [DateTime]::UtcNow.ToString('o')
    game_root = $GameRoot
    assembly_reference_present = $true
    output_directory = $absoluteOutput
    dll_sha256 = (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash
    runtime_loaded = $false
    harmony_hooks_observed = $false
}
$result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $absoluteOutput 'build-result.json') -Encoding utf8
$result | ConvertTo-Json -Depth 4
