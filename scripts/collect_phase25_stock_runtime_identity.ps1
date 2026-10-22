param(
    [Parameter(Mandatory = $true)]
    [string]$AssemblyCSharpPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$resolvedDll = (Resolve-Path -LiteralPath $AssemblyCSharpPath).Path
if ([IO.Path]::GetFileName($resolvedDll) -ne 'Assembly-CSharp.dll') {
    throw 'AssemblyCSharpPath must name Assembly-CSharp.dll'
}
$resolvedOutput = [IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Refusing to overwrite existing output: $resolvedOutput"
}
$hash = Get-FileHash -LiteralPath $resolvedDll -Algorithm SHA256
$record = [ordered]@{
    format = 'dmdod-phase25-stock-runtime-identity-v0.1'
    captured_utc = [DateTimeOffset]::UtcNow.ToString('o')
    assembly_csharp_path = $resolvedDll
    assembly_csharp_sha256 = $hash.Hash.ToUpperInvariant()
    audited_sha256 = '0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD'
    hash_matches_audited_dll = $hash.Hash -eq '0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD'
    runtime_behavior_verified = $false
}
$record | ConvertTo-Json | Set-Content -LiteralPath $resolvedOutput -Encoding utf8NoBOM
Write-Output $resolvedOutput
