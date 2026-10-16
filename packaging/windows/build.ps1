param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$InnoCompiler,
    [string]$SourceRef = 'v0.11.0',
    [string]$OutputRoot = 'output/windows-installer-v0.11.0'
)
$ErrorActionPreference = 'Stop'
& $Python "$PSScriptRoot/build.py" --source-ref $SourceRef --output $OutputRoot --inno $InnoCompiler
if ($LASTEXITCODE -ne 0) { throw "Windows build failed: $LASTEXITCODE" }
