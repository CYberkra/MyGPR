param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$InnoCompiler,
    [Parameter(Mandatory=$true)][string]$SourceRef,
    [Parameter(Mandatory=$true)][string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
& $Python "$PSScriptRoot/build.py" --source-ref $SourceRef --output $OutputRoot --inno $InnoCompiler
if ($LASTEXITCODE -ne 0) { throw "Windows build failed: $LASTEXITCODE" }
