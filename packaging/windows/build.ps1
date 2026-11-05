param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$InnoCompiler,
    [Parameter(Mandatory=$true)][string]$SourceRef,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [string]$UpdatePublicKeyFile,
    [string]$PreviewFeed,
    [string]$CertificateThumbprint
)
$ErrorActionPreference = 'Stop'
$buildArgs = @("$PSScriptRoot/build.py", '--source-ref', $SourceRef, '--output', $OutputRoot, '--inno', $InnoCompiler)
if ($UpdatePublicKeyFile) { $buildArgs += @('--update-public-key-file', $UpdatePublicKeyFile) }
if ($PreviewFeed) { $buildArgs += @('--preview-feed', $PreviewFeed) }
if ($CertificateThumbprint) { $buildArgs += @('--certificate-thumbprint', $CertificateThumbprint) }
& $Python @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Windows build failed: $LASTEXITCODE" }
