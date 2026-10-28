param(
    [Parameter(Mandatory = $true)]
    [string]$RepoUrl,
    [string]$Message = "HunyuanWorld-Mirror Vulkan inference"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here

if (-not (Test-Path (Join-Path $root ".git"))) {
    git -C $root init -b main
}

# Never publish the 5 GB weights or build artifacts.
if (Test-Path (Join-Path $root "weights")) {
    Write-Host "weights/ exists locally and is ignored; it will not be committed."
}

git -C $root add .
git -C $root commit -m $Message
$remotes = git -C $root remote
if ($remotes -contains "origin") {
    git -C $root remote set-url origin $RepoUrl
} else {
    git -C $root remote add origin $RepoUrl
}
git -C $root push -u origin main
