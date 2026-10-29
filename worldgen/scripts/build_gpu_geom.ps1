$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root "gpu-geom")
cargo build --release
Pop-Location
New-Item -ItemType Directory -Force (Join-Path $root "tools/gpu-geom") | Out-Null
Copy-Item (Join-Path $root "gpu-geom/target/release/gpu-geom.exe") (Join-Path $root "tools/gpu-geom/gpu-geom.exe") -Force
Write-Host "gpu-geom -> tools/gpu-geom/gpu-geom.exe"
