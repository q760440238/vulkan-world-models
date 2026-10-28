$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root "gpu-splat")
cargo build --release
Pop-Location
New-Item -ItemType Directory -Force (Join-Path $root "tools/gpu-splat") | Out-Null
Copy-Item (Join-Path $root "gpu-splat/target/release/vulkan-splat.exe") (Join-Path $root "tools/gpu-splat/gpu-splat.exe") -Force
Write-Host "gpu-splat -> tools/gpu-splat/gpu-splat.exe"
