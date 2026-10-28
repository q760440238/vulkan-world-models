$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root "rust")
cargo build --release
Pop-Location
New-Item -ItemType Directory -Force (Join-Path $root "tools/worldgen-vk") | Out-Null
Copy-Item (Join-Path $root "rust/target/release/worldgen-vk.exe") (Join-Path $root "tools/worldgen-vk/worldgen-vk.exe") -Force
Write-Host "worldgen-vk -> tools/worldgen-vk/worldgen-vk.exe"
