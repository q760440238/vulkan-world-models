# Build stable-diffusion.cpp with the Vulkan backend.
# Requires: Visual Studio 2022, CMake, Ninja, Vulkan SDK.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "third_party/stable-diffusion.cpp"
$build = Join-Path $root "third_party/stable-diffusion.cpp/build"

if (-not (Test-Path $src)) {
    git clone https://github.com/leejet/stable-diffusion.cpp $src
}
# Commit used for the validation runs.
git -C $src checkout 525fbe1

cmake -S $src -B $build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DSD_VULKAN=ON
cmake --build $build --config Release -j

New-Item -ItemType Directory -Force (Join-Path $root "tools") | Out-Null
Copy-Item (Join-Path $build "bin/sd-cli.exe") (Join-Path $root "tools/sd-cli.exe")
Write-Host "sd-cli -> tools/sd-cli.exe"
