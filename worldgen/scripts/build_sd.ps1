param([string]$Source = "")

# Build stable-diffusion.cpp with the Vulkan backend.
# Requires: Visual Studio 2022, CMake, Ninja, Vulkan SDK.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$cmakeDir = Join-Path $vs "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin"
$ninjaDir = Join-Path $vs "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"
$env:PATH = "$cmakeDir;$ninjaDir;$env:PATH"
$vulkanSdk = Get-ChildItem "C:/VulkanSDK/*" -Directory | Sort-Object Name -Descending | Select-Object -First 1
if ($vulkanSdk) { $env:VULKAN_SDK = $vulkanSdk.FullName }
if ($Source -ne "") {
    $src = (Resolve-Path $Source).Path
} else {
    $src = Join-Path $root "third_party/stable-diffusion.cpp"
    if (-not (Test-Path $src)) {
        git clone https://github.com/leejet/stable-diffusion.cpp $src
    }
    # Commit used for the validation runs.
    git -C $src checkout 525fbe1
}
$build = Join-Path $root "build-sd"

cmake -S $src -B $build -G "Visual Studio 17 2022" -A x64 `
  -DSD_VULKAN=ON
cmake --build $build --config Release --parallel

New-Item -ItemType Directory -Force (Join-Path $root "tools") | Out-Null
$exe = Join-Path $build "bin/Release/sd-cli.exe"
if (-not (Test-Path $exe)) { $exe = Join-Path $build "bin/sd-cli.exe" }
Copy-Item $exe (Join-Path $root "tools/sd-cli.exe")
Write-Host "sd-cli -> tools/sd-cli.exe"
