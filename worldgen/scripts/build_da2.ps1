# Build ncnn (Vulkan, shared) and the DA-2 runner.
# Requires: Visual Studio 2022, CMake, Vulkan SDK, prebuilt ncnn source.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ncnn = Join-Path $root "third_party/ncnn"
$build = Join-Path $root "third_party/ncnn/build"

if (-not (Test-Path $ncnn)) {
    git clone https://github.com/Tencent/ncnn $ncnn
}
git -C $ncnn submodule update --init --recursive

cmake -S $ncnn -B $build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DNCNN_VULKAN=ON `
  -DNCNN_SHARED_LIB=ON `
  -DNCNN_BUILD_TOOLS=OFF `
  -DNCNN_BUILD_EXAMPLES=OFF
cmake --build $build -j

$ncnnSrc = Join-Path $ncnn "src"
$ncnnBuildSrc = Join-Path $build "src"
$vulkan = (Get-ChildItem "C:/VulkanSDK/*/Lib/vulkan-1.lib" | Sort-Object FullName -Descending | Select-Object -First 1).FullName
$vulkanInc = (Get-ChildItem "C:/VulkanSDK/*/Include" | Sort-Object FullName -Descending | Select-Object -First 1).FullName
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$vcvars = Join-Path $vs "VC/Auxiliary/Build/vcvars64.bat"

New-Item -ItemType Directory -Force (Join-Path $root "tools/da2-ncnn") | Out-Null
Push-Location $root
cmd /c "call `"$vcvars`" >nul && cl /nologo /std:c++17 /EHsc /O2 /MD /DNCNN_VULKAN /I`"$ncnnSrc`" /I`"$ncnnBuildSrc`" /I`"$vulkanInc`" da2/main.cpp `"$build/src/ncnn.lib`" `"$vulkan`" /Fe:tools/da2-ncnn/da2ncnn.exe"
Pop-Location
Copy-Item (Join-Path $build "src/ncnn.dll") (Join-Path $root "tools/da2-ncnn/ncnn.dll") -Force
Write-Host "da2ncnn -> tools/da2-ncnn/da2ncnn.exe"
