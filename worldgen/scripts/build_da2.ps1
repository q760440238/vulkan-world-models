param([string]$Source = "")

# Build ncnn (Vulkan, shared) and the DA-2 runner.
# Requires: Visual Studio 2022, CMake, Vulkan SDK.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -property installationPath
$cmakeDir = Join-Path $vs "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin"
$env:PATH = "$cmakeDir;$env:PATH"
$vulkanSdk = Get-ChildItem "C:/VulkanSDK/*" -Directory | Sort-Object Name -Descending | Select-Object -First 1
if ($vulkanSdk) { $env:VULKAN_SDK = $vulkanSdk.FullName }

if ($Source -ne "") {
    $ncnn = (Resolve-Path $Source).Path
} else {
    $ncnn = Join-Path $root "third_party/ncnn"
    if (-not (Test-Path $ncnn)) { git clone https://github.com/Tencent/ncnn $ncnn }
    git -C $ncnn submodule update --init --recursive
}
$build = Join-Path $root "build-ncnn"

cmake -S $ncnn -B $build -G "Visual Studio 17 2022" -A x64 `
  -DNCNN_VULKAN=ON `
  -DNCNN_SHARED_LIB=ON `
  -DNCNN_BUILD_TOOLS=OFF `
  -DNCNN_BUILD_EXAMPLES=OFF
cmake --build $build --config Release --parallel

$ncnnSrc = Join-Path $ncnn "src"
$ncnnBuildSrc = Join-Path $build "src"
$lib = Join-Path $build "src/Release/ncnn.lib"
if (-not (Test-Path $lib)) { $lib = Join-Path $build "src/ncnn.lib" }
$dll = Join-Path $build "src/Release/ncnn.dll"
if (-not (Test-Path $dll)) { $dll = Join-Path $build "src/ncnn.dll" }
$vulkan = (Get-ChildItem "C:/VulkanSDK/*/Lib/vulkan-1.lib" | Sort-Object FullName -Descending | Select-Object -First 1).FullName
$vulkanInc = (Get-ChildItem "C:/VulkanSDK/*/Include" | Sort-Object FullName -Descending | Select-Object -First 1).FullName
$vcvars = Join-Path $vs "VC/Auxiliary/Build/vcvars64.bat"

New-Item -ItemType Directory -Force (Join-Path $root "tools/da2-ncnn") | Out-Null
Push-Location $root
cmd /c "call `"$vcvars`" >nul && cl /nologo /std:c++17 /EHsc /O2 /MD /DNCNN_VULKAN /I`"$ncnnSrc`" /I`"$ncnnBuildSrc`" /I`"$vulkanInc`" da2/main.cpp `"$lib`" `"$vulkan`" /Fe:tools/da2-ncnn/da2ncnn.exe"
Pop-Location
Copy-Item $dll (Join-Path $root "tools/da2-ncnn/ncnn.dll") -Force
Write-Host "da2ncnn -> tools/da2-ncnn/da2ncnn.exe"
