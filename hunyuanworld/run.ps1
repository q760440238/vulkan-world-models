param(
    [Parameter(Mandatory = $true)]
    [string]$Image,
    [string]$Out = "",
    [string]$Mode = "pad"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Out)) {
    $Out = Join-Path $here "out"
}

cargo build --release --bin hwm-vk --manifest-path (Join-Path $here "Cargo.toml")
& (Join-Path $here "target\release\hwm-vk.exe") --image $Image --out $Out --mode $Mode
