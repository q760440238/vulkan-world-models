# End-to-end benchmark: t2s 1024x512, 20 steps.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "output/benchmark-t2s"
$t = Measure-Command {
    py -3 (Join-Path $root "worldgen/worldgen_t2s.py") `
        --prompt "a cozy wooden bedroom with warm lamps" `
        --out $out --width 1024 --height 512 --steps 20
}
"t2s 1024x512/20 steps: {0:N1} s" -f $t.TotalSeconds
