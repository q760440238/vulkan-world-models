# Model manifest

Weights are not distributed in this repository. Sizes/hashes below are from
the validation machine.

| Path | Size (bytes) | SHA256 |
| --- | ---: | --- |
| `flux-dev/flux1-dev-q4_k.gguf` | 6,925,526,176 | `962d536a2a51342b4793e46cb2c491f3ea5fa62a66fbfc38684638fa7bff646b` |
| `flux-dev/ae.safetensors` | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |
| `flux-dev/clip_l.safetensors` | 246,144,152 | `660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd` |
| `flux-dev/t5xxl_fp8_e4m3fn.safetensors` | 4,893,934,904 | `315fadec5f39e44cdaa20af56446fcb1c7c9b17091209ee7ba0f03d47160518f` |
| `flux-dev/worldgen_text2scene.safetensors` | 34,647,632 | `5349c785650892914b32e74d88d16baa5863981a63a8747b70332e67e0feb10b` |
| `flux-dev/worldgen_img2scene.safetensors` | 44,917,720 | `b10a1fa212fcff58c87a8264b1b9a8c01853febcd4d3f5c239ab7d7fd35d17ef` |
| `flux-fill/flux1-fill-dev.safetensors` | 23,804,922,408 | `03e289f530df51d014f48e675a9ffa2141bc003259bf5f25d75b957e920a41ca` |
| `flux-fill/flux-fill-official-q4.gguf` | 6,793,133,088 | `096c26692c17a99a8e86961bcc3aafe1de01af7ee7a1b56218a01f6c48624c5d` |
| `da2/model.safetensors` | 1,378,513,064 | `d8ea568fc3dfb7d7432e5b763de499dd03fb6dc1c2020d84639f35e2dfa4f78e` |
| `da2/da2_trace_1024.ncnn.param` | 46,856 | `8e24dd92f2bd6beb8e5053f798243aa7e9c2cc8d2884d7d5007d06743ac17c5c` |
| `da2/da2_trace_1024.ncnn.bin` | 744,102,084 | `e7773c23b2d117ac9f5085a92e8df06b46abba05011ac90da33c65b53ea479d9` |
| `da2/da2_trace_1600.ncnn.param` | 46,856 | `615c3ee66f625c0831e9b7200dc2f0afa929d6589f6ea6a06cdbb83a77ce0966` |
| `da2/da2_trace_1600.ncnn.bin` | 748,136,644 | `6bd2c96f93258cc5ff8821ce9db7780d331f11902aa4a4ab187f2fbcb1e0e5ba` |

Notes:

- The q4 filenames are local; `download_models.py` documents the expected
  upstream sources.
- `flux-fill-official-q4.gguf` is produced from the official safetensors with
  `sd-cli --mode convert -m ... --type q4_K`.
- FLUX weights are subject to the Black Forest Labs non-commercial license and
  are not redistributed here.
