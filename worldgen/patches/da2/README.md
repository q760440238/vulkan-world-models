# DA-2 export patches

These files are drop-in replacements for the upstream DA-2 model code. They
make the graph exportable by pnnx/ncnn:

- `vit_w_esphere.py`: removes `torch.quantile`; normalization is done by the
  caller.
- `base.py`: replaces fused KV + `unbind` with separate K/V linear projections.
- `dinov2/attention.py`: replaces fused QKV + indexing with separate Q/K/V
  projections.

Copy them over the corresponding files in `DA-2/src/da2/model/` before running
`scripts/export_da2.py`. They are derived from DA-2 and remain under DA-2's
upstream license.
