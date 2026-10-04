# third_party — vendored network definitions (Phase 3, P3-T2)

Only **architecture source files** needed to instantiate the already-acquired Tier A weights (P1-T9) are vendored.
Every file is **byte-identical** to the upstream file at the pinned commit. `src/fes_bench/archs.py` checks each
file's SHA-256 before loading it and supplies the missing training-time imports with inference-only shims, so no
vendored file is edited. No weights live here: weights stay in the git-ignored `models/` tree.

Fetched 2026-09-29 from `raw.githubusercontent.com/<repo>/<commit>/<path>`. The commit is the one recorded in
`docs/phase_1/model_acquisition_manifest.csv` for each model; BasicSR (not a Phase 1 asset) is pinned to the
commit that was on `master` on the fetch date.

| Dir | Upstream repo @ commit | Upstream path | License (copy kept here) | SHA-256 |
|---|---|---|---|---|
| `restoreformerpp/` | wzhouxiff/RestoreFormerPlusPlus @ `59d250f45d6b11a3e620d62e6e067ca1288f93a9` | `RestoreFormer/modules/vqvae/vqvae_arch.py` | Apache-2.0 (`LICENSE`) | `ba4a1275…` |
| `vqfr/` | TencentARC/VQFR @ `239c77a78152473eca83270c40be083bb0ae0f4a` | `vqfr/archs/vqfrv1_arch.py` | Apache-2.0 (`LICENSE`) | `63a1c500…` |
| | | `vqfr/archs/vqganv1_arch.py` | | `f705db43…` |
| | | `vqfr/archs/quantizer_arch.py` | | `f85f34de…` |
| | | `vqfr/ops/dcn/deform_conv.py` (only `ModulatedDeformConvPack`'s parameter layout is used; the forward runs `torchvision.ops.deform_conv2d`, no compiled extension) | | `ff5dc4db…` |
| `basicsr/` | XPixelGroup/BasicSR @ `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a` | `basicsr/archs/rrdbnet_arch.py` (RRDBNet for Real-ESRGAN x4plus) | Apache-2.0 (`LICENSE.txt`) | `6aa8eb3f…` |
| | | `basicsr/archs/arch_util.py` | | `be2cec64…` |
| `swinir/` | JingyunLiang/SwinIR @ `6545850fbf8df298df73d81f3e8cba638787c8bd` | `models/network_swinir.py` | Apache-2.0 (`LICENSE`) | `9e143898…` |
| `fbcnn/` | jiaxi-jiang/FBCNN @ `54d1831927506b3247e2d4d245abb4f4dab1a1cd` | `models/network_fbcnn.py` | Apache-2.0 (`LICENSE`) | `e1afc1f6…` |
| `nafnet/` | megvii-research/NAFNet @ `2b4af71ebe098a92a75910c233a3965a3e93ede4` | `basicsr/models/archs/NAFNet_arch.py`, `arch_util.py`, `local_arch.py` | MIT (`LICENSE`) | `01b22270…` / `5a11af2e…` / `c4df2ba4…` |
| `gfpgan/` (Phase 5 Rapid, **research-only**, 2026-10-04) | TencentARC/GFPGAN @ `7552a7791caad982045a7bbe5634bbf1cd5c8679` | `gfpgan/archs/gfpganv1_clean_arch.py`, `gfpgan/archs/stylegan2_clean_arch.py` | Apache-2.0 + NVIDIA NC / CC BY-NC-SA parts (`LICENSE`) → production REVIEW_REQUIRED | `576f44f3…` / `274f6a66…` |
| `codeformer/` (Phase 5 Rapid, **research-only**, 2026-10-04) | sczhou/CodeFormer @ `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b` | `basicsr/archs/codeformer_arch.py`, `basicsr/archs/vqgan_arch.py` | **S-Lab License 1.0 — non-commercial** (`LICENSE`) → never production without a commercial licence | `dfccf9e9…` / `5ef00125…` |

Full SHA-256 values: `VENDORED_SHA256` in `src/fes_bench/archs.py` (authoritative; checked at load time).

## Shims (in `src/fes_bench/archs.py`, not in the vendored files)

| Missing import | Shim | Why safe |
|---|---|---|
| `timm.models.layers` (`DropPath`, `to_2tuple`, `trunc_normal_`) | identity DropPath at inference; `torch.nn.init.trunc_normal_` | init functions are overwritten by the checkpoint; DropPath is identity in eval |
| `einops.rearrange` | two permute patterns used by the VQFR v1 quantizer | any other pattern raises |
| `sklearn.cluster` | raises | only used by codebook re-estimation during training |
| `vqfr.utils`, `basicsr.utils` (logger, registry, `all_gather`) | logger / dict registry / raises | no numerical effect |
| `basicsr.ops.dcn` (for BasicSR `arch_util`) | class that raises if instantiated | RRDBNet never builds a DCN |

## License notes

All vendored code is under permissive licenses that allow redistribution with the license text (kept alongside).
**This vendoring does not change any production-license status**: model weights and training data remain
`REVIEW_REQUIRED` for production (P1-T3 R3/R4) until a license reviewer is named (open item G-4).
