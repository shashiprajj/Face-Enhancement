# P3-T2 — Candidate Model Adapters & Environment Validation

| Field | Value |
|---|---|
| Phase / Task | Phase 3 / P3-T2 |
| Status | Complete |
| Date | 2026-09-29 |
| Adapter code | `src/fes_bench/adapters.py`, `src/fes_bench/archs.py`, `src/fes_bench/geometry.py` |
| Vendored architectures | `third_party/` (byte-identical upstream files + licenses; `third_party/PROVENANCE.md`) |
| Validation report | `data/manifests/phase3/model_validation_report.json` (`scripts/p3_validate_models.py`) |
| Tests | `tests/test_bench_harness.py` — 23 passed (geometry, mask, paste-back, baseline contract/determinism) |
| Environment | `fes-dev:cpu-p2` (unchanged from Phase 2; no new packages installed) |

---

## 1. What Was Missing After Phase 1

P1-T9 acquired **weights only**. The pinned image has no `basicsr`, `timm`, `einops`, `facexlib` or any repository code, so none of the candidates could run.

**Resolution (no new pip packages, no model downloads):**
- Only the **architecture source files** needed to instantiate the acquired weights were fetched from the official repositories, at the **same commits recorded in P1-T9**, into `third_party/`. There are 12 files.
- BasicSR (RRDBNet for Real-ESRGAN) is not a P1 asset. Its `master` commit on the fetch date is pinned.
- Every vendored file is **unmodified**. `archs.py` checks each file's SHA-256 at import and provides inference-only import shims instead of edits (§4).
- The code licenses are Apache-2.0 / MIT (license copies are kept).
- **Vendoring changes no production-license status.** Weights and training data remain `REVIEW_REQUIRED` (P1-T3, G-4).

## 2. Common Adapter Interface

```
adapter = fes_bench.adapters.build(config_id); adapter.load()
out_bgr_uint8_224, meta = adapter.enhance(degraded_bgr_uint8_sxs, {"s": s, "M_e2f": 2x3})
```

- Output is always 224×224×3 uint8 BGR in the evaluation frame (P3-T1 §4).
- The only per-reference context is the recorded alignment transform. **No oracle degradation information** reaches any adapter.
- Model-specific range, channel order, padding, and scale are handled **inside** the adapter.
- Weights are verified by SHA-256 at every load and loaded with `torch.load(weights_only=True)`. Missing keys are a hard error.
- **Weights are never altered.**

## 3. Per-Model Specification (as implemented; from each repo's official inference path)

| Config | Network (asset) | Upstream reference for settings | Input frame / size | Colour / range | Padding / tiling | Output → E-frame |
|---|---|---|---|---|---|---|
| A1 RestoreFormer++ | `VQVAEGANMultiHeadTransformer(head_size=4, ex_multi_scale_num=1)` (A01) | `RestoreFormer/RestoreFormer.py` @59d250f | FFHQ-aligned **512×512** (F-frame) | RGB, (x−0.5)/0.5 → [−1, 1] | none | `out[0]`, clamp [−1, 1], round → F→E warp |
| A2 VQFR v1 | `VQFRv1` with the official test config (A05) | `options/test/VQFR/test_vqfr_v1.yml` @239c77a | F-frame 512 | RGB, [−1, 1] | none | `out['main_dec']` (fidelity decoder, as in `demo_util.py`) |
| A3 Real-ESRGAN | `RRDBNet(23 blocks, nf 64, gc 32, ×4)` `params_ema` (A02) | `inference_realesrgan.py` @a4abfb2 (tile 0, pre_pad 0) | D-frame, native size (14–224) | RGB, [0, 1] | none (×4 needs no mod-pad) | clamp [0, 1], round; resize ×4 result to 224 |
| A4 SwinIR-L | `SwinIR(real_sr, large: embed 240, depths 6×9, heads 8×9, window 8, nearest+conv, 3conv)` `params_ema` (A03) | `main_test_swinir.py` @6545850 | D-frame, native | RGB, [0, 1] | flip-pad to a multiple of 8 (upstream rule), crop to 4h×4w | as A3 |
| A5 stage 1 | `FBCNN(nc [64,128,256,512], nb 4)` (A04) | `main_test_fbcnn_color_real.py` @54d1831 | D-frame, native | RGB, [0, 1] | internal replicate-pad to a multiple of 8 | blind QF prediction (recorded as `fbcnn_pred_qf`) |
| A5 stage 2 | `NAFNet(w 64, enc [2,2,4,8], mid 12, dec [2,2,2,2])` SIDD (A07) | `options/test/SIDD/NAFNet-width64.yml` @2b4af71 | same | RGB, [0, 1] | internal pad to a multiple of 16 | clamp [0, 1] (float, no re-quantization between stages) |
| A5 stage 3 | Real-ESRGAN (A02) | as A3 | same | same | — | as A3 |
| (validated, not configured) | `NAFNetLocal(w 64, enc [1,1,1,28], mid 1, dec [1,1,1,1])` GoPro (A06) | `options/test/GoPro/NAFNet-width64.yml` | — | — | TLC local pooling | Not part of A5 (P3-T1 §6) |

**Baselines** (B0, B1, B2, B2c; no weights) follow P3-T1 §6.

## 4. Import Shims (instead of editing vendored code)

| Upstream import | Shim | Why numerically neutral |
|---|---|---|
| `timm.models.layers` | `DropPath` = identity in eval; `trunc_normal_` = `torch.nn.init.trunc_normal_`; `to_2tuple` | Init is overwritten by the checkpoint; drop-path is off at inference |
| `einops.rearrange` | Exactly the two permutes used by the VQFR v1 quantizer | Any other pattern raises |
| `sklearn.cluster`, `all_gather` | Raise | Training-only (codebook re-estimation, distributed) |
| Repo registries / loggers | Dict registry / std logger | No numerical role |
| `vqfr.ops.dcn` | Upstream `deform_conv.py` (vendored) | The v1 forward uses `torchvision.ops.deform_conv2d` (torchvision ≥ 0.9 branch). No compiled CUDA extension is needed. |

## 5. Load Validation Results

Measured 2026-09-29 in `fes-dev:cpu-p2`. Details: `model_validation_report.json`.

| Network | Asset | Params | Weights SHA-256 = P1-T9 | Missing keys | Unexpected keys | Forward on real input | Status |
|---|---|---|---|---|---|---|---|
| RestoreFormer++ | A01 | 73,472,579 | ✅ | 0 | 0 (after stripping `vqvae.`) | ✅ 512→512 | **Runs** |
| VQFR v1 | A05 | 76,563,583 | ✅ | 0 | 0 (strict) | ✅ 512→512 | **Runs** |
| Real-ESRGAN x4plus | A02 | 16,697,987 | ✅ | 0 | 0 (strict) | ✅ ×4 | **Runs** |
| SwinIR-L real-SR x4 | A03 | 28,013,059 | ✅ | 0 | 0 (strict) | ✅ ×4 | **Runs** |
| FBCNN colour | A04 | 71,921,796 | ✅ | 0 | 0 (strict) | ✅ | **Runs** |
| NAFNet-SIDD w64 | A07 | 115,982,915 | ✅ | 0 | 0 (strict) | ✅ | **Runs** |
| NAFNet-GoPro w64 | A06 | 67,888,835 | ✅ | 0 | 0 (strict) | ✅ | Runs (not in a configuration) |

**Every shortlisted candidate is runnable. None is blocked.** All 12 vendored source files match their recorded SHA-256.

The adapter-level checks (output contract, byte-identical repeat at a fixed thread count) are in the report's `adapters` section. The per-pair trial and main run manifests additionally record the pixel SHA-256 of every output.

## 6. Problems Found and How They Were Handled (not hidden)

| # | Finding | Handling |
|---|---|---|
| F1 | RestoreFormer++'s official loader uses `load_state_dict(strict=False)`, which could hide a partial load | The harness loads with `strict=False` (upstream behaviour) **but reports and asserts** key sets. Result: 0 missing, 0 unexpected, so the load is complete. |
| F2 | VQFR `demo_util.py` names quantizer `L2VectorQuantizer` with arguments that class does not accept. The upstream demo cannot build v1 as written. | Used the official **test config** (`L2VectorQuantizerKmeans`). Strict load: 0 missing/unexpected, which confirms the architecture matches the weights. |
| F3 | Upstream helpers auto-download weights / need facexlib + a face parser (`parsing_parsenet`, not acquired: NC lineage, P1-T9 §4) | Not used. Alignment and paste-back come from the shared harness geometry (the recorded transform and the facexlib template/fill values), identical for all candidates. |
| F4 | **PyTorch intra-op threads > 8 collapse on this host.** Real-ESRGAN at 56 px takes 1.5 s at 4 threads and 40–68 s at 12–16. Contributing factor: the Docker VM is shared with other containers using ~4–7 cores. | Threads fixed per worker. Throughput measured: 4 workers × 2 threads = 0.46 img/s, vs 0.27 for 3 × 4 and 0.16 for 1 × 4. The runtime was changed to 4 × 2 before any main output existed (`p3-protocol-1.0.1`, runtime-only). |
| F5 | Under Docker-VM memory pressure (≈ 4.5 GB available, swap 84% used by other workloads), reads through the Windows bind mount sporadically fail with `ENOMEM` (weight loading, PNG reads) | Worker initialisation failures are re-spawned by the pool. File reads retry (`fes_bench.read_bytes_retry`), and content is still SHA-verified. A pair that still fails is recorded `FAILED` and re-run on resume, never silently dropped. |
| F6 | `torch.meshgrid` deprecation warning from SwinIR | Harmless; the upstream code is unchanged |

## 7. Safe-Loading and Integrity Practices (unchanged from Phase 1)

- `torch.load(weights_only=True)` for every checkpoint. No pickled code executes.
- The weights' SHA-256 is re-verified at each load against the P1-T9 manifest values (`adapters.WEIGHTS`).
- Vendored code is SHA-verified at import (`archs.VENDORED_SHA256`).
- **No replacement or additional model was downloaded.**
- The one new asset is the LPIPS AlexNet backbone. It is the evaluation-metric dependency announced in P1-T9 §4 and recorded as **A15**: `models/X01_torchvision/hub/checkpoints/alexnet-owt-7be5be79.pth`, SHA-256 `7be5be79…cdee02`, torchvision BSD-3 code, ImageNet-trained, **evaluation-only**. The LPIPS linear head ships inside the pinned `lpips==0.1.4` wheel (`weights/v0.1/alex.pth`, SHA-256 `df73285e…`).

## 8. Done-When Check

The harness invokes every available candidate through one interface. All 7 acquired networks load completely, run, and are validated. No candidate is blocked, and every problem encountered is documented above. ✅
