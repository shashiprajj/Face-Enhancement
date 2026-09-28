# P1-T5 — Restoration Model Shortlist & Test Design

| Field | Value |
|---|---|
| Phase / Task | Phase 1 / P1-T5 |
| Status | Complete |
| Date | 2026-09-28 |
| Depends on | P1-T2 (survey), P1-T3 (license gate), P1-T4 (preprocessing) |
| Nature | Shortlist + benchmark design. **No final model; no benchmark implemented.** |

---

## 1. Entry Rule

Only candidates that are **ELIGIBLE (acquisition/evaluation)**, or **REVIEW_REQUIRED pending explicit reviewer approval**, are considered (task instruction 1). INELIGIBLE candidates (CodeFormer, DifFace, SUPIR, …) are out, whatever their technical merit.

## 2. Candidate Comparison Against Phase 0 Requirements

Legend: ● strong / expected · ◐ partial / unknown · ○ weak / not designed for. Quality and identity entries are **author claims or architectural expectations**, to be verified in Phase 2.

| Candidate | Type | Low-res | Blur | Noise | JPEG | Lighting | Identity risk | Deterministic | CPU dev | GPU prod | Memory | License (eval / prod) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **RestoreFormer++** (M03) | Face-specific, VQ dictionary + Transformer | ● | ● | ● | ● | ○ | ◐ (codebook prior may substitute details) | ● | ◐ (single 512 pass; seconds on CPU, TBC) | ● | ~294 MB ckpt | ELIGIBLE / REVIEW |
| **VQFR** (M05) | Face-specific, VQ codebook + parallel decoder | ● | ● | ● | ● | ○ | ◐ (fidelity branch intended to reduce this) | ● | ◐ | ● | TBC | ELIGIBLE / REVIEW |
| RestoreFormer (M04) | Face-specific (predecessor) | ● | ● | ● | ● | ○ | ◐ | ● | ◐ | ● | 291 MB | ELIGIBLE / REVIEW |
| **Real-ESRGAN x4plus** (M23) | General SR (GAN) | ● | ◐ | ◐ | ◐ | ○ | ◐ (no face prior; texture hallucination) | ● | ● (small RRDB) | ● | small | ELIGIBLE / REVIEW |
| **SwinIR real-SR (L)** (M24) | General SR (Transformer) | ● | ◐ | ◐ | ◐ | ○ | ◐ | ● | ◐ (large) | ● | 285 MB | ELIGIBLE / REVIEW |
| **FBCNN** (M30) | JPEG removal | ○ | ○ | ○ | ● | ○ | ● (low hallucination) | ● | ● | ● | 288 MB | ELIGIBLE / REVIEW |
| **NAFNet** (M28) | Deblur / denoise | ○ | ● (motion) | ● | ○ | ○ | ● | ● | ● | ● | TBC | ELIGIBLE / REVIEW |
| BSRGAN (M25), HAT (M26), Restormer (M27), SCUNet (M29), DRUNet (M31) | General | varies | | | | | | ● | | | | ELIGIBLE / REVIEW |
| GFPGAN v1.4 (M01) | Face-specific, StyleGAN2 prior | ● | ● | ● | ● | ◐ | ◐ | ● | ● (clean arch) | ● | ~350 MB | **REVIEW / REVIEW** |
| PMRF (M09) | Face-specific, rectified flow (distortion-optimal) | ● | ● | ● | ● | ○ | ● (designed to minimize distortion) | ◐ (seeded sampling) | ○ (25 flow steps) | ● | 923 MB | **REVIEW / REVIEW** |
| DiffBIR v2.1 (M08) | Multi-stage diffusion | ● | ● | ● | ● | ◐ | ○ (strong generative prior) | ◐ (needs seed) | ○ | ◐ (high VRAM) | ~5 GB+ | **REVIEW / REVIEW** |
| Classical (C01) | Bicubic/Lanczos, NLM, bilateral, unsharp, CLAHE | ◐ | ◐ | ◐ | ◐ | ● (CLAHE) | ● (no hallucination) | ● | ● | ● | none | ELIGIBLE / ELIGIBLE |

Lighting/contrast: no candidate targets it specifically. Classical CLAHE is the only option. Per master C7, it is evaluated as an **optional pre-step**.

## 3. Shortlist

### Tier A — benchmark in Phase 2 (all ELIGIBLE for evaluation; download in P1-T9)

| # | Candidate | Category | Why included |
|---|---|---|---|
| **A1** | **RestoreFormer++** | Face-specific restoration | Only face-specific candidate with Apache-2.0 code **and** weights in the same repo's GitHub release. Handles all core degradations (DG1–DG5, DG8). Deterministic single forward pass. |
| **A2** | **VQFR** | Face-specific restoration | Second independent face-specific family. Its explicit fidelity branch matches the identity-preservation principle. Apache-2.0 with a clean third-party list. |
| **A3** | **Real-ESRGAN x4plus** | General SR (baseline B3) | Tests whether a face prior adds value over general blind SR. Also the standard **background upsampler** for paste-back. Small, CPU-friendly. |
| **A4** | **SwinIR real-SR (x4, L)** | General SR (baseline B3, Transformer) | A second general family, to separate "face prior" from "architecture" effects. |
| **A5** | **Degradation-specific chain:** FBCNN → NAFNet → Real-ESRGAN | Multi-stage general pipeline | Tests whether dedicated, low-hallucination components reach face-model quality with **lower identity risk**. Directly addresses P0-T3 R-D6 (conservative behavior). |
| **B0–B2** | Identity (degraded input), bicubic/Lanczos, classical restoration (NLM + unsharp + optional CLAHE) | Baselines | Required by P0-T4 §6 |

5 learned configurations plus 3 baselines. **Small enough to benchmark on limited GPU time, broad enough** to cover face-prior vs general vs chained approaches and two face-specific families.

### Tier B — conditional (only if reviewer approves; **not downloaded** in Phase 1)

| # | Candidate | Why it matters | Condition |
|---|---|---|---|
| B-1 | **GFPGAN v1.4** | Industry-standard reference; most published comparisons include it | Reviewer clears NC third-party components (P1-T3 §6 Q4) |
| B-2 | **PMRF** | Designed for minimal distortion. Possibly the best identity/fidelity trade-off. | Reviewer clears embedded DifFace weights, **or** the MMSE stage is retrained on licensed data |
| B-3 | **DiffBIR v2.1** | Representative of diffusion restoration (to quantify the hallucination risk the spec warns about) | Reviewer accepts Open RAIL++-M; SD2.1 provenance resolved |

### Reserves (ELIGIBLE, not shortlisted)

RestoreFormer v1 (superseded by A1), BSRGAN and HAT (overlap with A3/A4), Restormer (overlaps NAFNet), SCUNet/DRUNet (weight location TBC; overlap NAFNet). Promote a reserve only if a Tier A candidate fails to run.

### Excluded

All INELIGIBLE (P1-T3 §4.2). REVIEW_REQUIRED candidates not in Tier B (DAEFR, DR2, BFRffusion, HiFaceGAN, OSDFace, AuthFace, InstantIR, OSEDiff, SeeSR, PiSA-SR, DreamClear, FaithDiff) are excluded for license ambiguity **plus** at least one of: heavy SD/SDXL dependency, no code license, low maturity, or general-purpose focus.

## 4. Expected Strengths / Weaknesses (hypotheses for Phase 2 to test)

| Candidate | Expected strengths | Expected weaknesses / risks |
|---|---|---|
| A1 RestoreFormer++ | Strong on severe blind degradation; realistic texture | Dictionary prior may replace fine identity details (U2); trained on aligned FFHQ, so pose-sensitive |
| A2 VQFR | Better fidelity than pure codebook methods (claim) | Older (2022); Google Drive weights without checksum; setup complexity |
| A3 Real-ESRGAN | Fast, robust, no facial "invention" of structure | Over-smoothing / painterly faces at high degradation; weak identity recovery at very low resolution |
| A4 SwinIR | Strong general SR | Heavy on CPU; no face prior |
| A5 Chain | Low hallucination, interpretable per degradation | Error compounding; needs degradation estimation to switch stages; lower perceptual quality |
| Classical | Zero hallucination; fast | Low perceptual gain |

## 5. Standardized Benchmark Configuration (fair comparison, P0-T4 §7)

### 5.1 Held constant across all candidates

| Aspect | Setting |
|---|---|
| Face detector / landmarks | **One fixed detector** for the main track (proposed: D05 facexlib RetinaFace-R50, the pipeline most candidates were trained with). A second track with D01 YuNet tests sensitivity. |
| Alignment | FFHQ 5-point template, similarity transform, 512×512, `cv2.INTER_LINEAR` warp, `BORDER_CONSTANT` (P1-T4 §4) |
| Input normalization | 8-bit sRGB → float32 [0,1] RGB (converted to each model's expected range/channel order inside a thin adapter) |
| Face-specific models (A1, A2, B-x) | Aligned **512×512** input. LR faces are **bicubic-upsampled to 512** before the model (standard BFR protocol). |
| General SR models (A3, A4, A5-SR stage) | Fed the **aligned LR crop at native resolution**, upscaled ×4, then resized (Lanczos) to exactly 512×512. Mild-only inputs (no downscale) → ×1 via downscale-after-SR, documented. |
| Paste-back | Same inverse affine + same soft mask + same background handling (**background = bicubic** for all, so that face quality is isolated). Real-ESRGAN background is a separate, labeled track. |
| Output | 8-bit PNG, sRGB |
| Precision | FP32 for quality/identity runs. FP16 only in the performance track, reported separately. |
| Determinism | `torch.manual_seed`, `numpy`/`random` seeds fixed; `torch.backends.cudnn.deterministic=True`; per-image seed = hash(image_id) for any stochastic model (master C4) |
| Tiling | Disabled for 512 faces. If needed for memory, the same tile size and overlap for all. |
| Model settings | **Author defaults** first. At most one documented alternative per model (e.g., a fidelity weight), tuned on the validation split only. |
| Weights | Exact files and SHA-256 from the P1-T9 manifest |

### 5.2 Required preprocessing (implemented once, shared)

1. Decode → EXIF orientation → sRGB 8-bit (P0-T2 §2.3).
2. Synthetic degradation generation from the versioned pipeline (P1-T7).
3. Detection + alignment (fixed component).
4. Candidate adapter: model-specific range, channel order, and resize **inside the adapter only**.

### 5.3 Measurements per run (from P0-T4 §3)

Quality (PSNR, SSIM, LPIPS; NIQE/MUSIQ on real sets) · identity (evaluators E01 + one more, P1-T4 §5 gap) · landmark distance · per-stage latency · peak RAM/VRAM · failure/warning behavior on the edge-case suite.

## 6. Required Hardware

| Track | Hardware | Available? |
|---|---|---|
| Correctness, adapters, small-sample quality/identity | Local CPU (Core Ultra 7 255H, 31 GB RAM) | ✅ Yes |
| Full quality/identity runs | CPU feasible for A1–A5 at moderate sample sizes. Estimated seconds per face; **timing to be measured, not assumed**. | ✅ (slow) |
| Performance (latency, P95, throughput, VRAM, FP16) | NVIDIA GPU of the production class (T4/L4/A10G class, TBD) | ❌ **Not available** (P1-T1 B-1) |
| Tier B diffusion (DiffBIR) | NVIDIA GPU ≥ 12–16 GB VRAM (estimate) | ❌ |

## 7. Unresolved Questions

1. Reviewer decisions on Tier B (GFPGAN, PMRF, DiffBIR).
2. Second eligible identity evaluator (P1-T4 §5).
3. Production GPU class (A-3) for the performance track.
4. Whether the degradation-specific chain (A5) needs a degradation classifier, or can be evaluated with oracle degradation labels only (proposed: oracle labels in Phase 2, estimator later).
5. Detector choice for the main track. Proposal: facexlib (matches training). YuNet is a sensitivity track.

## 8. Acceptance Criteria

| Criterion | Status |
|---|---|
| Shortlist small but meaningful | ✅ 5 learned configurations + 3 baselines; 3 conditional |
| Every included candidate has a reason | ✅ §3 |
| License eligibility visible | ✅ §2 last column; tiers tied to status |
| Standardized config + constant pre/post-processing | ✅ §5 |
| Required hardware documented | ✅ §6 |
| No final winner | ✅ |

**Benchmark prerequisites:** P1-T9 acquisition of Tier A weights with checksums · P1-T7 evaluation data · P1-T8 environment · a second eligible identity evaluator · NVIDIA GPU for the performance track.
