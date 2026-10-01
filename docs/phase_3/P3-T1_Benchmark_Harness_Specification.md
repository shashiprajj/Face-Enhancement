# P3-T1 — Benchmark Baseline & Harness Specification

| Field | Value |
|---|---|
| Phase / Task | Phase 3 / P3-T1 |
| Status | Complete. **Protocol frozen before any candidate output was produced.** |
| Date | 2026-09-29 |
| Protocol | `configs/phase3/benchmark_p3.yaml` (`p3-protocol-1.0.0`) |
| Schemas | `configs/phase3/schema/output_record.schema.json`, `metric_record.schema.json` |
| Populations | `data/manifests/phase3/population_{trial,main,noise_floor,impostor_pool}.csv.gz` + `populations_report.json` |
| Code | `src/fes_bench/` (`geometry.py`, `adapters.py`, `metrics.py`, `archs.py`), `scripts/p3_build_populations.py` |

---

## 1. Inputs Read

| Source | What was taken from it |
|---|---|
| P0-T4 | Dimensions A–D reported **separately**, no composite score. Metric list (§3). Baselines B0–B4 (§6). Fair-comparison rules (§7). TH-1…TH-11 are **TBD with a method** (§10). |
| P0 master / P0-T3 | Identity is a hard gate. The evaluator is test-only (C1). Never invent a face (C2). Determinism (C4). CLAHE is an optional pre-step (C7). |
| P1-T5 | Tier A shortlist A1–A5 + baselines (§3). Held-constant configuration (§5.1). Oracle-free chain question (§7 Q4). |
| P1-T7 | ID-1…ID-5 designs (§6). Validation → test use (§7). Sample-size method (§8). |
| P1-T9 | Weights and SHA-256 per asset. `weights_only=True` loading. |
| P2 final report + P2-T3/T4/T6/T7 | `fes-syn-v1`. References are 224-px FairFace chips with a recorded FFHQ-512 transform. Adapters must convert framing identically and exclude out-of-source pixels (P2 §7.4). dev_pilot is used first; test_core is used once per frozen configuration. |

## 2. Inventory of the Actual `fes-syn-v1` Manifests (verified, not assumed)

Before use, every manifest SHA-256 was re-checked against the P2-T8 evidence (`p2t8_hashes_after.txt`). All match (`populations_report.json → inputs`).

| Manifest | Rows | References | Cells | Input sizes per reference |
|---|---|---|---|---|
| `splits.csv.gz` | 9,850 | dev 3,043 / test 6,352 / excluded 455 | — | — |
| `pairs_dev_pilot.csv.gz` | 3,584 | 112 (8 × 14 strata) | 32 | 224 ×22 · 112 ×3 · 56 ×3 · 28 ×3 · 14 ×1 |
| `pairs_test_core.csv.gz` | 22,400 | 700 (50 × 14 strata) | 32 | same |

The 32 cells: `lowres` ×4, `gaussblur` ×4, `motionblur` ×4, `noise` ×4, `jpeg` ×4, `underexposure` ×3, `lowcontrast` ×3, `combined` ×3, `realistic2` ×3.

## 3. Evaluation Unit, Populations, and Trial / Test Separation

**Evaluation unit:** one *pair* = (clean reference, degraded input of one cell). Every configuration produces exactly one output per pair. **Resampling unit for statistics:** the *reference*, since the 32 pairs of a reference share one identity.

| Population | Source (split) | Selection (seeded, `sha256(seed|salt|ref_id)` order within race × gender) | Refs | Pairs | Use |
|---|---|---|---|---|---|
| **trial** | dev_pilot (**dev**) | 1 per stratum, capped at 4 | 4 | 128 | Validate every code path and output before the main run (P3-T4). Never reported as a result. |
| **main** | test_core (**test**) | 2 per stratum (all 14 strata) | **28** | **896** | The one controlled population every configuration is run on. Headline Phase 3 numbers. |
| noise_floor | dev_pilot (dev) | all | 112 | — | Evaluator noise floor (TH-1 method). Clean references only, no candidate. |
| impostor_pool | test_core (test) | all | 700 | — | Clean-reference impostor scores → the TAR/FAR operating point. |

The build script asserts:
- trial ∩ main = ∅;
- main ⊂ test split;
- trial ⊂ dev split;
- every reference and degraded file of every population matches its SHA-256 (924 files for main).

Main population manifest SHA-256: `840ee52d…`.

## 4. Input / Output Contract and Linkage

### 4.1 Linkage

```
population row (pair_id) ─┬─ ref_path + ref_sha256 ............ clean reference (E-frame, 224×224)
                          ├─ degraded_path + degraded_sha256 .. degraded input (D-frame, s×s)
                          ├─ det_landmarks, ffhq512_transform_2x3 (from splits.csv.gz; P2-T3)
                          └─ run record (config_id, pair_id) → output_path + output_sha256 → metric record
```

`pair_id` is the join key everywhere. Outputs are written to `experiments/phase3/outputs/<population>/<config>/<cell>/<ref_id>.png` (git-ignored). Only manifests are tracked.

### 4.2 Adapter contract (`fes_bench.adapters`)

`enhance(degraded uint8 BGR s×s, ctx={s, M_e2f}) → (uint8 BGR 224×224 in the E-frame, meta)`

- **No oracle information.** Adapters never see the clean reference, the degradation label, or its parameters. The only per-reference data is the recorded alignment transform, the same landmarks any detector would provide.
- Model-specific range, channel order, padding, and scale live **inside the adapter**.

### 4.3 Frames and resampling (`fes_bench.geometry`, identical for every candidate)

| Step | Rule | Reason |
|---|---|---|
| D → E (any LR use) | `x_E = (x_D+0.5)·224/s − 0.5` | Inverse of the INTER_AREA downscale that created the input |
| Face models: D → F (512) | One INTER_CUBIC warp composed of (D→E)·(E→F). Border `BORDER_CONSTANT (135,133,132)` | P1-T5 "bicubic to 512". The fill is facexlib's `align_warp_face` value, i.e. both face candidates' native pipeline. |
| Face models: F → E | INTER_AREA pre-reduction + INTER_LINEAR | Anti-aliased, no overshoot (same rule as `preprocess.align_face`) |
| E pixels not covered by the 512 canvas | Filled from the bicubic degraded input | P1-T5 paste-back rule (background = bicubic) |
| General SR: D → ×4 → E | Native-resolution input. INTER_AREA when 4s > 224, INTER_LANCZOS4 when 4s < 224 | P1-T5 §5.1 (Lanczos resize; "downscale-after-SR" for same-size inputs) |
| B0 LR → E | INTER_CUBIC | Same rule as the P2-T4 input statistics |

### 4.4 Evaluation mask

The FFHQ-512 window does not fully contain the chip for some references. This affects 3 of the 28 main references, with minimum coverage 98.4%, and 72 of the 700 test_core references.

The **evaluation mask** is the E-frame pixels covered by the 512 canvas, eroded by 2 px. It is computed from the reference alone and applied **to every configuration**. So pixels a face model never saw are excluded for all candidates (P2 finding §7.4).

## 5. Compute Budget and Population Size (measured, not assumed)

Warm single-image latency, `fes-dev:cpu-p2`, fp32, measured 2026-09-29 (details in P3-T2 and P3-T8):

| Threads | Real-ESRGAN, 56 px input |
|---|---|
| 1 / 2 / 4 | 3.2 / 2.1 / **1.5 s** |
| 8 | 3.5 s |
| 12 / 16 | **40 / 68 s** (oversubscription collapse; the host also runs other containers using ~4–7 cores) |

At 4 threads:

| Model | Input | Latency |
|---|---|---|
| RestoreFormer++ | 512 | 10.9 s |
| VQFR v1 | 512 | 33.2 s |
| Real-ESRGAN | 224 | 16.4 s |
| SwinIR-L | 224 | 56.2 s |
| SwinIR-L | 56 | 2.4 s |
| FBCNN | 224 | 2.3 s |
| NAFNet-SIDD | 224 | 1.4 s |

All five learned configurations together cost **≈ 1 CPU-hour per reference** (32 cells).

**Decision:**
- **main = 28 references (2 per race × gender stratum) × 32 cells = 896 pairs per configuration.** That is ≈ 10 h of wall time with 3 workers × 4 threads.
- The full test_core (700 refs) would need ≈ 3 weeks on this CPU, so it is deferred to the GPU track (G-1).

This is a **compute-bound** size. It is not a power-analysis result: TH-1/TH-3 (the minimum meaningful differences) are still TBD, so a formal power analysis would require inventing them. Instead, P3-T5/T6 report the **achieved CI half-widths**, so that readers can see what the sample can and cannot resolve. Per-stratum results (2 identities per stratum) are **descriptive only**.

Runtime is fixed at `torch_threads_per_worker: 4` and `workers: 3`.

## 6. Configurations (identical population, author defaults, no tuning)

| ID | Configuration | Weights (P1-T9) | Notes |
|---|---|---|---|
| B0 | Degraded input (no enhancement) | — | Minimum bar (P0-T4 B0) |
| B1 | Lanczos-4 upscaling | — | Classical upscaling (B1). Identical to B0 for same-size cells. |
| B2 | Lanczos → NLM (h=5, hColor=5, 7/21) → unsharp (σ 1.0, amount 0.5) | — | Classical restoration (B2). Parameters fixed a priori from OpenCV defaults/common values, **not tuned**. |
| B2c | B2 with CLAHE (clip 2.0, 8×8 on Lab L) after upscaling | — | Optional contrast pre-step (master C7) |
| A1 | RestoreFormer++ | A01 | Face-specific, F-frame |
| A2 | VQFR v1 (`main_dec` output) | A05 | Face-specific, F-frame. v1 has no fidelity control. |
| A3 | Real-ESRGAN x4plus | A02 | General SR |
| A4 | SwinIR-L real-SR x4 GAN | A03 | General SR |
| A5 | FBCNN (blind QF) → NAFNet-SIDD → Real-ESRGAN x4plus | A04, A07, A02 | **Blind**: every stage always runs, with no oracle routing (P1-T5 Q4). The order undoes the capture chain (JPEG last applied → removed first; sensor noise; then blur + resolution by SR). NAFNet-GoPro (A06) is load-validated in P3-T2 but **not** part of A5 (one NAFNet per the Phase 1 definition). |

The tuning allowance of P0-T4 §7.3 ("at most one documented tuning, on validation") is **not used** in Phase 3. Every candidate runs at author defaults.

## 7. Metrics

| Dimension | Metric | Definition | Separation |
|---|---|---|---|
| A quality | PSNR | RGB 8-bit, masked | P3-T5 |
| | SSIM | Luma BT.601, Gaussian σ 1.5, masked mean | P3-T5 |
| | LPIPS | AlexNet v0.1 (`lpips` 0.1.4), masked spatial mean | P3-T5 |
| | sharpness ratio, colour shift | Laplacian-variance ratio; ΔE76 of mean Lab colour. **Descriptive only.** | P3-T5/T7 |
| Geometry | YuNet detected; landmark NME | Detector on the output; 5-point error / reference inter-ocular distance | P3-T5/T7 |
| B identity | cos to reference | SFace, **fixed alignment with the reference landmarks** for every image, so identity content is measured independently of detector failures | P3-T6 only |
| | ID gain | cos(output, ref) − cos(B0, ref) per pair | P3-T6 |
| | impostor cos | output vs the clean references of the other 27 main references | P3-T6 |
| | TAR / FAR at FAR 1e-2 | Threshold from clean-reference impostor pairs of the 700-ref impostor pool | P3-T6 |
| C performance | latency (warm), load time, peak RSS | Separate controlled measurement | P3-T8 only |

Not used, with reason:
- **NIQE/MUSIQ:** no-reference metrics; real-world sets only (P0-T4 §4).
- **FID/KID:** need large N.
- **DISTS:** not installed. Adding a package is out of scope.

**Image-quality and identity results are reported in separate documents** (P3-T5 vs P3-T6).

## 8. Aggregation and Statistics

- Primary tables report, **per configuration × cell**: median, mean, p5, and p95, plus the mean paired difference vs B0 and its 95% CI.
- Rollups are **per family** (cells pooled) and **per severity**. There is no cross-dimension composite.
- **CI method:** cluster bootstrap over references. B = 2000, percentile, seed 20260928.
- **Tie rule:** if the CI of the paired difference contains 0, the result is "no detectable difference". There is no p-value hunting and no multiple-testing claims.
- **Failures:** FAILED/MISSING pairs are counted and reported per configuration and are never silently dropped. Aggregates state n.

## 9. Reproducibility Requirements

| Requirement | How |
|---|---|
| Same inputs | Population manifests with SHA-256. Each degraded file is re-hashed before inference. |
| Same code | Git head in every run record. Vendored architectures are SHA-checked at import (`archs.VENDORED_SHA256`). |
| Same weights | Weights SHA-256 re-verified at every load (P1-T9 values). `torch.load(weights_only=True)`. |
| Same environment | `fes-dev:cpu-p2`, fp32, 4 threads/worker. Versions recorded in `_run.json`. |
| Determinism | No stochastic candidate. `torch.manual_seed(0)` anyway. The pixel SHA-256 of every output is recorded, and a rerun check is part of P3-T2/P3-T10. |
| Resumability | The runner skips pairs with an existing output **and** an OK record, so a restart never changes an existing output. |

## 10. What Is Deliberately Not Defined Here

- No pass/fail threshold of any kind. TH-1…TH-11 stay TBD.
- The noise floor (TH-1 method) and the FAR operating point are **measured and reported**. They are not used as gates.
- There is no ranking, winner, or production recommendation.

## 11. Change Control

Any change to `benchmark_p3.yaml`, the geometry, or an adapter after main-run outputs exist requires:
1. a protocol version bump;
2. re-running **all** configurations on the affected population;
3. a note in this section.

**Changes after freeze:**
- `p3-protocol-1.0.1` (runtime only, before any main output): 4 workers × 2 threads (§5, P3-T2 F4).
- **Runtime-only deviation during the main run, 2026-09-30:**
  - What: A2 (VQFR) was OOM-killed at 4 workers, so its remaining 774 main pairs ran at 2 workers × 4 threads. No protocol, geometry, adapter or weight change.
  - Effect, measured in P3-T10: outputs reproduce exactly at their own thread count, and a thread-count change alters ≤ 1 grey level on ≤ 0.004 % of pixels.
  - Details: P3-T4 §5.

No re-run was required under the rule above because nothing in `benchmark_p3.yaml`'s protocol fields, the geometry, or an adapter changed.

## 12. Deviations From Phase 1 Plans (with evidence)

| Plan | Deviation | Evidence |
|---|---|---|
| P1-T5 §5.1: detector-based alignment to 512 for face models | The recorded per-reference transform (P2-T3) is used instead of re-detecting on degraded inputs | References are pre-aligned chips. Detection on degraded inputs fails at noise-extreme (0%, P2-T4). Identical alignment isolates the model effect (P0-T4 §7.2). |
| P1-T5 §5.1: general-SR output resized to 512 | Resized to the 224 E-frame | GT is 224 px (P2-T3). Metrics are computed at GT resolution. |
| P1-T7 §8: ~100 refs per cell pilot, power-analysed N | 4 trial refs; 28 main refs | CPU budget (§5). TH-1/TH-3 TBD. Consequence: wide CIs, reported explicitly. |
| P0-T4 §5.2: ≥ 2 identity evaluators | One (SFace) | G-2 open. Stated in every identity result. |

## 13. Done-When Check

Every candidate is evaluated through one adapter contract, on one frozen population, with one geometry, one mask, one metric implementation, and one statistics rule, all fixed before any output existed. ✅
