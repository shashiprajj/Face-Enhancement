# P2-T3 — Clean Reference Face Dataset

| Field | Value |
|---|---|
| Phase / Task | Phase 2 / P2-T3 |
| Status | Complete |
| Date | 2026-09-28 |
| Pipeline | `scripts/p2_build_references.py` + `src/fes_data/preprocess.py` (YuNet detection, pose, transform) |
| Config | `configs/phase2/references.yaml` (`ref-v1`) |
| Output | `datasets/derived/DS01_fairface/references/ref-v1/{val,train}/<sample_id>.png` (9,850 PNG, 224×224) |
| Manifest | `data/manifests/references_v1.csv.gz` (all 16,946 candidates, accepted or not) |
| Report | `data/manifests/references_v1_report.json` |
| Design evidence | `data/manifests/reference_size_probe_v1.json`, `data/manifests/framing_probe_v1.json` |

---

## 1. Candidate Pools

| Pool | Source | Selection | Candidates |
|---|---|---|---|
| test_pool | FairFace **val** | all | 10,954 |
| dev_pool | FairFace **train** | seeded (`20260928`) sample, stratified by race × gender (428 per stratum) | 5,992 |

The pools come from different official FairFace splits. P2-T7 removes the cross-split duplicates found in P2-T2 and screens for same-person leakage.

## 2. Key Design Decision — The Reference Is the Source Chip (No Resampling)

**Plan (P1-T5):** re-align every face to the FFHQ 5-point template, the framing FFHQ-trained restoration models expect.

**What the data showed (evidence recorded):**

1. *Probe 1* (`reference_size_probe_v1.json`, 2,000 seeded images):
   - median inter-ocular distance is **79 px**;
   - YuNet landmark order is correct in 99.85% of detections;
   - detection rate is 99.8%.
2. *First build* (FFHQ framing at 256, "no upsampling" rule): **0 of 16,946 accepted.** 16,776 failed `window_outside_source`. The FFHQ window needs ≈ 40% more context on each side than a FairFace chip (padding 0.25) contains. Padding would put ~50% synthetic pixels into the "clean" reference.
3. *Probe 2* (`framing_probe_v1.json`): 42 combinations of output size (192/224/256), template zoom (1.0–1.8), and centring. Even the best combination keeps only **≈ 46–53%** of faces without upsampling or padding, and it introduces a resampling step.

**Decision:** FairFace chips are already aligned faces: dlib 5-point alignment, 224×224, padding 0.25, as documented in the FairFace README. The **clean reference is the source chip itself**: the decoded JPEG pixels, saved as lossless PNG. The pipeline asserts that the PNG decodes to exactly the source pixels. No warp, resize, sharpening, or enhancement touches a reference.

YuNet and the alignment code are used to:
- **verify** the chip geometry and reject outliers (§3);
- **record** per reference the FFHQ-512 similarity transform (`ffhq512_transform_2x3`), its scale, and its out-of-source fraction. Benchmark adapters can then convert to FFHQ framing with one identical, traceable step for every candidate (P1-T5 §5.1).

**Consequences**
- Reference resolution is **224 px** (FairFace's native). Low-resolution factors ×2/×4/×8/×16 give integer sizes 112/56/28/14.
- High-resolution (≥ 512) GT still requires DS02 (open gap G-5).
- All references carry FairFace's uniform **JPEG Q≈75 floor** (P2-T2). They are "clean" relative to the degradations applied later, not pristine.

## 3. Objective Acceptance Rules

| Rule | Value | Source of the number |
|---|---|---|
| Stage-1 suitable (readable, color, labeled, not exact duplicate) | required | P2-T2 |
| Not in any near-duplicate group | required | P2-T2 pHash |
| Exactly one detected face (YuNet, score ≥ 0.7) | required | config |
| Detection score | ≥ 0.9 | config |
| Plausible landmarks | required | `landmarks_plausible` |
| Inter-ocular distance | 0.8–1.25 × pool median → **63.3–98.9 px** | pool median 79.12 px |
| Eye-midpoint offset from pool median (110.6, 68.0) | ≤ 0.15 × median eye distance → **11.9 px** | pool |
| Roll | ≤ 10° | config (chips are roll-normalized) |
| Sharpness (source Laplacian variance) | ≥ **8.48** (10th percentile of the stage-1 pool) | P2-T2 distribution |
| Exposure clipping | ≤ 5% pixels at luma ≤ 5 or ≥ 250 | config |
| JPEG quality | **not used** (all images Q≈75) | P2-T2 finding |

The exact resolved numbers are in `references_v1_report.json → resolved_thresholds`.

## 4. Results

| | Candidates | Accepted | Rate |
|---|---|---|---|
| test_pool (val) | 10,954 | **6,352** | 58% |
| dev_pool (train sample) | 5,992 | **3,498** | 58% |
| **Total** | 16,946 | **9,850** | 58% |

**Rejection reasons** (one image can have several): nonstandard position 3,010 · nonstandard scale 2,444 · clipped exposure 2,261 · nonstandard roll 1,682 · source blurry 1,628 · low detection score 1,382 · grayscale 432 · near duplicate 121 · multiple faces 75 · no face 50 · exact duplicate 46 · implausible landmarks 37.

**Fairness check: no group is disproportionately removed.** Acceptance by race is 0.54–0.60:

| Race | Rate |
|---|---|
| Black | 0.58 |
| East Asian | 0.59 |
| Indian | 0.60 |
| Latino/Hispanic | 0.60 |
| Middle Eastern | 0.54 |
| Southeast Asian | 0.59 |
| White | 0.57 |

By gender: Female 0.60, Male 0.56. Per-stratum counts are in the report (`accepted_strata`, `accepted_strata_by_pool`).

**Quality statistics of accepted references** (p5 / p50 / p95, from the report):
- detection score;
- eye distance;
- roll;
- yaw proxy (frontal ≈ 0; pose variation is kept, not filtered, so pose can be stratified later);
- Laplacian variance;
- mean luma;
- RMS contrast;
- FFHQ-512 scale and out-of-source fraction.

## 5. Traceability (per reference, in the manifest)

`sample_id`, `ref_id` (`<sample_id>.ref-v1`), `pool`, `split_source`, `source_path`, `source_sha256`, `ref_path`, `ref_sha256` (PNG bytes), `ref_pixel_sha256` (raw pixel buffer), `detector` (model file + thresholds), `preproc_version`, `n_faces`, `det_score`, `det_bbox_xywh`, `det_landmarks` (5×2, source px), `eye_dist`, `eye_mid_x/y`, `roll_deg`, `yaw_proxy`, `ffhq512_transform_2x3`, `ffhq512_scale`, `ffhq512_oob_frac`, quality statistics, `race`/`gender`/`age` (aggregate-only use), `license`, `accepted`, `reject_reasons`.

**Crop coordinates:** the reference is the full chip, so crop = identity over `[0,224)²`. The detection box and landmarks are given in the same coordinates.

**Reproduction:** rerunning the script re-verifies every existing PNG by hash (`non-reproducible reference` assertion) and rewrites the manifest.

## 6. Identity Separation

- Pools come from different official FairFace splits.
- Near- and exact-duplicate images are excluded from references.
- FairFace has no identity labels, so **same-person-different-photo leakage across pools is screened in P2-T7** with the test-only SFace evaluator (offline, never in the runtime).

## 7. Done-When Check

A reproducible clean reference set exists: 9,850 references. Every reference traces to its FairFace source (path + SHA-256) and to a zip member of the immutable raw archive, and no transformation other than lossless re-encoding is applied. ✅
