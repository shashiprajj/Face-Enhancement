# P2-T7 — Dataset Splits, Metadata & Reproducibility

| Field | Value |
|---|---|
| Phase / Task | Phase 2 / P2-T7 |
| Status | Complete |
| Date | 2026-09-28 |
| Benchmark | **`fes-syn-v1`** (synthetic, FairFace-based) |
| Config | `configs/phase2/benchmark_v1.yaml`, `configs/phase2/degrade_fes-syn-v1_{dev_pilot,test_core}.yaml` |
| Scripts | `scripts/p2_build_splits.py`, `scripts/p2_degrade.py`, `scripts/p2_validate_benchmark.py`, `scripts/p2_reproduce.sh` |
| Schemas | `configs/phase2/schema/reference_record.schema.json`, `configs/phase2/schema/pair_record.schema.json` |
| Final manifests | `data/manifests/benchmark/fes-syn-v1/` (`splits.csv.gz`, `selection.csv.gz`, `pairs_dev_pilot.csv.gz`, `pairs_test_core.csv.gz`, reports) |

---

## 1. Roles (only where needed)

| Role | Source | Purpose | Training? |
|---|---|---|---|
| **dev** | FairFace **train** (the P2-T3 dev pool) | Pilot variance, threshold calibration (TH-x), the one allowed tuning per model | No |
| **test** | FairFace **val** (the P2-T3 test pool) | Headline results, used once per frozen configuration | No |
| train | — | **None.** Phase 2 prepares evaluation data only (P0-T1 X8). A future fine-tuning set needs separate, identity-disjoint data. | — |

## 2. Leakage Prevention

| Layer | Rule | Result |
|---|---|---|
| Source | dev ⊂ FairFace train, test ⊂ FairFace val | validator `split_provenance` ✅ |
| Sample | no `sample_id` in both roles | `sample_disjoint` ✅ |
| Exact / near duplicates | P2-T2 found 21 exact and 63 near-duplicate groups **crossing FairFace's official split**. P2-T3 excluded every duplicate-group member from references. The split builder asserts that no group spans roles. | `duplicate_groups_disjoint` ✅ |
| **Same person, different photo** (no identity labels) | Offline screening with the **test-only** SFace evaluator. Any dev reference whose cosine similarity to *any* test reference is ≥ the threshold is **removed from dev**; test stays intact. Embeddings exist in memory only and are never written to disk. | 455 dev references removed; `identity_screen_dev_vs_test` ✅ |

**Threshold choice — evidence and correction.**
1. The first run used SFace's usual 1:1 value, **0.363**. It flagged **3,132 of 3,498 dev references (90%)**. That is a multiple-comparison artifact: each dev face is compared against 6,352 test faces.
2. The threshold is now **derived from the data**:
   - robust impostor model from the bulk of the 22.2M dev×test pairs: median 0.047, MAD-σ 0.101;
   - pick the smallest *t* for which the expected false exceedances over all pairs are < 1 (Bonferroni-style);
   - result: **t = 0.59**.
3. At 0.59, **1,159 pairs** exceed the threshold against ≈ 0.9 expected by chance. That large excess is real signal: FairFace's official train/val split is **not identity-disjoint**.
4. The full sweep (0.363 → 0.8) is in `splits_report.json`.

**Limitations**
- The Gaussian impostor model underestimates heavy tails, so some same-person pairs below 0.59 may remain.
- A test-only recognizer is a screening tool, not ground truth.
- 1,060 high-similarity pairs **within** test are recorded. They matter for future impostor-pair sampling (E3), not for dev/test leakage.

## 3. Final Splits & Subsets

| Split | References |
|---|---|
| dev | **3,043** |
| test | **6,352** |
| excluded_leakage | 455 |
| **total accepted references** | 9,850 |

| Subset (seeded, race × gender stratified) | Split | References | Pairs (× 32 cells) | Status |
|---|---|---|---|---|
| **dev_pilot** | dev | 112 (8 × 14 strata) | **3,584** | complete |
| **test_core** | test | 700 (50 × 14 strata) | **22,400** | complete |

Subset sizes are **provisional**. P1-T7 §8 fixes final N by power analysis on dev_pilot variance (TBD, not invented). Any larger test subset can be generated from the remaining 5,652 test references with the same code, without changing existing pairs (generation is per-reference deterministic).

## 4. Canonical Metadata

`reference_record.schema.json` — one row per reference (`splits.csv.gz`):
- **identity/group:** `ref_id`, `sample_id`, `split`, `subset`, `role`, `pool`, `split_source`, and race/gender/age (FairFace labels, aggregate strata only; no identity ID exists);
- **source:** `source_path`, `source_sha256`;
- **crop:** `ref_path`, `ref_sha256`, `ref_pixel_sha256`, `crop_box_xywh`, `ref_size`, `ref_version`;
- **detector/alignment:** `det_bbox_xywh`, `det_landmarks`, `det_score`, `detector`, `preproc_version`, `roll_deg`, `yaw_proxy`, `ffhq512_transform_2x3/_scale/_oob_frac`;
- **quality:** `lap_var`, `mean_luma`, `rms_contrast`, `frac_dark/_bright`;
- **leakage:** `exact_dup_group`, `near_dup_group`, `leak_flag`, `leak_max_cos_to_other_role` (floored), `leak_nearest_other_role`;
- **license/source:** `license`, `source_url`, `license_status`.

`pair_record.schema.json` — one row per clean/degraded pair (`pairs_<subset>.csv.gz`):
- `pair_id`, `ref_id`, `profile`, `severity`, `seed`, `params_json` (all exact parameters), `degrade_version`;
- `degraded_path`, `degraded_sha256`, `degraded_pixel_sha256`, `out_w/h`;
- input-quality statistics (PSNR, SSIM, YuNet detectability).

**Correction applied:** a stored similarity rounded to 0.59 (true value just below 0.59) tripped the validator. Stored similarities are now **floored**, so a stored value never exceeds the value used for flagging. The rebuilt selection is byte-identical (`92270e1e…`).

## 5. Raw → Derived → Benchmark Separation

```
datasets/raw/DS01_fairface/                      immutable, read-only (P2-T1)
datasets/work/DS01_fairface/v1/                  byte-identical working copy (P2-T2)
datasets/derived/DS01_fairface/references/ref-v1/   clean references (P2-T3)
datasets/benchmark/fes-syn-pilot-v1/             generator pilot (P2-T4)
datasets/benchmark/fes-syn-v1/degraded/          benchmark pairs (this task)
data/manifests/...                               tracked manifests, reports, schemas
```

## 6. Reproduction (script/config-driven, no manual steps)

`scripts/p2_reproduce.sh` rebuilds everything from the raw archive, in order: extract → analyze → references → pilot → splits → pairs (dev_pilot, test_core) → tests → full validation.

Each step:
- is deterministic;
- is idempotent (existing outputs are hash-verified; any difference aborts);
- records config SHA-256, git head, and library versions.

## 7. Validation (`scripts/p2_validate_benchmark.py`)

All **19 checks PASS** (sample reproducibility run, 500 pairs). The full `--repro-all` run is part of P2-T8.

The checks: schema (references, pairs) · unique IDs, reference content, pair IDs, degraded paths · 45,684 files exist and match SHA-256 · raw zip hash · pair→reference links · reference→raw traceability · license recorded · sample-disjoint · split provenance · duplicate groups disjoint · identity screen · subset membership · pair coverage (every selected reference × 32 cells) · degradation reproducibility.

## 8. Done-When Check

The benchmark dataset and its splits are reproducible from documented scripts and configuration. They are leakage-screened at the source, sample, duplicate, and identity levels, and validated by an automated checker. ✅
