# Phase 2 Final Report — Face Enhancement Service

| Field | Value |
|---|---|
| Date | 2026-09-28 |
| Phase | 2 — Dataset Acquisition & Preparation |
| Index | [`docs/phase_2/PHASE_2_INDEX.md`](../docs/phase_2/PHASE_2_INDEX.md) |
| Validation evidence | `data/manifests/benchmark/fes-syn-v1/validation_report.json`, `data/manifests/benchmark/fes-syn-v1/evidence/` |
| **Phase 2 gate** | **🟡 YELLOW — reproducible synthetic benchmark foundation ready; real-world evaluation data blocked** |

> No enhancement model was run or selected in Phase 2. No API. No Feature-Match.

---

## 1. Executive Summary

Phase 2 turned the FairFace asset from Phase 1 into a **verified, leakage-screened, fully reproducible synthetic benchmark (`fes-syn-v1`)**, built with a reusable, tested preprocessing and degradation toolkit:

| Stage | Output |
|---|---|
| Raw | 97,698 FairFace images; raw archive immutable and read-only |
| Verification | 0 corrupt; labels 1:1; duplicates and label conflicts found and handled |
| Clean references | **9,850**, each the original source chip (no resampling or enhancement), with objective acceptance rules and **even acceptance across demographic groups** |
| Degradation | 9 profiles × severities = **32 cells**, seeded per pair |
| Splits | dev **3,043** / test **6,352** after duplicate and identity leakage screening |
| Benchmark pairs | **25,984** (dev_pilot 112 refs, test_core 700 refs × 32 cells) |
| Tooling | preprocessing module with defined edge-case behavior, **55 tests passing**, executable real-world collection spec + validator |

**Reproducibility is demonstrated, not claimed:**
- A full rerun from the raw archive produced **byte-identical manifests at every stage**.
- **All 25,984 degraded images** were regenerated with 0 mismatches.

**Main gap:** there is **no real-world low-quality data**. No external source is approvable for commercial evaluation: SCface is now confirmed non-commercial, and Casual Conversations v2 is likely non-commercial. The internal collection is specified, but it needs Legal/Compliance approval before any capture. The Phase 0 spec makes real-world evaluation mandatory, and synthetic data is explicitly not a substitute. Hence **YELLOW**.

## 2. Task Status

| Task | Status | Key result |
|---|---|---|
| P2-T1 Baseline & inventory | ✅ Complete | Assets verified against Phase 1 hashes; raw moved (not re-downloaded) to `datasets/raw/` read-only; conventions defined |
| P2-T2 FairFace verification | ✅ Complete | 97,698 readable; 21 exact + 63 near-duplicate groups **cross FairFace's official split**; 63 exact-dup groups have **conflicting labels**; uniform JPEG Q≈75 |
| P2-T3 Clean references | ✅ Complete | 9,850 references = source chips (after data showed FFHQ re-framing needs ~40% padding or upsampling); acceptance by race 0.54–0.60 |
| P2-T4 Degradation generator | ✅ Complete | 32 cells, deterministic, monotonic severity; pilot 1,280 pairs verified |
| P2-T5 Real-world data | ⚠️ **Partial (plan only)** | Decision matrix; SCface → INELIGIBLE (agreement text); executable spec + validator + draft consent; **no data** |
| P2-T6 Preprocessing | ✅ Complete | All listed edge cases defined and tested; face-crop retry verified; occlusion is not detectable (documented) |
| P2-T7 Splits/metadata | ✅ Complete | JSON schemas; data-derived 1:N identity threshold 0.59 (a fixed 0.363 would have dropped 90% of dev); 19-check validator |
| P2-T8 Validation & gate | ✅ Complete | This report |

## 3. End-to-End Validation (source → detection → reference → degradation → metadata/split)

Run on 2026-09-28 in `fes-dev:cpu-p2`. Log: `evidence/p2t8_e2e_run.log`.

| Check | Result |
|---|---|
| Raw archive SHA-256 | ✅ |
| Working copy re-verified against zip bytes | ✅ 97,698 / 97,698 |
| File existence + SHA-256 (references, degraded, sources) | ✅ 45,684 / 45,684 |
| Counts: references / pairs | ✅ 9,850 / 25,984 (every selected reference × 32 cells) |
| Duplicate IDs / duplicate reference content / duplicate paths | ✅ none |
| Pair → reference → source → raw links | ✅ 0 broken |
| Leakage: sample, provenance, duplicate groups, identity screen | ✅ all disjoint; 0 violations at t = 0.59 |
| Metadata completeness vs schemas | ✅ 0 errors (9,850 + 25,984 rows) |
| **Degradation reproducibility** (`--repro-all`) | ✅ **25,984 / 25,984** |
| **Full-pipeline reproducibility** (rerun every stage, compare manifest SHA-256 before/after) | ✅ **7 / 7 manifests identical** (`evidence/p2t8_hashes_{before,after}.txt`) |
| Preprocessing failure handling (pytest) | ✅ 55 passed |
| License/source traceability | ✅ every reference carries license, source URL, and status |

**Final manifest hashes (SHA-256 prefix)**

| Manifest | Hash |
|---|---|
| `fairface_files_v1` | `2e864d44…` |
| `references_v1` | `6eebe9a3…` |
| `degrade_pilot_v1` | `c8bafe78…` |
| `splits` | `94bdb5cb…` |
| `selection` | `92270e1e…` |
| `pairs_dev_pilot` | `79cf9aa5…` |
| `pairs_test_core` | `d9a5e72d…` |

## 4. Comparison With Phase 0 Requirements

| Phase 0 requirement | Phase 2 status |
|---|---|
| Degradations DG1–DG8 (master §6) | ✅ all covered (low-res, blur, motion, noise, JPEG, exposure, contrast, two combined chains) |
| Severity levels Mild→Extreme | ✅ (Extreme where meaningful) |
| Identity must not be changed by data prep | ✅ references unmodified; only global degradation operators |
| Same-identity / different-quality testing | ⚠️ **synthetic only** (ref ↔ degraded). Cross-image same-identity pairs (E3) need DS02/DS03. |
| Real-world evaluation mandatory (master §8) | ❌ **blocked** (G-3) |
| Fairness strata | ✅ race × gender balanced subsets; demographic-neutral acceptance verified |
| Face sizes | ✅ via ×2/×4/×8/×16 (112 → 14 px) |
| Pose variation | ⚠️ recorded (yaw/roll proxies), not stratified yet; FairFace is mostly near-frontal |
| Lighting | ✅ synthetic exposure/contrast only |
| Edge cases (no face, multiple, tiny, occlusion, pose, crop) | ✅ preprocessing contract + tests (occlusion not detectable, documented) |
| Determinism (C4) | ✅ demonstrated |
| No-face → never invent (C2) | ✅ tested in both modes |
| Thresholds TBD, not invented | ✅ sample sizes provisional; identity-screen threshold derived from data with a stated method |
| High-res (≥ 512) GT | ❌ references are 224 px (FairFace native); DS02 not curated (G-5) |

## 5. Phase 1 Open Items — Status After Phase 2

| ID | Item | Status |
|---|---|---|
| G-1 | No NVIDIA GPU | Unchanged (not needed for Phase 2) |
| G-2 | Only one eligible identity evaluator | Unchanged. SFace used only for offline leakage screening. |
| G-3 | No real-world data | **Still open.** Legally reviewable path delivered (P2-T5). SCface confirmed INELIGIBLE. |
| G-4 | No license reviewer | Unchanged. Blocks DS03 approval, the Casual Conversations v2 review, and production eligibility of any model. |
| G-5 | No high-res public-domain portraits | Unchanged. FairFace-only references (224 px). |
| G-6 | Production target unconfirmed | Unchanged |

## 6. Completed · Partial · Blocked · Legally Unresolved

| Category | Items |
|---|---|
| **Completed** | FairFace verification; clean references; degradation generator; preprocessing module + tests; splits, schemas, validator; synthetic benchmark `fes-syn-v1`; reproduction script |
| **Partial** | Real-world data (plan/spec only); pose stratification (recorded, not stratified); final sample sizes (provisional, pending power analysis) |
| **Blocked** | Real-world evaluation data (Legal approval of DS03); high-res GT (DS02 curation); ≥ 2 identity evaluators (G-2) |
| **Legally unresolved** | DS03 consent form + DPDP review; Casual Conversations v2 agreement; FairFace per-photo (YFCC) rights (residual, P1-T6); training-data inheritance for all learned candidates (P1-T3 R4) |

## 7. New Findings Worth Carrying Forward

1. **FairFace's official train/val split is not leakage-free.** It has cross-split exact and near duplicates, and 1,159 dev×test pairs with identity similarity far above chance. Anyone using FairFace val as a test set without screening will get optimistic results.
2. **FairFace labels are inconsistent on duplicate images** (63 groups).
3. **All FairFace images share a Q≈75 JPEG floor.** JPEG-severity profiles were set below it.
4. **FFHQ framing does not fit FairFace chips.** Benchmark adapters must convert framing using the recorded transform, and exclude out-of-source pixels from metrics, identically for every candidate.
5. **Fixed 1:1 recognizer thresholds are invalid for 1:N leakage screening** (0.363 → 90% false flags). Thresholds must account for the number of comparisons.

## 8. What Is Ready for the Next Phase

- **`fes-syn-v1`**:
  - dev_pilot (112 × 32) for pilot variance and threshold calibration (TH-1…TH-11 method, P0-T4);
  - test_core (700 × 32) for provisional headline synthetic results;
  - all balanced across 14 race × gender strata.
- Reusable modules: `fes_data.preprocess` (detection/alignment contract), `fes_data.degrade`, validators, schemas.
- Per-reference FFHQ-512 transforms, ready for identical benchmark adapters (P1-T5 §5.1).

**Not ready:** real-world evaluation, high-res GT, a second identity evaluator, and GPU performance measurement. Per Phase 0 these are required **before a production model can be selected**.

## 9. Phase 2 Gate

### 🟡 YELLOW

**Evidence for "ready":** every completed item is verified by automated checks and a full byte-identical rerun (§3). The synthetic benchmark can drive candidate benchmarking on CPU (quality, synthetic identity self-consistency, fairness strata) immediately.

**Why not GREEN:**
1. No real-world low-quality data. It is mandatory per the Phase 0 master §8, and synthetic data is not a substitute.
2. No high-res GT.
3. Legal items are unresolved (DS03 consent, license reviewer).

These stop the project from reaching a **production model decision**. They do not stop the next benchmarking phase from starting on `fes-syn-v1`.

**Why not RED:** the dataset foundation exists, is traceable to the raw source, and is reproducible. All failure behavior is defined and tested. Nothing required for synthetic benchmarking is missing.

**This gate does not select or imply a final production model.**

## 10. Prerequisites / Recommendations for the Next Phase

1. **Name a license/compliance reviewer.** Approve the DS03 consent form and spec, then run the 30-participant pilot and validate it with `p2_validate_collection.py`.
2. Curate **DS02** (public-domain high-res portraits) for 512-px GT and cross-image identity pairs.
3. Approve a **second identity evaluator** for evaluation-only use (G-2).
4. Implement **benchmark adapters** that apply the recorded FFHQ transform identically for all candidates, with an out-of-source mask excluded from metrics.
5. Run the **dev_pilot** first. Fix the thresholds and final N (power analysis) **before** looking at test_core results.
6. Provision the **GPU** for the performance track (G-1).
7. Mirror datasets and manifests to an internal artifact store. Large files remain git-ignored.
