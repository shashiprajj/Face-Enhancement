# P2-T1 — Phase 2 Baseline & Dataset Inventory

| Field | Value |
|---|---|
| Phase / Task | Phase 2 / P2-T1 |
| Status | Complete |
| Date | 2026-09-28 |
| Starting commit | `c295b26` (Phase 1), baseline `80127d8` (Phase 0) |
| Inputs read | P0 Master Specification, Phase 0/1 final reports, P1 Master Inventory, P1-T3/T6 license matrices, P1-T7 evaluation plan, P1-T9 acquisition manifest |
| Machine-readable | [`data/registry/phase2_inventory.csv`](../../data/registry/phase2_inventory.csv) |

---

## 1. Starting Point

- **Phase 0:** requirements are authoritative. Key rules: identity preservation is a hard gate; real-world evaluation is mandatory; thresholds are TBD with defined methods.
- **Phase 1 gate: YELLOW.** A benchmark-ready Tier A candidate set is acquired. Open gaps: G-1 no GPU, G-2 one eligible identity evaluator, G-3 no real-world data, G-4 no license reviewer, G-5 public-domain portrait set not curated, G-6 production target unconfirmed.
- **Phase 2 scope:** dataset acquisition and preparation only. No enhancement, no model selection, no API, no Feature-Match.

## 2. Stakeholder Status (unchanged since 2026-09-28 Phase 1)

| Item | Status |
|---|---|
| Commercial use | CONFIRMED (project user, verbal) |
| License reviewer | **Not named** |
| NVIDIA GPU | **None** |
| Production target | UNCONFIRMED |
| Approval of any REVIEW_REQUIRED dataset (Casual Conversations v2, SCface, …) | **None** |

## 3. Verified Asset Inventory

Verification command (2026-09-28): `sha256sum` over all files. **Every hash matches the Phase 1 manifest.**

| Asset | Path (Phase 2) | Size | License | Phase 2 use |
|---|---|---|---|---|
| FairFace images (A12) | `datasets/raw/DS01_fairface/fairface-img-margin025-trainval.zip` | 578.2 MB | CC BY 4.0 | Source of clean references |
| FairFace train labels (A13) | `datasets/raw/DS01_fairface/fairface_label_train.csv` | 3.8 MB | CC BY 4.0 | Dev pool, aggregate strata |
| FairFace val labels (A14) | `datasets/raw/DS01_fairface/fairface_label_val.csv` | 0.45 MB | CC BY 4.0 | Test pool, aggregate strata |
| YuNet 2026may (A09) | `models/D01_yunet/` | 0.23 MB | MIT | **Phase 2 detector + 5 landmarks** |
| SFace (A11, test-only) | `models/E01_sface/` | 38.7 MB | Apache-2.0 | Offline leakage screening only |
| RetinaFace-R50 (A08) | `models/D05_facexlib/` | 109.5 MB | MIT | Deferred to benchmark phase |
| Restoration weights (A01–A07) | `models/M*/` | ≈ 1.8 GB | Apache/MIT/BSD (R3) | **Not used** in Phase 2 |

**The only dataset acquired in Phase 1 is FairFace.** DS02 (public-domain portraits) was not curated; DS03 (in-house) does not exist; all other datasets are REVIEW_REQUIRED or INELIGIBLE (P1-T6).

**Relocation, not re-download:** the FairFace files were **moved** from `datasets/DS01_fairface/` to `datasets/raw/DS01_fairface/`. The hashes were re-verified after the move and match. The files were set **read-only** (Windows `attrib +R`). The Phase 1 manifest keeps its historical paths; the Phase 2 path mapping is in `phase2_inventory.csv`, and `data/registry/DS01_fairface_files.csv` was updated.

## 4. Directory & Naming Conventions

```
datasets/                          (git-ignored contents)
  raw/<DSID>_<name>/               immutable, as acquired, read-only
  work/<DSID>_<name>/<version>/    verified working copy (e.g. extracted images), regenerable from raw by script
  derived/<DSID>_<name>/<stage>-<version>/   e.g. references/ref-v1 (aligned clean faces)
  benchmark/<bench-name>-<version>/          e.g. fes-syn-v1 (paired clean/degraded + manifests)
data/                              (tracked)
  registry/                        dataset registry, Phase 2 inventory
  manifests/                       canonical manifests (CSV) produced by scripts
configs/phase2/                    YAML configs (all pipeline parameters and seeds)
src/fes_data/                      reusable Python package (verification, detection/alignment, degradation, splits)
scripts/                           CLI entry points; scripts/dock.sh runs them in the pinned container
tests/                             pytest suite
```

**ID conventions**

| Entity | Pattern | Example |
|---|---|---|
| Source sample | `ff-<split>-<n>` (from the FairFace file name) | `ff-val-00123` |
| Clean reference | `<sample_id>.ref-v1` | `ff-val-00123.ref-v1` |
| Degraded sample | `<ref_id>.<profile>-<severity>` | `ff-val-00123.ref-v1.blur-moderate` |
| Pipeline version | semantic, in the config | `preproc 1.0.0`, `degrade 1.0.0` |

**Rules**
- Raw data is never written to.
- Every derived file is produced by a script from a config and carries its source path and SHA-256 in a manifest.
- Large files are git-ignored. Manifests, configs, and code are tracked.

## 5. Environment for Phase 2

- Docker image **`fes-dev:cpu-p2`** = the `cpu-p1` environment plus `pytest==9.1.1` (`requirements-dev.txt`) and `PYTHONPATH=/workspace/src`. `requirements.lock.txt` was regenerated.
- All scripts run via `scripts/dock.sh` (Smart App Control blocks PyTorch on the host; OpenCV works in both).
- No GPU. Nothing in Phase 2 needs one.

## 6. How Phase 2 Uses the Data

| Task | Use |
|---|---|
| P2-T2 | Verify the FairFace zip vs labels. Build a working copy under `datasets/work/` with per-file SHA-256. |
| P2-T3 | YuNet detection → FFHQ-template alignment at native scale (no upsampling beyond a stated limit) → clean references with quality stats |
| P2-T4 | Seeded degradation profiles (DG1–DG8 × severities) on references → paired set |
| P2-T5 | Real-world data: decision matrix + executable internal collection spec (no approved external source exists) |
| P2-T6 | Reusable detection/alignment module with edge-case behavior and tests |
| P2-T7 | Splits: FairFace **train → dev pool**, **val → test** (P1-T7 §7). Leakage screening, canonical schema, validators. |

## 7. Unresolved Issues Carried Into Phase 2

| ID | Issue | Effect on Phase 2 |
|---|---|---|
| G-1 | No NVIDIA GPU | None for data prep; blocks the later performance benchmark |
| G-2 | One eligible identity evaluator | SFace is used only for leakage screening here |
| G-3 | No real-world degraded data | P2-T5 produces a plan/spec only. **Synthetic data is not a substitute.** |
| G-4 | No license reviewer | FairFace is used under its declared CC BY 4.0. Production eligibility is unaffected by Phase 2. |
| G-5 | No high-res public-domain portraits | References are FairFace-only (≈ 224 px source) → the high-res (512) GT gap remains. Addressed in P2-T3 (native-scale references). |
| FairFace identity | No identity labels | Leakage cannot be excluded by label. Mitigated by embedding-based near-duplicate screening (P2-T7). |

## 8. Done-When Check

The next task can identify: **what** data exists (FairFace only, 97,698 images), **where** (`datasets/raw/DS01_fairface/`), its **source/license** (CC BY 4.0, ELIGIBLE for evaluation), and **how** Phase 2 uses it (§6). ✅
