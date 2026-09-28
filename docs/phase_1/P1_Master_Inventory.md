# Phase 1 — Master Inventory

| Field | Value |
|---|---|
| Date | 2026-09-28 |
| Authority | Consolidates P1-T1 … P1-T9. Phase 0 Master Specification remains authoritative for requirements. |
| Machine-readable sources | `model_candidates.csv`, `model_license_matrix.csv`, `detection_alignment_candidates.csv`, `dataset_license_matrix.csv`, `model_acquisition_manifest.csv`, `data/registry/*.csv` |

> **No final production model is selected.** This inventory defines a benchmark-ready candidate set only.

---

## 1. Reconciliation Register (Phase 1 inconsistencies, resolved explicitly)

| # | Where | Inconsistency | Resolution |
|---|---|---|---|
| I1 | P1-T1 §5 vs P1-T8 | P1-T1 planned the host `.venv` as the ML environment. P1-T8 found Windows Smart App Control blocks PyTorch DLLs there. | **Docker CPU image `fes-dev:cpu-p1` is the ML environment.** `.venv` is kept for non-PyTorch tooling only. The policy was not bypassed. |
| I2 | P1-T2 CSV `prelim_license_flag` vs P1-T3 | P1-T2 used preliminary flags (ELIGIBLE-EVAL / REVIEW). | **P1-T3 statuses are authoritative.** They are the same in substance; P1-T3 adds the production column. |
| I3 | P1-T4 CSV row E03 | First written as ELIGIBLE, then corrected to REVIEW_REQUIRED, because rule R3 needs the same rights holder and the original trainer of `resnet18_110` is unverified. | REVIEW_REQUIRED (final). |
| I4 | P1-T6 matrix DS01 size | Written first from the paper total (108,501). The acquired archive contains **97,698** images (86,744 train / 10,954 val). | Corrected to the verified count. |
| I5 | P1-T2/T3 counts | Early drafts stated 46 rows, 11 ELIGIBLE, 16 REVIEW. | Corrected to 45 rows; 12 ELIGIBLE (+C01), 15 REVIEW, 15 INELIGIBLE. Verified against the CSVs. |
| I6 | P0 master §8 / P0-T4 §5 (≥ 2 identity evaluators) vs P1-T4 §5 | Only **one** evaluator (SFace) is ELIGIBLE. | **Open gap G-2.** Phase 2 prerequisite. |
| I7 | P0 master §8 (real-world evaluation mandatory) vs P1-T6/T7 | No ELIGIBLE real-world degraded dataset exists yet. | **Open gap G-3.** DS03 in-house, or reviewer approval of DS05/DS12. |
| I8 | P1-T5 A2 "VQFR" vs P1-T9 | Only **VQFR v1** is in the official linked folder. | Benchmark VQFR v1; v2 recorded as unavailable. |
| I9 | P0 master A-4 (commercial use unconfirmed) | Confirmed by the project user on 2026-09-28 (P1-T1). | A-4 → CONFIRMED (verbal; written record recommended). |
| I10 | P1-T5 detector main track (facexlib) vs P1-T4 recommendation order (YuNet first) | Different emphasis. | Both are kept: **facexlib = main benchmark track** (matches how candidates were trained); **YuNet = sensitivity track and lean-service candidate**. |

## 2. Enhancement / Restoration Candidates

| Group | Candidates |
|---|---|
| **Approved for benchmarking (Tier A, acquired)** | RestoreFormer++ (M03, A01) · VQFR v1 (M05, A05) · Real-ESRGAN x4plus (M23, A02) · SwinIR-L real-SR x4 (M24, A03) · Chain FBCNN (M30, A04) → NAFNet GoPro/SIDD (M28, A06/A07) → Real-ESRGAN |
| **Baselines** | B0 identity · B1 bicubic/Lanczos · B2 classical (NLM/bilateral + unsharp + optional CLAHE) — OpenCV, ELIGIBLE |
| **Blocked (REVIEW_REQUIRED; Tier B)** | GFPGAN v1.4 (NC third-party code) · PMRF (embedded DifFace NC weights) · DiffBIR v2.1 (SD2.1 RAIL++-M, source withdrawn) |
| **Other REVIEW_REQUIRED (not shortlisted)** | DAEFR, DR2, BFRffusion, HiFaceGAN, OSDFace, AuthFace, InstantIR, OSEDiff, SeeSR, PiSA-SR, DreamClear, FaithDiff |
| **Rejected (INELIGIBLE)** | CodeFormer, DifFace, PGDiff, StableSR, ResShift, InvSR (S-Lab NC) · FaceMe (Pi-Lab NC) · DFDNet, DMDNet, PSFR-GAN, DT-BFR (CC BY-NC-SA) · MPRNet (academic) · SUPIR, HYPIR (proprietary NC) · GPEN (no license) |
| **Reserves (ELIGIBLE, not acquired)** | RestoreFormer v1, BSRGAN, HAT, Restormer, SCUNet*, DRUNet* (*weight location TBC) |
| **Excluded on scope** | FaceFusion (face swap) |

## 3. Detection / Alignment / Evaluator Inventory

| Component | Role | Status | Acquired |
|---|---|---|---|
| facexlib RetinaFace-R50 (D05) | Main-track detector + 5 landmarks | ELIGIBLE | ✅ A08 |
| YuNet (D01) | Sensitivity track / lean detector | ELIGIBLE | ✅ A09, A10 |
| MediaPipe (D03) | Pose estimation (optional) | ELIGIBLE | ❌ (Phase 2 if needed) |
| libfacedetection (D02) | C++ variant of YuNet | ELIGIBLE | — |
| dlib 5-pt, MTCNN, serengil RetinaFace, YOLO-face, DSFD, FAN, SPIGA, PIPNet | Alternatives | REVIEW_REQUIRED | — |
| InsightFace detectors, STAR | — | INELIGIBLE | — |
| **SFace (E01)** | Identity evaluator, **test-only** | ELIGIBLE | ✅ A11 |
| AdaFace (E02), resnet18_110 (E03), facenet-pytorch (E06) | 2nd evaluator options | REVIEW_REQUIRED | — |
| InsightFace ArcFace (E04), EdgeFace (E05) | — | INELIGIBLE | — |

Alignment: FFHQ 5-point similarity transform at 512×512, inverse transform kept for paste-back (P1-T4 §4).

## 4. Evaluation Datasets

| Dataset | Role | Status | Acquired |
|---|---|---|---|
| **DS01 FairFace** | E2 synthetic + fairness strata (224 px GT) | ELIGIBLE (CC BY 4.0) | ✅ A12–A14 |
| **DS02 public-domain portraits** | E1 high-res synthetic + E3 cross-image identity | ELIGIBLE per item | ❌ to curate (DB-1) |
| **DS03 in-house consented** | E4 real paired | ELIGIBLE once consented | ❌ does not exist (DB-2) |
| DS05 Casual Conversations v2, DS12 SCface, DS04 Unsplash Lite, DS06 FFHQ PD subset, DS09 LFW, DS10 XQLFW, DS11 TinyFace, DS13 QMUL-SurvFace, DS19 | Conditional | REVIEW_REQUIRED | — |
| FFHQ, CelebA*, WIDER FACE, UTKFace, DigiFace-1M, Face Synthetics, CelebRef-HQ, VGGFace2/MS-Celeb, AuthFace set | — | INELIGIBLE | — |

## 5. Synthetic Degradation Strategy

Seeded, versioned OpenCV/NumPy pipeline (P1-T7 §4–§5):
- DG1–DG7 single degradations at Mild/Moderate/Severe/Extreme;
- DG8a classical BFR chain;
- DG8b second-order realistic chain.

Each sample has a JSON sidecar (parameters, seed, pipeline version). Parameters are starting points, to be calibrated against real samples in Phase 2.

## 6. Licenses (summary)

| Status | Acquisition/eval | Production |
|---|---|---|
| ELIGIBLE | 12 models + classical; 4 detection/landmark; 1 evaluator; 3 datasets (1 acquired) | Classical baselines only (+ FairFace as evaluation data) |
| REVIEW_REQUIRED | 15 models; 9 detection/landmark; 3 evaluators; 9 datasets | **All learned models** (FFHQ / DIV2K / WIDER FACE training-data inheritance, R4) |
| INELIGIBLE | 15 models; 2 detection; 2 evaluators; 9 datasets | same |

## 7. Versions / Revisions

All acquired assets are pinned by URL, source-repo commit, and SHA-256 in `model_acquisition_manifest.csv`. Git baseline: `80127d8` (Phase 0). Phase 1 commit: see the final report.

## 8. Environment & GPU

| Item | Status |
|---|---|
| ML environment | Docker `fes-dev:cpu-p1` (Python 3.11.16, torch 2.14.0+cpu, OpenCV 5.0.0), digest-pinned base, lock file |
| Host venv | Non-PyTorch tooling only (Smart App Control) |
| NVIDIA GPU | ❌ **None.** `docker/Dockerfile.gpu` is a specification only. |

## 9. Phase 0 Requirement → Phase 1 Plan/Asset Traceability

| Phase 0 requirement | Phase 1 plan / asset | Status |
|---|---|---|
| Scope: restoration only, no recognition (master §3) | Evaluators test-only (P1-T4 §5); no recognition module acquired for runtime | ✅ |
| Feature-Match external | Not referenced anywhere in Phase 1 | ✅ |
| Input: full image + crop; face-count policy (master §4, C2) | Edge-case behavior mapped (P1-T4 §6) | ✅ planned |
| Detection/alignment supporting components | D05 + D01 acquired; alignment method (P1-T4 §4) | ✅ |
| Output: standard aligned crop + paste-back | Held-constant paste-back (P1-T5 §5.1) | ✅ planned |
| Degradations DG1–DG8 (master §6) | Degradation matrix (P1-T7 §4) | ✅ planned |
| Lighting/contrast conditional (C7) | DG6/DG7 + CLAHE baseline | ✅ planned |
| Identity preservation gate (master §7) | ID-1…ID-5 design (P1-T7 §6) | ⚠️ only **1** eligible evaluator (G-2) |
| ≥ 2 independent evaluators | SFace + ? | ⚠️ **G-2** |
| Evaluation: synthetic data | FairFace acquired; DS02 to curate | ⚠️ partial (high-res GT pending, DB-1) |
| Evaluation: real-world data (mandatory) | DS03 / conditional DS05, DS12 | ❌ **G-3** |
| Fairness strata | FairFace labels | ✅ |
| Baselines B0–B4 | Classical + Tier A | ✅ |
| Performance (latency, P95, VRAM) | GPU spec only | ❌ **G-1** (no GPU) |
| Reliability (fuzz/load/soak) | Phase 2 harness design | ✅ planned (GPU needed for load realism) |
| Determinism (C4) | Seeds + deterministic flags (P1-T5 §5.1) | ✅ planned |
| CPU runnable for dev (C5) | Docker CPU image; all Tier A are PyTorch CPU-capable | ✅ |
| Licensing hard gate (master §12) | P1-T3/T4/T6 screening; manifests | ✅ factual; ⚠️ **no reviewer** (G-4) |
| Security: safe weight loading (P0-T5 §8) | `weights_only=True` verified for all weights | ✅ |
| Privacy: no retention / no image logs | `logs/README.md` rule; datasets git-ignored | ✅ |
| Thresholds TH-1…TH-11 set before results | Validation/test split + pilot + power analysis (P1-T7 §7–§8) | ✅ planned |

## 10. Open Gaps

| ID | Gap | Blocks |
|---|---|---|
| G-1 | No NVIDIA GPU | Phase 2 performance track; final selection |
| G-2 | Only one eligible identity evaluator | Full identity gate (≥ 2 evaluators) |
| G-3 | No eligible real-world degraded data | Mandatory S-REAL evaluation; final selection |
| G-4 | No license reviewer | Production eligibility of any learned model; Tier B; REVIEW datasets |
| G-5 | DS02 not curated | High-res GT (E1) and cross-image identity (E3) |
| G-6 | Production target (A-3) unconfirmed | GPU class for the performance track |
