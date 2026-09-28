# Phase 1 Final Report — Face Enhancement Service

| Field | Value |
|---|---|
| Date | 2026-09-28 |
| Phase | 1 — Model & Dataset Discovery + Environment |
| Inventory | [`docs/phase_1/P1_Master_Inventory.md`](../docs/phase_1/P1_Master_Inventory.md) |
| **Decision gate** | **🟡 YELLOW — benchmark-ready with documented gaps. No final model selected.** |

---

## 1. Executive Summary

Phase 1 surveyed **42 restoration models** (plus reference, excluded, and classical entries), **15 detection/landmark components**, **6 identity evaluators**, and **21 datasets**. Every license was checked against primary sources, against the now **confirmed commercial-use requirement**.

Main finding: **the license gate is the dominant constraint.**
- Many strong face-restoration models are explicitly non-commercial, including CodeFormer, DifFace, SUPIR, and all S-Lab-licensed work.
- Two popular "permissive" models carry hidden non-commercial components. GFPGAN includes NVIDIA-NC and CC BY-NC-SA code; PMRF's checkpoint embeds DifFace weights.
- Nearly every learned model is trained on FFHQ, DIV2K, or WIDER FACE, whose terms are non-commercial or research-only.

So **12 models are eligible for internal evaluation, and no learned model is yet eligible for production.** Production eligibility needs a license reviewer, who has not been named.

A benchmark-ready Tier A set was shortlisted and **acquired with full provenance** (14 assets, ≈ 2.9 GB, SHA-256 verified, safe-loaded):
- **RestoreFormer++**, **VQFR v1** (face-specific)
- **Real-ESRGAN**, **SwinIR** (general SR)
- **FBCNN → NAFNet → Real-ESRGAN** chain
- classical baselines
- RetinaFace/YuNet detectors, the SFace test-only evaluator, and the FairFace dataset.

**Environment:** Windows Smart App Control blocks PyTorch on the host. ML work therefore runs in a **pinned Docker CPU image**. There is **no NVIDIA GPU**, so performance benchmarking is not possible yet.

## 2. Phase 1 Task Status

| Task | Deliverable(s) | Status |
|---|---|---|
| P1-T1 Open items & baseline | `P1-T1_Phase0_Open_Items_and_Baseline.md`; git `80127d8`; `.venv` | ✅ |
| P1-T2 Model survey | `P1-T2_Model_Survey.md`, `model_candidates.csv` | ✅ |
| P1-T3 License screening | `P1-T3_Model_License_Screening.md`, `model_license_matrix.csv` | ✅ (factual; no reviewer) |
| P1-T4 Detection/alignment | `P1-T4_Detection_Alignment_Survey.md`, `detection_alignment_candidates.csv` | ✅ |
| P1-T5 Shortlist & test design | `P1-T5_Model_Shortlist_and_Test_Design.md` | ✅ |
| P1-T6 Dataset survey | `P1-T6_Dataset_Survey_and_License_Screening.md`, `dataset_license_matrix.csv` | ✅ |
| P1-T7 Evaluation data plan | `P1-T7_Evaluation_Dataset_Plan.md` | ✅ |
| P1-T8 Environment | `P1-T8_Environment_Setup.md`, `ENVIRONMENT_MANIFEST.md`, `requirements*.txt`, `docker/` | ✅ CPU / ❌ GPU |
| P1-T9 Acquisition | `P1-T9_Acquisition_Manifest.md`, `model_acquisition_manifest.csv`, `data/registry/` | ✅ |
| P1-T10 Inventory & report | `P1_Master_Inventory.md`, this report, `PHASE_1_INDEX.md` | ✅ |

## 3. Stakeholder Status

| Item | Status |
|---|---|
| Commercial use | ✅ **CONFIRMED** by project user, 2026-09-28 (verbal; written record recommended) |
| Production hardware/deployment target | ❌ UNCONFIRMED |
| License reviewer | ❌ **Not named** |
| GPU access | ❌ **None** |

## 4. Model Inventory

| Tier | Models |
|---|---|
| **A — benchmark (acquired)** | RestoreFormer++ · VQFR v1 · Real-ESRGAN x4plus · SwinIR-L real-SR x4 · FBCNN → NAFNet (GoPro/SIDD) → Real-ESRGAN chain |
| Baselines | Identity · bicubic/Lanczos · classical (NLM/bilateral, unsharp, optional CLAHE) |
| B — conditional (reviewer) | GFPGAN v1.4 · PMRF · DiffBIR v2.1 |
| Reserves | RestoreFormer v1, BSRGAN, HAT, Restormer, SCUNet, DRUNet |

## 5. License-Screening Summary

| | Acquisition / evaluation | Production |
|---|---|---|
| ELIGIBLE | 12 models + classical | classical only |
| REVIEW_REQUIRED | 15 | 27 |
| INELIGIBLE | 15 | 15 |
| BLOCKED | 0 (commercial use confirmed) | 0 |

**Rules applied:** R1–R7 (P1-T3 §2).
- **R3:** weights inherit the permissive license of the same rights holder's repo.
- **R4:** training-data terms don't block evaluation, but they make production REVIEW_REQUIRED.

Both rules need reviewer confirmation.

## 6. Detection / Alignment Inventory

- **Main track:** facexlib RetinaFace-R50 (acquired).
- **Sensitivity / lean track:** YuNet (acquired; built into OpenCV).
- **Optional pose:** MediaPipe (ELIGIBLE; not acquired).
- InsightFace models are INELIGIBLE (non-commercial). The dlib 68-point model is unusable commercially.
- Alignment: FFHQ 5-point similarity transform to 512, with the inverse kept for paste-back.
- Edge-case behavior (no face, crop without face → reject per master C2, landmark failure, small/profile faces) is mapped in P1-T4 §6.

## 7. Dataset Inventory & Evaluation-Data Plan

| Subset | Data | Ready? |
|---|---|---|
| E2 SYN-FAIR | FairFace val (CC BY 4.0; 10,954 images; demographic strata) | ✅ acquired |
| E1 SYN-HR + E3 ID-PAIRS | Public-domain official portraits (to curate) | ❌ curation pending |
| E4 REAL-PAIRED | In-house consented capture | ❌ does not exist |
| E5 EDGE / E6 HQ-passthrough | Constructed from the above | ◐ partial (from FairFace now) |

**Not used:** FFHQ and CelebA. They are license-ineligible, and they are also the candidates' training data (contamination).

The degradation matrix covers DG1–DG8 at 4 severities, with a seeded, versioned pipeline. Splits are by identity/participant. Sample sizes come from a pilot plus power analysis (TBD, method defined).

## 8. Acquisition Manifest Summary

14 assets: 7 restoration weight files, 3 detector files, 1 evaluator, 3 FairFace files.

- All come from official GitHub releases, OpenCV Zoo pinned to a commit, or the authors' official Drive links.
- All SHA-256 values are recorded and re-verified against disk (14/14).
- VQFR matches the author's embedded hash prefix.
- All weights load with `torch.load(weights_only=True)`.
- LICENSE texts are stored next to the assets.
- No tokens were used, and no assets are in git.

## 9. Environment Summary

| Item | Value |
|---|---|
| ML runtime | Docker `fes-dev:cpu-p1` — Python 3.11.16, torch 2.14.0+cpu, torchvision 0.29.0, OpenCV 5.0.0, NumPy 2.4.6 (digest-pinned base, `requirements.lock.txt`) |
| Host | Windows 11, Core Ultra 7 255H (16C), 31.4 GB RAM, Intel Arc iGPU; **Smart App Control blocks PyTorch** on the host (not bypassed) |
| GPU image | `docker/Dockerfile.gpu` (CUDA 12.8 spec), **not built** |

## 10. GPU Readiness

❌ **Not ready.**
- No NVIDIA hardware and no CUDA. The Docker VM has no GPU passthrough.
- CPU is fine for correctness and for quality/identity evaluation, but **cannot** produce latency, P95, throughput, or VRAM results for production. Any CPU timing will be labeled "CPU dev only".

## 11. Risks / Blockers

| ID | Item | Severity | Blocks |
|---|---|---|---|
| G-1 | No NVIDIA GPU | High | Performance track; final selection (**not** the start of Phase 2) |
| G-2 | Only one eligible identity evaluator (SFace) | High | Full identity gate (≥ 2 evaluators, master §7) |
| G-3 | No eligible real-world degraded data | High | Mandatory real-world evaluation; final selection |
| G-4 | No license reviewer | High | Production eligibility of *any* learned model; Tier B; REVIEW datasets |
| G-5 | Public-domain portrait set not curated | Medium | High-res GT and cross-image identity tests |
| G-6 | Production target unconfirmed | Medium | GPU class choice |
| R-2 (P0) | **No commercially usable learned model may exist** after review (FFHQ/DIV2K inheritance) | High | Product viability. Fallbacks: commercial licenses, retraining a permissive architecture (RestoreFormer++/VQFR) on licensed data, or classical/chain approaches. |
| R-12 (new) | Google Drive-hosted weights may disappear | Low | Reproducibility. Mitigation: mirror to an internal artifact store (SHA-256 recorded). |
| R-13 (new) | Smart App Control limits host tooling | Low | Mitigated by Docker |

## 12. Rejected / Blocked Candidates

- **Rejected (INELIGIBLE, license):** CodeFormer, DifFace, PGDiff, StableSR, ResShift, InvSR, FaceMe, DFDNet, DMDNet, PSFR-GAN, DT-BFR, MPRNet, SUPIR, HYPIR, GPEN. Also InsightFace detectors/ArcFace, EdgeFace, STAR; datasets FFHQ, CelebA family, WIDER FACE, UTKFace, DigiFace-1M, Face Synthetics, CelebRef-HQ, VGGFace2/MS-Celeb, AuthFace set.
- **Blocked pending reviewer:** GFPGAN, PMRF, DiffBIR (Tier B), plus 12 other REVIEW_REQUIRED models. Detector/evaluator alternatives (AdaFace, resnet18_110, dlib 5-pt, MediaPipe full text). Datasets Casual Conversations v2, SCface, Unsplash Lite, FFHQ PD subset, LFW/XQLFW, TinyFace, QMUL-SurvFace.
- **Excluded on scope:** FaceFusion (face swapping).

## 13. What Phase 2 Should Benchmark

**Candidates:**
- A1 RestoreFormer++
- A2 VQFR v1
- A3 Real-ESRGAN x4plus
- A4 SwinIR-L real-SR x4
- A5 FBCNN → NAFNet(GoPro/SIDD) → Real-ESRGAN chain
- Baselines B0, B1, B2 (+ CLAHE variant)

**Pipeline (held constant):**
- facexlib RetinaFace-R50 main track; YuNet sensitivity track
- FFHQ 5-point alignment at 512; bicubic upsampling of LR input for face models, native LR ×4 for SR models
- identical paste-back with a bicubic background
- FP32; fixed seeds and threads (P1-T5 §5)

**Data and metrics:**
- E2 SYN-FAIR (FairFace val) over the full degradation matrix, now. Then E1/E3 once DS02 is curated, E4 once DS03 exists, and E5/E6.
- PSNR, SSIM, LPIPS; ID-1 identity gain (SFace, plus a second evaluator when approved); landmark distance; stratified fairness; edge-case status correctness.
- CPU-dev timings, labeled as such. The GPU performance track runs once G-1 is resolved.

**Rule:** thresholds TH-1…TH-11 are fixed on the validation split **before** test results are seen. No final winner is chosen on CPU-only or single-evaluator evidence.

## 14. Phase 2 Prerequisites

**To start Phase 2** (harness + CPU quality/identity on FairFace), everything is in place:
1. Commit the Phase 1 baseline (done at the end of this task).
2. Implement the candidate adapters in Phase 2. `basicsr` needs a pinned compatibility fix with the current torchvision.
3. Record the LPIPS AlexNet backbone download in the manifest on first use.

**To *complete* Phase 2** (select a production candidate):
1. **G-1:** Provision an NVIDIA GPU of the production class (P1-T1 §6; recommended: a cloud VM of the target class). Build `docker/Dockerfile.gpu`.
2. **G-2:** Approve a second identity evaluator (AdaFace or resnet18_110, for evaluation-only use), or find another permissive one.
3. **G-3:** Start the in-house consented capture (DS03), or get reviewer approval for Casual Conversations v2 / SCface.
4. **G-4:** Name a license reviewer. Resolve R3/R4 (P1-T3 §6), the Tier B questions, and the dataset questions (P1-T6 §5).
5. **G-5:** Curate the public-domain portrait set with per-item provenance.
6. **G-6:** Confirm the production target (A-3).
7. Mirror acquired assets to an internal artifact store.

## 15. Phase 1 Decision Gate

### 🟡 YELLOW — Benchmark-ready, with documented gaps

**Why not RED:**
- A license-screened Tier A candidate set, detectors, one evaluator, and an eligible evaluation dataset are **acquired, verified, and runnable** in a reproducible environment.
- The Phase 2 harness and CPU quality/identity benchmarking can start now.

**Why not GREEN:**
- **G-1** (no GPU) blocks the performance track.
- **G-2** (one evaluator) blocks the full identity gate.
- **G-3** (no real-world data) blocks the mandatory real-world evaluation.
- **G-4** (no reviewer) means no candidate can become production-eligible.

These gaps don't block *starting* Phase 2, but they **must** be resolved before Phase 2 can select a production candidate.

**This YELLOW does not imply that any model has been selected.**
