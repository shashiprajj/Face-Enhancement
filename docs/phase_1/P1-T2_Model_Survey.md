# P1-T2 — Hugging Face / Open-Source Model Survey

| Field | Value |
|---|---|
| Phase / Task | Phase 1 / P1-T2 |
| Status | Complete |
| Date checked | 2026-09-28 |
| Depends on | P1-T1 |
| Nature | **Discovery only — no final selection, no ranking of winners, no downloads** |
| Machine-readable | [`model_candidates.csv`](model_candidates.csv) (45 rows) |

---

## 1. Method

1. **GitHub REST API** (`/repos/{owner}/{repo}`) for every candidate. Recorded: detected SPDX license, stars, last push, default branch, archived flag.
2. **Raw LICENSE files** fetched from each official repository. This mattered because GitHub's automatic detection reports `NOASSERTION` for custom licenses: S-Lab, Pi-Lab, SUPIR, HYPIR, and the Tencent composite licenses.
3. **Official READMEs** grepped for license statements, weight hosting, and base-model dependencies.
4. **Hugging Face Hub API** (`/api/models/{id}`) for official model cards: `cardData.license`, commit `sha`, gating. Also HF search for "face restoration", "blind face", "GFPGAN", "CodeFormer", and "face super resolution".
5. **Web search** for 2025–2026 methods, to avoid tunnel vision on well-known models.
6. **Training-data terms** checked at the primary sources for the two datasets that recur across candidates (FFHQ, DIV2K).

Raw API responses and license texts were cached in the session scratchpad and are **not** committed. Every URL is in the CSV.

**Facts vs claims.** Licenses, stars, dates, release asset sizes, and HF license tags are **verified facts** (primary source, 2026-09-28). Supported degradations, quality, and identity preservation are **author claims** until Phase 2 measures them. Sizes marked "not verified" are estimates only.

## 2. Findings That Shape the Whole Pool

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| F1 | **Almost every learned face-restoration model is trained on FFHQ.** FFHQ images are licensed for "non-commercial purposes", and the dataset is CC BY-NC-SA 4.0. NVIDIA also states it is "not intended for … development or improvement of facial recognition technologies". | https://github.com/NVlabs/ffhq-dataset | Training-data inheritance is a **reviewer question for every face model**. It is tracked separately from code/weight licenses (P1-T3). |
| F2 | **DIV2K**, used by most general SR/restoration models, is "made available for academic research purpose only". | https://data.vision.ee.ethz.ch/cvl/DIV2K/ | Same issue for general models (Real-ESRGAN, SwinIR, FBCNN, HAT, …). |
| F3 | **The official Stable Diffusion 2.1 weights are no longer available** from `stabilityai` on Hugging Face. The API returns an error, and the OSDFace README says: "Since the official Stable Diffusion 2.1 is not available… download from ModelScope". | HF API; OSDFace README | Every SD2.1-based candidate (DiffBIR, OSEDiff, OSDFace, BFRffusion, PiSA-SR) would need **third-party mirrors**, which is a provenance risk. The base model is also under CreativeML Open RAIL++-M (use-based restrictions). |
| F4 | **Many strong face models are explicitly non-commercial:** S-Lab License (CodeFormer, DifFace, StableSR, ResShift, PGDiff, InvSR), Pi-Lab (FaceMe), CC BY-NC-SA (DFDNet, DMDNet, PSFR-GAN, DT-BFR), Academic (MPRNet), proprietary (SUPIR, HYPIR). | LICENSE files | Immediate license rejection for commercial use (commercial use confirmed in P1-T1). |
| F5 | **GFPGAN's LICENSE is composite.** The main project is Apache-2.0, but it lists StyleGAN2-derived code under the **NVIDIA license ("non-commercially … research or evaluation purposes only")** and code "largely modified from DFDNet" under **CC BY-NC-SA 4.0**. | GFPGAN LICENSE, §"Other dependencies" | GFPGAN is **not** a simple Apache-2.0 candidate → REVIEW. |
| F6 | **The PMRF checkpoint (MIT) embeds the SwinIR MMSE model trained by DifFace**, and DifFace is S-Lab non-commercial. README: "the MMSE model is also in the PMRF checkpoint". | PMRF README | PMRF → REVIEW, not ELIGIBLE, despite its MIT tag. |
| F7 | **Hugging Face mostly hosts unofficial re-uploads** (GFPGAN, CodeFormer, ONNX/CoreML conversions). Only a few official model cards exist: PMRF, DiffBIR-v2, InstantIR, DAEFR, OSDFace. | HF search | Re-uploads are **not primary sources** and are excluded from acquisition. |
| F8 | **Weights usually carry no separate license.** They are published as GitHub release assets or on Google Drive, linked from the official README. | Release/README review | P1-T3 needs an explicit rule for "weights inherit repo license" (P1-T3 rule R3). |

## 3. Broad Candidate Pool (45 entries)

### 3.1 Face-specific restoration (22)

| ID | Candidate | Family | Code license (verified) | Weights | Preliminary flag |
|---|---|---|---|---|---|
| M01 | GFPGAN v1.3/1.4 | StyleGAN2 prior | Apache-2.0 **+ NVIDIA NC + CC BY-NC-SA parts** | GitHub release | REVIEW |
| M02 | CodeFormer | VQ codebook + Transformer | S-Lab 1.0 (NC) | — | INELIGIBLE |
| M03 | RestoreFormer++ | VQ dictionary + cross-attn | Apache-2.0 | GitHub release v1.0.0 (294 MB) | ELIGIBLE-EVAL |
| M04 | RestoreFormer | VQ dictionary | Apache-2.0 | same release | ELIGIBLE-EVAL |
| M05 | VQFR | VQ codebook + parallel decoder | Apache-2.0 (3rd-party: OpenCV) | Google Drive (author) | ELIGIBLE-EVAL |
| M06 | GPEN | GAN prior | **No LICENSE** | — | INELIGIBLE |
| M07 | DifFace | Diffusion | S-Lab (NC) | — | INELIGIBLE |
| M08 | DiffBIR v2/2.1 | SwinIR + IRControlNet on SD2.1 | Apache-2.0 | HF apache-2.0 + SD2.1 RAIL++-M | REVIEW |
| M09 | PMRF | Posterior-mean rectified flow | MIT | HF MIT, **embeds DifFace SwinIR** | REVIEW |
| M10 | DAEFR | Dual-branch VQ | **No LICENSE** | HF apache-2.0 | REVIEW |
| M11 | DR2 | Diffusion + enhancer | MIT | not verified | REVIEW |
| M12 | BFRffusion | SD2.1 prior | MIT | SD2.1 dependency | REVIEW |
| M13 | PGDiff | Guided diffusion | S-Lab (NC) | — | INELIGIBLE |
| M14 | DFDNet | Face dictionary | CC BY-NC-SA 4.0 | — | INELIGIBLE |
| M15 | DMDNet | Dual memory dictionary | CC BY-NC-SA 4.0 | — | INELIGIBLE |
| M16 | PSFR-GAN | Semantic GAN | CC BY-NC-SA 4.0 | — | INELIGIBLE |
| M17 | HiFaceGAN | GAN | GPL-3.0 | not verified | REVIEW |
| M18 | OSDFace (CVPR'25) | One-step diffusion (SD2.1) | **No LICENSE** | HF creativeml-openrail-m | REVIEW |
| M19 | AuthFace (MM'25) | Face diffusion prior | Apache-2.0 file **vs "Non-commercial" badge** | HF, no tag | REVIEW |
| M20 | DT-BFR (WACV'25) | Diffusion prior | CC BY-NC-SA 4.0 | — | INELIGIBLE |
| M21 | FaceMe (AAAI'25) | Reference-based diffusion | Pi-Lab (NC) | — | INELIGIBLE |
| M22 | InstantIR | SDXL + DINOv2 | Apache-2.0 | HF apache-2.0; SDXL RAIL++-M | REVIEW |

### 3.2 General restoration / SR / deblur / denoise / JPEG (20)

| ID | Candidate | Task | Code license | Weights | Preliminary flag |
|---|---|---|---|---|---|
| M23 | Real-ESRGAN | Real-world SR | BSD-3 | GitHub release | ELIGIBLE-EVAL |
| M24 | SwinIR | SR / denoise / JPEG | Apache-2.0 | GitHub release v0.0 | ELIGIBLE-EVAL |
| M25 | BSRGAN | Blind SR | Apache-2.0 | Google Drive | ELIGIBLE-EVAL |
| M26 | HAT | SR | Apache-2.0 | Google Drive | ELIGIBLE-EVAL |
| M27 | Restormer | Deblur / denoise | MIT | Google Drive | ELIGIBLE-EVAL |
| M28 | NAFNet | Deblur / denoise | MIT | Google Drive | ELIGIBLE-EVAL |
| M29 | SCUNet | Real denoise | Apache-2.0 | source TBC | ELIGIBLE-EVAL (TBC) |
| M30 | FBCNN | JPEG artifacts | Apache-2.0 | GitHub release v1.0 | ELIGIBLE-EVAL |
| M31 | DRUNet (KAIR) | Denoise | MIT | source TBC | ELIGIBLE-EVAL (TBC) |
| M32 | MPRNet | Deblur/denoise | Academic (NC) | — | INELIGIBLE |
| M33 | StableSR | Diffusion SR | S-Lab (NC) | — | INELIGIBLE |
| M34 | ResShift | Diffusion SR (+face) | S-Lab (NC) | — | INELIGIBLE |
| M35 | InvSR | Diffusion SR | S-Lab (NC) | — | INELIGIBLE |
| M36 | SUPIR | SDXL restoration | Proprietary NC | — | INELIGIBLE |
| M37 | HYPIR | Diffusion-init GAN | Proprietary NC | — | INELIGIBLE |
| M38 | OSEDiff | One-step SR (SD2.1) | Apache-2.0 | SD2.1 dependency | REVIEW |
| M39 | SeeSR | SD2 SR | Apache-2.0 | SD2 dependency | REVIEW |
| M40 | PiSA-SR | SD2.1 SR | **No LICENSE** | SD2.1 dependency | REVIEW |
| M41 | DreamClear | PixArt restoration | Apache-2.0 | base not verified | REVIEW |
| M42 | FaithDiff | SDXL SR | MIT | SDXL dependency | REVIEW |

### 3.3 Other entries

| ID | Entry | Status |
|---|---|---|
| X01 | FaceFusion | **Excluded on scope.** Face swapping/manipulation contradicts P0-T1 X11. |
| X02 | NTIRE 2025 Real-World Face Restoration repo | **Reference only** (evaluation scripts, MIT). Its model zoo mixes third-party models. |
| C01 | Classical OpenCV methods | Baselines B1/B2 (Apache-2.0, no weights) |

## 4. Candidates Requiring Immediate License Rejection (commercial use)

CodeFormer, DifFace, PGDiff, StableSR, ResShift, InvSR (S-Lab NC) · FaceMe (Pi-Lab NC) · DFDNet, DMDNet, PSFR-GAN, DT-BFR (CC BY-NC-SA) · MPRNet (Academic) · SUPIR, HYPIR (proprietary; commercial license only by agreement) · GPEN (no license grant; author cites "commercial issues").

> CodeFormer is technically among the most-cited references. Its rejection is **purely licensing**, not technical. If the business wants it, the only route is a commercial license from the rights holder (S-Lab, NTU).

## 5. Candidates Needing Deeper License Verification

| Candidate | Open question |
|---|---|
| GFPGAN | Are the NVIDIA-NC and DFDNet-derived code paths part of the v1.3/1.4 "clean" inference architecture? Does the StyleGAN2 FFHQ prior carry NVIDIA's NC weight terms? |
| PMRF | Can the checkpoint be used commercially when it embeds DifFace's S-Lab-licensed SwinIR weights? |
| DiffBIR, OSEDiff, SeeSR, BFRffusion, PiSA-SR, OSDFace | SD2/2.1 base: acceptable provenance (mirror) and acceptability of Open RAIL++-M use restrictions |
| InstantIR, FaithDiff | SDXL base: Open RAIL++-M use restrictions |
| DAEFR, OSDFace, PiSA-SR | No code license |
| AuthFace | Conflicting Apache-2.0 file vs "Non-commercial" badge |
| DR2, DreamClear | Weight/base-model terms not verified |
| HiFaceGAN | GPL-3.0 copyleft implications for a network service (likely acceptable for internal use; reviewer to confirm) |
| **All learned models** | Training-data inheritance (FFHQ NC-SA, DIV2K academic-only) |

## 6. Technically Promising Candidates (technical view only, independent of license)

Based on author claims and architecture fit to Phase 0 (aligned-face restoration, identity preservation, CPU-feasible for development):

- **Face-specific, non-diffusion (fast, deterministic):** RestoreFormer++, VQFR, GFPGAN, CodeFormer*, DAEFR.
- **Face-specific, fidelity-oriented:** PMRF (explicitly optimizes for minimal distortion, which suits the identity principle).
- **Diffusion / generative:** DiffBIR, OSDFace, InstantIR. Strong perceptual quality, but a **higher hallucination risk** (P0-T3 U1/U2), heavy GPU needs, and stochasticity (C4).
- **General components** (baselines B3 and helpers): Real-ESRGAN (also background upsampler), SwinIR, BSRGAN; FBCNN (JPEG), NAFNet/Restormer (deblur/denoise), SCUNet (denoise).

*CodeFormer is technically promising but license-rejected.

## 7. Information Gaps

1. Exact parameter counts and VRAM/CPU latency for most models (Phase 2 measurement).
2. VQFR, BSRGAN, HAT, Restormer, NAFNet weights are on Google Drive with **no published checksums**. Provenance will rest on our own SHA-256 recorded at acquisition.
3. SCUNet and DRUNet weight hosting location is not yet confirmed.
4. DR2 weight source and terms.
5. Whether GFPGAN's "clean" architecture file contains NVIDIA-derived code (source-level review needed).
6. Independent (non-author) evidence of identity preservation for any candidate. None exists yet; this is exactly what Phase 2 measures.
7. The MediaPipe BlazeFace model card PDF was only partly machine-readable (P1-T4).

## 8. Acceptance Criteria

| Criterion | Status |
|---|---|
| Broad discovery (not tunnel vision) | ✅ 42 models + 2 reference/excluded + classical, across face-specific, general SR, deblur, denoise, JPEG, diffusion; includes 2025–2026 methods |
| Primary sources recorded | ✅ official repo URLs in CSV; HF official cards only |
| Licenses not guessed | ✅ from LICENSE files/READMEs/HF API. Unknowns marked "not verified" |
| Technical suitability separate from licensing | ✅ §6 vs §4–§5; separate CSV columns |
| No final winner | ✅ |

**Candidates moving to P1-T3:** all 42 models (M01–M42) plus classical baselines. P1-T3 formally assigns ELIGIBLE / INELIGIBLE / REVIEW_REQUIRED / BLOCKED.
