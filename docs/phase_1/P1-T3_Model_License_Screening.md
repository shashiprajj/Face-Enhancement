# P1-T3 — Model License & Commercial-Use Screening (HARD GATE)

| Field | Value |
|---|---|
| Phase / Task | Phase 1 / P1-T3 |
| Status | Complete (factual screening). **Not legal advice. Not a reviewer sign-off.** |
| Date checked | 2026-09-28 (all sources) |
| Depends on | P1-T1 (commercial use CONFIRMED by project user; license reviewer NOT named), P1-T2 |
| Matrix | [`model_license_matrix.csv`](model_license_matrix.csv) |

---

## 1. Basis of Screening

- **Intended use:** commercial product (network service). Confirmed by the project user on 2026-09-28 (P1-T1 §2). Screening is therefore **not conditional**, and the BLOCKED status is not needed.
- **License reviewer:** **none named.** Every status below is a *factual finding*. **ELIGIBLE means only that the evidence found supports the use; it is not approval.** Owner/reviewer sign-off is still required (P0 master §12).
- **Primary sources only:** official GitHub LICENSE files, official READMEs, official Hugging Face model cards (API `cardData.license`), dataset owners' pages. Unofficial HF re-uploads were ignored.

## 2. Screening Rules

| Rule | Statement |
|---|---|
| **R1** | Code license, weight terms, training-data terms, and inference-path third-party components are assessed **separately**. |
| **R2** | An explicit non-commercial, academic, or proprietary term on **code or weights in the inference path** → **INELIGIBLE**. |
| **R3** | If weights have **no separate license** and are distributed by the **same rights holder** as a release asset of, or from a location linked in the README of, a repository under a permissive license (MIT/BSD/Apache), the weights are *treated as covered by that license* **for acquisition and internal evaluation**. The reviewer must confirm R3 for production. |
| **R4** | **Training-data terms** (e.g., FFHQ NC-SA, DIV2K academic-only) do **not** block acquisition for internal evaluation. They **do** make the production status REVIEW_REQUIRED, because whether dataset terms carry over to trained weights is an unsettled legal question and is not interpreted here. |
| **R5** | No license file at all → no grant of rights → INELIGIBLE, unless other artifacts carry an explicit license. In that case → REVIEW_REQUIRED (partial grant). |
| **R6** | Conflicting statements, use-based-restriction licenses (Open RAIL / RAIL++), copyleft, unverified base models, or unavailable official sources → REVIEW_REQUIRED. |
| **R7** | Only candidates that are **ELIGIBLE for acquisition/evaluation** may be downloaded in P1-T9 (the project user's instruction: "Download ELIGIBLE only"). |

Two status columns are therefore reported: **`status_acquisition_eval`** (may we download and benchmark internally?) and **`status_production`** (may we ship it?).

## 3. Results Summary

| Status | Acquisition / evaluation | Production |
|---|---|---|
| **ELIGIBLE** | 12 (M03, M04, M05, M23, M24, M25, M26, M27, M28, M29*, M30, M31*) + C01 | **1** (C01 classical OpenCV only) |
| **INELIGIBLE** | 15 | 15 |
| **REVIEW_REQUIRED** | 15 | 27 (all learned models not INELIGIBLE) |
| **BLOCKED** | 0 (commercial use confirmed) | 0 |

\* M29 SCUNet and M31 DRUNet: ELIGIBLE on license, but the official weight location must be confirmed at acquisition time.

> **No learned model is production-eligible today.** That is expected: it follows from R4 (FFHQ/DIV2K training data) and the absence of a license reviewer. It is the main reviewer question for Phase 2 (see §6).

## 4. Candidate Status Detail

### 4.1 ELIGIBLE (acquisition & internal evaluation)

| ID | Candidate | Code | Weights (rule) | Training data flag | Production |
|---|---|---|---|---|---|
| M03 | **RestoreFormer++** | Apache-2.0 | GitHub release v1.0.0 (R3) | FFHQ | REVIEW_REQUIRED |
| M04 | RestoreFormer | Apache-2.0 | same release (R3) | FFHQ | REVIEW_REQUIRED |
| M05 | **VQFR** | Apache-2.0 (3rd-party: OpenCV only) | Author Google Drive via README (R3) | FFHQ | REVIEW_REQUIRED |
| M23 | **Real-ESRGAN** | BSD-3 | GitHub releases (R3) | DIV2K (academic) | REVIEW_REQUIRED |
| M24 | **SwinIR** | Apache-2.0 | GitHub release v0.0 (R3) | DIV2K etc. | REVIEW_REQUIRED |
| M25 | BSRGAN | Apache-2.0 | Google Drive (R3) | DIV2K/FFHQ etc. | REVIEW_REQUIRED |
| M26 | HAT | Apache-2.0 | Google Drive (R3) | DF2K/ImageNet | REVIEW_REQUIRED |
| M27 | Restormer | MIT | Google Drive (R3) | GoPro/SIDD etc. | REVIEW_REQUIRED |
| M28 | **NAFNet** | MIT (+Apache BasicSR) | Google Drive (R3) | GoPro/SIDD | REVIEW_REQUIRED |
| M29 | SCUNet | Apache-2.0 | location TBC | not verified | REVIEW_REQUIRED |
| M30 | **FBCNN** | Apache-2.0 | GitHub release v1.0 (R3) | DIV2K/Flickr2K | REVIEW_REQUIRED |
| M31 | DRUNet (KAIR) | MIT | location TBC | not verified | REVIEW_REQUIRED |
| C01 | OpenCV classical | Apache-2.0 | n/a | n/a | **ELIGIBLE** |

### 4.2 INELIGIBLE (evidence conflicts with commercial use)

| ID | Candidate | Reason (primary-source evidence) |
|---|---|---|
| M02 | CodeFormer | S-Lab License 1.0: "use for non-commercial purpose" |
| M06 | GPEN | No LICENSE (R5); README: model withdrawn "due to commercial issues" |
| M07 | DifFace | S-Lab License 1.0 |
| M13 | PGDiff | S-Lab License 1.0 |
| M14 | DFDNet | CC BY-NC-SA 4.0 |
| M15 | DMDNet | CC BY-NC-SA 4.0 (+ CelebRef-HQ non-commercial) |
| M16 | PSFR-GAN | CC BY-NC-SA 4.0 |
| M20 | DT-BFR | CC BY-NC-SA 4.0 |
| M21 | FaceMe | Pi-Lab License 1.0 (non-commercial) |
| M32 | MPRNet | Academic Public License: "Commercial Use ✗" |
| M33 | StableSR | S-Lab License 1.0 |
| M34 | ResShift | S-Lab License 1.0 |
| M35 | InvSR | S-Lab License 1.0 |
| M36 | SUPIR | SUPIR Software License Agreement: commercial use "strictly prohibited without express written permission" |
| M37 | HYPIR | HYPIR Software License Agreement (same licensor, same terms) |

### 4.3 REVIEW_REQUIRED (ambiguity or missing evidence)

| ID | Candidate | Specific question for reviewer |
|---|---|---|
| M01 | **GFPGAN** | LICENSE lists NVIDIA StyleGAN2 code ("non-commercially… research or evaluation purposes only") and DFDNet-derived code (CC BY-NC-SA). Are these in the v1.3/1.4 clean inference path? Does the FFHQ StyleGAN2 prior carry NVIDIA NC terms? |
| M09 | **PMRF** | MIT-tagged checkpoint embeds DifFace's S-Lab-licensed SwinIR MMSE model |
| M08 | DiffBIR | SD2.1 base: Open RAIL++-M use restrictions; official source withdrawn |
| M10 | DAEFR | Code has no license; weights tagged Apache-2.0 |
| M11 | DR2 | Weight terms not verified |
| M12 | BFRffusion | SD2.1 base |
| M17 | HiFaceGAN | GPL-3.0 copyleft; weights unverified |
| M18 | OSDFace | No code license; weights tagged CreativeML OpenRAIL-M; SD2.1 base |
| M19 | AuthFace | LICENSE file Apache-2.0 vs README badge "Non-commercial" |
| M22 | InstantIR | SDXL base (Open RAIL++-M) |
| M38–M42 | OSEDiff, SeeSR, PiSA-SR, DreamClear, FaithDiff | Base-model RAIL terms / no code license / unverified base |

## 5. Obligations for ELIGIBLE Assets (factual, from license texts)

- **Apache-2.0:** retain LICENSE and NOTICE, state changes. Includes an explicit patent grant, which terminates on patent litigation.
- **MIT / BSD-3:** retain copyright and permission notices. BSD-3 also forbids using contributor names for endorsement.
- All: record the license text alongside each acquired asset (P1-T9) and in a future NOTICE file (P0-T5 §6).

## 6. Questions for the Designated Reviewer (when named)

1. **Rule R3.** May weights with no separate license, released by the same author from a permissively licensed repo, be treated as licensed under the repo license?
2. **Rule R4 / training data.** Is commercial use of weights trained on FFHQ (images NC; dataset CC BY-NC-SA 4.0; "not intended for facial recognition") or DIV2K ("academic research purpose only") acceptable? If not, the realistic options are:
   - (a) negotiate commercial licenses,
   - (b) retrain a permissively licensed architecture (e.g., RestoreFormer++, VQFR) on commercially licensed data,
   - (c) classical or other non-learned approaches.
3. **Open RAIL++-M.** Are its use-based restrictions acceptable? (Relevant only if a diffusion candidate is pursued.)
4. **GFPGAN.** Source-level review of NC-derived components.
5. **Evaluation-only use of INELIGIBLE models.** Could NC models such as CodeFormer be used *only* as internal reference points in benchmarking? This task assumes **no**, because evaluation that selects a commercial product is arguably commercial activity. Nothing INELIGIBLE will be downloaded.

## 7. Acceptance Criteria

| Criterion | Status |
|---|---|
| Every shortlisted candidate has a documented license status | ✅ 42 + C01, in matrix |
| No model called commercially safe without evidence | ✅ No learned model is production-ELIGIBLE; acquisition-ELIGIBLE rests on explicit licenses + documented R3 |
| Evidence traceable to primary sources with URLs and dates | ✅ matrix columns `code_license_source`, `weights_source`, `date_checked` |
| Ambiguity not treated as approval | ✅ 15 REVIEW_REQUIRED |
| Only ELIGIBLE may proceed to acquisition, subject to sign-off | ✅ R7 |

**Eligible for the next stage (P1-T5 shortlisting / P1-T9 acquisition):** M03 RestoreFormer++, M04 RestoreFormer, M05 VQFR, M23 Real-ESRGAN, M24 SwinIR, M25 BSRGAN, M26 HAT, M27 Restormer, M28 NAFNet, M29 SCUNet*, M30 FBCNN, M31 DRUNet*, C01 OpenCV classical.
