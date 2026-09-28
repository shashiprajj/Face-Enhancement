# P1-T9 — Approved Model & Dataset Acquisition

| Field | Value |
|---|---|
| Phase / Task | Phase 1 / P1-T9 |
| Status | Complete |
| Date | 2026-09-28 |
| Manifest | [`model_acquisition_manifest.csv`](model_acquisition_manifest.csv) (14 assets) |
| Registry | `data/registry/datasets.csv`, `data/registry/DS01_fairface_files.csv` |

---

## 1. Acquisition Policy Applied

1. **Re-check before every download.** Each asset was confirmed **ELIGIBLE (acquisition/evaluation)** in P1-T3 (models), P1-T4 (detection/evaluators), or P1-T6 (datasets), **and** included in the P1-T5 Tier A shortlist or the P1-T7 plan.
2. The project user's instruction: download **ELIGIBLE only**. No REVIEW_REQUIRED or INELIGIBLE asset was downloaded.
3. **Official sources only:** the repository's GitHub releases, or Google Drive links from the official README / model table, or the OpenCV Zoo repository pinned to a commit. No HF re-uploads, no mirrors.
4. **No authentication was needed.** No tokens were created, stored, or used.
5. Weights were verified with `torch.load(..., weights_only=True)` (no arbitrary code execution from pickles; P0-T5 §8).
6. Large assets are stored **outside git** (`models/`, `datasets/` are git-ignored). The manifest and registry are tracked.

## 2. What Was Acquired

| Asset | Candidate | File | Size | SHA-256 (prefix) | Source | Verification |
|---|---|---|---|---|---|---|
| A01 | RestoreFormer++ (M03) | `RestoreFormer++.ckpt` | 294.0 MB | `613fe528…` | GitHub release v1.0.0 | size = asset; safe-load ✓ |
| A02 | Real-ESRGAN x4plus (M23) | `RealESRGAN_x4plus.pth` | 67.0 MB | `4fa0d389…` | GitHub release v0.1.0 | size ✓; safe-load ✓ |
| A03 | SwinIR-L real-SR x4 GAN (M24) | `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth` | 142.5 MB | `99adfa91…` | GitHub release v0.0 | size ✓; safe-load ✓ |
| A04 | FBCNN color (M30) | `fbcnn_color.pth` | 287.8 MB | `8b0e4ef2…` | GitHub release v1.0 | size ✓; safe-load ✓ |
| A05 | VQFR v1 (M05) | `VQFR_v1-33a1fac5.pth` | 306.4 MB | `33a1fac5…` | Official Drive folder | **SHA-256 prefix matches author's filename tag**; safe-load ✓ |
| A06 | NAFNet GoPro w64 (M28) | `NAFNet-GoPro-width64.pth` | 271.8 MB | `329d3ab4…` | Official Drive (README table) | safe-load ✓ (no published checksum) |
| A07 | NAFNet SIDD w64 (M28) | `NAFNet-SIDD-width64.pth` | 464.2 MB | `cd685efa…` | Official Drive (README table) | safe-load ✓ (no published checksum) |
| A08 | RetinaFace-R50 (D05 facexlib) | `detection_Resnet50_Final.pth` | 109.5 MB | `6d1de9c2…` | facexlib release v0.1.0 | size ✓; safe-load ✓ |
| A09 | YuNet 2026may (D01) | `face_detection_yunet_2026may.onnx` | 0.23 MB | `ebafce4e…` | opencv_zoo @ `47534e2` | `cv2.FaceDetectorYN` runs ✓ |
| A10 | YuNet 2023mar (D01) | `face_detection_yunet_2023mar.onnx` | 0.23 MB | `8f2383e4…` | opencv_zoo @ `47534e2` | runs ✓ |
| A11 | SFace (E01, **test-only**) | `face_recognition_sface_2021dec.onnx` | 38.7 MB | `0ba9fbfa…` | opencv_zoo @ `47534e2` | `cv2.FaceRecognizerSF` 128-d ✓ |
| A12 | FairFace images (DS01) | `fairface-img-margin025-trainval.zip` | 578.2 MB | `654c9b04…` | Official Drive (README) | zip test ✓; 97,698 JPEGs (86,744 train / 10,954 val) |
| A13 | FairFace train labels | `fairface_label_train.csv` | 3.8 MB | `90624451…` | Official Drive | 86,744 rows ✓ |
| A14 | FairFace val labels | `fairface_label_val.csv` | 0.45 MB | `19906f69…` | Official Drive | 10,954 rows ✓ |

Full SHA-256 values, URLs, source commits, and license sources are in the CSV. **Total ≈ 2.9 GB.** The license text of each source repository, at the pinned commit, is saved next to its weights (`models/<id>/LICENSE`, `datasets/DS01_fairface/LICENSE.txt`).

## 3. Storage Layout

```
models/
  M03_restoreformerpp/  M05_vqfr/  M23_realesrgan/  M24_swinir/
  M28_nafnet/  M30_fbcnn/  D01_yunet/  D05_facexlib/  E01_sface/
datasets/
  DS01_fairface/   (zip kept compressed; extraction + per-file registry in Phase 2)
```

## 4. Not Acquired (and why)

| Asset | Reason |
|---|---|
| GFPGAN, PMRF, DiffBIR (Tier B) | REVIEW_REQUIRED, awaiting the license reviewer |
| CodeFormer and all other INELIGIBLE models | Non-commercial / proprietary licenses |
| facexlib `parsing_parsenet.pth` | Lineage likely PSFR-GAN (CC BY-NC-SA). Not needed yet. |
| facexlib `recognition_arcface_ir_se50.pth` | Recognition is out of scope for the runtime; evaluator provenance not verified |
| VQFR v2 | Not present in the linked official folder. Only v1 was found. |
| AdaFace, VQFR `resnet18_110` (2nd identity evaluator) | REVIEW_REQUIRED |
| SCUNet, DRUNet, BSRGAN, HAT, Restormer | Reserves, not shortlisted |
| LPIPS AlexNet backbone | Auto-downloaded by `lpips` on first use; will be recorded when first used (Phase 2) |
| DS02 public-domain portraits | Requires curation (P1-T7 DB-1). Not a single download. |
| DS03 in-house | Does not exist yet (consent program) |
| All REVIEW_REQUIRED / INELIGIBLE datasets | Per P1-T6 |

## 5. Reproducibility

Every asset can be re-fetched from the manifest's `download_url` and checked against `sha256`:
- GitHub release URLs are stable per tag.
- OpenCV Zoo URLs are pinned to commit `47534e27…`.
- Google Drive links are the authors' official ones, but may disappear. The SHA-256 lets any re-hosted copy be verified.

**Recommendation:** mirror these exact files to an internal artifact store (not git) before Phase 2, keeping the LICENSE files.

## 6. Acceptance Criteria

| Criterion | Status |
|---|---|
| Every acquired asset has documented provenance and license status | ✅ 14/14 in manifest |
| No unresolved-license asset downloaded | ✅ ELIGIBLE only |
| Assets can be reproduced/retrieved later | ✅ URLs + commits + SHA-256 |
| Secrets not committed | ✅ none created |
| No training or benchmarking | ✅ Only load/sanity checks |
