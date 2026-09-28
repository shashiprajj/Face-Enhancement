# P2-T6 — Face Detection, Cropping & Alignment Pipeline

| Field | Value |
|---|---|
| Phase / Task | Phase 2 / P2-T6 |
| Status | Complete |
| Date | 2026-09-28 |
| Code | `src/fes_data/preprocess.py` (`preproc-1.0.0`) |
| Config | `configs/phase2/preprocess.yaml` |
| CLI | `scripts/p2_preprocess.py` (+ `scripts/p2_make_preprocess_examples.py`) |
| Tests | `tests/test_preprocess.py`, `tests/test_degrade.py`, `tests/test_collection_validator.py` — **55 passed** |
| Example metadata | `data/manifests/preprocess_examples_full_image_v1.json`, `data/manifests/preprocess_examples_face_crop_v1.json` |

---

## 1. Components

| Step | Implementation | Deterministic? |
|---|---|---|
| Decode | OpenCV → 8-bit BGR; `None` → `DECODE_ERROR` | ✅ |
| Detection | **YuNet** (`face_detection_yunet_2026may.onnx`, MIT, P1-T4 D01): bbox + score + 5 landmarks. Runs on a copy downscaled to ≤ 640 px; coordinates are mapped back. | ✅ (fixed order: area ↓, score ↓, x, y) |
| Selection | `largest` (default) / `all` / `index` | ✅ |
| Sanity | `landmarks_plausible`: eyes left→right, nose/mouth below the eye line | ✅ |
| Alignment | Closed-form **Umeyama** similarity to the FFHQ 5-point template at `out_size` (512 default). No RANSAC/LMEDS, so no random sampling. | ✅ |
| Warp | Anti-aliased: `INTER_AREA` pre-reduction when shrinking, then `INTER_LINEAR` warp. **No cubic/Lanczos**, so no sharpening overshoot. | ✅ |
| Records | detector id (model file + thresholds), score, bbox, landmarks, 2×3 transform, scale, roll, yaw proxy, out-of-source fraction, `preproc_version`, input mode, retry flag | ✅ |

**Detection is separate from enhancement.** The module has no restoration code and never calls a model other than the detector. RetinaFace (D05) is acquired but kept for the benchmark phase's native-pipeline track (P1-T5 §5.1), since it needs PyTorch architecture code. YuNet runs with OpenCV alone.

## 2. Defined Behavior per Case

| Case | Behavior | Status / warnings | Verified by |
|---|---|---|---|
| One face | Detect → align | `OK` | test + example 01 |
| Multiple faces | Policy `largest` / `all` / `index`; `n_faces` recorded; out-of-range index rejected | `OK` + `multiple_faces`; `INVALID_FACE_INDEX` | test + example 02 |
| No face detected | Nothing is cropped or invented | `NO_FACE_DETECTED`, `crop=None` | tests (flat, noise; both modes) + example 03 |
| Very small face | Inter-ocular distance < `min_eye_dist_px` (12) → not processed | `FACE_TOO_SMALL` (or `NO_FACE_DETECTED` if undetectable) | test + example 04 (`FACE_TOO_SMALL`, `low_confidence`) |
| Partial face | Aligned, flagged when the canonical window leaves the image (> 2%) | `OK` + `partial_face` | test + example 05 |
| Occluded face | Processed if detected. **Occlusion itself is not detectable** by this component; a low score gives `low_confidence`. | `OK` (+ possible `low_confidence`) or `NO_FACE_DETECTED` | test + example 06 (eye bar: `OK`, **not** flagged — documented limitation) |
| Already-cropped face (`face_crop` mode) | If missed, pad 50% (constant) and retry at relaxed threshold 0.5; coordinates reported in the original crop frame | `OK` + `detected_after_crop_retry` | test `test_tight_crop_needs_face_crop_mode_retry`: tight crops fail in `full_image`, recovered in `face_crop` |
| Crop with no face | Reject; never restore a non-face (P0 master C2) | `NO_FACE_DETECTED` | test + example 03 in `face_crop` mode |
| Face found, landmarks implausible | `face_crop`: unaligned resize + `no_landmarks_found`; `full_image`: `LANDMARK_FAILURE` | as stated | code path (P1-T4 §6) |
| Unusual pose | \|yaw proxy\| > 0.25 | `unusual_pose` | code path |
| Undecodable input | — | `DECODE_ERROR` | test |

## 3. Observations From the Examples

- **FairFace chips always get `partial_face` at 512 FFHQ framing.** The FFHQ window extends beyond the tightly framed chip. This is the same fact that drove the P2-T3 design (references = chips, with the FFHQ transform recorded). Benchmark adapters must therefore handle out-of-source regions identically for every candidate, and exclude them from metrics.
- **Occlusion is not flagged** (eye-bar example returns `OK` without warnings). A dedicated occlusion estimator (e.g., face parsing) would be needed; that is a later-phase item and is recorded as an open limitation.

## 4. Determinism & Traceability

- `test_deterministic`: two runs give identical crop pixels and identical records.
- Each result's `record()` holds everything needed to re-derive the crop from the source: input SHA-256 (CLI), landmarks, transform, out size, detector id, and version.

## 5. Tests (run in `fes-dev:cpu-p2`)

```
scripts/dock.sh python -m pytest -q -p no:cacheprovider tests      # 55 passed
```

| Suite | Tests | Covers |
|---|---|---|
| `test_preprocess.py` | 14 | geometry, decode error, no face ×4, one face, determinism, multi-face policies, tiny, partial, occluded, already-cropped, tight-crop retry |
| `test_degrade.py` | 35 | all 32 cells deterministic + reference untouched + sizes; seed keys; blur monotonicity; kernel normalization |
| `test_collection_validator.py` | 6 | valid collection + participant-level split; PII column; consent scope; cross-participant pairing; GPS EXIF; low-res HQ reference |

Face-bearing tests use git-ignored FairFace references and are **skipped** (not failed) when the dataset hasn't been prepared. No face images are committed.

## 6. Done-When Check

Benchmark images pass through consistent, traceable preprocessing, and failure behavior is defined for every case the task lists. ✅
