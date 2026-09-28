# P1-T4 — Face Detection / Alignment Component Survey

| Field | Value |
|---|---|
| Phase / Task | Phase 1 / P1-T4 |
| Status | Complete |
| Date checked | 2026-09-28 |
| Depends on | P1-T1, P1-T2; license rules R1–R7 from P1-T3 |
| Matrix | [`detection_alignment_candidates.csv`](detection_alignment_candidates.csv) |

---

## 1. Role in the Pipeline (from Phase 0)

Detection and alignment are **supporting** components (P0-T1 S7–S9). They **locate** faces and **normalize** them for the restoration model, which expects aligned square faces (FFHQ-style, typically 512×512 from 5 landmarks). They also supply the inverse transform for paste-back (P0-T2 §3, steps 4–6, 9).

**No recognition in the runtime.** Identity embeddings appear **only** in the offline evaluation harness (§5), never in FES itself (P0-T1 X1–X3; master C1).

## 2. Key Licensing Findings

| # | Finding | Source |
|---|---|---|
| L1 | **InsightFace** code is MIT, but "the models… are available for non-commercial research purposes only". This rules out SCRFD, the InsightFace RetinaFace models, and the ArcFace models. | insightface README §License |
| L2 | **WIDER FACE**, the training set of nearly every detector, is **CC BY-NC-ND 4.0** per its HF dataset card (CUHK-CSE). This is the same training-data inheritance question as FFHQ (P1-T3 R4). | huggingface.co/datasets/CUHK-CSE/wider_face |
| L3 | **dlib 68-point** landmarks: the dlib-models README states the 300-W training data license "excludes commercial use… the trained model therefore can't be used in a commercial product". The 5-point model has no such statement, but no explicit license either. | dlib-models README |
| L4 | **EdgeFace** weights on HF are **cc-by-nc-sa-4.0**, even though the code is BSD-3. | HF API `Idiap/EdgeFace-*` |
| L5 | **OpenCV Zoo** directories carry their own licenses: **YuNet → MIT**, **SFace → Apache-2.0**. The ONNX files are in those directories. | opencv_zoo READMEs |
| L6 | **facexlib** is an MIT *collection*. Its components have their own licenses (RetinaFace MIT, AdaptiveWingLoss Apache-2.0, face-parsing MIT, **SORT GPL-3.0**, **hyperIQA unknown**). | facexlib README table |
| L7 | YOLO-face variants are GPL-3.0 and depend on Ultralytics (AGPL-3.0), which carries network-service obligations. | GitHub API |

## 3. Candidate Options

### 3.1 Detectors (+ 5-point landmarks)

| ID | Component | Landmarks | Small faces | CPU | Eval status | Notes |
|---|---|---|---|---|---|---|
| **D01** | **YuNet (OpenCV Zoo)** | 5 | Designed for ~10–300 px faces | Very fast; built into `cv2` | **ELIGIBLE** | Detection + landmarks in one component. Only needs `opencv-python`. Large faces need a downscaled input. |
| **D04** | **RetinaFace (biubug6)** | 5 | Strong (R50) | Moderate (R50) / fast (mnet) | **ELIGIBLE** | De-facto standard in face-restoration code; stale repo |
| **D05** | **facexlib** (RetinaFace + helper) | 5 (+98 AWing, parsing) | Strong | Moderate | **ELIGIBLE** (detection, parsing, helper only) | `FaceRestoreHelper` does alignment + inverse-affine paste-back. It is the reference pipeline most candidates were trained and evaluated with. |
| **D03** | **MediaPipe** BlazeFace + Face Landmarker | 6 / 478 3D + pose matrix | Weak (short-range model) | Fast | **ELIGIBLE** | Useful for **pose estimation** (P0-T3 E6 warnings) and dense landmarks |
| D02 | libfacedetection | 5 | Good | Very fast | ELIGIBLE | C++; D01 is the Python route to the same model family |
| D07 | dlib (HOG/CNN + 5-pt) | 5 | HOG weak | HOG fast | REVIEW_REQUIRED | 5-pt model terms unstated; never use the 68-pt model (L3) |
| D09 | MTCNN | 5 | Moderate | Moderate | REVIEW_REQUIRED | Weight terms unverified |
| D10 | serengil/retinaface | 5 | Strong | Moderate | REVIEW_REQUIRED | Weight lineage from NC InsightFace |
| D11 | YOLO-face | 0/5 | Strong | GPU-oriented | REVIEW_REQUIRED | AGPL/GPL (L7) |
| D15 | DSFD | 0 | Strong | Slow | REVIEW_REQUIRED | No root LICENSE found |
| D06 | InsightFace SCRFD/RetinaFace | 5 | Excellent | Fast | **INELIGIBLE** | L1 |

### 3.2 Dense landmark / pose components (optional)

| ID | Component | Use | Status |
|---|---|---|---|
| D03 | MediaPipe Face Landmarker | Head pose (yaw/pitch/roll) for `unusual_pose` warning and evaluation stratification | ELIGIBLE |
| D05 | facexlib AWing (98 pts) | Landmark-distance metric (P0-T4 §3A) | ELIGIBLE (R3) |
| D08 | face-alignment (FAN) | 68-pt 2D/3D | REVIEW_REQUIRED |
| D12 | SPIGA | Landmarks + pose | REVIEW_REQUIRED |
| D13 | PIPNet | 68/98 pts | REVIEW_REQUIRED |
| D14 | STAR | 98 pts | INELIGIBLE (no license) |

### 3.3 Can one component do both detection and landmarks?

**Yes.** YuNet, RetinaFace, and facexlib each return a bounding box **and** the 5 landmarks (eyes, nose tip, mouth corners). That is exactly what FFHQ-style similarity alignment needs. Dense landmarks (MediaPipe/AWing) are optional and only needed for pose warnings and evaluation metrics.

## 4. Alignment Method (candidate, to be held constant in benchmarking)

1. Detect → 5 landmarks.
2. Estimate a **similarity transform** (rotation, uniform scale, translation; least-squares, e.g., `cv2.estimateAffinePartial2D`) to the canonical FFHQ 5-point template scaled to the model's input size (512 by default; D-2).
3. Warp the face crop with that transform, and **store the inverse** for paste-back.
4. Paste-back: inverse-warp the restored face and blend with a soft (eroded + blurred) mask. Optionally use a face-parsing mask (facexlib parsing, MIT) for cleaner seams (R-D9).
5. The same template, border mode, interpolation, and blending must be used for **every** restoration candidate (P0-T4 §7.2; P1-T5).

## 5. Identity Evaluators (TEST-ONLY — never in the runtime)

P0-T4 §5 needs **≥ 2 independent evaluators**. Findings:

| ID | Evaluator | Status | Note |
|---|---|---|---|
| **E01** | **SFace (OpenCV Zoo, MobileFaceNet)** | **ELIGIBLE** (evaluation only) | Apache-2.0 ONNX; runs in `cv2` |
| E02 | AdaFace (MIT, Google Drive weights) | REVIEW_REQUIRED | Strong on low-quality faces (author claim). Trained on research-only or withdrawn datasets. |
| E03 | resnet18_110 ArcFace (VQFR Drive; the "Deg" metric in BFR papers) | REVIEW_REQUIRED | Original rights holder not verified. It is the metric most comparable with published results. |
| E06 | facenet-pytorch InceptionResnetV1 | REVIEW_REQUIRED | VGGFace2/CASIA training |
| E04 | InsightFace ArcFace | INELIGIBLE | L1 |
| E05 | EdgeFace | INELIGIBLE | L4 |

> ⚠️ **Gap:** only **one** evaluator (SFace) is currently ELIGIBLE. The ≥ 2-evaluator requirement (master §7.3, §8) needs either reviewer approval of AdaFace/E03 **for evaluation-only use**, or another permissively licensed evaluator. This is a **Phase 2 prerequisite** (tracked in P1-T10).

## 6. Pipeline Behavior for Phase 0 Edge Cases

Mapped to P0-T2 §2.6, P0-T3 §8, and master C2. Thresholds are TBD (D-3, D-4) and set from Phase 2 data.

| Case | Detection outcome | Behavior | Status / warning |
|---|---|---|---|
| **No face** (`full_image`/`auto`) | 0 boxes ≥ threshold | Reject | `NO_FACE_DETECTED` |
| **One face** | 1 box | Align (5-pt) → restore → optional paste-back | success |
| **Multiple faces** | N boxes | Follow `face_selection` (`largest` by box area, ties by confidence; `all` up to max; `index`) | `faces_detected=N`; `face_limit_reached` if capped |
| **Face crop, no face detected** | 0 boxes even at the **relaxed** threshold | **Reject** (master C2: never restore a non-face) | `NO_FACE_DETECTED` |
| **Face crop, detector box found but landmarks unreliable** | box found, landmark confidence low or geometry implausible (e.g., eye distance ≈ 0) | Process the whole crop **without alignment** (resize/pad to model size) | `no_landmarks_found` |
| **Landmark failure in full image** | box without usable landmarks | Skip that face (per-face status in `all` mode) or reject if it is the only face | `PROCESSING_ERROR` per face / `no_landmarks_found` |
| **Small face** | box below min face size (TBD, D-3) | Warn or reject per policy; restore conservatively | `face_too_small` / `FACE_TOO_SMALL` |
| **Side-profile / extreme pose** | box found; yaw from landmark geometry or MediaPipe pose above threshold (TBD) | Process and warn. 5-pt alignment degrades on profiles; measured in Phase 2 | `unusual_pose` |
| **Large face** (> YuNet's range) | Detector runs on a downscaled copy; coordinates rescaled | Normal | — |
| **Occlusion** (mask/sunglasses) | Detection usually succeeds; landmarks may shift | Process; optional parsing-based occlusion hint | `occlusion_suspected` (if detectable) |

## 7. Recommendation for Deeper Evaluation (not a final choice)

Carry into Phase 2 benchmarking (all ELIGIBLE for evaluation):

1. **D01 YuNet.** Lightest dependency (OpenCV only), MIT, fast on CPU. Strong primary candidate for a lean service.
2. **D05 facexlib / D04 RetinaFace.** Reference detector + alignment used by most restoration candidates. Needed for a fair "native pipeline" comparison track (P0-T4 §7.2).
3. **D03 MediaPipe.** Pose estimation and dense landmarks for `unusual_pose` and evaluation stratification.

Metrics to compare (Phase 2): detection recall/precision on the evaluation sets by face-size bucket, landmark error vs annotations where available, alignment stability under degradation (landmark jitter on degraded vs clean images of the same face), CPU/GPU latency, and downstream effect on restoration quality and identity.

No production license status is ELIGIBLE yet, because of the WIDER FACE training data (R4). This goes to the reviewer together with FFHQ/DIV2K.

## 8. Acceptance Criteria

| Criterion | Status |
|---|---|
| Detection and alignment options documented | ✅ §3 (15 detection/landmark + 6 evaluator entries) |
| Licensing considered (code and weights separately) | ✅ §2, CSV columns |
| Runtime behavior for Phase 0 edge cases mapped | ✅ §6 |
| Comparable in later benchmarking | ✅ §7 metrics |
| No recognition functionality introduced | ✅ Evaluators are test-only (§5) |
| No Feature-Match integration | ✅ |
