# P2-T6 — Face Detection, Cropping & Alignment Pipeline

## Objective
Build the reusable preprocessing pipeline that standardizes face inputs for enhancement benchmarking.

## Work
Define and implement behavior for:
- one face
- multiple faces
- no detected face
- very small face
- partial/occluded face
- already-cropped face

Use Phase 1 detector options where appropriate. Keep detection separate from enhancement. Make crop/alignment deterministic. Record detector, confidence, bounding box, landmarks/alignment and processing version. Never invent a face when detection fails. Add unit/smoke tests for edge cases.

## Deliverables
- `docs/phase_2/P2-T6_Preprocessing_Pipeline.md`
- Reusable preprocessing code.
- Configuration and tests.
- Example metadata.

## Done When
Benchmark images can pass through consistent, traceable preprocessing with defined failure behavior.
