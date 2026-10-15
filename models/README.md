# models/ (git-ignored contents)
Model files used by the service. Never commit weights; SHA-256 values are pinned in `src/fes_service/models.py`
(NAFNet ONNX exports, also checked against the source weights recorded in their metadata).

- `D01_yunet/face_detection_yunet_2026may.onnx` — YuNet face detector (OpenCV Zoo; licence REVIEW_REQUIRED)
- `M28_nafnet/onnx/nafnet_gopro.onnx` — NAFNet-GoPro width-64 (megvii-research/NAFNet, MIT), profile r6
- `M28_nafnet/onnx/nafnet_face_ft.onnx` — its face-blur fine-tune (FairFace CC BY 4.0), profile r6_ft

The source PyTorch weights (.pth) and every other model were moved to `D:/ReBIT/FES-archive/models/`.
