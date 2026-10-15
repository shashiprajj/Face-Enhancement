# Face Enhancement Service (FES)

Guarded face deblurring on CPU. One photo in, one face chip out — or the photo unchanged when enhancing it would not
be safe. Research candidate, not production-approved.

## What it runs

| Step | Component |
|---|---|
| Face detection | YuNet (`models/D01_yunet/face_detection_yunet_2026may.onnx`) |
| Input gate | sealed thresholds `configs/phase5/r6_nafnet_candidate.json` (front) + `side_face_route.json` (turned faces) |
| Deblurring | NAFNet-GoPro (`r6`) and its face-blur fine-tune (`r6_ft`), ONNX Runtime, `models/M28_nafnet/onnx/` |
| Safety | 4 mirror runs per model (any unstable run → original), artifact guard, local-patch guard, colour kept from input |
| Choice | `best`: the sharper accepted output of the two models |
| Admin blur range | `configs/deployment_settings.json` on the 0-100 % scale `configs/phase5/blur_scale.json` |

Endpoints, profiles and environment settings are documented at the top of `src/fes_service/app.py`; the pipeline in
`src/fes_service/pipeline.py`.

## Run

```sh
docker run -d --name fes-ui -p 127.0.0.1:8080:8080 \
  -v "D:/ReBIT/Face-Enhancement:/workspace" -v "D:/ReBIT/Face-Enhancement/models:/workspace/models:ro" \
  -w /workspace -e PYTHONPATH=src fes-dev:cpu-p5-onnx uvicorn fes_service.app:app --host 0.0.0.0 --port 8080
```

Demo page: http://localhost:8080/ui/best.html · readiness: http://localhost:8080/ready (ready after ~25 s).

The image is built by the chain `docker/Dockerfile.cpu` (tag `fes-dev:cpu-p2`) → `Dockerfile.cpu-p4` →
`Dockerfile.cpu-p5` → `Dockerfile.cpu-p5-onnx`.

## Test

```sh
FES_IMAGE=fes-dev:cpu-p5-onnx scripts/dock.sh sh -c 'PYTHONPATH=src python -m pytest -q tests'
node --test tests/ui/fes_best.test.mjs tests/ui/fes_best_polish.test.mjs
```

Tests that need face images read them from `tests/fixtures/` (local only, never committed) and skip without them.

## History

Research code, benchmarks, other models (A4/SwinIR, GFPGAN, CodeFormer, …), datasets and reports were removed on
2026-10-15. Code is in git history (last full state: commit `7cbc3c9`); files outside git were moved to
`D:/ReBIT/FES-archive/`.
