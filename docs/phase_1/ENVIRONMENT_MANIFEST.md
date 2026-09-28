# Environment Manifest — Phase 1

Recorded 2026-09-28. Update this file whenever the environment changes.

## Host (development machine) — facts

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language, build 10.0.26200, 64-bit |
| CPU | Intel Core Ultra 7 255H, 16 cores / 16 threads |
| RAM | ≈ 31.4 GB |
| GPU | Intel Arc 140T integrated (shared memory) |
| NVIDIA GPU / CUDA | **None / not installed** (`nvidia-smi`, `nvcc` absent) |
| Security policy | **Smart App Control ON** (`HKLM\...\CI\Policy\VerifiedAndReputablePolicyState = 1`). It blocks unsigned native DLLs of PyTorch (`torch_global_deps.dll`) and pandas in the Windows venv. **Not bypassed.** |
| Docker | Docker Desktop 29.8.0, WSL2 backend; VM has 16 CPUs and ≈ 16.4 GB memory |
| Git | 2.52.0 |

## Environment A — Docker CPU image (primary for ML work)

| Item | Value |
|---|---|
| Image tag | `fes-dev:cpu-p1` |
| Image ID | `sha256:c9850ffbeabf0216794240e61883adeccac32f747d259c8e8c539924e909554d` (≈ 2.1 GB) |
| Base | `python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b` |
| Dockerfile | `docker/Dockerfile.cpu` |
| Kernel | Linux 6.18.33.1-microsoft-standard-WSL2, glibc 2.36 |
| Python | **3.11.16** |
| PyTorch | **2.14.0+cpu** (CPU wheel index); `torch.cuda.is_available() = False`; 16 threads |
| torchvision | 0.29.0+cpu |
| OpenCV | 5.0.0 (`opencv-python-headless==5.0.0.93`); `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF` available |
| NumPy / SciPy / scikit-image | 2.4.6 / 1.17.1 / 0.26.0 |
| Pillow / pandas | 12.3.0 / 3.0.6 |
| lpips / gdown / PyYAML / tqdm | 0.1.4 / 6.4.0 / 6.0.3 / 4.70.1 |
| Full lock | `requirements.lock.txt` (`pip freeze` from the image, 34 packages) |
| User | non-root `fes` (uid 1000) |
| Run | `docker run --rm -it -v "<repo>:/workspace" fes-dev:cpu-p1 python ...` (Git Bash: prefix `MSYS_NO_PATHCONV=1`, use `$(pwd -W)`) |

Smoke test (2026-09-28): all imports OK. A 3→64 3×3 conv at 512×512 took ~171 ms mean on CPU. This is a *sanity check only, not a benchmark*.

## Environment B — Host venv `.venv/` (non-PyTorch tooling only)

| Item | Value |
|---|---|
| Python | 3.11.0 (Windows) |
| Works | numpy 2.4.6, opencv-python 5.0.0.93, pillow 12.3.0, scipy 1.17.1, scikit-image 0.26.0 |
| **Blocked** | torch 2.14.0+cpu, pandas 3.0.6 (Smart App Control) |
| Use | Lightweight scripts (file hashing, manifests, OpenCV-only degradation prototyping). **All model inference runs in Environment A.** |

## Environment C — GPU benchmark image (specification only)

| Item | Value |
|---|---|
| Dockerfile | `docker/Dockerfile.gpu` — **not built, not tested** |
| Base | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` (confirm against target driver at provisioning) |
| PyTorch | 2.14.0 from the `cu128` wheel index (confirm) |
| Requires | NVIDIA GPU (class TBD, P1-T1 §6), compatible driver, NVIDIA Container Toolkit |
| Status | ❌ **No GPU benchmark readiness.** No GPU exists locally. |

## Reproducibility Notes

1. Rebuild: `docker build -f docker/Dockerfile.cpu -t fes-dev:cpu-p1 .` (base pinned by digest; packages pinned in `requirements.txt` and the Dockerfile).
2. Exact resolved set: `requirements.lock.txt`.
3. CPU numerics can differ slightly across CPU models and thread counts. For determinism tests, fix `torch.set_num_threads()` and seeds (P1-T5 §5.1).
4. `lpips` downloads torchvision AlexNet ImageNet weights on first use. Treat that as an asset download: record it in the acquisition manifest when first used (Phase 2).
5. No secrets in images or git. HF tokens, if ever needed, go through `huggingface-cli login` inside the container session only, and are never baked into images.
