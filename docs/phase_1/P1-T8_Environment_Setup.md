# P1-T8 — Local ML Environment & Infrastructure Setup

| Field | Value |
|---|---|
| Phase / Task | Phase 1 / P1-T8 |
| Status | Complete — **CPU environment ready; GPU NOT ready** |
| Date | 2026-09-28 |
| Manifest | [`ENVIRONMENT_MANIFEST.md`](ENVIRONMENT_MANIFEST.md) |

---

## 1. What Was Set Up

| Item | Result |
|---|---|
| Project-local Python env | `.venv/` (Python 3.11.0), created in P1-T1 |
| **Finding** | **Windows Smart App Control blocks PyTorch's native DLLs** in the host venv (`OSError: [WinError 4551] An Application Control policy has blocked this file … torch_global_deps.dll`). pandas is also blocked. NumPy, OpenCV, SciPy, scikit-image, and Pillow work. The policy was **not bypassed or modified**. |
| **Resolution** | ML work runs in a **Docker Linux container** (`fes-dev:cpu-p1`) on the existing Docker Desktop / WSL2 install. It is isolated from machine-wide packages, reproducible, and matches the production deployment direction (P0-T5 §3). |
| Dependencies | Pinned in `requirements.txt`; PyTorch pinned in the Dockerfiles; full lock in `requirements.lock.txt` |
| GPU spec | `docker/Dockerfile.gpu` written for the future NVIDIA machine. **Not built.** |
| Directories | `models/`, `datasets/`, `experiments/`, `evaluation/`, `logs/` (+ READMEs), `data/registry/`, existing `reports/` |
| `.gitignore` | Weights, datasets, logs, and secrets ignored. The directory READMEs are tracked. |

## 2. Dependency Justification (only what the shortlist needs)

| Package | Why |
|---|---|
| torch 2.14.0 (CPU), torchvision 0.29.0 | Inference for Tier A models (RestoreFormer++, VQFR, Real-ESRGAN, SwinIR, FBCNN, NAFNet). All are PyTorch. |
| opencv-python-headless 5.0.0.93 | YuNet detector / SFace evaluator (P1-T4); classical baselines B1/B2; degradation operators (P1-T7) |
| numpy, scipy, pillow | Image I/O and math |
| scikit-image | PSNR / SSIM |
| lpips | LPIPS (P0-T4 §3A) |
| pandas, pyyaml, tqdm | Manifests, configs, progress |
| gdown | Official Google Drive weight links (P1-T9) |

**Not installed yet:**
- `basicsr` / `facexlib` — model-specific. They go in with the candidate adapters in Phase 2. Note: `basicsr` has known incompatibilities with recent torchvision and will need a pinned fix.
- `mediapipe` — optional pose track.
- `huggingface_hub` — no ELIGIBLE asset is hosted on HF.
- Any diffusion stack.

## 3. Capability & GPU Readiness

| Check | Result |
|---|---|
| CPU | 16 threads available to the container |
| NVIDIA CUDA | **Not available** (`torch.cuda.is_available() == False`; no NVIDIA hardware) |
| Intel iGPU acceleration | Not configured. Possible later via OpenVINO or PyTorch XPU, but **not representative of NVIDIA production** (P1-T1 §6 Option E) |
| **GPU benchmark readiness** | ❌ **NOT READY.** Performance numbers (latency, P95, throughput, VRAM) **cannot** be produced locally. CPU timings will be labeled "CPU dev only". |

## 4. Security

- No tokens or secrets created, stored, or committed.
- The container runs as non-root `fes`.
- Smart App Control was left enabled. Turning it off is the machine owner's decision; Microsoft documents it as not easily re-enabled. This work does not need it off.

## 5. Acceptance Criteria

| Criterion | Status |
|---|---|
| Isolated from machine-wide packages | ✅ Docker image; the venv uses no global packages |
| Versions recorded | ✅ manifest + lock |
| Reproducible | ✅ digest-pinned base, pinned requirements, Dockerfile |
| GPU limitations explicit | ✅ §3 |
| No unapproved asset downloaded | ✅ Only Python packages and the Python base image. No models or datasets. |
| No FastAPI, no Feature-Match | ✅ |
