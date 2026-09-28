# P0-T5 — Technical & Deployment Constraints

| Field | Value |
|---|---|
| Phase / Task | Phase 0 / P0-T5 |
| Status | Complete |
| Date | 2026-09-28 |
| Depends on | P0-T1 … P0-T4 (reviewed) |
| Nature | Constraints only — nothing installed, no models downloaded |

---

## 1. Environment Facts (development machine)

Collected on 2026-09-28 with read-only system queries (`Get-CimInstance`, `python --version`, `pip list`, `docker --version`, `nvidia-smi`, `nvcc`).

| Item | **Fact** (observed) |
|---|---|
| OS | Microsoft Windows 11 Home Single Language, build 10.0.26200, 64-bit |
| Python | 3.11.0 (64-bit), only interpreter registered with the `py` launcher |
| CPU | Intel Core Ultra 7 255H — 16 cores / 16 threads |
| RAM | ≈ 31.4 GB visible |
| GPU | **Intel Arc 140T integrated GPU** ("16GB" reported = shared system memory, not dedicated VRAM) — driver 32.0.101.8132 |
| NVIDIA GPU | **None** — `nvidia-smi` not found |
| CUDA toolkit | **Not installed** — `nvcc` not found |
| Dedicated VRAM | **None** (integrated GPU uses shared RAM) |
| NPU | Core Ultra 200H series includes an NPU (inferred from CPU model; not queried) |
| Storage | D: ≈ 229.5 GB free (project drive); C: ≈ 149.8 GB free; E: ≈ 79 GB free |
| Docker | Docker 29.8.0 installed; WSL2 default distribution is `docker-desktop` |
| Git | 2.52.0 installed; **project directory is not a git repository** |
| Python packages present (relevant) | `fastapi 0.111.0`, `numpy 2.4.2`, `pillow 12.1.0` (global env; **not** installed by this project). No `torch`, `onnxruntime`, `openvino`, `opencv` present. |
| Repository contents | Only Phase 0 task prompts plus these docs. No existing code. |

### Implications of facts

1. **No CUDA locally.** Most face-restoration reference implementations target PyTorch + CUDA. Local development can run them on **CPU** (slow) or via **Intel GPU paths** (PyTorch XPU backend, OpenVINO, ONNX Runtime with DirectML/OpenVINO EP) — feasibility to verify in Phase 1.
2. **Local latency numbers will not represent production GPU performance.** Performance evaluation (P0-T4 §8) needs access to a representative NVIDIA GPU server (or whichever production target is chosen) before final model selection.
3. 31 GB RAM and ample disk are sufficient for model weights (typically < 1–2 GB each) and evaluation datasets of moderate size.
4. Some reference code relies on custom CUDA extensions; such models may be hard or impossible to run locally — a practical selection factor.

## 2. Facts vs. Assumptions

| Statement | Fact / Assumption |
|---|---|
| Dev machine specs in §1 | **Fact** |
| Production will run on an NVIDIA GPU server | **Assumption** (A-T1) |
| Production will be containerized with Docker | **Assumption** (A-T2), consistent with P0-T1 A3 |
| Linux will be the production OS | **Assumption** (A-T3) |
| A CPU-only fallback is desirable | **Assumption** (A-T4); viability depends on model cost |
| Service processes privacy-sensitive images | **Assumption** treated as requirement (P0-T1 A6) |
| Commercial use is intended | **Assumption** treated as requirement (P0-T1 A4) |
| Expected request volume | **Unknown** (P0-T1 Q1) |

## 3. Deployment Targets (conceptual)

| Target | Purpose | Notes |
|---|---|---|
| **Local development** (this Windows machine) | Prototyping, correctness, unit/functional tests, small evaluations | CPU / Intel iGPU only. Use a project-local virtual environment; do not rely on global packages. |
| **Docker** | Reproducible build and runtime; parity between dev and prod | Linux containers via WSL2. GPU containers require NVIDIA Container Toolkit on the host — not available locally; CPU image testable locally. |
| **GPU server** (primary production target, assumed) | Production inference | NVIDIA GPU class, VRAM, and CUDA version **TBD**. Model export options (ONNX/TensorRT, FP16) evaluated in Phase 1/2. |
| **CPU fallback** (possible) | Low-volume or degraded-mode serving; environments without GPU | Only if latency on CPU is acceptable for some consumer; OpenVINO/ONNX Runtime likely paths. Decision after benchmarking. |

## 4. Hardware Constraints

| Constraint | Status |
|---|---|
| Model must be runnable (even slowly) on CPU for dev/testing | **Requirement** — needed to develop on this machine |
| Model should not *require* custom CUDA kernels without a CPU/pure-PyTorch fallback | **Strong preference** |
| Peak VRAM per worker on GPU target | **TBD** (depends on GPU class; measured per P0-T4) |
| Peak RAM per worker | **TBD** |
| Model footprint (weights + detector + runtime) | Recorded per candidate; smaller preferred |
| Must support running ≥ 1 model instance per GPU with bounded memory | Requirement |

## 5. Performance Targets / TBDs

No hard numbers are fixed without consumer requirements and measurements (P0-T4 §10 TH-8).

| Target | Initial expectation (non-binding) | Final value |
|---|---|---|
| End-to-end P95 latency, single face, GPU | "Interactive" range (order of sub-second to ~1–2 s) is the working goal for a synchronous API | **TBD** |
| End-to-end latency, CPU | Expected to be several seconds or more per face for learned models | **TBD**; may determine that CPU is dev-only |
| Throughput per GPU | — | **TBD** |
| Cold start | Model load time acceptable within container start-up | **TBD** |
| Concurrency | Bounded worker queue per GPU | **TBD** |

How they will be set: consumer SLA inputs (P0-T1 Q1) + measured capability on reference hardware (P0-T4 §8) → targets agreed and frozen before production candidate selection.

## 6. Licensing / Commercial Requirements

Licensing is a **hard selection gate** (P0-T4 §9 E1). A candidate is ineligible unless **all** of the following are verified and recorded:

| Item | Requirement |
|---|---|
| **Model code license** | Permits commercial use, modification, and deployment as a network service (note AGPL network-use obligations). |
| **Model weight license** | Checked **separately** from code — weights are often under different (frequently non-commercial) terms than the code. |
| **Training-data terms** | Check whether the weight license inherits restrictions from training datasets (e.g., non-commercial dataset terms); record risk. |
| **Face detector / landmark models** | Same checks — detectors and landmark models are dependencies with their own weight licenses. |
| **Identity evaluator models** (test-only, P0-T4 §5) | Must permit evaluation use; if only research-licensed, record that they are used **only offline in evaluation** and confirm that is permitted. |
| **Third-party dependencies** | All runtime libraries (image codecs, inference runtimes, CUDA/cuDNN redistribution in images) reviewed; SBOM produced in implementation phases. |
| **Redistribution** | Whether weights can be baked into Docker images and distributed; otherwise downloaded at deploy time from an approved internal store. |
| **Attribution / notices** | NOTICE file listing all licenses. |

Known risk pattern to verify in Phase 1 (not a conclusion): several popular open face-restoration and face-analysis projects release code or weights under **non-commercial / research-only** licenses, and some depend on components with restrictive terms (e.g., certain custom CUDA ops or pretrained recognition models). Each candidate must be checked individually from its primary license texts.

## 7. Operational Constraints

| Area | Constraint |
|---|---|
| **Max image size** | Enforce encoded-size, decoded-pixel, and side-length limits *before* full decoding (decompression-bomb protection). Values TBD (P0-T2 D2-1). |
| **Temporary-file handling** | Prefer in-memory processing. If temp files are needed: private temp dir, random names, restrictive permissions, deleted in `finally`/on error; no persistence across requests. |
| **Data retention** | Default: **no retention** of input or output images after the response. Any debug capture must be opt-in, time-limited, and access-controlled. |
| **Resource isolation** | Model inference in bounded worker(s); per-request memory/time limits; container CPU/memory limits; one GPU context per worker. |
| **Request limits** | Max request size, max faces per request (P0-T2 D2-5), per-client rate limits, bounded queue with fast rejection (`RESOURCE_EXHAUSTED`) when full. |
| **Timeouts** | Per-stage and overall processing timeout (`TIMEOUT`). |
| **Logging** | Structured logs with `request_id`, timings, status, warnings, model version. **Never log image content, base64 payloads, or face crops.** Face bounding boxes are acceptable in logs only if approved by compliance. |
| **Monitoring** | Metrics: request rate, latency histograms (P50/P95/P99), error rate by category, queue depth, GPU/CPU utilization, memory/VRAM, model-load status. Health and readiness endpoints in API phase. |
| **Versioning** | Model and service version reported in every response (P0-T2 §5) and logs. |
| **Model artifacts** | Weights stored with checksums; loaded from an approved location; integrity verified at start-up. |

## 8. Security Considerations

- **Untrusted input**: image decoders are an attack surface → keep codec libraries patched; restrict formats (P0-T2 §2.2); validate by magic bytes; enforce limits pre-decode; run as non-root in container.
- **Model loading**: prefer weight formats that do not execute code on load (e.g., `safetensors`, ONNX) over pickle-based checkpoints; if pickle is unavoidable, load only verified checksummed files.
- **Privacy**: faces are personal/biometric-adjacent data. Minimize data, strip EXIF/GPS from outputs, encrypt in transit (TLS), no retention by default (§7). Legal review of applicable law (e.g., India DPDP Act 2023) pending (P0-T1 Q4).
- **Misuse**: output may be mistaken for an original photo → response metadata marks the image as enhanced; consider watermark/provenance (P0-T1 Q6).
- **AuthN/AuthZ**: required for any network deployment; mechanism decided in API phase.
- **Supply chain**: pin dependency versions; SBOM; vulnerability scanning of container images.
- **Denial of service**: size limits, rate limits, bounded queues, timeouts.

## 9. Feature-Match Independence — Explicit Confirmation

- The Feature-Match codebase was **not inspected, imported, modified, or referenced** in any Phase 0 work.
- FES has **no** code, library, schema, data, model, configuration, or deployment dependency on Feature-Match.
- Identity evaluation uses independent evaluators (P0-T4 §5), not Feature-Match.
- Feature-Match may become a consumer through the generic API only (P0-T1 §7).

## 10. Open Questions

| ID | Question | Needed by |
|---|---|---|
| Q5-1 | Which production GPU class / VRAM / CUDA version, and on-prem vs cloud? | Phase 1 (for performance evaluation) |
| Q5-2 | Is access to an NVIDIA GPU available for Phase 1 benchmarking? | Phase 1 start |
| Q5-3 | Is CPU fallback a real requirement for any consumer? | Phase 1 |
| Q5-4 | Expected request volume, peak concurrency, and latency SLA | Phase 1 |
| Q5-5 | Data-residency / legal requirements for face images | Before production |
| Q5-6 | Approved license list / legal reviewer for model licenses | Phase 1 (before candidate shortlisting) |
| Q5-7 | Is the project to be placed under version control now? (Recommended: `git init` before Phase 1.) | Phase 1 start |
| Q5-8 | Intel iGPU/NPU acceleration (OpenVINO) worth pursuing for local dev or edge? | Phase 1 |

## 11. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| AC1 | Previous outputs reviewed | ✅ |
| AC2 | Environment facts recorded (OS, Python, CPU, RAM, GPU, VRAM, CUDA, storage) | ✅ §1 |
| AC3 | Facts separated from assumptions | ✅ §2 |
| AC4 | Deployment targets: local, Docker, GPU server, CPU fallback | ✅ §3 |
| AC5 | Performance expectations without unrealistic hard limits | ✅ §5 |
| AC6 | Licensing: model, weights, code, third-party, redistribution/commercial | ✅ §6 |
| AC7 | Operational constraints: max size, temp files, isolation, request limits, logging/monitoring | ✅ §7 |
| AC8 | Security considerations | ✅ §8 |
| AC9 | Feature-Match independence explicitly confirmed | ✅ §9 |
| AC10 | No models downloaded/installed | ✅ |
