# P1-T1 — Phase 0 Open Items & Project Baseline

| Field | Value |
|---|---|
| Phase / Task | Phase 1 / P1-T1 |
| Status | Complete |
| Date | 2026-09-28 |
| Authoritative input | `docs/phase_0/P0_Master_Specification.md` v1.0, `reports/Phase_0_Final_Report.md` (gate: YELLOW) |

---

## 1. Authority

The Phase 0 Master Specification is treated as **authoritative**. Nothing in Phase 1 changes a Phase 0 decision silently. Any change is recorded in the relevant Phase 1 document with a reference to the Phase 0 section it affects.

## 2. Stakeholder Status Check (2026-09-28)

These answers come from the project user in the Claude Code session on 2026-09-28. Nothing here was inferred.

| Item | Phase 0 ref | Status | Evidence |
|---|---|---|---|
| **Commercial-use requirement** | A-4, Q-4 | ✅ **CONFIRMED — commercial use** | Stated by the project user (shashiprajj@gmail.com) on 2026-09-28: "Yes, commercial confirmed". Recommendation: record a written product-owner confirmation in the project's decision log. |
| **Production hardware / deployment target** | A-3, A-7 | ❌ **UNCONFIRMED** | The GPU server as primary target and Linux as production OS are still assumptions. |
| **License reviewer / owner** | Q-4 | ❌ **UNCONFIRMED — none named** | "Not yet". All license findings in Phase 1 are **factual screening, not approvals**. |
| **GPU access** | Q-2 | ❌ **NOT AVAILABLE** | "No GPU access yet". Verified locally: no NVIDIA GPU, no `nvidia-smi`, no CUDA (P0-T5 §1). |
| **Download policy for Phase 1** | — | Set by project user | Download **only clearly ELIGIBLE** assets (permissive code **and** weights / data terms). Skip REVIEW_REQUIRED. |

Effect of the commercial-use confirmation: licensing screening (P1-T3, P1-T6) checks candidates **against commercial use**, not conditionally. Non-commercial or research-only assets become **INELIGIBLE** for production. For **offline evaluation only** they are REVIEW_REQUIRED, and only if their terms allow that use.

## 3. Phase 0 Open Items Affecting Phase 1

| ID | Item | Phase 1 handling | Owner | Status |
|---|---|---|---|---|
| Q-1 | Consumers, volume, latency SLA | Needed for performance targets (TH-8); does not block the survey or evaluation design | Product | OPEN |
| Q-2 | NVIDIA GPU access | Gap and options in §6. Blocks Phase 2 **performance** benchmarking only | Infra | OPEN — **blocker for Phase 2 performance** |
| Q-3 | Lighting/contrast wanted? | P1-T5 treats lighting as optional (C7) | Product | OPEN |
| Q-4 | Commercial use / license reviewer | Commercial use **confirmed**; reviewer **not named** | Product / Legal | PARTIAL |
| Q-5 | Data protection / residency | Affects dataset handling (P1-T6/T7): no private data used | Legal | OPEN |
| Q-6 | Watermark/provenance | API phase | Product | OPEN (not Phase 1) |
| Q-7 | Human-review panel | Needed for Phase 2 human evaluation | Product | OPEN |
| Q-8 | Licensed/consented evaluation datasets | P1-T6 / P1-T7 | Data / Legal | IN PROGRESS (Phase 1) |
| Q-9 | Fine-tuning allowed later | Not needed in Phase 1 | Product | OPEN |
| Q-10 | CPU production fallback | Measured, not decided, in Phase 2 | Product | OPEN |
| D-2, D-3, D-4, D-6, D-7, D-9, D-12, D-15 | Output size, min face size, detector, extreme policy, lighting, runtime QA, thresholds, model choice | Inputs gathered in Phase 1; decided with Phase 2 evidence | — | OPEN (by design) |
| Master §18.1 | Product-owner sign-off of master spec | Commercial use is confirmed; full spec sign-off is not recorded | Product | **UNCONFIRMED** |

## 4. Git Baseline

| Item | Value |
|---|---|
| Repository | Initialized in `D:\ReBIT\Face-Enhancement` on 2026-09-28, default branch `main` |
| **Phase 0 baseline commit** | **`80127d820c9e25111c7c20003ab5afe229b68288`** (`80127d8`), "Phase 0: requirements and specification baseline" |
| Contents | `Tasks/` (Phase 0 & 1 prompts), `docs/phase_0/`, `reports/Phase_0_Final_Report.md`, `.gitignore` |
| Remote | None configured |
| `.gitignore` | Excludes `.venv/`, model/dataset directories, weight files (`*.pth`, `*.safetensors`, `*.onnx`, …), logs, archives, and **secrets** (`.env`, token files, `.huggingface/`) |

## 5. Python Environment Approach

| Item | Value |
|---|---|
| Type | Project-local `venv` at `.venv/` (git-ignored) |
| Interpreter | Python 3.11.0 (64-bit), from the only registered interpreter |
| Installed | `pip 26.2.1`, `setuptools 65.5.0` — **nothing else** |
| Global packages | Not used. The machine-wide `fastapi`, `numpy`, `pillow` are ignored |
| Model-specific packages | **Not installed** (per task rules). Installed in P1-T8 only for the approved shortlist |
| Reproducibility | Pinned `requirements*.txt` produced in P1-T8 |
| Activation | `.venv\Scripts\Activate.ps1` (PowerShell) or `.venv/Scripts/python.exe` directly |

## 6. GPU Status & Access Plan

**Exact gap:** the only development machine has an Intel Arc 140T integrated GPU (shared memory) and **no NVIDIA GPU, no CUDA, and no dedicated VRAM**. So:

- ✅ Possible locally: model survey, license screening, CPU inference for correctness, **quality and identity evaluation on CPU** (slow but valid), and evaluation-harness development.
- ❌ **Not possible locally:** GPU latency, P95, throughput, VRAM, FP16/TensorRT measurements. Local CPU timings **must not** be reported as production performance.

**Options** (to be decided by Infra/Product; costs change often and must be checked at the time):

| Option | Pros | Cons | Fit |
|---|---|---|---|
| **A. On-demand cloud GPU VM** (AWS/GCP/Azure, e.g., NVIDIA T4/L4/A10G-class) | Can match the likely production GPU class; reproducible images | Account/billing setup; data-governance review if evaluation images leave premises | **Recommended for Phase 2 benchmarking** |
| B. GPU marketplace / rental (e.g., RunPod, Lambda, Vast.ai) | Cheapest and fastest to start | Weaker data-governance guarantees; hardware varies | Quick feasibility checks with public data only |
| C. Internal / on-prem GPU server | Best data control; can be the production target | Availability unknown | Preferred if it exists |
| D. Buy a workstation GPU | Permanent dev capacity | Capital cost; may not match production GPU | Longer term |
| E. Intel iGPU/NPU (OpenVINO) locally | Available now | **Not representative of NVIDIA production**; only an optional CPU/edge track | Supplementary only |

**Recommended path:** (1) Product/Infra choose the production GPU class (resolves A-3). (2) Provision Option A or C with the **same GPU class** for Phase 2. (3) Use only public, license-eligible data on rented GPUs until data governance (Q-5) is settled.

## 7. Unresolved Blockers

| # | Blocker | Blocks | Severity |
|---|---|---|---|
| B-1 | No NVIDIA GPU access | Phase 2 performance benchmarking; final model selection | **High** (not blocking for P1-T2 … P1-T9) |
| B-2 | No license reviewer named | Final sign-off of any ELIGIBLE asset; resolution of REVIEW_REQUIRED items | **High** (Phase 1 continues with factual screening; downloads limited to clearly permissive assets per user direction) |
| B-3 | Production target unconfirmed (A-3) | Choice of GPU class for benchmarking | Medium |
| B-4 | Full spec sign-off not recorded | Formal GREEN for Phase 0 | Low–Medium |

## 8. Recommended Resolution Path

1. Product owner records commercial-use confirmation in writing (already given verbally in session) and signs off the master spec.
2. Name a license reviewer before P1-T9 downloads are relied on for anything beyond internal evaluation.
3. Infra decides the GPU class and access option (§6) before Phase 2 starts.
4. Continue P1-T2 … P1-T8 now. They do not need a GPU or a reviewer.

## 9. Acceptance Criteria

| Criterion | Status |
|---|---|
| Phase 0 treated as authoritative | ✅ §1 |
| Open items explicitly tracked | ✅ §3 |
| No unverified approval claimed | ✅ §2 (commercial use attributed to user statement; everything else UNCONFIRMED) |
| Git baseline exists | ✅ `80127d8` |
| Local environment strategy documented | ✅ §5 |
| No model/dataset downloaded | ✅ |

**P1-T2 can safely begin:** yes. It is a discovery task and needs no GPU, reviewer, or downloads.
