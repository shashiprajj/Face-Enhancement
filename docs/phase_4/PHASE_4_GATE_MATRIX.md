# Phase 4 Gate Matrix (updated 2026-10-04, P4-T8)

| Gate | Required evidence | Status | Evidence | What closes it |
|---|---|---|---|---|
| **G-1 GPU** | Real GPU performance/throughput | 🔴 **UNRESOLVED.** No NVIDIA GPU; CPU only (A4 ≈ 45 s/face) | P3-T8; `t7/router_validation.json`; `perf/gate_latency.json` | GPU access (Q-2); P95/throughput/cold start of gate + A4 on target hardware |
| **G-2 Second evaluator** | Independent identity evaluator | 🟢 **ADDRESSED** (evaluation-only). E07 dlib ResNet validated 7/7 pre-registered criteria and used throughout P4-T4/T6/T7, reported separately from SFace. | `evaluator_E07_validation.json`; P4-T3 | Reviewer confirmation of evaluation-only use under R4 (part of G-4) |
| **G-3 Real-world data** | Approved real-world evaluation | 🔴 **BLOCKED.** No source passes the legal gate; no real-world evaluation performed. | `p4_realworld_sources.csv`; `realworld/*/BLOCKED.json`; P4-T1 | Legal approval of DS03 consent/spec (or a vendor contract) → collect → `p4_freeze_realworld.py` |
| **G-4 Production licensing** | Code/weight/data licence review | 🔴 **UNRESOLVED.** No reviewer. All candidate weights + YuNet/RetinaFace are REVIEW_REQUIRED (R4). | `model_acquisition_manifest.csv`; P1-T3 | Named reviewer; decision on A4 + YuNet first (only assets on the routed path) |
| **G-5 Human review** | Completed blinded review | 🔴 **NOT PERFORMED.** The kit `rv1` is ready, but a kit alone is not completion. | `review/kit_rv1.json`, `analysis_rv1.json` (INSUFFICIENT) | ≥ 2 qualified human reviewers complete `rv1`; analysis status EVIDENCE |

| Phase 4 task | Status | Commit(s) |
|---|---|---|
| P4-T1 Dataset & governance | ✅ framework; data blocked (G-3) | `42f451c` |
| P4-T2 Taxonomy & quality metadata | ✅ | `f2b4c0b` |
| P4-T3 Second evaluator | ✅ | `52b18bd` |
| P4-T4 Benchmark | ✅ synthetic two-evaluator; real-world arm blocked | `b613aa5` |
| P4-T5 Human review | ⚠️ kit only; review pending (G-5) | `0f1ab8e` |
| Infra resume (dev inference) | ✅ all 9 configs validated | `4ebceb6` |
| P4-T6 Do-Not-Enhance gate | ✅ frozen → evaluated once | `e504dfa`, `bc5ef80` |
| P4-T7 Selective routing | ✅ frozen → evaluated once | `785a4b2`, `ad79455` |
| P4-T8 Evidence synthesis & gate | ✅ **YELLOW** | this commit |

Gate semantics are those of `Tasks/.../PHASE_4_GATES.md`. A gate is not marked GREEN because implementation exists.
