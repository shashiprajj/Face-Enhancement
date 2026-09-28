# Phase 0 Final Report — Face Enhancement Service

| Field | Value |
|---|---|
| Date | 2026-09-28 |
| Phase | 0 — Requirements & Specification |
| Authoritative spec | [`docs/phase_0/P0_Master_Specification.md`](../docs/phase_0/P0_Master_Specification.md) |
| **Decision gate** | **🟡 YELLOW — proceed to Phase 1 with documented open items** |

---

## 1. Executive Summary

Phase 0 defined the standalone **Face Enhancement Service (FES)**. FES takes an image containing a face and returns a **faithfully restored** face image with quality metadata. It is explicitly **not** a recognition or verification service. **Feature-Match is external**: it was not inspected, imported, or depended on, and may only ever call FES through the generic API.

Six tasks were completed in order. Together they produced a scope definition, a conceptual input/output contract, a degradation taxonomy with identity-preservation requirements, an evaluation methodology, a technical/licensing constraints record, and a consolidated master specification. The master spec reconciles eight overlaps or conflicts between the task documents, and each one is listed openly.

All ten GREEN criteria are met on paper. The gate is still **YELLOW** for three reasons:

- the spec was written without stakeholder confirmation of key assumptions (commercial use; GPU production target);
- the dev machine has **no NVIDIA GPU or CUDA**, and no GPU access plan exists yet;
- no license reviewer has been named, although licensing is a hard gate for model selection.

None of these requires redefining the product. All three must be resolved during early Phase 1, before final model selection.

## 2. Tasks Completed

| Task | Title | Status | Acceptance |
|---|---|---|---|
| P0-T1 | Product Scope & Boundaries | ✅ Complete | 8/8 criteria met |
| P0-T2 | Input / Output Requirements | ✅ Complete | 10/10 criteria met |
| P0-T3 | Enhancement & Quality Requirements | ✅ Complete | 10/10 criteria met |
| P0-T4 | Evaluation & Success Criteria | ✅ Complete | 10/10 criteria met |
| P0-T5 | Technical & Deployment Constraints | ✅ Complete | 10/10 criteria met |
| P0-T6 | Master Specification & Decision Gate | ✅ Complete | See §10 |

Phase 0 boundary respected: no models or datasets downloaded, no FastAPI implemented, no production code written, no Feature-Match access. The only system actions were read-only environment queries for P0-T5.

## 3. Documents Produced

| Path | Content |
|---|---|
| `docs/phase_0/P0-T1_Product_Scope_and_Boundaries.md` | Scope, out-of-scope, integration boundary |
| `docs/phase_0/P0-T2_Input_Output_Specification.md` | Input/output contract, flow diagram, metadata, errors |
| `docs/phase_0/P0-T3_Enhancement_Requirements.md` | Degradation taxonomy, behavior rules, edge cases, test-case seeds |
| `docs/phase_0/P0-T4_Evaluation_and_Success_Criteria.md` | Evaluation dimensions, metrics, baselines, selection evidence, TBDs |
| `docs/phase_0/P0-T5_Technical_Constraints.md` | Environment facts, deployment, licensing, ops, security |
| `docs/phase_0/P0_Master_Specification.md` | Authoritative consolidated spec + conflict register |
| `docs/phase_0/PHASE_0_INDEX.md` | Index |
| `reports/Phase_0_Final_Report.md` | This report |

## 4. Key Decisions

1. **FES is restoration only.** It never returns embeddings, identity scores, or match results.
2. **Feature-Match is external**, with zero dependency. It can only couple to FES through a generic versioned API.
3. **Input: accept both full images and face crops** (`auto` default). FES owns detection and alignment, because restoration quality depends on consistent alignment.
4. **Default output:** an aligned, restored face crop, PNG, metadata stripped. Paste-back into the full image is optional.
5. **Multi-face default:** enhance the largest face. `all` and `index` are available.
6. **Identity preservation is a hard gate.** Critical defects are identity change, fabricated details, and attribute shifts. Beautification is a defect.
7. **Target severity:** Mild–Moderate is primary, Severe is secondary. Extreme inputs are flagged or declined, never silently hallucinated.
8. **Evaluation:** four dimensions reported separately and ranked lexicographically (identity → reliability → performance → quality). There must be ≥ 2 independent, test-only identity evaluators.
9. **Baselines:** degraded input, classical upscaling and restoration, general learned SR, face-specific candidates.
10. **No invented thresholds.** Every numeric threshold is TBD, with a method for setting it. Thresholds are fixed before candidate results are seen.
11. **Licensing is a hard gate.** Code, weights, training-data terms, detectors, and dependencies are each checked separately.
12. **No retention and no image logging by default.** Candidates must be runnable on CPU for development.
13. **Conflict resolutions C1–C8** (master spec §2). The most important is C2: in crop mode with no detectable face, reject rather than risk fabricating a face.

## 5. Assumptions

| ID | Assumption | Status |
|---|---|---|
| A-1 | Suitable pretrained restoration models exist | Unverified — Phase 1 |
| A-2 | Still images, one per request | Proposed |
| A-3 | Docker + NVIDIA GPU server as primary production target | **Unconfirmed** |
| A-4 | Commercial use intended | **Unconfirmed** (treated as requirement) |
| A-5 | Human faces only | Proposed |
| A-6 | Face images are privacy-sensitive; no retention | Treated as requirement |
| A-7 | Linux production OS | Unconfirmed |
| A-8 | No prior code to preserve | Fact |

## 6. Open Questions

| ID | Question | Owner |
|---|---|---|
| Q-1 | First consumers, volume, latency SLA | Product |
| Q-2 | NVIDIA GPU access for Phase 1 (class/VRAM/CUDA) | Infra |
| Q-3 | Lighting/contrast correction wanted? | Product |
| Q-4 | Commercial use confirmed? Who reviews licenses? | Product / Legal |
| Q-5 | Data-protection obligations and data residency | Legal / Compliance |
| Q-6 | Watermark/provenance on enhanced output? | Product / Compliance |
| Q-7 | Human-review panel availability | Product |
| Q-8 | Licensed/consented evaluation datasets; fairness attributes | Data / Legal |
| Q-9 | Fine-tuning allowed later; in-domain data? | Product / Data |
| Q-10 | CPU production fallback needed? | Product |

## 7. Risks (top)

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R-1 | Identity hallucination by generative models | Critical | Identity gate, multiple evaluators, human review, guardrails |
| R-2 | No commercially licensed model of adequate quality | High | License screen first; permissive alternatives; classical fallback |
| R-3 | No NVIDIA GPU for benchmarking (currently true) | Medium | Secure access early; do quality/identity work on CPU in the meantime |
| R-4 | Evaluator bias / metric gaming | High | Diverse evaluators, human review |
| R-6 | Demographic disparity | High | Stratified evaluation, fairness gate |
| R-7 | Privacy/regulatory non-compliance | High | No retention; legal review |
| R-8 | Enhanced output misused as an original | High | `enhanced` metadata; provenance decision |

Full register: master spec §16.

## 8. Deferred Decisions

D-1 … D-15 (master spec §14). Key ones:

- input limits, standard output size, and minimum face size;
- detector and thresholds;
- extreme input: flag vs. decline;
- lighting correction;
- fidelity option exposure;
- runtime quality-score method;
- API transport details;
- all numeric thresholds TH-1 … TH-11;
- production GPU and runtime;
- CPU fallback;
- **model choice**.

## 9. Environment Facts (summary)

| Item | Value |
|---|---|
| OS | Windows 11 Home SL, build 26200, 64-bit |
| Python | 3.11.0 |
| CPU | Intel Core Ultra 7 255H, 16 cores |
| RAM | ≈ 31.4 GB |
| GPU | Intel Arc 140T iGPU (shared memory) — **no NVIDIA GPU, no CUDA, no dedicated VRAM** |
| Storage | ≈ 229 GB free on D: |
| Docker | 29.8.0 (WSL2) |
| VCS | Git installed; **project not yet a git repository** |

## 10. Acceptance Checklist (GREEN criteria)

| # | GREEN criterion | Met? | Evidence |
|---|---|---|---|
| 1 | Unambiguous scope | ✅ | Master §3; T1 §4–§5 |
| 2 | Feature-Match explicitly external | ✅ | Master §3; T1 §7; T5 §9 |
| 3 | Input/output documented | ✅ | Master §4–§5; T2 |
| 4 | Degradation types documented | ✅ | Master §6; T3 §2 |
| 5 | Identity preservation as a core requirement | ✅ | Master §7; T3 §6 |
| 6 | Evaluation methodology defined | ✅ | Master §8; T4 |
| 7 | Technical/deployment constraints documented | ✅ | Master §11; T5 |
| 8 | Commercial licensing as a selection criterion | ✅ | Master §12; T4 §9 E1; T5 §6 |
| 9 | Open items explicitly identified | ✅ | Master §14, §17 |
| 10 | No fundamental product definition required before Phase 1 | ✅ | Remaining items are confirmations and resources, not definition gaps |

Phase 0 boundary checks:

| Check | Result |
|---|---|
| No models/datasets downloaded | ✅ |
| No FastAPI implemented | ✅ |
| No Feature-Match inspection/integration | ✅ |
| All tasks executed in the prescribed order | ✅ |
| Conflicts resolved explicitly, not silently | ✅ (Master §2, C1–C8) |

## 11. Decision Gate

### 🟡 YELLOW — Can proceed to Phase 1 with documented open items

**Why not GREEN:** the specification is complete, but three items affect Phase 1 outcomes and are not yet in place:

1. **Stakeholder confirmation missing.** The spec is based on assumptions A-3 (GPU production target) and A-4 (commercial use), which have not been confirmed. A-4 decides which models are even eligible.
2. **No GPU access.** The dev machine has no NVIDIA GPU or CUDA. Performance evaluation, and therefore final model selection, cannot be completed without a production-class GPU.
3. **No license reviewer named.** Licensing is a hard gate, and many face-restoration components carry non-commercial terms.

**Why not RED:** no scope, contract, or requirement is undefined. All open items are confirmations, resources, or empirically determined values that Phase 1 is designed to produce.

**Upgrade to GREEN when:** the product owner signs off the master spec (confirming A-3 and A-4), a license reviewer is named, and a GPU access plan exists.

## 12. Phase 1 — Exact Prerequisites & Recommendations

**Prerequisites (before or at Phase 1 kickoff)**

1. Product owner signs off `P0_Master_Specification.md`, confirming scope, A-3, and A-4.
2. `git init` the repository and commit the Phase 0 documents as the baseline.
3. License reviewer named, and commercial-use requirement confirmed (Q-4).
4. GPU access plan (Q-2), with GPU class, VRAM, and CUDA version recorded.
5. Project-local Python environment created (do not use the global packages).

**Recommended Phase 1 sequence**

1. **Candidate survey + license screen first.** List face-restoration candidates, detectors/landmarkers, and identity evaluators. Record code, weight, and data licenses from the primary sources. Only license-eligible items may be downloaded.
2. **Evaluation plan document:** freeze datasets, the degradation pipeline (seeded and versioned), metrics, repeated-run counts, CI method, α, and the methods for TH-1 … TH-11, *before* running any candidates.
3. **Evaluation harness:** baselines B0–B3 first, then candidates. Quality and identity work runs on the CPU/iGPU locally; performance runs on the GPU.
4. **Measure noise floors**, then set the thresholds.
5. **Stratified results + blind human review pilot** (Q-7).
6. **Candidate report** against evidence requirements E1–E8 (T4 §9), and a recommendation for D-15 (model choice) plus D-2, D-3, D-6, and D-7.
7. Still no FastAPI in Phase 1 unless the phase plan says otherwise. The API contract in T2 is the design input for the later API phase.
