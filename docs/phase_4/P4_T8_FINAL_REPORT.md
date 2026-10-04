# Phase 4 Final Report — Real-World Validation & Enhancement Gating (P4-T8)

| Field | Value |
|---|---|
| Phase | 4 — Real-world validation & enhancement gating |
| Date | 2026-10-04 |
| Baseline | Phase 3 commit `fdfce67` (gate YELLOW), protocol `p3-protocol-1.0.1`, unchanged |
| Index | [docs/phase_4/PHASE_4_INDEX.md](../docs/phase_4/PHASE_4_INDEX.md) · gate matrix [docs/phase_4/PHASE_4_GATE_MATRIX.md](../docs/phase_4/PHASE_4_GATE_MATRIX.md) · Phase 5 scope [docs/phase_4/PHASE_5_SCOPE.md](../docs/phase_4/PHASE_5_SCOPE.md) |
| **Phase 4 gate** | **🟡 YELLOW.** Selective enhancement is supported by reproducible two-evaluator **synthetic** evidence, and identity risk is controllable by gating. The real-world, human-review, licensing and GPU gates remain unresolved. |

> Every quantitative result in this report is on **synthetic** degradations of FairFace chips (`fes-syn-v1`). There is **no real-world evidence** (G-3) and **no human review** (G-5). No model is selected, ranked, or approved for production.

---

## 1. Executive Summary

1. **Unconditional enhancement is ruled out.** It is confirmed by two independent identity evaluators: SFace (E01) and dlib ResNet (E07, new in P4-T3). On the untouched synthetic test population, **every candidate used always-on lowers identity similarity vs the untouched input under both evaluators.** This holds for A1 RestoreFormer++, A2 VQFR v1, A3 Real-ESRGAN, A4 SwinIR-L and A5 FBCNN → NAFNet-SIDD → Real-ESRGAN.
2. **Mild inputs are always harmed by always-on enhancement** (135–245 of 252 mild pairs). The frozen Do-Not-Enhance gate (P4-T6) enhances **0 of 252** mild inputs.
3. **Selective routing (P4-T7) is the only policy with a positive identity effect under both evaluators and no identity flips:**
   - quality analysis → frozen gate → original, or A4 for NOISE/BLUR buckets;
   - 7 % of inputs enhanced;
   - ΔID +0.0044 (SFace) / +0.0005 (dlib), both CIs > 0;
   - 38 confirmed benefits vs 2 confirmed harms;
   - 0 flips under either evaluator.
4. **The benefit is narrow:** noise and Gaussian blur only, through one general-SR model. Face-generative models (A1, A2) and the blind chain (A5) earned **no** route.
5. **Production is blocked** by:
   - G-3: no approved real-world data;
   - G-5: human review not performed;
   - G-4: every candidate weight and the YuNet detector the gate depends on are `REVIEW_REQUIRED`;
   - G-1: no GPU; A4 takes about 45 s per face on CPU.

## 2. Proven Findings (what the evidence establishes)

| # | Finding | Evidence |
|---|---|---|
| P1 | No candidate improves mean identity when applied to every input | P4-T4 (test) and dev: all ΔID CIs < 0 under E01 **and** E07 |
| P2 | Always-on enhancement harms mild inputs | Mild: 0 confirmed benefits for any candidate; 135–245 of 252 harmed (P4-T6 §2.2) |
| P3 | A deterministic, reference-free gate can prevent mild-input damage | T6: 0/252 mild enhanced, frozen before evaluation, reproduced byte-identical |
| P4 | Gated, degradation-specific routing gives a small multi-evaluator identity gain with no flips (synthetic) | T7 route ▲/▲, 0/0 flips, 3 harm-any of 896; dev in-sample agrees |
| P5 | Noise is the family with the clearest identity benefit; Gaussian blur under gating is second | P4-T4 (A3/A4 ▲/▲ on noise); T7 per-family (noise 18−0, gaussblur 19−1) |
| P6 | Phase 3's single-evaluator "extreme-severity gain" for A1/A3 is not confirmed by the second evaluator | P4-T4 §4 (A1 ▲/≈, A3 ▲/▼; only A4 ▲/▲) |
| P7 | No candidate corrects exposure/contrast or JPEG artifacts in identity terms | P4-T4 photometric ▼/▼; T7 dev utility ≤ 0 in PHOTOMETRIC/COMPRESS |
| P8 | The pipeline is reproducible | Byte-identical re-scores (E07 A4, dev A3), T6/T7 decision logs re-run identical, 133 tests pass, run validation `all_ok` |

## 3. Unproven / Uncertain Findings

| # | Item | Why uncertain |
|---|---|---|
| U1 | Any real-world effect | No approved real-world data (G-3). Thresholds are calibrated on synthetic dev and explicitly invalid for real inputs. |
| U2 | Human-perceived identity substitution / fabrication | Kit built, review not performed (G-5) |
| U3 | Stability of the gate's learned rules | Dev CV for the chosen K = 4: per-fold utility −15 / +7 / +15 / +9 (one fold net-harmful). The test result is positive but the margin is small. |
| U4 | Fairness | 2 identities per race × gender stratum; no demographic conclusion possible |
| U5 | Production-path identity with non-oracle alignment | Candidate inputs still use the recorded FairFace alignment (P3-T1 §12); only gating uses the real detection contract |
| U6 | Effect size relevance | Pooled +0.004 SFace is small. Its operational significance depends on consumer use and the TH-1 threshold (product decision). |

## 4. Identity Preservation (test, synthetic; SFace / dlib separately)

| Policy | Enh. rate | ΔID SFace | ΔID dlib | Conf. benefit | Harm (any) | Flips E01 / E07 |
|---|---|---|---|---|---|---|
| Always original (B0) | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| Always A1 / A2 / A3 / A4 / A5 | 1.0 | all ▼ (−0.008 … −0.080) | all ▼ (−0.001 … −0.011) | 25–120 | 440–795 | 12–72 / 9–65 |
| T6 gate + A4 | 0.10 | +0.0041 ▲ | +0.0006 ▲ | 45 | 12 | 1 / 1 |
| **T7 route** | **0.07** | **+0.0044 ▲** | **+0.0005 ▲** | **38** | **3** | **0 / 0** |

## 5. Mild-Input Damage

| Policy | Mild inputs enhanced | Mild harmed (any evaluator) |
|---|---|---|
| Always A1 / A2 / A3 / A4 / A5 | 252 / 252 | 245 / 245 / 145 / 135 / 195 |
| T6 + any candidate | 0 | 0 |
| T7 route | 0 | 0 |

## 6. Severe / Noise Findings

- **Noise:** the clearest benefit. A3 and A4 are ▲/▲ (always-on, P4-T4). Under T7, noise inputs are routed to A4: 33 enhanced, 18 confirmed benefits, 0 harms.
- **Severe/extreme:** most severe/extreme inputs **fail the detection contract** (0 % on 14-px lowres and noise-extreme; 14–25 % on combined/realistic2/lowres-severe). They are UNSAFE_OR_UNCERTAIN, so the original is returned and escalated. This is where Phase 3 saw identity substitution (A1/A2) and A5 collapse. Gating removes them from the enhancement path rather than "rescuing" them.
- **Missed opportunity:** 13 extreme-severity WORTHY pairs are not enhanced (T6). That is the price of the conservative policy.

## 7. Real-World Findings

**None.** P4-T1:
- every candidate source fails the legal gate;
- DS03 (consented capture) is `PENDING_LEGAL_REVIEW`;
- `BLOCKED.json` evidence is recorded.

The freeze tool, leakage checks, dataset card and executable plan are ready (P4-T1 §5, P4-T4 §6). One harness gap must be closed first: candidate alignment from the detection contract instead of recorded transforms.

## 8. Human Review

**Not performed (G-5).** The blinded kit `rv1` is built, reproducible and tested:
- 228 items: 156 random + 20 targeted + 40 controls + 12 repeats;
- an offline tool;
- pre-registered analysis (qualification ≥ 0.85, Krippendorff's α, paired vs B0).

`analysis_rv1.json` status: INSUFFICIENT (0 reviewers). The model's own visual judgement is not used as human evidence.

## 9. Cross-Evaluator Agreement

| Measure | Value |
|---|---|
| Same-direction degradation sensitivity (ref-vs-B0, dev) | Spearman 0.963 |
| Impostor ranking (dev) | Spearman 0.384, i.e. independent identity errors |
| Verdict κ on always-on candidates (test) | 0.21 (A2) – 0.54 (A4); lowest for the generative face models |
| Opposite-direction verdicts | A1 76, A2 106 always-on. **0 on T7-routed pairs** (74.6 % exact agreement; the rest are one-evaluator-neutral). |

Single-evaluator identity claims about generative face models are unreliable. The routing claim holds under both evaluators.

## 10. Routing Performance

| Metric | Value |
|---|---|
| Decision mix (test) | DO_NOT_ENHANCE 63.1 % · UNSAFE_OR_UNCERTAIN 26.7 % · ENHANCE → original 3.2 % · ENHANCE → A4 7.0 % |
| Route table | NOISE → A4, BLUR → A4, all else → original |
| Fallbacks triggered on test | 0 |
| Runtime router validation (dev, real adapters) | 5/5 checks: deterministic, equal to harness, original exact, fault injection → original, plan = replay |
| Gate overhead | 2–12 ms p50 per input (CPU, 1 thread) |
| A4 inference | ≈ 45 s per face (CPU, 2 threads, shared host); not GPU-representative |

## 11. Licensing

| Asset | Evaluation use | Production |
|---|---|---|
| A1–A5 candidate weights (+ NAFNet) | ELIGIBLE | **REVIEW_REQUIRED** (FFHQ/DIV2K training data, R4) |
| YuNet detector (used by the gate, buckets and output guard), facexlib RetinaFace | ELIGIBLE | **REVIEW_REQUIRED** (WIDER FACE, R4) |
| E01 SFace, E07 dlib ResNet | ELIGIBLE (evaluation only) | n/a: never in the runtime |
| FairFace (DS01) | ELIGIBLE (CC BY 4.0) | ELIGIBLE; per-image YFCC rights noted |
| dlib runtime (BSL-1.0), OpenCV, PyTorch | permissive | review with the stack |

**No licence reviewer is named.** That makes G-4 a hard blocker for any production path, including the gate itself (YuNet).

## 12. GPU / Performance

- **No NVIDIA GPU (G-1).** All timings are CPU on a shared laptop.
- A4, the only routed model, takes about 45 s per face on CPU. Phase 3 measured 49 s at 224 px, 2 threads.
- The gate and features cost milliseconds.
- **Throughput, P95 and cold-start on target hardware are unmeasured.** TH-8 is still TBD.

## 13. Limitations

1. Synthetic data only. 28 test identities, 28 dev identities. Per-stratum results are descriptive.
2. 224-px FairFace references; candidates use oracle (recorded) alignment.
3. The gate's learned rules show fold variance on dev (U3). Thresholds are synthetic-calibrated.
4. Both identity evaluators are imperfect, and they agree only moderately on generative outputs.
5. Thread-based timeout cannot kill a hung model (needs process isolation).
6. Dev inference was interrupted by host memory pressure and laptop standby. It was resumed from checkpoints with only the worker count changed (threads unchanged), and completion validation passed (`RESUME_LOG.md`).

## 14. Open Blockers

| Gate | Blocker | Owner | Unblocks |
|---|---|---|---|
| **G-3** | No approved real-world data (DS03 consent/DPDP review, or a vendor contract) | Legal / Compliance + Data | Real-world benchmark, threshold re-calibration, true TAR/FAR |
| **G-5** | Blinded human review not performed (≥ 2 qualified reviewers) | Product (Q-7) | Fabrication / substitution evidence, TH-5 |
| **G-4** | No licence reviewer. All candidate weights + YuNet are REVIEW_REQUIRED for production. | Product / Legal (Q-4) | Any production path |
| **G-1** | No GPU; no target-hardware latency/throughput | Infra (Q-2) | TH-8, operational viability of the A4 route |
| — | Product thresholds TH-1 (min ID gain), TH-5, TH-8 undefined | Product | Pass/fail criteria |

## 15. Evidence-Based Phase 5 Scope

See [docs/phase_4/PHASE_5_SCOPE.md](../docs/phase_4/PHASE_5_SCOPE.md).

**May proceed:**
- service skeleton around the **frozen gate + router contract**, with the original as default and escalation for UNSAFE;
- process-isolated candidate execution with hard timeouts;
- GPU benchmarking of A4 and the gate;
- real-world harness extension (detection-based alignment);
- running the human review;
- licence review of A4 + YuNet;
- re-calibration once a real dev split exists.

**Must not proceed:**
- production model selection;
- any always-on enhancement;
- routing to A1/A2/A5;
- exposure/JPEG "enhancement" claims;
- fairness claims;
- deployment using the synthetic-calibrated thresholds on real traffic.

## 16. Decision

### 🟡 YELLOW

**Why not GREEN:** the required production gates are unresolved: G-3 real-world, G-5 human review, G-4 licensing, G-1 GPU. All results are synthetic. Missing evidence is treated as unresolved, not as a pass.

**Why not RED:** the dominant identity risks found in Phase 3 are **controllable**:
- mild-input harm;
- extreme-input identity substitution;
- A5 collapse.

A frozen, reproducible gate + router eliminates mild damage, sends unreliable inputs back unchanged with an escalation flag, and yields 0 identity flips with a small positive gain under two independent evaluators.

**Selective routing is supported, on synthetic evidence, as the design direction. No model is approved for production.**
