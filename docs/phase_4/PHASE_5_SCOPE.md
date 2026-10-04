# Evidence-Based Phase 5 Scope (from the Phase 4 gate, 2026-10-04)

Phase 4 closed **YELLOW**. Phase 5 may build on what is proven. It may not convert unproven items into assumptions.

## In scope (supported by evidence)

| # | Work item | Basis |
|---|---|---|
| S1 | **Service skeleton around the frozen gate + router contract.** Default = return the original; UNSAFE_OR_UNCERTAIN → original + escalation flag; `enhanced: true` metadata only when a candidate output is returned. | P4-T6/T7: original-by-default is the only always-safe policy; routing evidence is positive only for gated use |
| S2 | **Process-isolated candidate execution** with hard timeouts, memory limits, and the four fallbacks (MODEL_FAILURE, TIMEOUT, RESOURCE_FAILURE, INVALID_OUTPUT) | T7 §1.3: thread timeout cannot kill a hung model |
| S3 | **GPU benchmark** of the gate + A4 route (latency P50/P95, throughput, cold start, memory) | G-1; A4 ≈ 45 s/face on CPU |
| S4 | **Real-world harness extension:** candidate alignment from the detection contract (not recorded transforms), versioned as a protocol bump, validated on synthetic first | P4-T4 §6 adapter gap |
| S5 | **Run the blinded human review `rv1`** (≥ 2 qualified reviewers), then extend it to routed outputs | G-5 |
| S6 | **Licence review prioritised to the routed path:** A4 SwinIR-L weights + YuNet detector (+ dlib/SFace evaluation-only confirmation) | G-4; these are the only assets on the T7 path |
| S7 | **Real-world data acquisition** under an approved DS03 consent/spec, then freeze `rw-v1` → re-calibrate gate thresholds and buckets on its **dev** split → single test evaluation | G-3; thresholds are synthetic-only |
| S8 | Robustness of the gate: more dev identities (e.g. the remaining dev_pilot refs) to re-check K and the thresholds before any production decision | T6 CV fold variance (U3) |

## Out of scope / not supported

| Item | Reason |
|---|---|
| Production model selection or "best model" claims | G-1/3/4/5 unresolved; P4 evidence is synthetic |
| Always-on enhancement | All candidates harmful always-on under both evaluators (P1/P2) |
| Routing to A1 RestoreFormer++, A2 VQFR, A5 chain | No positive dev utility in any bucket; substitution/collapse risk |
| Exposure/contrast or JPEG enhancement claims | No candidate improves identity there (P7). Exposure correction needs new, licence-screened candidates (Q-3). |
| Fairness claims | 2 identities per stratum |
| Using synthetic-calibrated thresholds on real traffic | Explicitly invalid until re-calibrated (taxonomy `calibration` block) |
| Feature-Match integration | Phase 0 boundary |

## Entry criteria for any later production-selection phase

1. G-3: real-world `rw-v1` frozen, with the gate/router re-calibrated on its dev split and evaluated once on its test split under E01 + E07.
2. G-5: human review status EVIDENCE, with substitution/fabrication rates on routed outputs and TH-5 agreed.
3. G-4: production licence decision for every asset on the routed path.
4. G-1: target-hardware performance within TH-8 (defined with product).
5. TH-1 (minimum meaningful identity gain) defined by product before results are compared.
