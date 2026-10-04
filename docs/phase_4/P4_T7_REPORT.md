# P4-T7 — Selective Model Routing: Report

| Field | Value |
|---|---|
| Phase / Task | Phase 4 / P4-T7 (completion pack) |
| Date | 2026-10-04 |
| Status | **Complete:** routing implemented and tested, calibrated on DEV on top of the frozen T6 gate, runtime router validated on DEV, frozen (commit `785a4b2`), evaluated once on untouched TEST, and the evaluation reproduced |
| Frozen config | `configs/phase4/t7_routing.json`, seal `2f026f229039…`, built on the T6 seal `b8ee2c43…` (verified at load) |
| Code | `src/fes_bench/routing.py` (`p4-routing-1.0.0`): buckets, table fit, runtime `Router` with fallbacks |
| Scripts | `p4_route_calibrate.py` (DEV) · `p4_router_validate.py` (DEV, real adapters) · `p4_route_evaluate.py` (single TEST evaluation) |
| Tests | `tests/test_p4_routing.py`: 21 passed (buckets, table rules, every fallback by fault injection, timeout recovery). Full suite: 133 passed. |
| Decision log | `data/manifests/phase4/t7/decision_log_main.csv.gz` (per input: decision, reason, bucket, route, used, fallback, escalate, ΔID per evaluator, verdict) |
| Data | **SYNTHETIC** (`p4dev` calibration, `main` test). Not real-world. |

---

## 1. Routing Contract

```
input ─► quality features (p4-quality-1.0.0, reference-free)
      ─► frozen T6 gate ─┬─ UNSAFE_OR_UNCERTAIN ─► ORIGINAL + escalate flag
                         ├─ DO_NOT_ENHANCE      ─► ORIGINAL
                         └─ ENHANCE_CANDIDATE   ─► degradation bucket ─► frozen table ─► candidate | ORIGINAL
                                                     candidate run with guards; any failure ─► ORIGINAL
```

Rules (from the completion pack):
- The original (B0) is always valid.
- Nothing is enhanced just because a model exists.
- Any uncertainty returns the original.
- Routing chooses only among the five already-benchmarked candidates.

### 1.1 Degradation buckets (evidence-based, no outcome data)

Thresholds are the 2 % / 98 % quantiles of the features of the **112 clean dev references**, i.e. what "not degraded" looks like:

| Flag | Rule |
|---|---|
| LOWRES | `iod_px < 71.4` |
| NOISE | `noise_sigma > 1.08` and not LOWRES (small native images inflate the noise estimate, P4-T2) |
| BLUR | `lap_norm_c < 0.0231` and not LOWRES |
| COMPRESS | `blockiness > 1.52` |
| PHOTOMETRIC | `luma_mean < 47.1` or `rms_contrast < 25.4` |

Bucket priority: PHOTOMETRIC > MIXED (≥ 2 structural flags) > LOWRES > NOISE > BLUR > COMPRESS > NONE.

How the buckets match the true synthetic family (test, gated inputs):
- NOISE: 33/33 noise.
- COMPRESS: 12/12 jpeg.
- BLUR: 29/30 gaussian or motion blur.
- PHOTOMETRIC on gated inputs: blur/jpeg with darkening side effects.

### 1.2 Route table (DEV, pre-registered rule)

Per bucket, among DEV inputs the T6 gate sends to ENHANCE, the candidate with the largest U = #confirmed benefit − 3·#harm is chosen. It needs U > 0 and n ≥ 10; ties go to the cheaper candidate. Otherwise the route is ORIGINAL.

| Bucket | DEV n | U: A1 / A2 / A3 / A4 / A5 | **Route** |
|---|---|---|---|
| NOISE | 35 | −60 / −93 / +1 / **+16** / −49 | **A4 SwinIR-L** |
| BLUR | 38 | −93 / −97 / −24 / **+4** / −87 | **A4 SwinIR-L** |
| COMPRESS | 16 | −39 / −48 / −4 / −4 / −18 | ORIGINAL |
| PHOTOMETRIC | 8 | −24 / −21 / −14 / +5 / −17 | ORIGINAL (n < 10) |
| LOWRES | 5 | all < 0 | ORIGINAL |
| MIXED / NONE | 0 / 1 | — | ORIGINAL |

### 1.3 Fallbacks (runtime `Router`)

| Failure | Detection | Result |
|---|---|---|
| MODEL_FAILURE | any exception from the adapter | ORIGINAL + reason |
| TIMEOUT | > `timeout_s` (120 s frozen) | ORIGINAL. The thread can't be killed in-process, so production must isolate it (Phase 5). |
| RESOURCE_FAILURE | `MemoryError` / allocator out-of-memory text | ORIGINAL |
| INVALID_OUTPUT | wrong shape/dtype, non-finite, near-constant (std < 1), **or no face detected on the output** | ORIGINAL. The no-face check catches colour-noise collapse like Phase 3's A5 dark-input failure. |
| UNSAFE_OR_UNCERTAIN / DO_NOT_ENHANCE | T6 | ORIGINAL (+ escalate flag for UNSAFE) |

## 2. Runtime Validation on DEV (real adapters; `router_validation.json`)

| Check | Result |
|---|---|
| R1 decisions/routes identical on two passes | ✅ |
| R1b runtime plan from freshly computed features = offline replay from stored features | ✅ |
| R2 routed outputs pixel-identical across passes **and** equal to the benchmark harness output SHA-256 | ✅ |
| R3 ORIGINAL paths return the B0 image exactly | ✅ |
| R4 injected TIMEOUT (1 ms) and MODEL_FAILURE end in ORIGINAL with the correct reason | ✅ |

**Resources:** peak RSS 1.44 GB (one candidate loaded at a time).

**Latency** (shared CPU, 2 threads):

| Path | Latency |
|---|---|
| ORIGINAL (features + gate) | 1–5 ms |
| A4 route | 44–52 s per face |

## 3. Single Test Evaluation (`main`, untouched; re-run byte-identical, decision log `79094637…`)

### 3.1 Routing behaviour

| | Count (of 896) |
|---|---|
| UNSAFE_OR_UNCERTAIN → original (escalated) | 239 (26.7 %) |
| DO_NOT_ENHANCE → original | 565 (63.1 %) |
| ENHANCE_CANDIDATE → bucket → original | 29 (3.2 %) |
| **ENHANCE_CANDIDATE → A4** | **63 (7.0 %)**: 33 NOISE, 30 BLUR |
| Fallbacks triggered | 0 (all 63 A4 outputs had a detectable face) |
| Mild inputs enhanced | **0 / 252** |

### 3.2 Policy comparison (P4-T7 §1–4; SFace E01 and dlib E07 reported separately)

| Policy | Enh. rate | ΔID SFace | ΔID dlib | Conf. benefit | Harm (any) | Conf. harm | Rescues/flips E01 | Rescues/flips E07 | MILD harm |
|---|---|---|---|---|---|---|---|---|---|
| 1 Always original | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 | 0/0 | 0 |
| 2 Always A1 | 1.00 | −0.0492 ▼ | −0.0054 ▼ | 56 | 726 | 540 | 16/12 | 17/9 | 245 |
| 2 Always A2 | 1.00 | −0.0608 ▼ | −0.0065 ▼ | 25 | 795 | 584 | 10/13 | 10/18 | 245 |
| 2 Always A3 | 1.00 | −0.0205 ▼ | −0.0040 ▼ | 58 | 533 | 301 | 15/38 | 8/27 | 145 |
| 2 Always A4 | 1.00 | −0.0078 ▼ | −0.0013 ▼ | 120 | 440 | 260 | 15/38 | 11/18 | 135 |
| 2 Always A5 | 1.00 | −0.0802 ▼ | −0.0110 ▼ | 45 | 677 | 481 | 11/72 | 6/65 | 195 |
| 3 T6 + A3 | 0.10 | +0.0016 ▲ | −0.0001 ≈ | 17 | 25 | 13 | 1/1 | 0/3 | 0 |
| 3 T6 + A4 | 0.10 | +0.0041 ▲ | +0.0006 ▲ | 45 | 12 | 7 | 0/1 | 1/1 | 0 |
| **4 T7 route (gated, degradation-specific)** | **0.07** | **+0.0044 ▲** [+0.0033, +0.0055] | **+0.0005 ▲** [+0.0003, +0.0007] | **38** | **3** | **2** | **0/0** | **0/0** | **0** |

The DEV in-sample reference agrees: T7 route +0.0049 ▲ / +0.0006 ▲, 38 benefits, 6 harm (any), 0 flips.

### 3.3 By family (T7 route, test)

| Family | Enhanced | Benefit − harm | ΔID E01 / E07 |
|---|---|---|---|
| noise | 33 / 112 | 18 − 0 | ▲ / ▲ |
| gaussblur | 23 / 112 | 19 − 1 | ▲ / ▲ |
| motionblur | 6 / 112 | 1 − 1 | ≈ / ≈ |
| lowcontrast | 1 / 84 | 0 − 1 | ≈ / ≈ |
| lowres, jpeg, underexposure, combined, realistic2 | 0 | 0 − 0 | ≈ / ≈ (original returned) |

### 3.4 Evaluator agreement on routed pairs (n = 63)

| | Count |
|---|---|
| Both BENEFIT | 37 |
| E01 BENEFIT / E07 NEUTRAL | 14 |
| E01 NEUTRAL / E07 BENEFIT | 1 |
| Both NEUTRAL | 8 |
| Both HARM | 2 |
| E01 HARM / E07 NEUTRAL | 1 |
| **Opposite verdicts (DISAGREE)** | **0** |
| Exact agreement | 74.6 % |

### 3.5 Failure cases (all routed harms)

| Cell | Route | ΔE01 | ΔE07 | Verdict |
|---|---|---|---|---|
| motionblur-extreme | A4 | −0.068 | −0.013 | CONFIRMED_HARM |
| lowcontrast-moderate | A4 | −0.048 | −0.004 | CONFIRMED_HARM |
| gaussblur-severe | A4 | −0.025 | −0.002 | PARTIAL_HARM |

**Phase 3 failure modes, followed up:**

| Failure mode | Outcome under routing |
|---|---|
| A5 dark-image collapse | A5 is never routed (negative dev utility in every bucket). Dark inputs fall into PHOTOMETRIC → original. The INVALID_OUTPUT face check is a second guard. |
| Extreme-low-resolution identity substitution (A1/A2) | 14/28 px inputs fail detection and become UNSAFE_OR_UNCERTAIN → original. LOWRES is routed to original. A1/A2 are never routed. |
| Mild-input harm | 0 mild inputs enhanced. |
| Exposure/contrast | No candidate helps (P4-T4), so PHOTOMETRIC → original. Exposure correction remains an unmet product need (Q-3). |
| Compression | No positive candidate on dev (COMPRESS → original). JPEG remains unaddressed. |

## 4. Findings

1. **Selective routing is supported on synthetic evidence.** The frozen route has the best risk profile of all policies:
   - positive mean identity change under **both** evaluators;
   - 38 confirmed benefits vs 2 confirmed harms (3 harm-any);
   - **0 identity flips** under either evaluator;
   - 0 mild inputs touched.

   Every always-on policy is negative under both evaluators.
2. **The benefit is narrow:** noise and Gaussian blur only, enhancing 7 % of inputs, via one general-SR model (A4). The pooled gain is small (+0.004 SFace), because 93 % of inputs are deliberately left untouched.
3. **Face-specific generative models (A1, A2) and the blind chain (A5) earned no route.** On this evidence they should not be in a production path.
4. **A4 is slow on CPU** (about 45 s per face). A route is only operationally viable with GPU performance evidence (G-1).
5. **This is a routing result, not a model selection.** A4 is the only candidate whose gated use showed a multi-evaluator benefit here. That is not a production approval: licence REVIEW_REQUIRED (G-4), no real-world evidence (G-3), no human review (G-5).

## 5. Deliverables Checklist

| Required | Status |
|---|---|
| Routing implementation | ✅ `src/fes_bench/routing.py` |
| Config (frozen) | ✅ `configs/phase4/t7_routing.json` |
| Tests | ✅ 21 (+ 12 gate tests) |
| Decision manifest/log | ✅ `decision_log_main.csv.gz` (+ dev) |
| Determinism and resource validation on DEV | ✅ `router_validation.json` |
| Single test evaluation; enhancement rate, route distribution, ΔID, rescue/damage, agreement, failure cases | ✅ §3 |

## 6. Limits

- Synthetic only, 28 test identities. The CIs are clustered by identity, and per-stratum fairness is not assessable.
- Timeout fallback is thread-based, so it is not a hard kill.
- Buckets and thresholds are calibrated on synthetic dev data and need re-calibration on a real dev split (G-3).
- YuNet (used for gating and the output face check) has production licence status REVIEW_REQUIRED.
