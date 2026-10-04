# P4-T6 — Do-Not-Enhance Gate: Report

| Field | Value |
|---|---|
| Phase / Task | Phase 4 / P4-T6 (completion pack `Tasks/Face_Enhancement_Phase_4_Completion`) |
| Date | 2026-10-04 |
| Status | **Complete:** implemented, tested, calibrated on DEV, frozen, evaluated once on untouched TEST, and the evaluation reproduced |
| Frozen gate | `configs/phase4/t6_gate.json`. Seal `b8ee2c4309e5…`, file SHA-256 `e8b8e72e…`, frozen in commit `e504dfa` **before** the test evaluation |
| Code | `src/fes_bench/gate.py` (`p4-gate-1.1.0`) · `src/fes_bench/quality.py` (`p4-quality-1.0.0`) · `src/fes_bench/p4eval.py` |
| Scripts | `scripts/p4_gate_calibrate.py` (DEV only) · `scripts/p4_gate_evaluate.py` |
| Tests | `tests/test_p4_gate.py`: 12 passed (unit + image→decision integration). Full suite: 133 passed. |
| Artifacts | `data/manifests/phase4/t6/{calibration_report.json, evaluation_main.json/.md, decisions_main.csv.gz, evaluation_p4dev.*}` |
| Data | **SYNTHETIC** `fes-syn-v1`. Calibration on `p4dev` (dev split, 28 refs × 32 cells). Evaluation on `main` (Phase 3 test population, 28 refs × 32 cells), opened once. |

> These results are synthetic-degradation evidence only. They are **not** real-world evidence (G-3), and the thresholds are explicitly invalid for real inputs until re-calibrated on a real dev split.

---

## 1. Gate Definition

The decision order uses reference-free features of the input only.

| Decision | Rule | Calibrated? |
|---|---|---|
| **UNSAFE_OR_UNCERTAIN** | The FES detection contract returned no aligned face (no face, IOD < 12 px, landmark failure), **or** detector confidence < 0.85, **or** \|yaw proxy\| > 0.25 / implausible landmarks | **No.** These are P2-T6 constants defined before Phase 4 (P0 C2/C6: never invent a face; flag or decline extreme inputs). The original is returned and escalated. |
| **ENHANCE_CANDIDATE** | Any of the 4 frozen rules below | Yes (DEV) |
| **DO_NOT_ENHANCE** | Otherwise (default; the original/B0 is returned untouched) | — |

**Implemented quality features** (P4-T2, `p4-quality-1.0.0`):
- effective face resolution: `iod_px`, `in_min_side`;
- blur/focus: `lap_norm_c`, `hf_ratio_c`;
- noise: `noise_sigma`;
- exposure: `luma_mean`, `dark_clip`;
- contrast: `rms_contrast`, `dyn_range`;
- compression: `blockiness`;
- detection confidence: `det_score`;
- alignment validity: `det_status`, `align_resid`, `yaw_proxy`.

**Frozen rules** (a condition on a missing/NaN feature never fires):

| # | Rule | Reading |
|---|---|---|
| 0 | `noise_sigma ≥ 9.84` | visibly noisy input |
| 1 | `hf_ratio_c ≤ 0.000167` | almost no high-frequency content (heavy blur or low resolution) |
| 2 | `blockiness ≥ 2.42` ∧ `noise_sigma ≤ 0.283` | strong JPEG blocking on an otherwise clean signal |
| 3 | `lap_norm_c ≤ 0.0198` ∧ `noise_sigma ≤ 0.283` | strong blur on a clean signal |

### Calibration (pre-registered in `gate.py`, DEV only)

- **Labels** come from all five candidates' multi-evaluator verdicts vs B0. SFace (E01) and dlib (E07) are each judged against their own δ (0.0215 / 0.0026).
  - WORTHY: at least one candidate is a confirmed benefit.
  - UNSAFE: every candidate shows harm.
- **Objective:** U = #WORTHY − 3·#UNSAFE among enhanced inputs, built as a greedy rule list. K is chosen by 4-fold CV grouped by reference.
- **DEV labels:** 86 WORTHY / 243 UNSAFE / 381 NEUTRAL over 710 fit pairs. The other 186 pairs are UNSAFE_OR_UNCERTAIN by rule.
- **CV utility by K:** 0 → 0.0, 1 → −1.8, 2 → +3.0, 3 → +2.8, 4 → +4.0, so **K = 4**.
- **Robustness caveat (recorded, not tuned):** the per-fold CV utility for K = 4 is **−15 / +7 / +15 / +9**. One held-out fold was net harmful. The dev evidence for the learned rules is weak to moderate. Only "never enhance" (K = 0) is never negative.

## 2. Single Test Evaluation (`main`, untouched, run once; re-run byte-identical)

### 2.1 Decisions and gate correctness (candidate-agnostic labels)

| Group | n | DO_NOT_ENHANCE | ENHANCE | UNSAFE/UNCERTAIN | False enhancement (ENH on all-harm pair) | Missed enhancement (DNE on WORTHY) / WORTHY |
|---|---|---|---|---|---|---|
| **all** | 896 | 565 (63 %) | 92 (10.3 %) | 239 (26.7 %) | 9 | 29 / 162 |
| **mild** | 252 | 223 | **0** | 29 | 0 | 0 / 0 |
| moderate | 252 | 204 | 18 | 30 | 2 | 15 / 28 |
| severe | 252 | 101 | 40 | 111 | 5 | 1 / 44 |
| extreme | 140 | 37 | 34 | 69 | 2 | 13 / 90 |
| noise | 112 | 32 | 33 | 47 | 0 | 3 / 64 |
| underexposure | 84 | 70 | 0 | 14 | 0 | 0 / 0 |
| lowcontrast | 84 | 68 | 1 | 15 | 1 | 0 / 0 |

UNSAFE_OR_UNCERTAIN reasons, mostly `no_valid_face`: concentrated in lowres-extreme/severe, noise-extreme and combined/realistic2-severe. These are exactly the inputs where Phase 3 saw identity substitution and A5 colour collapse.

### 2.2 Identity outcome: gate + candidate vs always-original vs always-candidate

SFace and dlib are reported separately and never pooled. ▲/▼/≈ means the 95 % cluster-bootstrap CI is above, below or contains 0.

| Policy | Enh. rate | ΔID SFace (E01) | ΔID dlib (E07) | Conf. benefit | Harm (any) | Conf. harm | Rescues/flips E01 | Rescues/flips E07 | **MILD harm / MILD enhanced** |
|---|---|---|---|---|---|---|---|---|---|
| always original (B0) | 0 | 0 | 0 | 0 | 0 | 0 | — | — | **0 / 0** |
| always A1 RestoreFormer++ | 1.00 | −0.0492 ▼ | −0.0054 ▼ | 56 | 726 | 540 | 16/12 | 17/9 | **245 / 252** |
| T6 + A1 | 0.10 | −0.0044 ▼ | −0.0003 ≈ | 6 | 66 | 33 | 0/0 | 1/0 | 0 / 0 |
| always A2 VQFR v1 | 1.00 | −0.0608 ▼ | −0.0065 ▼ | 25 | 795 | 584 | 10/13 | 10/18 | **245 / 252** |
| T6 + A2 | 0.10 | −0.0046 ▼ | −0.0004 ▼ | 0 | 74 | 35 | 0/2 | 0/1 | 0 / 0 |
| always A3 Real-ESRGAN | 1.00 | −0.0205 ▼ | −0.0040 ▼ | 58 | 533 | 301 | 15/38 | 8/27 | **145 / 252** |
| T6 + A3 | 0.10 | +0.0016 ▲ | −0.0001 ≈ | 17 | 25 | 13 | 1/1 | 0/3 | 0 / 0 |
| always A4 SwinIR-L | 1.00 | −0.0078 ▼ | −0.0013 ▼ | 120 | 440 | 260 | 15/38 | 11/18 | **135 / 252** |
| **T6 + A4** | 0.10 | **+0.0041 ▲** | **+0.0006 ▲** | **45** | 12 | 7 | 0/1 | 1/1 | 0 / 0 |
| always A5 chain | 1.00 | −0.0802 ▼ | −0.0110 ▼ | 45 | 677 | 481 | 11/72 | 6/65 | **195 / 252** |
| T6 + A5 | 0.10 | −0.0053 ≈ | −0.0009 ▼ | 10 | 52 | 30 | 0/6 | 0/7 | 0 / 0 |

The DEV in-sample reference (`evaluation_p4dev.*`) shows the same pattern: T6 + A4 ▲/▲ (+0.0048 / +0.0006), T6 + A3 ▲/≈, and 0 mild inputs enhanced.

## 3. Findings

1. **The gate removes mild-input damage completely.** Always-on candidates harm 135–245 of 252 mild inputs. Under the gate, 0 mild inputs are enhanced, so mild damage is 0 for every candidate.
2. **Gating turns the best general-SR candidate from net-harmful to net-beneficial under both evaluators.** A4 goes from ▼/▼ to ▲/▲ (45 confirmed benefits vs 7 confirmed harms). A3 goes from ▼/▼ to ▲/≈.
3. **Gating does not rescue the generative face models (A1, A2) or the A5 chain.** They remain harmful or neutral inside the gate's ENHANCE region. The gate decides *whether*; *which* candidate matters, which is T7.
4. **The gate errs conservatively.**
   - 9 false enhancements (about 1 % of inputs).
   - 29 of 162 WORTHY pairs are missed; 13 of those are extreme.
   - 239 inputs are UNSAFE/UNCERTAIN and returned as the original. Most of them fail detection. Identity rescue on them is impossible without inventing a face.
5. **Effect sizes are small** (ΔE01 +0.004 pooled), because 90 % of inputs are left untouched by design. The benefit is concentrated in the noise family and a subset of blur.
6. **Weak robustness signal from CV** (one of four dev folds net harmful at K = 4). The test result is positive, but with 28 identities the margin is not large. More dev data, or a real dev split, should re-validate K and the thresholds.

## 4. Completion-Pack Checklist

| Requirement | Status |
|---|---|
| Resume partial inference, no recompute | ✅ `RESUME_LOG.md`; validation `all_ok` |
| Conservative workers; A2 memory-safe | ✅ A2 1 → 2 → 4 workers with a memory watchdog; threads unchanged |
| Implemented, testable quality features | ✅ P4-T2 module, 8 tests |
| DO_NOT_ENHANCE / ENHANCE_CANDIDATE / UNSAFE_OR_UNCERTAIN | ✅ |
| Calibrated on DEV only | ✅ the calibration script never reads `main` |
| Identity preservation / rescue / damage; mild damage explicit | ✅ §2.2 (SFace and dlib separately) |
| Frozen before evaluation | ✅ commit `e504dfa` precedes the evaluation |
| Evaluated once on untouched data; reproducible | ✅ re-run gives byte-identical decisions (`420e0d0d…`) and identical metrics |
| Deterministic unit/integration tests | ✅ 12 |

## 5. Evidence Status and Limits

- **Proven (synthetic, two evaluators):** the gate eliminates mild damage, and gate + A4 gives a small, positive, multi-evaluator identity gain on the synthetic test population.
- **Not proven:**
  - real-world validity (G-3; thresholds must be re-calibrated);
  - human-perceived identity or fabrication (G-5, review not performed);
  - fairness (2 identities per stratum);
  - stability of K and the thresholds (CV fold variance).
- **The gate depends on YuNet for detection.** YuNet's production licence is `REVIEW_REQUIRED` (WIDER FACE training data, R4; G-4).
- **No production model is selected.**
