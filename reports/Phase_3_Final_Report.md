# Phase 3 Final Report — Face Enhancement Service

| Field | Value |
|---|---|
| Phase | 3 — Benchmarking & Baseline Evaluation |
| Date | 2026-10-01 |
| Benchmark | `fes-syn-v1` (synthetic), protocol `p3-protocol-1.0.1` (frozen 2026-09-29 before any output) |
| Population | `main`: 28 FairFace test identities (2 per race × gender stratum) × 32 degradation cells = 896 pairs per configuration |
| Configurations | B0, B1, B2, B2c (baselines) · A1 RestoreFormer++ · A2 VQFR v1 · A3 Real-ESRGAN · A4 SwinIR-L · A5 FBCNN → NAFNet-SIDD → Real-ESRGAN |
| Index | [docs/phase_3/PHASE_3_INDEX.md](../docs/phase_3/PHASE_3_INDEX.md) |
| **Phase 3 gate** | **🟡 YELLOW — reproducible synthetic benchmark record complete for all 5 candidates; no candidate is production-eligible, and the evidence required for selection is blocked by G-1…G-4** |

---

## 1. Executive Summary

Every shortlisted candidate ran on the **same** 896 controlled pairs with **zero failed or missing outputs**. Each output was scored for image quality, identity (SFace), geometry and failure modes, and CPU performance was measured separately under one method. Everything is reproducible from tracked manifests, and the reproducibility re-checks are in §5.

**What the benchmark shows (synthetic data, one identity evaluator):**

1. **No candidate preserves identity better than the unenhanced input on average.** ΔID vs B0 ranges from −0.008 (A4) to −0.080 (A5), all with CIs excluding 0. Identity improves only on **noise** (all five) and at **extreme** severity (A1, A3, A4).
2. **Every candidate makes mildly degraded images worse**, both perceptually and in identity. Leaving the input alone (B0) is the best option on mild inputs.
3. **Perceptual quality and identity diverge.** On mixed and severe degradations, every candidate improves LPIPS while lowering identity similarity. At 14 px, the general SR models reach better LPIPS than the input while identity collapses (cos 0.22–0.27 vs 0.45).
4. **Face-specific models (A1, A2) substitute identity at extreme inputs.** They produce a sharp, plausible *different* person, removing beards, glasses and jewellery. Their TAR stays at B0 level and detection never fails, but similarity between *different* people rises (impostor cos 0.034 → 0.046).
5. **General SR models (A3, A4)** are closest to identity-neutral on mild to severe inputs and best on noise, but distort faces at ≤ 28 px.
6. **The blind always-on chain (A5)** is below its own last stage (A3) in every family. It collapses into colour noise on dark and 14-px inputs and has the most misidentification-risk outputs (68 of 896 closer to another person).
7. **No candidate corrects exposure or contrast.** All are worse than the input on photometric degradations.
8. **CPU cost** at 224 px and 2 threads ranges from 5.5 s (A1) to 20 s (A2) for the face models and from 12.6 s (A3) to 49 s (A4) for general SR. Peak memory reaches 3.3 GB (A2). The CPU numbers are not GPU-representative.

**No model is selected, ranked, or recommended.** These are descriptive synthetic results, and the Phase 0 evidence needed for a selection (real-world data, a second identity evaluator, licence review, GPU performance, human review) is not available.

## 2. Task Status

| Task | Deliverable | Status |
|---|---|---|
| P3-T1 | Harness specification, frozen protocol, schemas, populations | ✅ |
| P3-T2 | Adapters for all 7 networks, validation report | ✅ All runnable, none blocked |
| P3-T3 | Baselines B0–B2c | ✅ 896/896 each |
| P3-T4 | Candidate inference, trial then main | ✅ 5 × 896/896, 0 failed. Incidents I1–I6 were infrastructure only. |
| P3-T5 | Image quality (PSNR / SSIM / LPIPS) | ✅ |
| P3-T6 | Identity (SFace, single evaluator) | ✅ Single-evaluator limitation stated |
| P3-T7 | Failure taxonomy F1–F10, statistics, sample index | ✅ |
| P3-T8 | CPU performance | ✅ CPU only; GPU pending (G-1) |
| P3-T9 | Evidence synthesis (no winner) | ✅ |
| P3-T10 | Validation, report, gate | ✅ This document |

## 3. Results at a Glance (pooled, vs B0, 95 % cluster-bootstrap CI over references)

| | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| ΔLPIPS (↓ better) | −0.048 ▲ | −0.033 ▲ | −0.062 ▲ | −0.078 ▲ | −0.003 ≈ |
| ΔPSNR dB | −4.85 ▼ | −4.16 ▼ | −1.77 ▼ | −1.71 ▼ | −3.16 ▼ |
| ΔID (SFace) | −0.049 ▼ | −0.061 ▼ | −0.021 ▼ | −0.008 ▼ | −0.080 ▼ |
| TAR @ FAR 1e-2 (B0 0.959) | 0.963 | 0.955 | 0.933 | 0.933 | 0.891 |
| Identity flips / rescues | 12 / 16 | 13 / 10 | 38 / 15 | 38 / 15 | 72 / 11 |
| Mild ΔLPIPS / ΔID | ▼ / ▼ | ▼ / ▼ | ▼ / ▼ | ▼ / ▼ | ▼ / ▼ |
| Extreme ΔLPIPS / ΔID | ▲ / ▲ | ▲ / ≈ | ▲ / ▲ | ▲ / ▲ | ▲ / ▼ |
| CPU warm latency @224 (2 thr) / peak RSS | 5.5 s / 1.3 GB | 20.1 s / 3.3 GB | 12.6 s / 1.2 GB | 49.4 s / 1.3 GB | 12.7 s / 2.1 GB |

Full tables: `data/manifests/phase3/results/main/tables.md`. By family and severity: P3-T5 §2–3, P3-T6 §2–3, P3-T9 §2.

## 4. Comparison With Phase 0 Requirements (P0-T4 §9 evidence list; factual, no thresholds invented)

| # | Evidence required for a production candidate | Phase 3 status |
|---|---|---|
| E1 | Licence review permits commercial use | ❌ All candidate weights are `REVIEW_REQUIRED`. No reviewer (G-4). |
| E2 | ID gain ≥ 0 (mean) with CI; drop rate ≤ TBD; no FAR increase; S-SYN **and** S-ID/S-REAL; ≥ 2 evaluators | ❌ On S-SYN **no candidate has mean ID gain ≥ 0**. S-ID/S-REAL is not available (G-3). Only 1 evaluator (G-2). Drop-rate threshold TBD. |
| E3 | No demographic stratum significantly worse | ⚠️ Not assessable: 2 identities per stratum (descriptive only) |
| E4 | Significant improvement over B0 and the best classical baseline on primary-target severities; **not worse than B0 on mild inputs** | ⚠️ LPIPS improves at severe/extreme for A1–A4 (and A5 vs B0). PSNR mostly does not. ❌ **Every candidate is worse than B0 on mild inputs.** |
| E5 | Human review of fabrication / artifacts | ❌ Not performed. F1/F2 (identity substitution, attribute removal) were found by visual inspection only. |
| E6 | Latency / throughput / memory targets on target hardware | ⚠️ CPU measured. Targets TBD (TH-8). No GPU (G-1). |
| E7 | Reliability: failure rate, fuzz/load/soak, determinism | ⚠️ 0 model failures on 4,480 learned-model inferences. Determinism is byte-identical at a fixed thread count (36/36 sampled pairs pixel-identical on re-run; ≤ 1 grey level across thread counts). Load/soak testing belongs to the service phase. |
| E8 | Reproducible evaluation report with configs, seeds, versions | ✅ |

## 5. Validation and Reproducibility (P3-T10 checks)

| Check | Result |
|---|---|
| Output counts and manifests | ✅ 9 configs × 896 pairs, all `OK`. Every output's file SHA-256 was verified at scoring. Trial 9 × 128. |
| Same evaluation population | ✅ All 9 metric files share an identical pair order (asserted by `p3_aggregate.py`). One population manifest (`840ee52d…`). |
| Leakage | ✅ trial ∩ main = ∅. main ⊂ test split and trial ⊂ dev split. All 28 main references are `leak_flag = False` and have no near-duplicate group. Note: the main references are among the 700 impostor-pool references used for the FAR operating point (clean references only, no outputs). |
| Consistent preprocessing | ✅ One geometry module, one mask, and the recorded transform for all configurations (P3-T1 §4). Inputs were re-hashed before inference. |
| Feature-Match | ✅ No reference in `src/`, `scripts/`, `configs/`, `tests/` or `third_party/`. No dependency was introduced. |
| Re-run of a sample of outputs | ✅ 36/36 sampled main pairs (4 per configuration, seeded) re-run from the degraded input are **pixel-identical** at 2 threads. A2 across thread counts: pairs reproduce exactly at their original thread count, and a 2 → 4 thread change alters ≤ 1 grey level on ≤ 0.004 % of pixels (PSNR ≥ 95 dB). (`repro/rerun_t2.json`, `rerun_A2_t4.json`) |
| Metric scoring re-run | ✅ A1 and A4 re-scored: byte-identical metric files (`1f9d62c0…`, `ef12e127…`). Trial re-score was also identical. |
| Aggregation re-run | ✅ cells / deltas / identity / strata byte-identical (seeded bootstrap) |
| Harness unit tests | ✅ `tests/test_bench_harness.py` 23 passed |
| Model-load validation report | ✅ 7/7 networks VALIDATED (weights SHA-256, 0 missing/unexpected keys, finite forward), 12/12 vendored sources SHA ok, 9/9 adapters pass the output contract and repeat identically (`model_validation_report.json`) |

Evidence: `data/manifests/phase3/repro/`, `model_validation_report.json`.

## 6. Limitations (explicit)

1. **No real-world dataset (G-3).** Every result is on synthetic degradations of FairFace chips. Phase 0 makes real-world evaluation mandatory, and synthetic results are not a substitute.
2. **One identity evaluator (G-2).** All identity results are SFace-only. Identity substitution (F1) in particular needs a second evaluator and human review.
3. **No GPU (G-1).**
   - The main population is 28 identities, so CIs are wide: per-cell LPIPS ±0.07 and ΔID ±0.12 at worst.
   - There is no demographic conclusion and no GPU performance.
4. **Production licensing unresolved (G-4).** Every candidate's weights are `REVIEW_REQUIRED`.
5. **224-px ground truth (G-5).** Face models synthesise at 512 px and are scored at 224 px against sometimes-soft references (F10).
6. **Oracle-quality alignment.** All candidates used the recorded reference transform. A production pipeline would detect faces on degraded inputs, which fails at noise-extreme and 14 px (P2-T4), so these results are an upper bound for the alignment-dependent face models.
7. **Thresholds TBD.** ▲/▼/≈ mean only that a 95 % CI excludes or contains 0.

## 7. Phase 3 Gate

### 🟡 YELLOW

**Why not GREEN:**
- The Phase 0 evidence a production decision needs (E1, E2 multi-evaluator and real-world, E3, E5, E6) cannot be produced with the current data, evaluators, hardware and legal status.
- On the evidence that does exist, no candidate meets the Phase 0 identity expectation (mean ID gain ≥ 0) or the "not worse than B0 on mild inputs" expectation.

**Why not RED:**
- The benchmark is complete, controlled and reproducible for all runnable candidates.
- It isolates where candidates help (noise, extreme severity) and where they harm (mild inputs, photometric degradations, ≤ 28 px identity, F1/F2 substitution).
- Nothing required for synthetic benchmarking is missing.

**This gate does not select, rank, approve or reject any model for production.**

## 8. What the Next Phase Can Do — and What Remains Blocked

**Can proceed now:**
- Use the harness, protocol and frozen populations for any further CPU evaluation. It is ready for new candidates or a GPU re-run, which only needs a new population size.
- Design and evaluate a **"do not enhance" decision rule** (input-quality gating), since B0 wins on mild inputs. This needs TH-6/TH-7 calibration on dev data.
- Prepare a **human-review protocol** for fabrication and identity substitution (E5, TH-5) using the P3-T7 sample index.
- Add **fidelity-controllable** or exposure-aware candidates through a Phase 1 shortlist revision, with licence screening first.
- Investigate A5's dark-image collapse, which is attributed to NAFNet-SIDD on very dark inputs (P3-T7 §4), or drop the always-on chain design.

**Blocked:**

| Blocked item | Blocked by |
|---|---|
| Real-world evaluation | G-3 / DS03 legal approval |
| Second identity evaluator | G-2 |
| Fairness conclusions and full test_core | G-1 |
| GPU performance (E6) | G-1 |
| Production eligibility | G-4 licence reviewer |
| **Production model selection** | All of the above (P0-T4 §9) |
