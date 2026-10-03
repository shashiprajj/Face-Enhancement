# P4-T4 — Benchmark (Real-World Arm Blocked; Multi-Evaluator Synthetic Arm Complete)

| Field | Value |
|---|---|
| Phase / Task | Phase 4 / P4-T4 |
| Date | 2026-10-03 |
| Real-world arm | **BLOCKED (G-3).** No frozen `rw-v1` manifest exists (P4-T1), and nothing real-world was evaluated. |
| Synthetic arm | **Complete.** Phase 3 `main` (test, 28 refs × 32 cells = 896 pairs, 9 configurations) re-analysed with **two validated evaluators** (E01 SFace + E07 dlib), with B0 as the mandatory baseline |
| Harness | Phase 3 unchanged: same outputs, adapters, weights, geometry and protocol `p3-protocol-1.0.1`. Every output was re-verified by SHA-256 at scoring. |
| Scripts | `scripts/p4_evaluate_e07.py`, `scripts/p4_benchmark_analysis.py`, `src/fes_bench/p4eval.py` |
| Evidence | `data/manifests/phase4/metrics_e07/main/` · `data/manifests/phase4/results/main/{summary.json, verdicts.csv.gz, tables.md}` · `data/manifests/phase4/repro/p4t4_checks.json` |

> **Every number below is SYNTHETIC** (fes-syn-v1 degradations of FairFace chips). None of it is real-world evidence.

---

## 1. Method

- **ΔID per evaluator** = cos(config output, reference) − cos(B0, reference), paired per pair, with a 95 % cluster-bootstrap CI over references (B = 2000).
- **The two evaluators' scores are never pooled** (P4-T3 §4).
- **Per-pair verdict** per evaluator: BENEFIT if Δ > δ_e, HARM if Δ < −δ_e (δ: E01 0.0215, E07 0.0026; dev noise floors).
  - **Multi-evaluator:** CONFIRMED_BENEFIT / CONFIRMED_HARM when both evaluators agree, PARTIAL when only one moves, DISAGREE when they point in opposite directions.
- **TAR** at each evaluator's FAR 1e-2 point from 700 clean test references: E01 0.3133, E07 0.9336.
- **Flips** are B0 ≥ thr while the output falls below it. **Rescues** are the reverse.
- **Closer-to-impostor:** the output is more similar to another identity than to its own reference.
- **Detection:** the non-oracle FES detection contract on the degraded input (P4-T2), plus YuNet on the output.
- **Perceptual metrics** (PSNR/SSIM/LPIPS) are unchanged from Phase 3 (P3-T5) and are valid only because a reference exists. **They are not identity evidence.**

## 2. Pooled Results (vs B0)

| Config | ΔE01 [95 % CI] | ΔE07 [95 % CI] | Conf. benefit / harm / disagree (of 896) | TAR E01 B0→c | TAR E07 B0→c | Flips/rescues E01 · E07 | Closer-to-impostor E01 · E07 |
|---|---|---|---|---|---|---|---|
| B1 Lanczos | −0.0006 ▼ | −0.0001 ≈ | 5 / 6 / 4 | .959→.954 | .911→.911 | 5/1 · 1/1 | 10 · 42 |
| B2 classical | −0.0360 ▼ | −0.0038 ▼ | 22 / 427 / 12 | .959→.950 | .911→.906 | 13/5 · 9/5 | 13 · 42 |
| B2c +CLAHE | −0.1084 ▼ | −0.0146 ▼ | 3 / 835 / 14 | .959→.921 | .911→.875 | 36/2 · 33/1 | 28 · 68 |
| A1 RestoreFormer++ | −0.0492 ▼ | −0.0054 ▼ | 56 / 540 / 76 | .959→.963 | .911→.920 | 12/16 · 9/17 | 16 · 45 |
| A2 VQFR | −0.0608 ▼ | −0.0065 ▼ | 25 / 584 / **106** | .959→.955 | .911→.902 | 13/10 · 18/10 | 16 · 41 |
| A3 Real-ESRGAN | −0.0205 ▼ | −0.0040 ▼ | 58 / 301 / 49 | .959→.933 | .911→.889 | 38/15 · 27/8 | 34 · 56 |
| A4 SwinIR-L | −0.0078 ▼ | −0.0013 ▼ | **120** / 260 / 34 | .959→.933 | .911→.903 | 38/15 · 18/11 | 25 · 56 |
| A5 chain | −0.0802 ▼ | −0.0110 ▼ | 45 / 481 / 50 | .959→**.891** | .911→**.845** | **72/11 · 65/6** | **68 · 93** |

Full CIs: `results/main/tables.md`.

## 3. Where Enhancement Helps or Harms (confirmed benefit − confirmed harm; E01/E07 marks)

| Config | noise | extreme severity | mild severity | photometric (under-exp. / low-contrast) | lowres |
|---|---|---|---|---|---|
| A1 | 25−37, ▲/≈ | 33−24, ▲/≈ | 0−226, ▼/▼ | 0−83 / 0−74, ▼/▼ | 9−58, ▼/≈ |
| A2 | 5−37, ▲/▼ | 14−35, ≈/≈ | 0−234, ▼/▼ | ▼/▼ | 7−70, ▼/▼ |
| A3 | **46−3, ▲/▲** | 26−35, ▲/▼ | 0−83, ▼/▼ | ▼/▼ | 0−96, ▼/▼ |
| A4 | **63−5, ▲/▲** | **69−29, ▲/▲** | 0−76, ▼/▼ | ▼/▼ | 0−76, ▼/▼ |
| A5 | 25−10, ▲/≈ | 24−55, ▼/▼ | 0−148, ▼/▼ | 0−64 / 0−67, ▼/▼ | 0−107, ▼/▼ |

## 4. Findings

1. **The second evaluator confirms Phase 3's central finding.** Under both SFace and dlib, **no candidate preserves identity better than the untouched input on average.** All five pooled ΔID CIs are below 0 for both evaluators.
2. **Noise is the only family with a confirmed identity benefit**: A3 (46 confirmed benefits vs 3 harms) and A4 (63 vs 5), ▲ under both evaluators. A1's noise benefit is single-evaluator only (▲/≈).
3. **Phase 3's single-evaluator "extreme ▲" is only partly confirmed.** A4 at extreme severity is ▲/▲. A1 is ▲/≈ and A3 is ▲/▼, so E07 does **not** confirm them. This is exactly the kind of claim G-2 existed to test.
4. **Mild inputs: zero confirmed benefits for any candidate**, with 76–234 confirmed harms each. The Do-Not-Enhance default is supported by both evaluators.
5. **Photometric degradations** (underexposure, low contrast): every candidate is ▼/▼ with 0 confirmed benefits. No candidate should be routed there (P4-T7).
6. **Identity substitution signal (F1).** A1 and A2 have the most evaluator **disagreements** (76 and 106), i.e. pairs where one evaluator sees a gain and the other a loss. That pattern fits a sharp, plausible face whose identity evaluators read differently. A5 has the most outputs closer to another identity (68 E01 / 93 E07; B0 ≈ 10 / 42). **Human review (G-5) is needed to tell substitution apart from evaluator noise.**
7. **Evaluator agreement is moderate:**
   - Cohen's κ on 3-class verdicts ranges 0.25 (A2) – 0.54 (A4).
   - Spearman of Δ ranges 0.54 – 0.73.
   - Agreement is lowest for the generative face models. **Single-evaluator identity claims on these models are not reliable on their own.**
8. **The detection contract fails exactly where enhancement is most dangerous.** On degraded inputs, detection succeeds 0 % on lowres-extreme and noise-extreme, 14–25 % on combined/realistic2/lowres-severe, and 82 % on noise-severe.
   - Where detection fails, B0 TAR drops to 0.76 (E01) / 0.50 (E07), compared with 0.99 / 0.98 where it succeeds.
   - Phase 3's oracle alignment hid this. A production pipeline would REJECT/ESCALATE these inputs (P4-T6), which removes the extreme-lowres F1 cases from the enhancement path.
9. **Output detection:** YuNet finds a face on 93 % (A5) to 99.6 % (A1) of outputs. That is not evidence of identity, since A1/A2 always "produce a face".

## 5. P4-T4 Measure Checklist

| Measure | Status |
|---|---|
| Identity similarity | ✅ two evaluators, per pair |
| TAR/FAR | ✅ at each evaluator's FAR 1e-2 point from test-split clean impostors (FairFace has no identity labels, so cross-photo genuine pairs are impossible: G-3) |
| Identity flips / rescues | ✅ per evaluator |
| Evaluator agreement | ✅ κ, Spearman, verdict cross-tabs |
| Detection / alignment failures | ✅ non-oracle input contract + output detection |
| Family / severity results | ✅ |
| Perceptual metrics with valid references only | ✅ Phase 3 P3-T5 (unchanged; not identity evidence) |
| Latency / memory | ↪ Phase 3 P3-T8 (CPU; unchanged outputs). New: E07 ≈ 150 ms/embedding; gate 2–12 ms p50 per input (`perf/gate_latency.json`). GPU: G-1. |
| Clustered CIs | ✅ cluster bootstrap over references |
| Reproducibility | ✅ E07 re-score of A4 byte-identical (`6a717270…`). Analysis re-run gives byte-identical verdicts (`946a7244…`). |
| Frozen manifests / hashes | ✅ population `840ee52d…`, E07 weights `55533b28…`, run-manifest output SHA-256 verified per pair |

## 6. Real-World Arm — Ready to Run When G-3 Opens

1. Freeze `rw-v1` (P4-T1 §5). Compute P4-T2 features. Re-calibrate gates on the rw **dev** split.
2. **Adapter gap (must be closed first):** the Phase 3 adapters take the recorded FairFace→FFHQ transform (`M_e2f`). Real captures need it from the detection contract (`preprocess.canonical_transform`), with an E-frame defined per capture. This is a harness extension that needs a protocol version bump. It is not done, because without data it cannot be validated.
3. Identity: cross-photo genuine pairs (LQ capture vs HQ_REF of another session) and true TAR/FAR with participant identity labels, scored by E01 + E07.
4. Run B0 + candidates on the rw test split **once**.

## 7. Limits

- Synthetic only. 28 identities, so per-stratum results are descriptive.
- Both evaluators are imperfect and agree only moderately on generative outputs.
- Oracle alignment still drives the candidate inputs (P3-T1 §12), so identity results for face models are an upper bound.
