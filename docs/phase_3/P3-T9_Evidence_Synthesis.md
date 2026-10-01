# P3-T9 — Cross-Candidate Evidence Synthesis

| Field | Value |
|---|---|
| Phase / Task | Phase 3 / P3-T9 |
| Status | Complete — **evidence organised for a later decision gate; no model selected, ranked, or recommended** |
| Date | 2026-10-01 |
| Sources | P3-T3 (baselines), P3-T5 (quality), P3-T6 (identity, SFace only), P3-T7 (failures), P3-T8 (CPU), P1-T3/P1-T9 (licensing) |
| Data | `results/main/{cells,deltas,identity,strata}.csv.gz`, `failure_stats.csv`, `data/manifests/phase3/perf/cpu_benchmark_t2.json` |

**Rules followed:**
- No composite score (P0-T4 §2).
- Dimensions are shown side by side, never summed.
- Licensing is kept in its own column and never weighed against performance.
- All results are synthetic (`fes-syn-v1`, 28 test identities) and measured by one identity evaluator.

---

## 1. Side-by-Side Evidence (pooled over all 896 pairs)

| | A1 RestoreFormer++ | A2 VQFR v1 | A3 Real-ESRGAN | A4 SwinIR-L | A5 FBCNN→NAFNet→R-ESRGAN | B2 classical |
|---|---|---|---|---|---|---|
| **Quality** ΔLPIPS vs B0 | −0.048 ▲ | −0.033 ▲ | −0.062 ▲ | −0.078 ▲ | −0.003 ≈ | +0.030 ▼ |
| ΔPSNR vs B0 (dB) | −4.85 ▼ | −4.16 ▼ | −1.77 ▼ | −1.71 ▼ | −3.16 ▼ | −1.29 ▼ |
| ΔLPIPS mild / extreme | +0.104 ▼ / −0.278 ▲ | +0.094 ▼ / −0.200 ▲ | +0.056 ▼ / −0.229 ▲ | +0.055 ▼ / −0.276 ▲ | +0.105 ▼ / −0.165 ▲ | +0.046 ▼ / +0.010 ▼ |
| **Identity** (SFace) ΔID vs B0 | −0.049 ▼ | −0.061 ▼ | −0.021 ▼ | −0.008 ▼ | −0.080 ▼ | −0.036 ▼ |
| TAR @ FAR 1e-2 (B0 0.959) | 0.963 | 0.955 | 0.933 | 0.933 | 0.891 | 0.950 |
| Flips / rescues | 12 / 16 | 13 / 10 | 38 / 15 | 38 / 15 | 72 / 11 | 13 / 5 |
| Closer to another person | 16 | 16 | 34 | 25 | 68 | 13 |
| Mean impostor cos (B0 0.034) | 0.046 | 0.044 | 0.040 | 0.040 | 0.038 | 0.036 |
| **Robustness** detection lost / landmark shift | 0 / 38 | 0 / 45 | 16 / 34 | 17 / 17 | 49 / 83 | 11 / 36 |
| Main failure modes (P3-T7) | F1 substitution, F2 attribute removal, F7 colour cast, F10 | F1, F2, F5 mesh texture | F3 distortion at ≤ 28 px, F6 | F3 at 14 px, F6, F2 | F4 collapse (dark, 14 px), F3 | F6 |
| **CPU** (2 threads, warm, per image; P3-T8) | 5.5 s (224 and 56) | 20.1 s | 12.6 s @224 · 0.76 s @56 | 49.4 s @224 · 2.3 s @56 | 12.7 s @224 · 0.80 s @56 | 0.05 s |
| Peak RSS | 1.3 GB | 3.3 GB | 1.2 GB | 1.3 GB | 2.1 GB | 0.25 GB |
| **Licence** (separate; P1-T3) | eval ELIGIBLE · production **REVIEW_REQUIRED** | same | same | same | same (3 assets) | n/a |

## 2. Where Each Candidate Helps or Hurts (vs B0; Q = LPIPS, I = SFace identity)

| Family | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| lowres | Q≈ I▼ | Q≈ I▼ | Q≈ I▼▼ | Q▲ I▼ | Q▼ I▼▼▼ |
| gaussblur | Q≈ I▼ | Q≈ I▼ | Q▼ I▼ | Q≈ I≈ | Q▼ I▼ |
| motionblur | Q≈ I▼ | Q≈ I▼ | Q▼ I▼ | Q≈ I▼ | Q▼ I▼ |
| **noise** | **Q▲ I▲** | **Q▲ I▲** | **Q▲ I▲** | **Q▲ I▲** | **Q▲ I▲** |
| jpeg | Q▼ I▼ | Q▼ I▼ | Q▼ I▼ | Q▼ I≈ | Q▼ I▼ |
| underexposure | Q▼ I▼▼ | Q▼ I▼ | Q▼ I▼ | Q▼ I▼ | Q▼ I▼▼ |
| lowcontrast | Q▼ I▼ | Q▼ I▼▼ | Q▼ I▼ | Q▼ I▼ | Q▼ I▼▼ |
| combined | Q▲ I▼ | Q▲ I▼ | Q▲ I▼ | Q▲ I▼ | Q▲ I▼ |
| realistic2 | Q▲ I▼ | Q▲ I▼ | Q▲ I▼ | Q▲ I▼ | Q▲ I▼ |

| Severity | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| mild | Q▼ I▼ | Q▼ I▼ | Q▼ I▼ | Q▼ I▼ | Q▼ I▼ |
| moderate | Q≈ I▼ | Q≈ I▼ | Q▲ I▼ | Q▲ I▼ | Q≈ I▼ |
| severe | Q▲ I▼ | Q▲ I▼ | Q▲ I▼ | Q▲ I▼ | Q▲ I▼▼ |
| extreme | Q▲ I▲ | Q▲ I≈ | Q▲ I▲ | Q▲ I▲ | Q▲ I▼ |

(▼▼ marks ΔID < −0.1 and ▼▼▼ marks < −0.2. Full numbers and CIs are in P3-T5 §2–3 and P3-T6 §2–3.)

## 3. Trade-offs (descriptive; no winner)

1. **Perceptual quality vs identity.** These are the central tension. In the combined/realistic2 families and on severe inputs, every candidate improves LPIPS **and** lowers SFace identity similarity. A single quality metric would point the wrong way for identity-sensitive use.
2. **Face-specific vs general SR.**
   - The face models (A1, A2) keep detection (0 lost), hold TAR at B0 level, and stay at B0 identity at 14 px. But they drift identity on most pairs, raise similarity between different people, and confidently synthesise a *different* plausible person with altered attributes (F1/F2) at extreme inputs.
   - The general SR models (A3, A4) are closest to identity-neutral on mild to severe inputs and best on noise. But they produce visibly distorted faces at ≤ 28 px (F3) and lose detection on some pairs.
3. **Do-nothing is a strong option.** For mild inputs, which is most of any realistic traffic, B0 beats every candidate on both dimensions. Any system design needs a way to **not** enhance. Whether that is a quality gate or a fidelity control is an open design question.
4. **No candidate handles photometric degradations** (under-exposure, low contrast). All are ▼ on both dimensions.
5. **Chain A5** scores below its own last stage run alone (A3) on mean LPIPS **and** mean identity in **every** family. It adds the largest failure counts (F4) and the most impostor-closer outputs. On this evidence, the blind always-on chain is the weakest configuration. That is a statement about this configuration, not about its stages.
6. **Compute.** CPU cost differs by about 9× between candidates at 224 px (A1 5.5 s to A4 49 s; P3-T8). General-SR cost scales with input area; face-model cost is constant (always 512×512).

## 4. Candidates / Evidence Needing Further Testing

| Item | Why | Blocked by |
|---|---|---|
| A1, A2 identity substitution at extreme inputs (F1/F2) | The most consequential failure for identity-sensitive use. SFace alone cannot show whether humans or a second model agree. | G-2 (second evaluator), human review protocol |
| A3, A4 at ≤ 28 px (F3) | Distortion frequency at real small-face sizes is unknown | G-3 (real-world data) |
| A5 on dark images (F4) | Attributed to the NAFNet-SIDD stage (P3-T7 §4). Robustness to exposure needs real low-light data. | G-3 |
| Mild-input regression (all) | Needs a decision rule (when not to enhance) and thresholds | TH-1…TH-11 TBD |
| Fidelity-controllable models (e.g. a fidelity weight) | Not in the Phase 1 shortlist. P3 shows a fidelity knob would address the main trade-off, but no model was added in Phase 3 (scope rule). | Phase 1 shortlist revision, licensing |
| Demographic behaviour | 2 identities per stratum → no conclusion possible | G-1 (full test_core on GPU) |
| GPU latency / throughput | CPU only | G-1 |

## 5. Evidence-Gap List for Later Phases

| ID | Gap | Effect on Phase 3 evidence |
|---|---|---|
| G-1 | No GPU | Main = 28 references (CPU budget). Wide per-cell CIs. No GPU performance. |
| G-2 | One identity evaluator (SFace) | All identity findings are single-evaluator |
| G-3 | No real-world low-quality data | All findings are synthetic-only (Phase 0 makes real-world mandatory) |
| G-4 | No licence reviewer | Production eligibility of every candidate is unresolved (REVIEW_REQUIRED) |
| G-5 | No high-res GT (224-px FairFace) | Face models are scored at 224 px after 512-px synthesis. Soft references penalise sharpening (F10). |
| TH-1…TH-11 | Thresholds TBD | Results are descriptive (▲ / ▼ / ≈ only) |
| — | Thread-count determinism | A2 main ran at two thread counts (P3-T4 §5). Quantified in P3-T10. |

## 6. Done-When Check

Image quality, identity, robustness and CPU evidence are combined side by side, by family and severity. Trade-offs are described without a composite score, licensing is kept separate, the real-world limitation is explicit, no best or worst model is declared, no production recommendation is made, and the further-testing and evidence-gap lists are stated. ✅
