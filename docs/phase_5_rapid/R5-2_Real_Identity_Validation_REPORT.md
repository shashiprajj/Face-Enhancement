# R5-2 — Real Identity-Labelled Validation (research proxy)

| Field | Value |
|---|---|
| Phase / task | 5 Rapid Validation / R5-2 |
| Date | 2026-10-05 |
| Frozen state | commit `76577d5` · gate `p5-rwgate-1.0.0` (seal `78837a8d`, unchanged) · A4 SwinIR-L (weights `99adfa91…`) · R5-1 service |
| Identity metric | **SFace (E01) + dlib ResNet (E07) — research PROXY. This is NOT Feature-Match** (not available in this environment) |
| Data | (1) **LFW** research proxy: 80 identities, 880 queries · (2) the user's **real paired photos**: 1 identity, 5 degraded |
| Evidence | `data/manifests/phase5_rapid/lfw/` (gate, runs, eval rows, summary, tables) · `data/manifests/phase5_rapid/realperson/results.json` |
| Scripts | `scripts/r5_lfw_population.py`, `r5_lfw_gate.py`, `r5_lfw_evaluate.py`, `r5_realperson_eval.py` |

> **Not production evidence.** LFW is licence REVIEW_REQUIRED (no grant; news photos) and is used locally for research only. The real-person set is one identity. No gate, threshold, A4, T6/T7 or Feature-Match setting was changed.

## Answers to the decision questions

| Q | Answer (proxy evidence) |
|---|---|
| **Q1 Does A4 improve identity on degraded images at all?** | **Yes, but only for strong Gaussian blur and strong noise, and mainly in genuine similarity, not in match decisions.** Strong blur: SFace +0.032 ▲ / dlib +0.006 ▲ (both evaluators). Strong noise: SFace ▲, dlib ≈ (no change). Everywhere else forced A4 lowers identity. |
| **Q2 Does A4 improve the images the gate admits (G1, 144)?** | **Partly.** SFace +0.033 ▲, dlib +0.002 ≈. Matches: +2 / +2 (1 flip and 3 rescues per evaluator). Rank-1 unchanged. Pair verdicts are split: 47 helped vs 50 harmed. The cost is **more impostor-like scores: 36 new SFace false-accept crossings** (5 dlib). Blur cases are clearly positive (14 vs 3); noise cases are mixed (31 vs 37). |
| **Q3 Does A4 improve the "degraded, no A4 evidence" group (G2, 656)?** | **No, overall.** Forced A4: −0.0078 ▼ / −0.0012 ▼, 71 helped vs **378 harmed**, 10 dlib flips, 104 / 23 new false accepts. The gate is right to withhold A4 here, **with one exception: strong Gaussian blur just above the frozen blur threshold** (Q4). |
| **Q4 Is the current gate too conservative?** | **Only for strong Gaussian blur.** 52 strongly blurred queries were left original (`lapd1` 0.0026–0.009). Forced A4 helped them: +0.029 ▲ / +0.006 ▲, 24 helped vs 12 harmed. Every other G2 region is harmed by A4. Overall the gate adds value: gated +0.0059 ▲ / +0.0004 ▲ vs forced ≈ / ≈, with 428 harms avoided. |
| **Q5 Where should A4 be admitted?** | **Strong Gaussian/defocus blur** (positive under both evaluators in G1 and G2), down to about 35 px IOD. **Strong noise** only with caution: SFace gain, dlib neutral, higher false-accept tendency. |
| **Q6 Where should A4 stay blocked?** | Low resolution, mild grain, JPEG, the vintage mix (blur + grain + JPEG), moderate blur, motion blur, clean faces, and photometric (dark/low-contrast) faces: all ▼ or harm-dominated. Also unsafe inputs (extreme pose, low detector confidence, no face). |

## 1. Dataset / source

- **LFW**:
  - `datasets/raw/lfw/lfw.tgz`, SHA-256 `055f7d9c…`. This is the figshare file that scikit-learn downloads and pins; the official UMass site is unreachable.
  - 13,233 images / 5,749 identities, which are the canonical counts.
  - Phase 1 status DS09 **REVIEW_REQUIRED**. Used locally as a research proxy on the user's instruction; never for training.
- **Real-person set**:
  - `Dataset-and-code/Real-Person-Test-Data`: 1 clear reference (phone mirror selfie) and 5 degraded phone photos of the same person.
  - Personal data. **No images, crops or file names are committed**; only SHA-256 hashes, IDs D1–D5 and scores.

## 2. Number of identities

| | Identities | Queries | Gallery |
|---|---|---|---|
| LFW proxy | 80 selected (≥ 2 photos, YuNet score ≥ 0.85 on both) | 880 = 80 × 11 cells | 480 references (80 + 400 impostor-pool identities) |
| Real person | 1 | 5 | the same 480 LFW references as impostors + the person's reference |

The reference and the query are **different photos** of the same person (cross-photo matching). They are not the same picture degraded.

## 3. Degradation protocol (LFW; applied to the whole 250 × 250 query photo, seed = sha256(name | cell))

The cells are clean · grain-mild σ5 · noise-strong σ15 · noise-severe σ25 · blur-moderate σ1 · blur-strong σ2 · blur-strong+grain (σ2 then grain σ4) · motion (length 9) · lowres ×0.5 · JPEG q15 · vintage-mix (blur σ1 → grain σ6 → JPEG q40).

LFW faces are about 45 px IOD (median 44.8), so blur σ2 here is about σ4 relative to the face on 224-px FairFace chips. The real-person photos carry real degradations: low light, greyscale, small face, extreme pose.

**Conditions**
- **A original:** the service's original in the 224 E-frame.
- **B gated:** the frozen gate through the service.
- **C forced A4:** A4 on the same chip regardless of the gate. Not run on unsafe (G3) inputs.

**Scoring**
- One landmark set per query (YuNet on A, else on C) is used for every condition.
- Thresholds are FAR 1e-2 from 115k clean reference impostor pairs: SFace 0.288, dlib 0.908.
- **Service check:** 5 G1 queries re-run through the real R5-1 service with real isolated A4 give pixels **identical** to C. G1 output-guard failures: 0. A4 failures: 0 / 800.

## 4. Original vs gated vs forced-A4 (LFW, G1 + G2 = 800 queries)

| Condition | ΔID SFace | ΔID dlib | helped / harmed | match SFace (A→X) | match dlib (A→X) | flips S/d | rescues S/d | rank-1 SFace | new false accepts S/d |
|---|---|---|---|---|---|---|---|---|---|
| B gated (144 enhanced) | **+0.0059 ▲** | **+0.0004 ▲** | 47 / 50 | 766 → 768 | 764 → 766 | 1 / 1 | 3 / 3 | 766 → 766 | 36 / 5 |
| C forced A4 (800 enhanced) | −0.0005 ≈ | −0.0006 ≈ | 118 / **428** | 766 → 768 | 764 → **760** | 2 / **11** | 4 / 7 | 766 → **762** | **140 / 28** |

**Ceiling effect.** At FAR 1e-2, 96 % of degraded LFW queries already match their reference in the original condition. So verification flips and rescues are rare in either direction; the signal is in similarity and verdict counts. A harder threshold, or Feature-Match's own operating point, may change this.

## 5. Group 1 — gate → A4 (144)

| Cell | n | ΔID SFace | ΔID dlib | helped / harmed | rescues / flips (SFace) | new FA S/d |
|---|---|---|---|---|---|---|
| blur-strong | 23 | +0.038 ▲ | +0.0074 ▲ | **14 / 3** | 0 / 0 | 5 / 4 |
| blur-strong+grain | 14 | +0.016 ▲ | −0.0039 ≈ | 2 / 7 | 0 / 0 | 4 / 1 |
| noise-strong | 53 | +0.022 ▲ | +0.0002 ≈ | 11 / 21 | 1 / 0 | 9 / 0 |
| noise-severe | 50 | +0.051 ▲ | +0.0034 ≈ | 20 / 16 | 2 / 1 | 16 / 0 |
| motion | 4 | −0.025 ≈ | +0.0062 ▲ | 0 / 3 | 0 / 0 | 2 / 0 |
| **all** | **144** | **+0.033 ▲** | **+0.0022 ≈** | **47 / 50** | 3 / 1 | **36 / 5** |

## 6. Group 2 — degraded, left original (656); forced A4 shows what was withheld

| Cell (or gate reason) | n | ΔID SFace | ΔID dlib | helped / harmed |
|---|---|---|---|---|
| **blur-strong** (missed) | 52 | **+0.029 ▲** | **+0.0057 ▲** | **24 / 12** |
| blur-strong+grain | 58 | +0.011 ▲ | +0.0007 ≈ | 12 / 26 |
| noise-strong | 19 | +0.030 ▲ | −0.0019 ≈ | 4 / 8 |
| blur-moderate | 75 | −0.015 ▼ | −0.0004 ≈ | 2 / 43 |
| motion | 72 | −0.010 ▼ | −0.0010 ≈ | 9 / 42 |
| grain-mild | 73 | −0.009 ▼ | −0.0018 ▼ | 3 / 39 |
| JPEG q15 | 77 | −0.009 ▼ | −0.0027 ▼ | 12 / 49 |
| lowres ×0.5 | 78 | −0.027 ▼ | −0.0021 ▼ | 2 / 56 |
| vintage-mix | 76 | −0.017 ▼ | −0.0033 ▼ | 3 / 53 |
| clean | 75 | −0.018 ▼ | −0.0026 ▼ | 0 / 49 |
| *by reason:* `lowres` / `noise+lowres` / `blur+lowres` / photometric | 279 / 182 / 112 / 16 | ▼ / ▼ / ≈ / ≈ | ▼ / ▼ / ≈ / ≈ | 22/163 · 9/111 · 25/60 · 2/10 |
| **all G2** | **656** | **−0.0078 ▼** | **−0.0012 ▼** | **71 / 378** |

## 7. Group 3 — unsafe / unrecognizable (80)

- **Reasons:** pose unreliable 54, low detector confidence 17, no face 9. These are concentrated in noise-severe (29).
- **A4 was never run** on them, forced or not.
- The original still matches for 71/80 (SFace) and 70/80 (dlib). Returning the original costs nothing for these, so it is safe.
- No unsafe input was enhanced.

## 8. Benefit / harm counts (multi-evaluator verdicts vs original)

| | helped (both evaluators) | harmed (any evaluator) |
|---|---|---|
| G1 gated | 47 | 50 |
| G2 if A4 had been forced | 71 | 378 |
| Gate's net effect | admits 47/50; avoids 71/378 | — |

## 9. Identity flips

| | match → no-match (SFace / dlib) | no-match → match | rank-1 lost (SFace) |
|---|---|---|---|
| B gated | 1 / 1 | 3 / 3 | 0 net (132 → 132 in G1) |
| C forced A4 | 2 / 11 | 4 / 7 | 4 |

## 10. Gate misses

- **Strong Gaussian blur above `blur_max`:** 52 G2 queries with `lapd1` 0.0026–0.009, where A4 helps on average in every `lapd1` band (SFace +0.022 to +0.050, dlib +0.005 to +0.007).
- **But `lapd1` overlaps with blur-moderate** (harm 43 vs 2). The current feature cannot separate *strong blur on a soft photo* from *moderate blur on a sharp photo*.
- **Motion blur:** 4 queries are admitted, where A4 is neutral to harmful. This repeats the R5-3 limitation.
- No unsafe input was admitted.

## 11. Degradation-specific findings

| Degradation | Forced A4 effect | Gate today |
|---|---|---|
| Strong Gaussian blur | **positive under both evaluators** | partly admitted (23 of 75) |
| Strong / severe noise | SFace positive, dlib neutral, more impostor-like scores | admitted (103 of 123 safe) |
| Blur + grain | SFace positive, dlib neutral, harm > help | partly admitted |
| Moderate blur, motion blur | negative | left original ✔ |
| Mild grain, JPEG, low-res, vintage mix | negative under both | left original ✔ |
| Clean | negative (A4 alters good faces) | left original ✔ |

**Real-person set (1 identity, SFace/dlib proxy).**
- The gate routed **none** of the 5 photos to A4: D1 dark → photometric; D3 greyscale and D4 small face → acceptable; D2 low confidence and D5 no face → unsafe.
- Forced A4: SFace +0.029 (D1), +0.063 (D3), −0.015 (D4), with dlib within ±0.006.
- **No match decision changed.** D3 lost dlib rank-1 to an impostor.
- D2 (very low light) fails SFace in every condition. That is a photometric/low-light problem A4 does not address.
- With n = 4 scorable photos this is anecdotal. It shows the user's real problem set (low light, pose, small face) lies largely **outside** what A4 + gate can fix.

## 12. Human-recognition observations

- **Not performed by a human. The model's visual impression is not used as human evidence.**
- Review material is prepared in `experiments/phase5_rapid/outputs/lfw_review/`:
  - `review_G1.jpg`, `review_G2.jpg`, `review_G3.jpg`: query | A4 | reference, 24 / 24 / 12 samples;
  - `human_review_template.csv.gz`, with blank columns: recognizable, same person as reference (query / A4), fabrication visible, reviewer.
- The R5-3 real-photo sheets remain as before.
- Model note, for context only: A4 on strong noise produces smooth, "waxy" skin. That is consistent with higher genuine *and* impostor similarity.

## 13. Limitations

1. **Proxy evaluators, not Feature-Match.** SFace and dlib disagree on noise, and Feature-Match may behave differently.
2. **LFW** has news photos at about 45 px IOD and is already JPEG-compressed. The degradations are synthetic, applied to real photos. The licence is unresolved.
3. **Ceiling effect.** 96 % of queries match even when degraded, so match-level effects are small.
4. **The real-person set is 1 identity / 5 photos.** It is anecdotal.
5. **Only one A4 setting was tested.** No alternative model or post-processing was considered (out of scope).
6. **Human review is pending.**

## 14. Was actual Feature-Match available?

**No.** No Feature-Match service, API or code is accessible in this environment, and the Phase 0 boundary forbids touching it. Every identity number here comes from SFace/dlib and must not be read as Feature-Match results.

## 15. Recommendation for the next task

1. **Do not widen A4 broadly.**
   - The evidence supports A4 only for strong Gaussian blur. Gains on strong noise are proxy-dependent and raise false-accept tendency.
   - Most real degradations (low-res, grain, JPEG, motion, low light) are harmed or untouched by A4.
2. **The decisive next step is Feature-Match evaluation** on (a) the user's real paired photos, expanded to at least about 20 identities, and (b) the LFW G1/G2-blur subsets. Feature-Match's own threshold must be used. Measure flips, false accepts and rank-1 at its operating point. Request evaluation-only access.
3. **If a targeted gate change is wanted afterwards (a separate task):**
   - consider admitting strong Gaussian blur slightly above `blur_max` only with a feature that separates blur strength from the photo's native softness, validated on held-out data;
   - consider *tightening* noise admission if Feature-Match confirms the false-accept tendency.
4. **Model direction.** The user's real failure modes (low light, pose, small faces) are not addressed by A4. If enhancement value is required there, the next model search should target **identity-preserving low-light and small-face restoration**, evaluated with Feature-Match from the start. Generative face priors (GFPGAN/CodeFormer) stay excluded (R5-X).
5. **Human review** of the prepared sheets to complete step 7.

## Reproduce

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_lfw_population.py --ids 80
PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_lfw_gate.py
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/p3_run_inference.py --config configs/phase5/benchmark_p5.yaml \
  --config-id B0 --population lfwq --root data/manifests/phase5_rapid/lfw --workers 2 --threads 1      # then A4, 7×2
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_lfw_evaluate.py --verify-service 5
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_realperson_eval.py
```
