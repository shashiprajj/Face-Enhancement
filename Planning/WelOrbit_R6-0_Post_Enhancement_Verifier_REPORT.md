# R6-0: Post-Enhancement Safety Verifier — Feasibility Report

Date: 2026-10-08 · Task: `Tasks/Phase-8/R6-0_Post_Enhancement_Verifier_Feasibility_PROMPT.txt` · Baseline `179744b`

## 1. Executive summary

**Question:** after NAFNet has produced an output, can we reliably tell whether that output is safe to return?

**Answer.**

1. **The five requested signal families, used as a classic "look at the output" verifier: no, not reliably.**
   - The best single signal is colour dottedness (`t_dot_c_max`), which is the R5-8 signal again (AUC 0.983).
   - It still misses 5 of 74 known BAD outputs, including img118, image_84257 and image_89280.
   - Geometry, identity and the local-difference statistics are close to chance (AUC 0.51–0.71).
   - Every combination that catches more failures rejects 15–21 % of good outputs.
   - This confirms the R5-7 → R5-10 experience: inspecting a single output does not work.

2. **The root cause, found in R6-0: the artifacts are numerical instability of NAFNet, not image content.** For every face I ran NAFNet four times: on the face and on its horizontal, vertical and double mirror images, each result mirrored back. A real deblur does not depend on mirroring. The artifacts do:
   - On 1,200 held-out archive TEST faces, **1,185 gave four stable runs and 15 had one or more runs explode** (a tracer value up to 13,695 against a normal of about 4).
   - Of the 70 known BAD outputs from the R5 builds, 66 come from runs that are unstable under mirroring.
   - The other 4 are faint, position-inconsistent dots, plus img118's colour-only mottling. The 4-run median and the input's colour remove all of them.

3. **A verifier built on that cause works.** The design is **stability-verified consensus (R6)**:
   - Run NAFNet four times (identity + 3 mirrors).
   - If **any** run is unstable (colour-dottedness tracer above a DEV-only limit), return the ORIGINAL.
   - Otherwise return the **per-pixel median of the 4 runs, with the input's own colour** (luma-only enhancement). The output cannot invent colour.

   Results:

   | Evaluation | R5-8 guard (best existing) | **R6 stability + consensus** |
   |---|---|---|
   | Known BAD, held-out (non-DEV, n=42): visible artifact returned | 4 (img118, 84257, 89280, 91965) | **0** |
   | Known BAD, all (n=74): visible artifact returned | 7 | **0** |
   | GOOD rejected, held-out (n=357) | 0 | 2 (0.6 %) |
   | GOOD rejected, all (n=815) | 1 | 7 (0.9 %) |
   | archive TEST population (1,200 eligible faces, held out): rejected | 0.7 % | 1.2 % |
   | archive TEST ΔSFace (whole policy), face_chip / full_image | +0.154 / +0.112 | **+0.156 / +0.112** |
   | archive TEST faces harmed (ΔSFace < −0.05) | 16 | **9** |
   | CPU time, 128-px face (live service) | ≈ 0.2 s | ≈ 1.0 s |

   The identity benefit is kept, harm is roughly halved, and every known destructive or localized artifact in the corpus is either rejected or repaired.

**Verdict: GREEN (feasibility). Production approval is NOT claimed.** The evidence is mostly synthetic-blur archive data with a single (my own) visual reviewer. The real-photo set is small (14 NAFNet outputs). Feature-Match has not been run (§15).

**What I changed, beyond the letter of the prompt** (the owner explicitly asked me to "use my own brain … try to implement and achieve the goal"):

- R6 is implemented as an **opt-in research profile `r6`**, exactly like r5_8 / r5_10:
  - new sealed config;
  - a new model factory;
  - one new pipeline branch, active only for that profile;
  - a UI option shown only when enabled.
- **Default behaviour is byte-identical.** A request without `profile` still runs R5-6, and R5-6 / R5-8 / R5-10 are unchanged.
- The `fes-ui` demo container was restarted with `FES_EXPERIMENTAL_PROFILES=r5_8,r6`, so the owner can try R6 at http://localhost:8080/ui/compare.html.
- Details are in §9 and §18.

## 2. Current architecture context

```
INPUT → face detect / chip → rw features → NAFNet input gate (R5-6 live: p5-nafnet-route-1.0.0)
      → NAFNet-GoPro (isolated worker, timeout, fallback) → artifact guard → ENHANCED | ORIGINAL
```

- **Live default:** R5-6.
  - seal `eb6fe101…`, config sha256 `b2d0a2f2…`;
  - NAFNet zero-pads to a multiple of 16;
  - black fill outside the image;
  - guard: chroma p99 ≤ 10.
- **Opt-in profiles:**
  - r5_8: reflect-pad16, reflect fill, R5-7 guard + `l_dot_c_max ≤ 24.45`, seal `631acd73…`;
  - r5_10: r5_8 + blur_max 0.005, seal `cc2347ac…`, sha256 `827c7e01…`.
- **fes-ui at start:** `FES_EXPERIMENTAL_PROFILES=r5_8`.
- **Git at start:** HEAD `179744b`. Pre-existing working-tree changes were left untouched: 2 modified `identity_calibration.json`, plus the old Phase-5 baseline files and task prompts.

## 3. R6-0 scope and non-goals

**In scope:** the 5 requested families, local difference, identity movement, combinations, FA/FR, the known-failure challenge.

**Additional families (my own):**
- **F**: flip-consistency maps;
- **run tracer**: the per-run colour dottedness of each mirror run.

**Respected:**
- no blur threshold changed;
- no model changed or added;
- no Feature-Match;
- no new dataset;
- no change to the live/default route, to R5-6/R5-8/R5-10 behaviour, or to any existing API field.

**Deviation (owner-authorized, see §1):** the R6 profile was implemented opt-in instead of stopping at a paper recommendation.

## 4. Evaluation corpus (existing R5 evidence only)

`scripts/r6_0/build_cases.py` → `data/manifests/phase5_rapid/r6_0/cases.json` (895 cases). For each case, `run_variants.py` re-created the exact output of the build that produced it. **Determinism check:** 887 of 895 are bit-identical to the saved R5 output. The other 8 are real-photo full_image chips, and their saved files were stored pasted back at 224 px rather than as chips (verified: the chips are identical).

| Source | What | Build |
|---|---|---|
| `dev` | all 493 R5-8 DEV visual labels (archive val) | R5-8 candidate |
| `test_r57` | the 8 R5-7 TEST escapes | R5-8 candidate |
| `test_r56` | the 29 live R5-6 outputs above the R5-8 local limit (R5-8 §13) **+ 120 random returned R5-6 outputs** (seed 0) | live R5-6 |
| `test_r58` / `test_r510` | image_89280, image_91965 (R5-8 escapes); image_84257 (R5-10 escape) | candidate |
| `real_r56` / `real_r510` | every NAFNet output ever produced on the 200 CC0 real photos (14, incl. img118) | live / candidate |
| `dev_mod` | 80 random R5-10 moderate-blur outputs (archive val) | candidate |
| `p4` | all 119 R5-10 outputs of the p4dev/r5gap synthetic families (gauss/motion blur mild–extreme, JPEG, low contrast, under-exposure) | candidate |
| `clean` | 30 random NAFNet outputs on SHARP faces (suspicious change on an input that needs nothing) | candidate |

**Population sets (held-out benefit and rates, not labelled one by one):**

- `population.py` ran the R5-8 candidate path on the first 600 gate-eligible faces per mode from a seeded shuffle of archive **val** (DEV) and **test** (held out): 1,200 + 1,200 faces × 6 output variants.
- Identity is measured against the sharp GT with the R5-ARCHIVE method.
- The artifact rate in the population sets was checked visually on the tails, near-threshold, rejected and random samples (§10).

## 5. Label definitions

The label applies to the **output the build returned** (v0).

- **GOOD:** no visible artifact at 2× nearest-neighbour zoom.
- **BAD:**
  - *clear*: visible dotted / rainbow / grid patch, edge streak or colour mottling;
  - *faint*: visible only at zoom;
  - *destructive*: whole-face texture or grid;
  - *stress*: halftone prints that the live service sends to A4, never NAFNet. NAFNet was forced there in R5-10's run. Reported separately and excluded from FA/FR.
- **UNCERTAIN:** kept separate, never counted as GOOD or BAD.

**Label sources:**
- R5-8 DEV labels are carried over verbatim.
- Every other case was labelled visually by me in R6-0 (`labels_r6.json`, 392 cases, sheet and note per case).
- Separately, `labels_final_output.json` holds the judgement on the **R6 output** (median + luma) for every BAD case R6 would not reject.

**A finding from the labelling:** of the 120 random returned R5-6 outputs, **120 are GOOD**. That is consistent with the ≈0.2 % localized-artifact rate, so the failures are rare but real.

## 6. Exact dataset counts

| Group | GOOD | BAD clear | BAD faint | BAD destructive | BAD stress | UNCERTAIN | Total |
|---|---|---|---|---|---|---|---|
| A — good-source pools (DEV LEGITIMATE + random R5-6 TEST) | 578 | — | — | — | — | — | 578 |
| B — known failures (DEV artifacts, R5-7/8/10 escapes, R5-6 flagged) | — | 48 | 12 | 12 | — | 2 | 74 |
| C — borderline (moderate blur, p4 families, clean faces, real photos) | 237 | 2 | — | — | 4 | — | 243 |
| **Total** | **815** | **50** | **12** | **12** | **4** | **2** | **895** |

**By source:**

| Source | Counts |
|---|---|
| dev | GOOD 458, BAD 32, UNCERTAIN 2 |
| test_r56 | GOOD 120, BAD 29 |
| test_r57 | BAD 8 |
| test_r58 | BAD 2 |
| test_r510 | BAD 1 |
| dev_mod | GOOD 80 |
| p4 | GOOD 118, BAD 1 (gaussblur-extreme@168, magenta dotted cheek) |
| clean | GOOD 30 |
| real | GOOD 9, BAD 1 clear (img118) + 4 stress |

**FA/FR denominators:**
- DEV: 458 GOOD / 32 BAD.
- **Held-out (everything not DEV): 357 GOOD / 42 BAD.**

**Population:**
- val: 600 fc + 600 fi;
- test: 600 fc + 600 fi;
- each × 6 variants, 4 NAFNet runs per face.

## 7. Signal definitions (`scripts/r6_0/signals.py`; Lab space; 8×8 windows; out-of-image pixels excluded)

| Family | Signals |
|---|---|
| **G geometry** | YuNet on input vs output: score drop, max landmark displacement / IOD, bbox IoU, face still found |
| **P photometric** | max channel-mean shift, luminance shift, chroma Δ p99 and p99.9, 8×8 local chroma Δ max, colour invented over a grey input, fraction of pixels changed > 64 |
| **T texture / frequency** | HF luma energy ratio, HF colour / luma ratio, HF colour growth, colour dottedness `t_dot_c_max` (= R5-8 `l_dot_c_max`), luma dottedness, spectral peak of the luma residual (grid / period detector) |
| **L local difference** on \|out − in\| | changed-pixel %, largest connected changed region, top-2 % window energy concentration, share of change energy in the 8-px border, 8×8 max change and its excess over the median, luma 8×8 max |
| **I identity** | SFace and dlib cos(input chip, output) with the input's landmarks: a SAFETY signal only |
| **F flip consistency** (new) | per-pixel disagreement between NAFNet(x) and flip⁻¹(NAFNet(flip x)) for h / v / hv: luma and chroma 8×8 max and p99.9, disagreement / change ratio, mean |
| **Run tracer** (new) | for each of the 4 runs, colour dottedness vs the input; `run_dot_c_max` = the maximum over runs |

**Threshold rule (fixed before evaluation; research only):** for every signal, the DEV-GOOD 99th percentile (n=458), oriented so that larger means worse. Non-DEV numbers are therefore held out. **No threshold was tuned on non-DEV data.** The single R6 threshold is `run_dot_c_max` > 20.2863.

## 8. Individual signal results (GOOD n=815, BAD n=70 excl. stress; borderline = group C)

| Signal | GOOD median / p95 | BAD median | Borderline median / p95 | AUC | Recall @DEV-p99 (all / held-out) | FR held-out | Known misses |
|---|---|---|---|---|---|---|---|
| g_score_drop | −0.005 / 0.003 | −0.001 | −0.002 / 0.003 | 0.71 | 0.15 / 0.05 | 0.3 % | 63 |
| g_lm_disp | 0.086 / 0.27 | 0.10 | 0.048 / 0.24 | 0.58 | 0.03 / 0.00 | 0.3 % | 72 |
| g_bbox_iou | 0.94 / 0.82 | 0.93 | 0.97 / 0.85 | 0.56 | 0.03 / 0.00 | 0.3 % | 72 |
| p_chroma_p999 | 7.3 / 22.2 | 18.2 | 4.5 / 12.1 | 0.85 | 0.36 / 0.21 | 0 % | 47 |
| p_chroma_local_max | 5.6 / 19.2 | 12.4 | 3.5 / 11.9 | 0.77 | 0.24 / 0.07 | 0 % | 56 |
| p_chan_shift | 0.50 / 2.17 | 1.41 | 0.28 / 1.41 | 0.79 | 0.15 / 0.02 | 0.3 % | 63 |
| p_colour_invented | 1.0 / 5.0 | 1.4 | 0.6 / 2.7 | 0.59 | 0.16 / 0.00 | 0 % | 62 |
| t_hf_colour_growth | 1.05 / 1.20 | 1.52 | 1.05 / 1.12 | 0.95 | 0.55 / 0.40 | 0 % | 33 |
| **t_dot_c_max** (R5-8 signal) | 3.8 / 7.2 | 103 | 3.6 / 7.3 | **0.983** | **0.93 / 0.93** | 0.3 % | **5** |
| t_dot_l_max | 40 / 191 | 671 | 37 / 299 | 0.96 | 0.70 / 0.62 | 2.8 % | 22 |
| t_spec_peak | 68 / 148 | 37 | 48 / 131 | 0.21 (inverted) | 0.00 | 0.8 % | 74 |
| l_changed_pct | 0.30 / 0.60 | 0.33 | 0.10 / 0.51 | 0.56 | 0.01 / 0.00 | 0 % | 73 |
| l_largest_cc | 0.028 / 0.14 | 0.037 | 0.006 / 0.076 | 0.57 | 0.03 / 0.00 | 0 % | 72 |
| l_top2_conc | 0.16 / 0.28 | 0.16 | 0.18 / 0.33 | 0.54 | 0.14 / 0.07 | 7.6 % | 64 |
| l_border_share | 0.21 / 0.42 | 0.25 | 0.18 / 0.37 | 0.53 | 0.15 / 0.26 | 3.4 % | 63 |
| l_win_max | 34 / 66 | 36 | 19 / 56 | 0.59 | 0.16 / 0.00 | 0 % | 62 |
| i_sface_io | 0.80 / 0.47 (p5) | 0.80 | 0.96 / 0.64 | 0.54 | 0.03 / 0.00 | 0.3 % | 72 |
| i_dlib_io | 0.977 / 0.935 (p5) | 0.973 | 0.994 / 0.947 | 0.57 | 0.03 / 0.00 | 0.3 % | 72 |
| f_chroma_p999 | 5.8 / 15.3 | 26.5 | 4.1 / 10.2 | 0.95 | 0.55 / 0.45 | 0.6 % | 33 |
| f_luma_p999 | 34 / 80 | 77 | 21 / 71 | 0.88 | 0.27 / 0.17 | 0.3 % | 54 |
| **run_dot_c_max** (new) | 4.5 / 8.9 | **258** | 4.1 / 8.5 | 0.977 | **0.95 / 0.93** | 0.6 % | **4** |

**Reading:**

- **Geometry, identity and local difference do not separate.** BAD artifacts are small patches: they move landmarks and embeddings no more than a legitimate strong deblur does. Identity similarity of a destroyed output is often *higher* than that of a good strong deblur, which changes the face a lot. So **identity cannot be a pass signal**, which confirms the prompt's warning.
- The spectral-peak "grid detector" is *inverted*: legitimate deblurs create sharper periodic structure (hair, teeth) than the artifacts do.
- **Only dottedness separates.** As an output check it hits the same ceiling R5-8 found (5 misses). See §13 for why.

## 9. Combination results (DEV-p99 thresholds; reject if any member fires)

| Verifier | DEV FR | DEV FA | Held-out FR | **Held-out FA** |
|---|---|---|---|---|
| R5-6 live guard | 157/458 (34 %) | 14/32 | 3/357 | **36/42** |
| R5-8 guard (global + l_dot_c_max) | 1/458 | 3/32 | 0/357 | **4/42** |
| photometric + local | 42 | 12 | 44 (12 %) | 21 |
| texture + local | 52 | 2 | 71 (20 %) | 2 |
| geometry + identity | 22 | 22 | 5 | 40 |
| photometric + texture + local | 68 | 1 | 73 (20 %) | 2 |
| all 5 requested families | 86 (19 %) | 1 | 77 (22 %) | 2 |
| best signal per family (5) | 24 | 2 | 3 | 2 |
| flip consistency (F) alone | 22 | 10 | 14 | 21 |
| run tracer `run_dot_c_max` alone | 5 | 1 | 2 | 3 |
| **R6: any unstable run → ORIGINAL, else median + input colour** | **5 (1.1 %)** | **0** | **2 (0.6 %)** | **0** |
| R6 variant "≥ 2 of 4 unstable" | 1 | 2 (faint residue) | 0 | 0 |

**How FA is counted for R6:**
- R6 returns the median/luma image, not v0.
- A BAD case that R6 does not reject counts as a false accept unless its **R6 output** was reviewed clean. Unreviewed counts as FA.
- Strict R6 accepts 4 BAD cases (all four runs stable): dev 68258 (coarse grid), img118 (mottling), 89280 (dotted cheek), 91965 (edge specks). Their R6 outputs were checked at 4× zoom: all clean. The grid, mottling and specks are gone because the median and the input's colour remove them.
- The "2 of 4" variant was rejected: two DEV destructive cases (75394, 76320) keep a faint dotted residue after the median.

**The 7 GOOD outputs R6 rejects:**
- DEV: 6804, 82463, 65775, 74459, 77614;
- TEST: 91875, 99928.

All are cases where v0 looked fine but a mirror run exploded (tracer up to 5,277). That is the conservative trade: the original is returned.

**Implementation (opt-in profile `r6`):**

| Item | Detail |
|---|---|
| config | `configs/phase5/r6_nafnet_candidate.json`, seal `9d73d36c…`: the R5-8 candidate unchanged + a `stability` section |
| route module | `fes_service.nafnet_route`: `dot_c_max`, `consensus_luma`, `stability_check` |
| model factory | `fes_service.models.nafnet_factory_r6` (4 runs in the isolated worker) |
| pipeline | a stability branch before the existing guards; fallback `STABILITY_GUARD` |
| app | `r6` gets its own worker, `FES_NAFNET_R6_TIMEOUT_S`, default 120 s |
| UI | compare page "NAFNet R6 — Stability-Verified Experimental" plus a "Stability check (4 runs)" row |

The R5-8 global and local guards still run after the stability check (defence in depth).

## 10. False-accept analysis

**Corpus:** R6 returns **0** visible artifacts among 70 known BAD outputs (DEV 0/32, held-out 0/42); all 4 stress cases are rejected. The existing R5-8 guard false-accepts 7/74, including 4 held-out cases.

**Population, archive TEST (held out, 1,200 faces):**
- Unstable-run histogram: 1,185 × 0 unstable runs, 12 × 1, 2 × 2, 1 × 4.
- Visually reviewed R6 outputs: 35 accepted outputs with the highest luma-dottedness or flip disagreement, the 14 accepted outputs nearest the threshold, and 28 random accepted outputs. **0 artifacts** in all of them.
- The 15 rejected cases were also reviewed: 12 show exploding runs (multi-colour dotted blocks in v0, or in the mean of the runs). Three are clean in v0/median, but one mirror run exploded.
- archive val (DEV, 1,200 faces): the same review found 0 artifacts in the accepted samples (13 rejected).

**What R6 cannot see (residual FA risk):**
1. A failure that is stable under mirroring **and** in luminance only: a consistent luma texture hallucination. None was found in the corpus, but the corpus has few such inputs.
2. Failures on input types the gate does not admit today. R6 keeps the conservative R5-8 gate; img118 is not even attempted under R6.
3. The FA = 0 result rests on my single-reviewer visual judgement of 4 accepted-BAD R6 outputs plus the review samples above.

## 11. False-reject analysis

| Measure | Value |
|---|---|
| Corpus FR, DEV | 5/458 (1.1 %) |
| Corpus FR, held-out | 2/357 (0.6 %) |
| Population reject rate, val fc / fi | 1.0 % / 2.0 % |
| Population reject rate, test fc / fi | 1.3 % / 1.2 % |

**The cost of the luma-only output (identity proxy, accepted outputs):**
- The median of 4 runs is slightly *better* than v0: DEV fc +0.1797 vs +0.1753 on labelled GOOD.
- Removing the colour change costs ≈ 0.005 ΔSFace.

**Net effect on archive TEST (whole policy, ΔSFace vs input [95 % CI]):**

| Mode | R5-8 | R6 |
|---|---|---|
| face_chip | +0.1538 [0.144, 0.164] | +0.1560 [0.146, 0.166] |
| full_image | +0.1117 [0.104, 0.120] | +0.1116 [0.104, 0.120] |

dlib moves the same way. **Harmed faces (ΔSFace < −0.05): 16 → 9.** The benefit is unchanged and the harm is lower.

On DEV, R6 is 0.001–0.002 below R5-8 (within the CI).

## 12. Known-failure challenge (every known BAD case; full list in `analysis.json` → `challenge`)

| Challenge | Cases | R5-8 guard | R6 |
|---|---|---|---|
| R5-7 rainbow / dotted escapes | 8 (95636, 88924, 8964, 96161, 97266, 90856, 85126, 96475) | 8 DETECTED | 8 REJECTED |
| R5-8 DEV localized artifacts | 21 | 18 DETECTED, 3 MISSED (77722, 68258, 77307) | 20 REJECTED; 68258 ACCEPTED, output clean |
| R5-8 DEV destructive | 11 | 11 DETECTED | 11 REJECTED |
| R5-8 TEST escapes | image_89280 (clear), image_91965 (faint) | 2 MISSED | both ACCEPTED, output clean (4× zoom) |
| R5-10 image_84257 | 1 | MISSED | **REJECTED** (a mirror run exploded) |
| R5-10 real **img118** (colour mottling) | 1 | MISSED | ACCEPTED as a stable run → **output clean** (colour comes from the input). Also not attempted at all under R6's gate. |
| R5-6 live flagged artifacts | 29 (17 clear, 11 faint, 1 destructive) | 29 DETECTED | 29 REJECTED |
| p4 gaussblur-extreme@168 (magenta dots) | 1 | DETECTED | REJECTED |
| motion blur (p4 motion mild–extreme, 28 outputs) | GOOD | 0 FR | 0 FR |
| moderate blur (80 archive val) | GOOD | 0 FR | 0 FR |
| grayscale (real grey prints) | 4 GOOD | pass | pass; luma-only keeps them grey |
| printed / halftone (img165, img72, forced) | 4 stress | all REJECTED | all REJECTED (all 4 runs unstable) |
| small faces | none admitted by the gate (`fi_chip_px` < 72: 0 in DEV) | n/a | n/a |
| JPEG / low contrast / under-exposure | 6 GOOD | pass | pass |

## 13. Missed-failure analysis

**Why output-only verifiers miss:**
- The missed cases are **low-amplitude**: coarse grid 68258, 89280, 91965.
- Or **colour-smooth**: img118 mottling, a low-frequency colour shift with no dots.
- Their dottedness overlaps the strongest legitimate deblurs: GOOD max 24.8 vs BAD 2.3–16.8.

No threshold separates them. This is the same wall R5-8 and R5-10 hit.

**Why R6 does not miss them:**
- **84257** sits on an input where NAFNet is unstable. A mirror run explodes (tracer far above 20), so the instability is visible even when v0's own artifact is faint.
- **68258, 89280, 91965, img118** are stable across runs but sit at different positions in different runs. The per-pixel median removes them (the faint dots are not consistent). The colour mottling is removed by construction, because R6 never changes colour.

**R6's own possible misses** are listed in §10 (stable luma-only hallucination; unadmitted inputs). No known case falls into them.

## 14. Recommended verifier architecture (R6)

```
chip ──(R5-8 gate, unchanged)──► NAFNet ×4: x, flipH, flipV, flipHV   (mirrored back; reflect-pad16)
                                  │ per-run colour-dottedness tracer vs input
                                  ├─ any run > 20.2863 ─────────────────────────► ORIGINAL  (STABILITY_GUARD)
                                  └─ all stable → median(4 runs) → Y(median) + CrCb(input)
                                        → R5-8 global guard → R5-8 local guard → ENHANCED | ORIGINAL
```

**Properties:**
- deterministic (bit-exact against the R6-0 study; real-NAFNet test);
- no new model or data; one DEV-derived constant;
- conservative failure modes: tracer missing, NaN or the wrong count → ORIGINAL; model failure or timeout → ORIGINAL;
- colour can never change; a grey input stays grey;
- cost: 4× NAFNet, ≈ 1.0 s per 128-px face on the shared CPU (measured live), against ≈ 0.25 s.

## 15. Limitations

- **Data:**
  - The archive is synthetic blur (CelebA-like, licence unknown; evaluation only).
  - Real photos: 14 NAFNet outputs, no identity references.
  - Grainy and printed real inputs remain thin. That is why R6 keeps the R5-8 gate and does not widen it.
- **Labels:** one reviewer (me), at 2–4× zoom. Faint artifacts below that visibility can exist. The population review covered signal tails, near-threshold, rejected and random samples, not all 2,400 outputs one by one.
- **Identity:** SFace and dlib proxies against the GT, not Feature-Match. Feature-Match validation (≥ 20 people) is still open.
- **Threshold:** one constant (DEV-GOOD p99). Its held-out FR (0.6 % corpus, 1.2 % population) and FA (0) were measured, not tuned.
- **Licensing:** NAFNet MIT, but production use is REVIEW_REQUIRED (no licence reviewer named).
- **Pre-existing test failure, not caused by R6-0:** `tests/test_p5_baseline.py` fails on `src/fes_bench/adapters.py`, a file R6-0 did not touch. It belongs to the old uncommitted 5.1 baseline-freeze files (memory note).
  - The rest of the suite: **345 passed, 9 skipped** (Docker `fes-dev:cpu-p5`, with the R6 changes).
  - R6 tests: 23 + 1 real-model test pass.
  - UI: 33/33.

## 16. Verdict: **GREEN — verifier feasible (research), not production-approved**

The evidence meets the GREEN definition:

- every known unsafe NAFNet output is either detected (rejected) or neutralised in what is returned: 0/70 visible, held-out 0/42;
- about 98.8 % of eligible faces are still enhanced;
- the identity gain is unchanged.

The five requested output-only families alone would be **YELLOW**: the best signal still misses 5 known cases. The GREEN rests on the stability check and consensus, which follow from the root cause found in R6-0.

## 17. Exact next step

1. **Owner review (now):** open http://localhost:8080/ui/compare.html. Profile **"NAFNet R6 — Stability-Verified Experimental"** is enabled. Rate R6 against R5-6 / R5-8 on 30–50 own real images (the R5-9 evaluation form now has R6 columns).
2. **If the owner agrees, R6-1** (separate task, owner approval required):
   - make `r6` the NAFNet default, or keep it opt-in for the Feature-Match run;
   - re-run the R5-6 service-level harness (39k requests) with `FES_NAFNET_PROFILE=r6`.
3. **Feature-Match validation** with R6 (≥ 20 identities), still blocked on access and data.
4. **Not recommended:**
   - any further output-only detector work;
   - widening the blur gate (R5-10) until real grainy data exists.

## 18. Reproduction

Host = `.venv/Scripts/python.exe`. Docker = `FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh` (or a detached `docker run` with the same mounts).

```
python scripts/r6_0/build_cases.py                                  # corpus (cases.json)
docker: python scripts/r6_0/run_variants.py --workers 4 --threads 3 # 4 NAFNet runs per case → outputs/r6_0/cache
PYTHONPATH=src python scripts/r6_0/compute_signals.py               # signals.csv.gz
docker: python scripts/r6_0/identity.py                             # identity.csv.gz
python scripts/r6_0/run_signals.py                                  # run_signals.csv.gz
PYTHONPATH=src python scripts/r6_0/baselines.py                     # R5-6 / R5-8 guard decisions
python scripts/r6_0/label_sheets.py / set_labels.py                 # visual labelling (labels_r6.json)
python scripts/r6_0/analyze.py                                      # analysis.json (thresholds, tables, challenge)
docker: python scripts/r6_0/population.py --split val  --n 600 --workers 4 --threads 2 --save
docker: python scripts/r6_0/population.py --split test --n 600 --workers 4 --threads 2 --save   # then gzip -9
python scripts/r6_0/popanalyze.py                                   # population_summary.json
python scripts/r6_0/pop_review.py {val|test} OUTDIR                 # review sheets
PYTHONPATH=src python scripts/r6_0/make_r6_config.py                # sealed r6 config (seal 9d73d36c…)
docker: python -m pytest -q tests/test_r6_candidate.py  (+ FES_RUN_NAFNET=1 for the real-model test)
node --test tests/ui/fes_compare.test.mjs tests/ui/fes_ui.test.mjs
```

**Demo service** (as running now): `docker run -d --name fes-ui -p 127.0.0.1:8080:8080 -v <repo>:/workspace -v <repo>/models:/workspace/models:ro -w /workspace -e PYTHONPATH=src -e FES_EXPERIMENTAL_PROFILES=r5_8,r6 … fes-dev:cpu-p5 uvicorn fes_service.app:app --host 0.0.0.0 --port 8080`. Requests without `profile` = R5-6, unchanged.

**Files:**
- `scripts/r6_0/*`
- `data/manifests/phase5_rapid/r6_0/*`
- `configs/phase5/r6_nafnet_candidate.json`
- `src/fes_service/{nafnet_route,models,pipeline,app}.py` and `ui/{fes_compare.js,compare.js,compare.html}` (additive, r6 only)
- `tests/test_r6_candidate.py`, `tests/ui/fes_compare.test.mjs`
- this report

## 19. Addendum (2026-10-09): blur range widened, second pass tested, fine-tuning cost measured

**Owner feedback:** the gate only attempted heavy blur, where little can be recovered, and rejected less-blurred faces that are easier to improve.

**1. The R6 blur range is widened** (config re-sealed `32e47852…`; R5-6 / R5-8 / R5-10 are unchanged). `blur_max = clean_lapd1_lo = 0.008`, so the mild-blur band is gone.

DEV evidence (archive val, R5-10 outputs, ΔSFace vs GT):

| Blur score (lapd1) | ΔSFace | Harmed |
|---|---|---|
| < 0.003 | +0.155 | 1.1 % |
| 0.003–0.005 | +0.141 to +0.153 | 0.3–0.6 % |
| 0.005–0.012 | +0.126 to +0.150 | ≤ 0.4 % |
| sharp GT faces | −0.02 | — |

Mild blur gains as much as strong blur. On 800 val faces, sharp GT faces have lapd1 p5 = 0.0073. The 0.008 limit admits 97.5 % of blurred faces and 6.5 % of sharp ones.

Live check (HTTP, fes-ui): 6/6 DEV faces with lapd1 0.0041–0.0078, all rejected by R5-6 as MILD_BLUR / NOT_BLUR, are enhanced by R6 with 0/4 unstable runs. The faces still rejected fail other gates: photometric (too dark / low contrast) or face size.

**2. Second / third NAFNet pass** (`scripts/r6_0/multipass.py`, 480 val faces, `multipass_val.json`):
- Visually sharper, but **no identity gain**: heavy +0.160 → +0.159 → +0.154; strong +0.173 → +0.171 → +0.167; moderate +0.146 → +0.144 → +0.138.
- Harm rises with each pass: moderate 0 % → 1.2 % → 4.4 %.
- **Not adopted.**

**3. Why outputs on soft blur change little.** NAFNet-GoPro was trained only on camera-shake video blur (GoPro dataset). Defocus / Gaussian face blur is outside its training. No threshold changes this; fine-tuning on face blur does.

**Fine-tuning measured on this laptop** (Docker, 8 threads, 128-px crops): **≈ 95–110 s per training step at batch 4**. 2,000 steps ≈ 2.5 days. Not practical on this CPU.

**Licensed data is available:** FairFace (DS01, CC BY 4.0, ELIGIBLE, ≈ 97k faces) with self-generated blur. NAFNet weights are MIT.
