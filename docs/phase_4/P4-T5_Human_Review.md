# P4-T5 — Blinded Human Identity & Fabrication Review

| Field | Value |
|---|---|
| Phase / Task | Phase 4 / P4-T5 |
| Date | 2026-10-03 |
| Status | **Review kit `rv1` built, reproducible and tested. REVIEW NOT PERFORMED. G-5 open: ≥ 2 human reviewers are needed.** The model's own visual judgement is not used as human review. |
| Builder | `scripts/p4_review_build.py` · tool template `scripts/p4_review_tool.html` |
| Analysis | `scripts/p4_review_analyze.py` · tests `tests/test_p4_review.py` (3 passed, simulated responses only) |
| Tracked | `data/manifests/phase4/review/key_rv1.csv.gz` (unblinding key), `kit_rv1.json`, `analysis_rv1.json` (status INSUFFICIENT) |
| Local kit (git-ignored) | `experiments/phase4/outputs/review_kit/rv1/index.html` + `img/` (faces never leave the machine) |
| User decision | 2026-10-03: kit only, review pending |

---

## 1. What Is Reviewed

Identity preservation and fabrication, **not attractiveness**. Every item shows the clean reference (left) and one image under review (right). The reviewer does not know how the right image was produced.

| # | Dimension (P4-T5) | Response scale |
|---|---|---|
| 1 | Same-person confidence | 1 definitely different … 3 cannot tell … 5 definitely same |
| 2 | Identity substitution | no / unsure / yes ("a different, specific person") |
| 3 | Invented facial structure | none / minor / major |
| 4 | Identity-relevant attributes removed or changed | none, facial hair, glasses, jewellery/piercing, marks (moles, scars, bindi), hair, eyes, nose, mouth/teeth |
| 5 | Age / appearance change | none, younger, older, gender presentation, skin tone, expression |
| 6 | Artefacts | none / minor / major |
| 7 | Reviewer confidence | low / medium / high |

The instructions tell reviewers explicitly that a blurry image of the same person is still the same person, and to judge identity, not sharpness.

## 2. Reproducible Sample (`rv1`, seed 20260928, sha256-ordered)

The source is the Phase 3 `main` outputs (test split, **SYNTHETIC**). Nothing was re-run.

| Role | Selection | n | Purpose |
|---|---|---|---|
| TEST / RANDOM | The same 26 pairs for each of B0, A1–A5 (severity-stratified: 6 mild, 6 moderate, 8 severe, 6 extreme) | 156 | Unbiased rates; **paired** vs B0 |
| TEST / TARGETED | 4 per candidate from the P3-T7 identity flags (id_flip, id_closer_to_impostor, id_drift) | 20 | Confirms or refutes F1/F2; reported separately (biased by construction) |
| CONTROL_SAME | The reference itself after JPEG q85 + a 2 % rescale | 20 | Known-same |
| CONTROL_DIFFERENT | The clean reference of another identity in the **same race × gender stratum** | 20 | Known-different (hard negative) |
| REPEAT | Exact duplicates of test items under new ids | 12 | Intra-rater consistency |
| **Total** | | **228** | About 45–60 min per reviewer |

**Blinding.** Item ids are random hashes, and the kit contains no configuration names. The unblinding key is tracked separately and is not shipped with the kit. Each reviewer gets a deterministic order derived from the kit version and their reviewer code. Both are recorded in the export.

**Reproducibility.** Rebuilding gives byte-identical `key_rv1.csv.gz` (`5264c93d…`) and `items.json` (`7e291b84…`).

## 3. Tool

A single offline HTML file:
- no network requests and no external resources (checked: 0 external references; JavaScript passes `node --check`);
- progress is saved in the browser's localStorage, and the reviewer exports a JSON file;
- reviewer codes only, never real names;
- every question must be answered, and "none" cannot be combined with other options.

**Not yet usability-tested with a real reviewer.**

## 4. Analysis Rules (pre-registered in `p4_review_analyze.py`)

- **Qualification:** control accuracy ≥ 0.85 (same-controls rated 4–5, different-controls rated 1–2). Unqualified reviewers are reported but excluded.
- **Item outcome:** the mean same-person score over qualified reviewers; majority for binary outcomes (ties → unsure).
- **Rates use RANDOM items only:** substitution, any or major invention, attribute change, appearance change, major artefacts. Each has a reference-clustered bootstrap CI and a paired difference vs B0. TARGETED items are reported separately.
- **Agreement:**
  - Krippendorff's α: ordinal for same-person, invented and artefacts; nominal for substitution;
  - repeat consistency: same-person within ±1, identical substitution answer;
  - a disagreement list: same-person range ≥ 2, or a split substitution answer.
- **Status:** `EVIDENCE` only with ≥ 2 qualified reviewers, otherwise `INSUFFICIENT`. There is no fabrication-rate threshold: TH-5 is TBD with product. Evaluator–human correlation (E01/E07 vs mean same-person score) is computed once responses exist.

## 5. How to Run the Review

1. Give each reviewer the folder `experiments/phase4/outputs/review_kit/rv1/` (local or internal share only; FairFace CC BY 4.0 faces, internal evaluation, no publication). Reviewers must be human and should not be the person who built the benchmark.
2. The reviewer opens `index.html`, enters a code (e.g. R01) and exports `review_rv1_R01.json`.
3. Collect the files in `experiments/phase4/outputs/review_responses/rv1/`, then run `scripts/dock.sh python scripts/p4_review_analyze.py --version rv1 --responses experiments/phase4/outputs/review_responses/rv1`.

## 6. Current Result

`analysis_rv1.json`: **INSUFFICIENT (G-5: < 2 qualified reviewers), 0 response files.** F1 (identity substitution) and F2 (attribute removal) therefore still rest on the Phase 3 visual inspection and the two automated evaluators (P4-T4). That is **not** human evidence.

## 7. Limits

- The items are synthetic degradations of 224 px FairFace chips. A real-world review kit can be built with the same code once `rw-v1` exists.
- Known-same controls are the same photo, which makes them easy. A cross-photo same-identity control needs identity-labelled data, which requires G-3.
- 26 random pairs per configuration from 28 identities gives wide CIs. They are descriptive per stratum.
