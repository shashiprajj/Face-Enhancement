# R5-4 — Actual Feature-Match Validation: BLOCKER REPORT

| Field | Value |
|---|---|
| Phase / task | 5 Rapid Validation / R5-4 (final decision gate) |
| Date | 2026-10-05 |
| Status | **BLOCKED. No Feature-Match verdict.** Both stop conditions of the task apply. |
| Frozen baseline (verified, untouched) | commit `4ea2ed3` · gate `p5-rwgate-1.0.0` from `76577d5`, seal `78837a8d` · A4 SwinIR-L · R5-1 service |
| Delivered | evaluation harness `scripts/r5_fm_validate.py` (ready, not run) · `tests/test_r5_fm_harness.py` · `data/manifests/phase5_rapid/feature_match_validation/results.json` (status BLOCKED) |

## 1. Executive verdict

**No GREEN / YELLOW / RED verdict can be given on Feature-Match.**

- **B1:** Feature-Match is not accessible from this environment.
- **B2:** the real paired dataset is 1 person / 5 photos; the task's minimum is 20 or more people.

As instructed, **SFace/dlib were not substituted**, and nothing in the frozen pipeline was changed. The question *"Does enhancement convert real degraded images into correct Feature-Match matches?"* remains **unanswered**.

The best available evidence is still the R5-2 **proxy** result:
- the gated A4 gives a small identity-similarity gain, with strong blur the clearest case;
- noise is mixed, with more wrong-person similarity;
- low light and pose are not helped.

That evidence is research-only and must not be read as a Feature-Match result.

## 2. Objective

Measure, with the actual Feature-Match system on real identity-labelled pairs, whether:
- **A:** the original;
- **B:** the frozen gate → original or A4;
- **C:** forced A4 (diagnostic)

change correct / wrong / no-match outcomes, including rescues, harms and new wrong-person matches.

## 3. Frozen baseline (checked before work)

- `git log -1` = `4ea2ed3`. The working tree has no tracked modifications; only untracked pre-existing items (old 5.1 files, task prompts, `Dataset-and-code/`).
- `configs/phase5/r5_rwgate.json`: `p5-rwgate-1.0.0`, seal `78837a8d…`.
- No source, config, A4, gate, detector or Feature-Match file was changed by this task (`git diff 4ea2ed3 -- src configs` is empty).

## 4. Feature-Match integration: what was checked

| Check | Result |
|---|---|
| This repository (code, configs, docs) | Feature-Match appears only in docs as an **external, never-inspected future consumer**: P0-T1 §5 X6 and §7, P0-T5 §9 *"not inspected, imported, modified, or referenced"* |
| `D:\ReBIT` | contains only `Face-Enhancement` |
| Depth-4 directory search of `D:\` and `C:\Users\SHASHI` for `*feature*match*` | nothing found |
| Docker images / containers | nothing matching |
| API endpoint, credentials, model version, threshold | none known |

There is no safe local way to invoke Feature-Match. Building one would require inventing an integration, so the task stops here.

## 5. Dataset / inclusion rules

| Required | Available |
|---|---|
| ≥ 20 real people, each with a clear reference + degraded queries | **1 person**, 1 reference + 5 degraded phone photos (`Dataset-and-code/Real-Person-Test-Data`) |
| blur, low light, noise, compression, small face, pose | low light (2), greyscale, small face, extreme pose; **no blur, noise or compression** |
| successes and failures | proxy: 3 match / 2 fail (low light, pose) |

The 1-person set may serve only as a supplementary pilot, as the task says. It cannot support any rate or decision.

## 6–12. Conditions, results, rescue/harm, degradation, gate groups, forced-A4, false accepts

**Not run.** There are no Feature-Match numbers. The harness that will produce every listed metric exists (§16):

| Metric | Harness field |
|---|---|
| Correct-match rate, original vs current pipeline, absolute change | `metrics.pipeline_B.rate_A / rate_X` |
| Rescued / harmed / unchanged | `metrics.pipeline_B.rescued / harmed / unchanged` |
| New wrong-person matches | `…new_wrong_person` (each output is also verified against every *other* person's reference) |
| Degradation breakdown | `metrics.by_degradation` (from `labels.csv`) |
| Gate-admitted vs gate-rejected | `metrics.gate_admitted` (B) · `metrics.gate_rejected_forced` (C) |
| Forced-A4 diagnostic | `metrics.forced_C`; never run on UNSAFE inputs |
| Per pair | anonymised id (P01-Q1 …), SHA-256, degradation, faces found, gate decision/reason, A4 ran, fallback, and score / threshold / correct / wrong-person count for A, B, C |

## 13. Limitations

- Without Feature-Match, every identity claim in Phase 5 Rapid rests on SFace/dlib proxies (R5-2).
- The harness assumes Feature-Match can compare a reference image with a query image. If it only accepts full photos (not 224-px face chips), condition B needs Feature-Match's guidance on input format. The current A4 output is a face chip.

## 14. Product-value decision

**Undecided, pending Feature-Match.** Under the task's decision rule, no product value may be claimed without measured rescues on Feature-Match. Current proxy evidence would place A4 + gate at most in a narrow class (strong blur). The user's observed real failures (low light, pose) are not addressed by A4 in any evidence so far.

## 15. Exactly what is needed to unblock (recommended next task)

**1. Feature-Match evaluation access**, as one of:
- **(a)** a Python entry point: package path, function to embed or compare, model version, configured threshold / operating point;
- **(b)** an evaluation endpoint: URL, request format (image upload fields), response fields (score, match, threshold), authentication;
- **(c)** a Docker image of Feature-Match for local evaluation.

Also needed:
- the **input expectation** (full photo or face crop; any alignment it performs);
- **1:1 verification vs 1:N identification** and the gallery to use;
- written confirmation that evaluation-only use with the test photos is permitted.

The adapter is a ~20-line class implementing `verify(reference, query)`. Feature-Match itself is not modified.

**2. Paired real data**, placed outside git:
- **≥ 20 people** (30+ preferred), each with 1 clear reference and 1–3 degraded photos;
- spread over blur, low light, noise/grain, compression, small face and pose, including some that already match;
- consent for evaluation use.

Layout: `<data>/<person>/reference/*`, `<data>/<person>/degraded/*`, optional `labels.csv` (`person_id,file,degradation`).

**3. Then run (about 1 h of CPU for about 60 queries):**

```bash
FES_FM_CLIENT=<module:factory> FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_fm_validate.py --data <dir>
```

The R5-4 report is then completed with the GREEN/YELLOW/RED decision under the task's rule.

## 16. Tests / commands

- `tests/test_r5_fm_harness.py` uses a fake matcher, with no faces and no model. It covers:
  - anonymisation and labels;
  - rescue / harm / unchanged / wrong-person accounting;
  - UNSAFE inputs never forced;
  - the blocked exit without a client.
- Focused run: `tests/test_r5_fm_harness.py tests/test_r5_rwgate.py tests/test_r5_service.py` → **52 passed, 1 skipped** (opt-in real-A4 test).

## 17. Commit

See the commit that adds this report (hash returned in the task response).
