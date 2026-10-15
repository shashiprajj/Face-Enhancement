# P5-T1 — Freeze the Phase 4 Baseline

| Field | Value |
|---|---|
| Phase / task | 5 / 5.1 |
| Date | 2026-10-04 |
| Baseline commit | `7aa8b29` (Phase 4 final, gate 🟡 YELLOW) |
| Manifest | [`configs/phase5/baseline_manifest.json`](../../configs/phase5/baseline_manifest.json), version `p5-baseline-1.0.0`, seal `9ce89a52…` |
| Tool | `scripts/p5_baseline_freeze.py` (`--write` / `--check`) |
| Tests | `tests/test_p5_baseline.py` (19 tests) |
| Data kind | **SYNTHETIC only.** The frozen thresholds are not valid for real-world inputs. |

**Outcome.** The exact Phase 4 behaviour is frozen and can be reproduced:
- **Decision logs:** T6 and T7 decisions replay byte-identically to the committed Phase 4 logs, on the synthetic test and dev populations.
- **A4 outputs:** A4 re-runs give pixel-identical outputs.
- **Assets:** every asset on the path matches its acquisition SHA-256.

Nothing was changed in this task: no routing, models, thresholds or Feature-Match integration.

## 1. Phase 4 commits verified

| Role | Commit |
|---|---|
| Phase 4 final (T8) | `7aa8b2905bf347a1a6e5375f976617d5fb11a0fe` |
| T7 freeze / evaluation | `785a4b21…` / `ad79455f…` |
| T6 freeze / evaluation | `e504dfa1…` / `bc5ef801…` |
| Phase 3 baseline | `fdfce67a…` |

- `t6_gate.json` was last changed in `e504dfa`, and `t7_routing.json` in `785a4b2`. Both are the freeze commits.
- The working tree had no tracked changes when the manifest was written (`git_head` = `7aa8b29`).

## 2. Frozen configuration (summary — full detail in the manifest)

| Item | Frozen value |
|---|---|
| T6 gate | `p4-gate-1.1.0`, seal `b8ee2c43…`, K = 4 rules. Safety: detection status OK, det score ≥ 0.85, \|yaw proxy\| ≤ 0.25. A NaN feature never enhances. Default DO_NOT_ENHANCE. |
| T7 routing | `p4-routing-1.0.0`, seal `2f026f22…`, built on the T6 seal. Table **NOISE → A4, BLUR → A4, everything else → B0 (original)**. |
| Fallbacks | DO_NOT_ENHANCE / MODEL_FAILURE / TIMEOUT (120 s) / RESOURCE_FAILURE / INVALID_OUTPUT → original. UNSAFE_OR_UNCERTAIN → original + escalate. |
| Must not route | A1 RestoreFormer++, A2 VQFR, A5 chain (A3 is also unrouted by the table) |
| Quality features | `p4-quality-1.0.0`. YuNet face_crop mode, out 112, pad 0.5, relaxed thr 0.5, C-frame 112. |
| Detector | YuNet `face_detection_yunet_2026may.onnx` `ebafce4e…`, thr 0.7, NMS 0.3, top-k 50, max side 640. Preprocess `preproc-1.0.0`. |
| A4 | SwinIR-L real-SR ×4 GAN. Weights `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth` `99adfa91…` (142 473 939 B, `params_ema`, strict load, 0 missing keys). Upstream SwinIR `6545850`. Vendored arch `network_swinir.py` `9e143898…`. fp32, CPU, 2 torch threads, tile 0. Window-8 flip-pad. Output → E-frame (area / Lanczos). |
| Alignment | A4 gets the **native degraded chip** and uses no alignment transform. Evaluation uses the E-frame with the **recorded (oracle) FairFace alignment**. This is the oracle that Task 5.3 must remove for real-world inputs. |
| Evaluators (test-only) | **E01 SFace** `0ba9fbfa…`: δ = 0.02154; FAR 1e-2 thr = 0.31333 (main) / 0.2843 (dev). **E07 dlib ResNet v1** `55533b28…` (dlib 20.0.1): δ = 0.00258; thr = 0.93363 / 0.92975. Neither is Feature-Match. |
| Runtime image | `fes-dev:cpu-p4` `sha256:cdf17906…`. torch 2.14.0+cpu, OpenCV 5.0.0, numpy 2.4.6, Python 3.11.16. |

## 3. Reproducibility checks

| Check | Result |
|---|---|
| T6 decisions `main` (896) | identical — `420e0d0d…` |
| T6 decisions `p4dev` (896) | identical — `b5ba0329…` |
| T7 decision log `main` (896) | identical — `79094637…` |
| T7 decision log `p4dev` (896) | identical — `349325f0…` |
| A4 re-run on 2 A4-routed dev pairs (seeded selection, both BLUR) | pixel-identical to the Phase 4 run manifest. 79–81 s/face on CPU with 2 threads. Shared host: WelOrbit containers were running, so this is not a performance figure. |
| Asset SHA-256 (A4, YuNet, SFace, dlib) | all match the P1-T9 acquisition manifest |
| `--check` re-derivation vs sealed manifest | seal OK, 0 drift |

Frozen behaviour reproduced on synthetic data:

| Population | DO_NOT_ENHANCE | ENHANCE_CANDIDATE | UNSAFE (escalated) | Used A4 | Fallbacks | Mild enhanced |
|---|---|---|---|---|---|---|
| main (test) | 565 | 92 | 239 | 63 | 0 | **0 / 252** |
| p4dev (dev) | 607 | 103 | 186 | 73 | 0 | **0 / 252** |

## 4. Regression tests (`tests/test_p5_baseline.py`)

| Guarantee | Test |
|---|---|
| Manifest, frozen configs and the frozen code path are unchanged (LF-normalised SHA-256) | `test_manifest_seal_and_frozen_files_unchanged` |
| Manifest agrees with the loaded seals, table, thresholds and versions | `test_manifest_matches_loaded_seals_and_versions` |
| A tampered table (e.g. PHOTOMETRIC → A1) or tampered gate rule is rejected by the seal | `test_tampered_routing_is_rejected` |
| Only A4 is routed, and only for NOISE/BLUR | `test_only_a4_is_routed_and_only_for_noise_blur` |
| Re-computed decisions, buckets and routes equal the committed T7 logs row by row | `test_replay_equals_committed_phase4_logs` |
| **Mild inputs are never enhanced.** All 504 mild inputs go through the runtime router; spy adapters are never called. | `test_mild_inputs_are_never_enhanced` |
| **A1/A2/A5 are never routed** on any of the 1 792 baseline inputs, even when their adapters are supplied | `test_a1_a2_a5_are_never_routed_on_any_baseline_input`, `test_a4_routed_input_never_calls_other_candidates` |
| **Unsafe/uncertain inputs fall back safely.** An A4-routed input with no face / too small / low score / NaN score / extreme or NaN yaw becomes UNSAFE → original + escalate, and no model is called. | `test_unsafe_or_uncertain_returns_original_and_escalates` (6 cases) |
| A4 exception, OOM, invalid output, or no face on the output → original | `test_a4_failure_falls_back_to_original` (5 cases) |

Full suite in `fes-dev:cpu-p4`: see §6.

## 5. How to reproduce

```bash
FES_IMAGE=fes-dev:cpu-p4 scripts/dock.sh python scripts/p5_baseline_freeze.py --check     # exit 0 = baseline intact
FES_IMAGE=fes-dev:cpu-p4 scripts/dock.sh python -m pytest -q tests/test_p5_baseline.py
# full re-freeze incl. A4 pixel check (only for a deliberate new baseline version):
FES_IMAGE=fes-dev:cpu-p4 scripts/dock.sh python scripts/p5_baseline_freeze.py --write --a4-check 2 --image-id "$(docker image inspect fes-dev:cpu-p4 --format '{{.Id}}')"
```

**Rule for later Phase 5 tasks:** new service code goes in new modules.
- Changing any file listed in `code_sha256_text` or `frozen_configs` fails the baseline tests.
- A deliberate change needs a new manifest version, with a report of what changed.

## 6. Validation

- Baseline regression tests: **19 passed**.
- Full suite: see the commit message for the exact count.

## 7. Limitations

- All baseline evidence is synthetic (FairFace chips). The frozen thresholds remain **invalid for real-world traffic**.
- The A4 pixel check covers 2 pairs, not all 73 routed dev pairs. The decision-level replay covers all 1 792 pairs.
- The T7 router's timeout is thread-based and cannot kill a hung model. This is recorded as-is; process isolation belongs to 5.4.
- Licensing is unchanged. A4 weights and YuNet are REVIEW_REQUIRED for production.
