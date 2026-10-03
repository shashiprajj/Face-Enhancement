# P4-T3 — Second Identity Evaluator (E07 dlib ResNet)

| Field | Value |
|---|---|
| Phase / Task | Phase 4 / P4-T3 |
| Date | 2026-10-03 |
| Status | **E07 VALIDATED (7/7 pre-registered criteria) for evaluation-only use.** G-2 is addressed technically. Production-licence confirmation of evaluation use rests with the reviewer (G-4). |
| Adapter | `src/fes_bench/evaluators.py` (`DlibResNet`) · tests `tests/test_p4_evaluator.py` (3 passed) |
| Image | `fes-dev:cpu-p4` = `fes-dev:cpu-p2` + dlib 20.0.1 compiled from the hash-pinned official sdist (`docker/Dockerfile.cpu-p4`, `requirements-p4.txt`) |
| Validation | `scripts/p4_validate_evaluator.py` → `data/manifests/phase4/evaluator_E07_validation.json` |
| Scoring | `scripts/p4_evaluate_e07.py` → `data/manifests/phase4/metrics_e07/<population>/` |
| Asset | A16 in `docs/phase_1/model_acquisition_manifest.csv` |
| User decision | 2026-10-03: use dlib ResNet as the second evaluator |

---

## 1. Investigation (before implementation)

| Item | Finding |
|---|---|
| Candidates re-screened | E02 AdaFace, E03 resnet18_110 and E06 facenet-pytorch remain REVIEW_REQUIRED (weights have no explicit terms). E04 InsightFace and E05 EdgeFace remain INELIGIBLE (NC). New: **E07 dlib ResNet** and **E08 OpenFace nn4.small2** (Apache-2.0, "source code and trained … model files"; weak, LFW ≈ 93 %; not adopted). |
| Capability | ResNet-29 metric-learning network, 128-d descriptor. LFW 99.38 % (author). |
| Code licence | dlib: Boost Software License 1.0 |
| Weight licence | dlib-models README @ `fd81b630`: *"anyone can do whatever they want with these model files as I've released them into the public domain."* Copy kept in `models/E07_dlib_resnet/LICENSE_STATEMENT.md`. |
| Commercial / production use | **Evaluation-only**, on the same basis as E01 SFace: explicit permissive weight terms. It is never part of the FES runtime (P0 C1). |
| Training data (R4) | About 3 M faces from FaceScrub + the VGG dataset + a web scrape (README). The inheritance caveat is recorded for the reviewer (G-4). |
| Preprocessing | 150 × 150 RGB chip in dlib `get_face_chip` geometry (padding 0.25). |
| Runtime | About 150 ms per embedding on the shared CPU (single thread, under load) |
| Reproducibility | Pinned dlib version, weights SHA-256 checked at load, bit-identical output on repeat |
| Limitations | Not trained for very low resolution. The cosine scale is compressed (dev impostor median 0.85, §4). The training-set overlap with FairFace identities is unknown (both are web-sourced). |

**Independence from SFace (E01):**

| | E01 SFace | E07 dlib |
|---|---|---|
| Architecture | MobileFaceNet | ResNet-29 |
| Loss | SFace sigmoid-constrained hypersphere loss | metric learning (pair hinge) |
| Training data | not stated (likely CASIA/MS1M) | FaceScrub + VGG + scrape |
| Framework | ONNX / OpenCV | dlib C++ |
| Alignment | 5-point warp to a 112 px crop | dlib chip geometry |

None of the Phase 3 candidates uses either network as a training loss.

## 2. Adapter (output contract)

- **Chip route** (every E-frame image): `cv2.resize(224 → 150, INTER_AREA)`. FairFace chips **are** dlib `get_face_chip` outputs with padding 0.25, so this reproduces dlib's own geometry exactly. It is identical for every image of a pair (the P3-T1 fixed-alignment principle), and E07 identity is independent of detector failures.
- **Template route** (any image): a similarity fit of 5 landmarks (YuNet convention) to `DLIB_CHIP_TEMPLATE_150`. The template is the mean landmarks of the 112 dev references, derived on dev and frozen in code. The 5-point dlib shape predictor, whose licence is unstated (P1-T4), is **not** used.
- **Output:** a 128-d L2-normalised descriptor, compared by cosine. Embeddings stay in memory and are never persisted.

## 3. Validation — criteria fixed before the run, dev data only

| # | Criterion | Result |
|---|---|---|
| V1 | weights SHA-256 + dlib 20.0.1 pinned | ✅ |
| V2 | 128-d, finite, raw norm in [0.5, 2] | ✅ raw norm 1.23–1.53 |
| V3 | bit-identical repeat (112 refs) | ✅ |
| V4 | chip vs template route, median cos ≥ 0.95 | ✅ median 0.994, min 0.968 |
| V5 | noise floor p99 (1−cos) < 0.05 | ✅ jpeg q90 0.0002 · down2-up 0.0003 · 1-px shift 0.0026 |
| V6 | mild-cell genuine vs dev impostor AUC ≥ 0.95 | ✅ 1.000 |
| V7 | independence reported | ✅ (§4) |

Data: the 112 dev_pilot references and the p4dev B0 inputs (896 pairs). **SYNTHETIC, dev only.**

## 4. Behaviour and Agreement With E01 (dev)

| Severity | AUC E01 | AUC E07 |
|---|---|---|
| mild | 1.000 | 1.000 |
| moderate | 1.000 | 0.999 |
| severe | 0.988 | 0.977 |
| extreme | 0.990 | 0.990 |

| Agreement measure | Value | Reading |
|---|---|---|
| Spearman, ref-vs-B0 cos (896 pairs) | 0.963 | Both evaluators see the same degradation-induced identity loss |
| Spearman, dev impostor pairs (6,216) | 0.384 | Each evaluator ranks *different* people differently, i.e. independent errors on identity structure |
| Dev impostor median / FAR 1e-2 threshold | E01 0.043 / 0.284 · E07 0.849 / 0.930 | |

**Scale rule.** E07 cosines are compressed toward 1, so the two evaluators' scores and deltas are **never pooled or compared numerically**. Each evaluator's effects are judged against **its own** dev noise floor (δ) and its own same-split FAR 1e-2 threshold. Multi-evaluator evidence = agreement of the *decisions* (benefit / harm / flip), not averaged scores.

Per-evaluator identity-change threshold δ (TH-1 method; p99 of 1−cos under identity-neutral perturbations, max over perturbations, dev):

| Evaluator | δ |
|---|---|
| E01 | 0.0215 |
| E07 | 0.0026 |

These are used in P4-T4/T6/T7 as the minimum change counted as a real identity effect. They are measured, not tuned.

## 5. Use in Phase 4

E07 scores are used only after `validated: true` (asserted in `p4_evaluate_e07.py`). They are used for:
- the Phase 3 `main` outputs (P4-T4 multi-evaluator re-analysis);
- `p4dev` (gate and routing calibration);
- the real-world population once G-3 opens.

## 6. Remaining Gate Items

- G-4: the reviewer confirms evaluation-only use of the E01/E07 weights given R4 training-data inheritance.
- A third, low-resolution-specialised evaluator (e.g. AdaFace) remains REVIEW_REQUIRED.
