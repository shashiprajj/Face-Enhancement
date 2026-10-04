# Phase 4 — Document & Evidence Index

Real-world validation and enhancement gating. **Gate: 🟡 YELLOW** ([final report](../../reports/Phase_4_Final_Report.md) · [gate matrix](PHASE_4_GATE_MATRIX.md) · [Phase 5 scope](PHASE_5_SCOPE.md)). Task prompts: `Tasks/Face_Enhancement_Phase_4/`, `Tasks/Face_Enhancement_Phase_4_Completion/`.

## Reports (execution order)

| Task | Report | Commit |
|---|---|---|
| P4-T1 Dataset & governance | [P4-T1_Real_World_Dataset_Governance.md](P4-T1_Real_World_Dataset_Governance.md) | `42f451c` |
| P4-T2 Taxonomy & quality metadata | [P4-T2_Degradation_Taxonomy_Quality_Metadata.md](P4-T2_Degradation_Taxonomy_Quality_Metadata.md) | `f2b4c0b` |
| P4-T3 Second evaluator | [P4-T3_Second_Identity_Evaluator.md](P4-T3_Second_Identity_Evaluator.md) | `52b18bd` |
| P4-T4 Benchmark | [P4-T4_Real_World_Benchmark.md](P4-T4_Real_World_Benchmark.md) | `b613aa5` |
| P4-T5 Human review | [P4-T5_Human_Review.md](P4-T5_Human_Review.md) | `0f1ab8e` |
| Infra resume | `data/manifests/phase4/runs/p4dev/{RUNTIME_LOG.md, RESUME_LOG.md, validation.json}` | `4ebceb6` |
| P4-T6 Do-Not-Enhance gate | [P4_T6_REPORT.md](P4_T6_REPORT.md) | `e504dfa` (freeze), `bc5ef80` |
| P4-T7 Selective routing | [P4_T7_REPORT.md](P4_T7_REPORT.md) | `785a4b2` (freeze), `ad79455` |
| P4-T8 Synthesis & gate | [P4_T8_FINAL_REPORT.md](P4_T8_FINAL_REPORT.md) = [reports/Phase_4_Final_Report.md](../../reports/Phase_4_Final_Report.md) | final commit |

## Frozen configurations

| File | Content | Seal / hash |
|---|---|---|
| `configs/phase4/p4_protocol.yaml` | Phase 4 protocol (extends `p3-protocol-1.0.1`) | — |
| `configs/phase4/taxonomy_v1.yaml` | `p4-taxonomy-1.0.0` | sha in `quality/*_report.json` |
| `configs/phase4/t6_gate.json` | Frozen Do-Not-Enhance gate | seal `b8ee2c43…` |
| `configs/phase4/t7_routing.json` | Frozen routing (on T6 seal) | seal `2f026f22…` |

## Evidence (under `data/manifests/phase4/`)

| What | Path |
|---|---|
| Real-world register + blocked evidence | `../../registry/p4_realworld_sources.csv`, `realworld/*/BLOCKED.json` |
| Quality metadata (dev/test) | `quality/{p4dev,main}.csv.gz` + reports |
| Second-evaluator validation | `evaluator_E07_validation.json` |
| Dev population + runs (9 × 896) + validation | `population_p4dev.csv.gz`, `runs/p4dev/` |
| Per-pair metrics, SFace / dlib | `metrics/p4dev/`, `metrics_e07/{main,p4dev}/` (main SFace: `../phase3/metrics/main/`) |
| Two-evaluator analysis | `results/{main,p4dev}/{summary.json, verdicts.csv.gz, tables.md}` |
| T6 calibration / evaluation / decisions | `t6/` |
| T7 calibration / router validation / evaluation / decision log | `t7/` |
| Human-review kit key + status | `review/` (kit images local only: `experiments/phase4/outputs/review_kit/rv1/`) |
| Reproducibility | `repro/p4t4_checks.json`; T6/T7 re-runs byte-identical (decision logs `420e0d0d…`, `79094637…`) |
| Performance | `perf/gate_latency.json`, `t7/router_validation.json` |

## Code and tests

`src/fes_bench/{quality,evaluators,p4eval,gate,routing}.py` · scripts `scripts/p4_*.py|sh` · tests `tests/test_p4_*.py`. Full suite: **133 passed** (`fes-dev:cpu-p4`).
