# Phase 3 — Document & Results Index

Face Enhancement Service — benchmarking and baseline evaluation on `fes-syn-v1` (synthetic). **Evaluation only:** no model is selected, approved, or recommended. Gate: [Phase 3 Final Report](../../reports/Phase_3_Final_Report.md).

## Documents (in execution order)

| Order | Task | Documents |
|---|---|---|
| 1 | P3-T1 | [P3-T1_Benchmark_Harness_Specification.md](P3-T1_Benchmark_Harness_Specification.md) · `configs/phase3/benchmark_p3.yaml` · `configs/phase3/schema/*.json` |
| 2 | P3-T2 | [P3-T2_Model_Adapters.md](P3-T2_Model_Adapters.md) · `src/fes_bench/` · `third_party/` (+ `PROVENANCE.md`) · `data/manifests/phase3/model_validation_report.json` |
| 3 | P3-T3 | [P3-T3_Baseline_Evaluation.md](P3-T3_Baseline_Evaluation.md) |
| 4 | P3-T4 | [P3-T4_Candidate_Inference.md](P3-T4_Candidate_Inference.md) |
| 5 | P3-T5 | [P3-T5_Image_Quality_Evaluation.md](P3-T5_Image_Quality_Evaluation.md) |
| 6 | P3-T6 | [P3-T6_Identity_Preservation_Evaluation.md](P3-T6_Identity_Preservation_Evaluation.md) |
| 7 | P3-T7 | [P3-T7_Failure_Analysis.md](P3-T7_Failure_Analysis.md) |
| 8 | P3-T8 | [P3-T8_CPU_Performance_Benchmark.md](P3-T8_CPU_Performance_Benchmark.md) |
| 9 | P3-T9 | [P3-T9_Evidence_Synthesis.md](P3-T9_Evidence_Synthesis.md) |
| 10 | P3-T10 | [../../reports/Phase_3_Final_Report.md](../../reports/Phase_3_Final_Report.md) · this index |

## Results Index (all under `data/manifests/phase3/`; tracked)

| What | Path |
|---|---|
| Populations (trial / main / noise_floor / impostor_pool) + build report | `population_*.csv.gz`, `populations_report.json` |
| Run manifests (per config: per-pair status, output + pixel SHA-256, meta) | `runs/{trial,main}/<config>.csv.gz`, `<config>_run.json`, `<config>.partial.jsonl` |
| Per-pair metrics | `metrics/{trial,main}/<config>.csv.gz`, `identity_calibration.json` |
| Evaluator noise floor | `identity_noise_floor.json` |
| Aggregates (cluster-bootstrap CIs) | `results/{trial,main}/cells.csv.gz`, `deltas.csv.gz`, `identity.csv.gz`, `strata.csv.gz`, `summary.json` |
| Rendered tables | `results/main/tables.md` |
| Failure taxonomy flags / stats / sample index | `results/main/failure_flags.csv.gz`, `failure_stats.csv`, `sample_index.csv` |
| CPU performance | `perf/cpu_benchmark_t2.json` |
| Reproducibility evidence | `repro/rerun*.json`, `repro/a5_stage_probe.json`, `repro/p3t10_checks.json` |
| Model-load validation | `model_validation_report.json` |

**Not tracked (git-ignored):** enhanced outputs `experiments/phase3/outputs/<pop>/<config>/<cell>/<ref>.png`, contact sheets `experiments/phase3/outputs/sheets/`, and logs `logs/phase3*`.

## Scripts

`scripts/p3_build_populations.py` → `p3_validate_models.py` → `p3_run_inference.py` / `p3_run_all.sh` → `p3_evaluate.py` → `p3_aggregate.py` → `p3_report_tables.py` → `p3_failure_index.py` · `p3_contact_sheet.py` · `p3_cpu_benchmark.py` · `p3_rerun_check.py` · `p3_a5_stage_probe.py` · `p3_identity_noise_floor.py`

Phase 0 ([index](../phase_0/PHASE_0_INDEX.md)) remains authoritative for requirements. Phase 1 and 2: [P1 index](../phase_1/PHASE_1_INDEX.md), [P2 index](../phase_2/PHASE_2_INDEX.md).
