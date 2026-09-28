# Phase 2 — Document Index

Face Enhancement Service — dataset acquisition & preparation. No enhancement model is selected; no API; no Feature-Match.

| Order | Task | Document | Main artifacts |
|---|---|---|---|
| 1 | P2-T1 | [P2-T1_Dataset_Inventory.md](P2-T1_Dataset_Inventory.md) | `data/registry/phase2_inventory.csv`, directory/naming conventions, `scripts/dock.sh`, image `fes-dev:cpu-p2` |
| 2 | P2-T2 | [P2-T2_FairFace_Verification.md](P2-T2_FairFace_Verification.md) | `scripts/p2_fairface_prepare.py`, `configs/phase2/fairface.yaml`, `data/manifests/fairface_files_v1.csv.gz`, `fairface_validation_v1.json` |
| 3 | P2-T3 | [P2-T3_Clean_Reference_Dataset.md](P2-T3_Clean_Reference_Dataset.md) | `scripts/p2_build_references.py`, `configs/phase2/references.yaml`, `references_v1.csv.gz` + report, framing/size probes |
| 4 | P2-T4 | [P2-T4_Degradation_Design.md](P2-T4_Degradation_Design.md) | `src/fes_data/degrade.py`, `scripts/p2_degrade.py`, `degrade_pilot_v1.csv.gz` + report |
| 5 | P2-T5 | [P2-T5_Real_World_Data_Plan.md](P2-T5_Real_World_Data_Plan.md) | `configs/phase2/realworld_collection_v1.yaml`, `scripts/p2_validate_collection.py`, [templates/](templates/) |
| 6 | P2-T6 | [P2-T6_Preprocessing_Pipeline.md](P2-T6_Preprocessing_Pipeline.md) | `src/fes_data/preprocess.py`, `configs/phase2/preprocess.yaml`, `scripts/p2_preprocess.py`, `tests/` |
| 7 | P2-T7 | [P2-T7_Splits_Metadata_Reproducibility.md](P2-T7_Splits_Metadata_Reproducibility.md) | `scripts/p2_build_splits.py`, `configs/phase2/benchmark_v1.yaml`, `configs/phase2/schema/`, `data/manifests/benchmark/fes-syn-v1/`, `scripts/p2_validate_benchmark.py`, `scripts/p2_reproduce.sh` |
| 8 | P2-T8 | [../../reports/Phase_2_Final_Report.md](../../reports/Phase_2_Final_Report.md) | `data/manifests/benchmark/fes-syn-v1/validation_report.json`, `logs/p2t8_*` evidence (copied to `data/manifests/benchmark/fes-syn-v1/evidence/`) |

**Rebuild everything from raw:** `scripts/p2_reproduce.sh`
**Run tests:** `scripts/dock.sh python -m pytest -q -p no:cacheprovider tests`

Phase 0 (requirements) and Phase 1 (candidates, licenses) remain authoritative: [Phase 0 index](../phase_0/PHASE_0_INDEX.md) · [Phase 1 index](../phase_1/PHASE_1_INDEX.md).
