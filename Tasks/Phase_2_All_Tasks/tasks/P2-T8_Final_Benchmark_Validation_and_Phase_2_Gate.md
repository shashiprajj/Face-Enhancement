# P2-T8 — Final Benchmark Validation & Phase 2 Gate

## Objective
Validate the complete Phase 2 dataset foundation and issue the Phase 2 gate.

## Work
Run end-to-end validation:
source → detection/crop/alignment → clean reference → degradation → metadata/split.

Check:
- file integrity
- counts
- duplicates
- identity/source leakage
- metadata completeness
- reproducibility
- degradation reproducibility
- preprocessing failure handling
- license/source traceability

Compare results with Phase 0 requirements and Phase 1 open items. Do not invent thresholds marked TBD. Clearly separate completed, partial, blocked and legally unresolved items. Decide GREEN/YELLOW/RED using evidence. Do not select the final production model.

## Deliverables
- `reports/Phase_2_Final_Report.md`
- `docs/phase_2/PHASE_2_INDEX.md`
- Final manifests and validation evidence.
- Explicit Phase 2 gate and open-item list.

## Done When
A reproducible benchmark-data foundation exists and the report clearly states what is ready for the next phase.
