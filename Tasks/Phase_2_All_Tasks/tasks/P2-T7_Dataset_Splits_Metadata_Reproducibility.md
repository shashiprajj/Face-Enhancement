# P2-T7 — Dataset Splits, Metadata & Reproducibility

## Objective
Create leakage-safe benchmark splits and a canonical metadata framework.

## Work
1. Define train/dev/test roles only where actually needed; distinguish training from evaluation.
2. Prevent identity/source leakage across evaluation boundaries.
3. Use deterministic split rules and fixed seeds.
4. Define canonical metadata including sample/source IDs, identity/group ID where legitimately available, source path, crop path, quality metrics, degradation profile/parameters, detector/alignment information and license/source status.
5. Separate raw, derived and benchmark data.
6. Make reconstruction script/config driven rather than manual.
7. Validate missing files, duplicate IDs, broken links, leakage and metadata inconsistencies.

## Deliverables
- `docs/phase_2/P2-T7_Splits_Metadata_Reproducibility.md`
- Canonical metadata schema.
- Split-generation and validation scripts.
- Final manifests.

## Done When
The benchmark dataset and splits can be reproduced from documented scripts/configuration.
