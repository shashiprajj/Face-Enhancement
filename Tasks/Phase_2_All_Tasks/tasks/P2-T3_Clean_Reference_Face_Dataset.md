# P2-T3 — Clean Reference Face Dataset

## Objective
Create clean/high-quality reference faces to act as the target side of later enhancement evaluation.

## Work
1. Define objective filtering rules for usable reference images.
2. Use Phase 1 detector/alignment components where appropriate.
3. Create standardized face crops/aligned faces while retaining source links.
4. Do not enhance, beautify, sharpen or hallucinate reference images.
5. Record source path, sample/identity ID where available, crop coordinates, alignment and quality information.
6. Keep evaluation identities separated to prevent leakage.
7. Preserve the original image and make every transformation traceable.

## Deliverables
- `docs/phase_2/P2-T3_Clean_Reference_Dataset.md`
- Reference-face generation pipeline.
- Reference manifest and quality statistics.

## Done When
A reproducible clean reference set exists and every derived face can be traced back to its source.
