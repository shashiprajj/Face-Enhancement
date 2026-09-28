# P2-T4 — Controlled Low-Quality / Degradation Generator

## Objective
Generate realistic, reproducible low-quality versions of clean reference faces for paired enhancement testing.

## Degradation Families
Evaluate appropriate profiles for:
- low resolution/downscaling
- Gaussian/defocus blur
- motion blur
- noise
- JPEG/compression artifacts
- brightness/exposure degradation
- contrast degradation where justified
- combinations of degradations

## Work
1. Never modify the clean reference.
2. Use deterministic seeds.
3. Define named profiles and severity levels.
4. Record exact degradation parameters per generated image.
5. Keep dimensions and formats compatible with candidate enhancement models.
6. Validate generated images and collect basic quality statistics.
7. Ensure degradation does not intentionally alter identity.

## Deliverables
- `docs/phase_2/P2-T4_Degradation_Design.md`
- Reproducible degradation code/configuration.
- Paired clean/degraded manifest.
- Generation statistics.

## Done When
Every degraded image can be reproduced from its clean source and recorded degradation profile.
