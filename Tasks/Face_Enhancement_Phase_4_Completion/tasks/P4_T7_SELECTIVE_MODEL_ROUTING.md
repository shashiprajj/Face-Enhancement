# P4-T7 — Selective Model Routing

## Objective
Route each input to original/B0 or an appropriate already-benchmarked enhancement candidate.

## Rules
- Original/B0 is always valid.
- Never enhance merely because a model exists.
- Never tune using evaluation/test results.
- If uncertain, fall back to original.

## Steps
1. Implement the routing contract:
   quality analysis → T6 gate → original or degradation-specific route.
2. Define evidence-based degradation buckets.
3. Initially route only among:
   - A1 RestoreFormer++
   - A2 VQFR
   - A3 Real-ESRGAN
   - A4 SwinIR-L
   - A5 FBCNN → NAFNet-SIDD → Real-ESRGAN
4. Define model failure, timeout, invalid-output and resource-failure fallbacks.
5. Validate determinism and resource behavior on development data.
6. Freeze routing.
7. Evaluate once on untouched evaluation data.
8. Report enhancement/no-enhancement rate, route distribution, identity delta, rescue/damage, evaluator agreement and failure cases.

## Required deliverables
- routing implementation
- config
- tests
- decision manifest/log
- P4_T7_REPORT.md
