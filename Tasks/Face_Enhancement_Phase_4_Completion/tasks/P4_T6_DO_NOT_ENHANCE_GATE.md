# P4-T6 — Do-Not-Enhance Gate

## Objective
Build a deterministic gate deciding whether a face should be enhanced at all.

## Steps
1. Inspect existing partial outputs and resume incomplete inference; do not recompute completed pairs.
2. Use conservative worker counts; prefer one worker for memory-heavy VQFR/A2.
3. Define implemented, testable quality features such as:
   - effective face resolution
   - blur/focus
   - noise
   - exposure/brightness
   - contrast
   - compression/artifacts
   - detection confidence
   - alignment validity
4. Define decisions:
   - DO_NOT_ENHANCE
   - ENHANCE_CANDIDATE
   - UNSAFE_OR_UNCERTAIN
5. Calibrate thresholds on development data only.
6. Measure identity preservation/rescue and identity damage; explicitly measure mild-input damage.
7. Freeze thresholds, feature definitions and logic before evaluation.
8. Evaluate once on untouched evaluation data.
9. Add deterministic unit/integration tests.

## Required deliverables
- implementation
- tests
- frozen config/calibration artifact
- evaluation artifact
- P4_T6_REPORT.md

## Gate
T6 is complete only when the gate is frozen and evaluation is reproducible.
