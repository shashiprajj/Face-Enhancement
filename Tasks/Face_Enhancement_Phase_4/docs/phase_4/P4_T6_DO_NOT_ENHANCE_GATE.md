# P4-T6 — Input-Quality / Do-Not-Enhance Gate

Candidate features:
face size, blur/sharpness, noise, exposure, contrast, compression, detector confidence, alignment, pose, occlusion.

Actions:
DO_NOT_ENHANCE
ENHANCE_CANDIDATE
REJECT / ESCALATE

Calibrate on development data, freeze thresholds, then evaluate on untouched data. Measure false enhancement, missed enhancement, identity gain/loss and reject rate. Optimize identity utility rather than perceptual quality.
