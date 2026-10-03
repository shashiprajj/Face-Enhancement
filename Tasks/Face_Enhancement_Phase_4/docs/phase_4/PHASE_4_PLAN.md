# Phase 4 Plan — Real-World Validation & Enhancement Gating

## Objective
Determine whether enhancement improves identity utility on real-world degraded faces and whether the system can safely decide when to enhance, when to leave the input untouched, and when to reject/escalate.

## Task order
1. P4-T1 Real-world dataset acquisition & governance
2. P4-T2 Real-world degradation taxonomy & quality metadata
3. P4-T3 Second identity evaluator
4. P4-T4 Real-world benchmark execution
5. P4-T5 Human identity/fabrication review
6. P4-T6 Input-quality / Do-Not-Enhance gate
7. P4-T7 Selective candidate routing
8. P4-T8 Evidence synthesis & Phase 4 gate

## Cross-cutting gates
G-1 GPU performance
G-2 second identity evaluator
G-3 real-world dataset/legal approval
G-4 production licence review
G-5 human review

## P4-T1 — Real-world dataset acquisition & governance
Create a legally usable, representative real-world evaluation population. Record source, permission/licence, identity provenance, degradation type, reference/degraded pairing, split assignment and hashes. Prevent leakage and freeze the final manifest before final evaluation.

Candidate real-world conditions may include ID/document captures, phone-camera images, screenshots/re-encodes, motion/defocus blur, low resolution, noise, compression, exposure, glare, occlusion, poor crop and mixed degradation.

## P4-T2 — Degradation taxonomy & quality metadata
Version a taxonomy for observed real-world degradation. Record where valid: image size, face size, blur/sharpness, noise, brightness, contrast, compression, pose, occlusion, detector confidence and alignment quality. Calibration must use development data only.

## P4-T3 — Second identity evaluator
Investigate a genuinely independent identity evaluator. Before use, document model/weights source, licence and commercial-use status, preprocessing, runtime, reproducibility and limitations. Pin versions and hashes. Add adapter, validation and tests.

## P4-T4 — Real-world benchmark
Reuse the Phase 3 harness and keep B0 untouched input as a mandatory baseline. Evaluate eligible candidates. Measure identity similarity, TAR/FAR where supported, identity flips/rescues, evaluator agreement, failures, degradation-family/severity behavior, perceptual metrics only where a valid reference exists, and performance. Use identity-level clustered confidence intervals and reproducibility checks.

## P4-T5 — Human review
Create blinded review focused on identity preservation and fabrication, not attractiveness. Review same-person confidence, identity substitution, invented/removed attributes, age/appearance changes, artefacts and confidence. Include controls and reproducible sample selection.

## P4-T6 — Do-Not-Enhance gate
Design a deterministic quality gate using measurable properties such as face size, blur, noise, exposure, contrast, compression, detection confidence, alignment, pose and occlusion. Initial outputs: DO_NOT_ENHANCE, ENHANCE_CANDIDATE, REJECT/ESCALATE. Calibrate on development data and freeze before final evaluation. Optimize identity utility, not LPIPS alone.

## P4-T7 — Selective routing
Compare:
1. always B0
2. always one enhancement candidate
3. quality-gated enhancement
4. quality-gated degradation-specific routing

Only retain a route when reproducible evidence shows identity benefit/protection with acceptable fabrication/failure risk. Investigate Phase 3 findings: A5 dark-image collapse, extreme low-resolution behavior, mild-input harm and exposure/contrast limitations.

## P4-T8 — Evidence synthesis & gate
Produce one decision record covering real-world evidence, second evaluator, human review, quality gate, routing, GPU, licensing, fairness limitations and reproducibility.

GREEN: evidence is sufficient to proceed to a later production-selection phase.
YELLOW: useful evidence exists but a production gate remains unresolved.
RED: unacceptable identity risk/failure cannot currently be controlled.

## Non-goals
- no Feature-Match integration
- no always-on enhancement
- no premature production model selection
- no ranking based only on perceptual metrics
- no fairness claims from underpowered samples
- no treating missing evidence as a pass
