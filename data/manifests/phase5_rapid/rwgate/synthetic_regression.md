# R5-3 synthetic regression — frozen T6/T7 vs rw gate (A4 outputs already scored)

## `main` — TEST (not used for calibration)

| Policy | A4 routed | ΔID SFace | ΔID dlib | conf. benefit | harm (any) | flips SFace/dlib | MILD enhanced |
|---|---|---|---|---|---|---|---|
| T6/T7 | 63 | +0.0044 [+0.0033, +0.0055] ▲ | +0.0005 [+0.0003, +0.0007] ▲ | 38 | 3 | 0/0 | 0 |
| rw | 50 | +0.0036 [+0.0023, +0.0048] ▲ | +0.0003 [+0.0002, +0.0005] ▲ | 29 | 5 | 0/0 | 3 |

Routes changed: 27 · added by rw: {'combined-mild:CONFIRMED_HARM': 1, 'lowres-mild:CONFIRMED_HARM': 1, 'noise-mild:CONFIRMED_HARM': 1, 'motionblur-extreme:NEUTRAL': 1, 'gaussblur-extreme:CONFIRMED_BENEFIT': 2, 'motionblur-extreme:CONFIRMED_HARM': 1} · removed by rw: {'gaussblur-extreme:CONFIRMED_BENEFIT': 10, 'gaussblur-severe:NEUTRAL': 3, 'motionblur-extreme:CONFIRMED_BENEFIT': 1, 'motionblur-severe:PARTIAL_BENEFIT': 1, 'motionblur-extreme:CONFIRMED_HARM': 1, 'gaussblur-severe:PARTIAL_HARM': 1, 'motionblur-extreme:NEUTRAL': 1, 'motionblur-moderate:NEUTRAL': 1, 'motionblur-severe:NEUTRAL': 1}

rw reasons: {'degraded_no_a4_evidence': 219, 'no_valid_face': 135, 'acceptable': 189, 'compression_no_a4_evidence': 80, 'photometric_no_a4_evidence': 108, 'strong_noise': 33, 'low_detection_confidence': 17, 'insufficient_face_size': 7, 'strong_blur': 17, 'alignment_unreliable': 87, 'degraded_below_a4_size': 2, 'beyond_evidence_extreme_blur': 2}

## `p4dev` — DEV (in-sample)

| Policy | A4 routed | ΔID SFace | ΔID dlib | conf. benefit | harm (any) | flips SFace/dlib | MILD enhanced |
|---|---|---|---|---|---|---|---|
| T6/T7 | 73 | +0.0049 [+0.0038, +0.0060] ▲ | +0.0006 [+0.0004, +0.0007] ▲ | 38 | 6 | 0/0 | 0 |
| rw | 55 | +0.0044 [+0.0034, +0.0053] ▲ | +0.0004 [+0.0003, +0.0005] ▲ | 31 | 1 | 0/0 | 0 |

Routes changed: 22 · added by rw: {'noise-severe:CONFIRMED_BENEFIT': 1, 'motionblur-extreme:NEUTRAL': 1} · removed by rw: {'gaussblur-extreme:CONFIRMED_BENEFIT': 8, 'gaussblur-severe:PARTIAL_BENEFIT': 1, 'motionblur-severe:DISAGREE': 1, 'motionblur-severe:PARTIAL_HARM': 2, 'motionblur-severe:NEUTRAL': 1, 'gaussblur-severe:NEUTRAL': 3, 'motionblur-moderate:NEUTRAL': 1, 'motionblur-extreme:PARTIAL_BENEFIT': 1, 'gaussblur-severe:PARTIAL_HARM': 1, 'motionblur-moderate:CONFIRMED_HARM': 1}

rw reasons: {'degraded_no_a4_evidence': 230, 'low_detection_confidence': 16, 'strong_blur': 19, 'acceptable': 205, 'compression_no_a4_evidence': 77, 'alignment_unreliable': 37, 'photometric_no_a4_evidence': 136, 'no_valid_face': 133, 'strong_noise': 36, 'insufficient_face_size': 7}

