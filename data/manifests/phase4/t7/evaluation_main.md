# T7 routing evaluation — `main` (SYNTHETIC, test; FINAL single evaluation of frozen routing)

T6 `b8ee2c4309e5…` · T7 `2f026f229039…` · table {"BLUR": "A4", "COMPRESS": "B0", "LOWRES": "B0", "MIXED": "B0", "NOISE": "A4", "NONE": "B0", "PHOTOMETRIC": "B0"}

Decisions {'DO_NOT_ENHANCE': 565, 'UNSAFE_OR_UNCERTAIN': 239, 'ENHANCE_CANDIDATE': 92} · routes {'B0': 833, 'A4': 63} · used {'B0': 833, 'A4': 63} · fallbacks {} · buckets {'BLUR': 30, 'NOISE': 33, 'PHOTOMETRIC': 6, 'COMPRESS': 12, 'LOWRES': 8, 'MIXED': 2, 'NONE': 1}

| Policy | enh. rate | ΔE01 | ΔE07 | conf. benefit | harm (any) | conf. harm | rescues/flips E01 | rescues/flips E07 | MILD harm / enhanced |
|---|---|---|---|---|---|---|---|---|---|
| 1_always_original | 0.000 | +0.0000 [+0.0000, +0.0000] ≈ | +0.0000 [+0.0000, +0.0000] ≈ | 0 | 0 | 0 | 0/0 | 0/0 | 0 / 0 |
| 2_always_A1 | 1.000 | -0.0492 [-0.0703, -0.0325] ▼ | -0.0054 [-0.0080, -0.0031] ▼ | 56 | 726 | 540 | 16/12 | 17/9 | 245 / 252 |
| 2_always_A2 | 1.000 | -0.0608 [-0.0773, -0.0466] ▼ | -0.0065 [-0.0085, -0.0047] ▼ | 25 | 795 | 584 | 10/13 | 10/18 | 245 / 252 |
| 2_always_A3 | 1.000 | -0.0205 [-0.0265, -0.0141] ▼ | -0.0040 [-0.0054, -0.0028] ▼ | 58 | 533 | 301 | 15/38 | 8/27 | 145 / 252 |
| 2_always_A4 | 1.000 | -0.0078 [-0.0139, -0.0010] ▼ | -0.0013 [-0.0024, -0.0002] ▼ | 120 | 440 | 260 | 15/38 | 11/18 | 135 / 252 |
| 2_always_A5 | 1.000 | -0.0802 [-0.1189, -0.0559] ▼ | -0.0110 [-0.0171, -0.0068] ▼ | 45 | 677 | 481 | 11/72 | 6/65 | 195 / 252 |
| 3_T6+A1 | 0.103 | -0.0044 [-0.0090, -0.0009] ▼ | -0.0003 [-0.0008, +0.0001] ≈ | 6 | 66 | 33 | 0/0 | 1/0 | 0 / 0 |
| 3_T6+A2 | 0.103 | -0.0046 [-0.0083, -0.0016] ▼ | -0.0004 [-0.0007, -0.0001] ▼ | 0 | 74 | 35 | 0/2 | 0/1 | 0 / 0 |
| 3_T6+A3 | 0.103 | +0.0016 [+0.0000, +0.0030] ▲ | -0.0001 [-0.0005, +0.0002] ≈ | 17 | 25 | 13 | 1/1 | 0/3 | 0 / 0 |
| 3_T6+A4 | 0.103 | +0.0041 [+0.0023, +0.0057] ▲ | +0.0006 [+0.0003, +0.0009] ▲ | 45 | 12 | 7 | 0/1 | 1/1 | 0 / 0 |
| 3_T6+A5 | 0.103 | -0.0053 [-0.0152, +0.0006] ≈ | -0.0009 [-0.0022, -0.0001] ▼ | 10 | 52 | 30 | 0/6 | 0/7 | 0 / 0 |
| 4_T7_route | 0.070 | +0.0044 [+0.0033, +0.0055] ▲ | +0.0005 [+0.0003, +0.0007] ▲ | 38 | 3 | 2 | 0/0 | 0/0 | 0 / 0 |

Evaluator agreement on routed pairs: {'n': 63, 'crosstab': {'BENEFIT|BENEFIT': 37, 'BENEFIT|NEUTRAL': 14, 'HARM|HARM': 2, 'HARM|NEUTRAL': 1, 'NEUTRAL|BENEFIT': 1, 'NEUTRAL|NEUTRAL': 8}, 'exact_agreement': 0.746}

## T7 route by family (benefit − harm(any), E01/E07 marks)

- lowres: enhanced 0/112, 0−0, ≈/≈
- gaussblur: enhanced 23/112, 19−1, ▲/▲
- motionblur: enhanced 6/112, 1−1, ≈/≈
- noise: enhanced 33/112, 18−0, ▲/▲
- jpeg: enhanced 0/112, 0−0, ≈/≈
- underexposure: enhanced 0/84, 0−0, ≈/≈
- lowcontrast: enhanced 1/84, 0−1, ≈/≈
- combined: enhanced 0/84, 0−0, ≈/≈
- realistic2: enhanced 0/84, 0−0, ≈/≈
