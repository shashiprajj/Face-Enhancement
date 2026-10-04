# T7 routing evaluation — `p4dev` (SYNTHETIC, dev; IN-SAMPLE dev reference)

T6 `b8ee2c4309e5…` · T7 `2f026f229039…` · table {"BLUR": "A4", "COMPRESS": "B0", "LOWRES": "B0", "MIXED": "B0", "NOISE": "A4", "NONE": "B0", "PHOTOMETRIC": "B0"}

Decisions {'DO_NOT_ENHANCE': 607, 'UNSAFE_OR_UNCERTAIN': 186, 'ENHANCE_CANDIDATE': 103} · routes {'B0': 823, 'A4': 73} · used {'B0': 823, 'A4': 73} · fallbacks {} · buckets {'BLUR': 38, 'COMPRESS': 16, 'NOISE': 35, 'LOWRES': 5, 'PHOTOMETRIC': 8, 'NONE': 1}

| Policy | enh. rate | ΔE01 | ΔE07 | conf. benefit | harm (any) | conf. harm | rescues/flips E01 | rescues/flips E07 | MILD harm / enhanced |
|---|---|---|---|---|---|---|---|---|---|
| 1_always_original | 0.000 | +0.0000 [+0.0000, +0.0000] ≈ | +0.0000 [+0.0000, +0.0000] ≈ | 0 | 0 | 0 | 0/0 | 0/0 | 0 / 0 |
| 2_always_A1 | 1.000 | -0.0518 [-0.0670, -0.0375] ▼ | -0.0064 [-0.0088, -0.0042] ▼ | 62 | 739 | 567 | 13/10 | 14/6 | 252 / 252 |
| 2_always_A2 | 1.000 | -0.0629 [-0.0741, -0.0526] ▼ | -0.0075 [-0.0093, -0.0058] ▼ | 28 | 817 | 617 | 15/12 | 12/11 | 252 / 252 |
| 2_always_A3 | 1.000 | -0.0253 [-0.0327, -0.0182] ▼ | -0.0042 [-0.0055, -0.0030] ▼ | 55 | 560 | 286 | 10/25 | 8/28 | 158 / 252 |
| 2_always_A4 | 1.000 | -0.0099 [-0.0168, -0.0027] ▼ | -0.0016 [-0.0027, -0.0004] ▼ | 111 | 458 | 262 | 11/28 | 10/20 | 141 / 252 |
| 2_always_A5 | 1.000 | -0.0752 [-0.0910, -0.0617] ▼ | -0.0109 [-0.0132, -0.0087] ▼ | 31 | 756 | 542 | 6/44 | 6/50 | 223 / 252 |
| 3_T6+A1 | 0.115 | -0.0055 [-0.0093, -0.0022] ▼ | -0.0005 [-0.0010, -0.0002] ▼ | 6 | 77 | 43 | 0/0 | 0/1 | 0 / 0 |
| 3_T6+A2 | 0.115 | -0.0050 [-0.0076, -0.0027] ▼ | -0.0006 [-0.0009, -0.0002] ▼ | 3 | 92 | 43 | 0/0 | 0/0 | 0 / 0 |
| 3_T6+A3 | 0.115 | +0.0019 [+0.0005, +0.0032] ▲ | +0.0001 [-0.0001, +0.0002] ≈ | 17 | 24 | 5 | 0/0 | 0/1 | 0 / 0 |
| 3_T6+A4 | 0.115 | +0.0048 [+0.0033, +0.0065] ▲ | +0.0006 [+0.0004, +0.0008] ▲ | 46 | 12 | 5 | 0/0 | 0/0 | 0 / 0 |
| 3_T6+A5 | 0.115 | -0.0029 [-0.0050, -0.0010] ▼ | -0.0005 [-0.0007, -0.0003] ▼ | 6 | 64 | 30 | 0/0 | 0/2 | 0 / 0 |
| 4_T7_route | 0.082 | +0.0049 [+0.0038, +0.0060] ▲ | +0.0006 [+0.0004, +0.0007] ▲ | 38 | 6 | 1 | 0/0 | 0/0 | 0 / 0 |

Evaluator agreement on routed pairs: {'n': 73, 'crosstab': {'BENEFIT|BENEFIT': 38, 'BENEFIT|NEUTRAL': 18, 'HARM|BENEFIT': 1, 'HARM|HARM': 1, 'NEUTRAL|BENEFIT': 2, 'NEUTRAL|HARM': 4, 'NEUTRAL|NEUTRAL': 9}, 'exact_agreement': 0.6575}

## T7 route by family (benefit − harm(any), E01/E07 marks)

- lowres: enhanced 0/112, 0−0, ≈/≈
- gaussblur: enhanced 27/112, 22−1, ▲/▲
- motionblur: enhanced 11/112, 0−5, ▼/≈
- noise: enhanced 35/112, 16−0, ▲/▲
- jpeg: enhanced 0/112, 0−0, ≈/≈
- underexposure: enhanced 0/84, 0−0, ≈/≈
- lowcontrast: enhanced 0/84, 0−0, ≈/≈
- combined: enhanced 0/84, 0−0, ≈/≈
- realistic2: enhanced 0/84, 0−0, ≈/≈
