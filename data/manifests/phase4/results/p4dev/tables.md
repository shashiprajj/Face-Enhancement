# P4-T4 multi-evaluator results — `p4dev` (SYNTHETIC, dev split)

n = 896 pairs, 28 references. δ: E01 0.02154, E07 0.00258. FAR 1e-2 thresholds: E01 0.2843, E07 0.9297. ▲/▼/≈: 95 % cluster-bootstrap CI above / below / containing 0. Scores of the two evaluators are never pooled.

## Pooled

| Config | ΔE01 [CI] | ΔE07 [CI] | conf. benefit | conf. harm | disagree | TAR E01 B0→c | TAR E07 B0→c | flips/rescues E01 | flips/rescues E07 | closer-to-impostor (both) | κ verdicts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | -0.0007 [-0.0011, -0.0003] ▼ | -0.0001 [-0.0002, +0.0000] ≈ | 3 | 7 | 3 | 0.960→0.961 | 0.921→0.917 | 0/1 | 5/2 | 11 | 0.3109 |
| B2 | -0.0403 [-0.0462, -0.0353] ▼ | -0.0044 [-0.0057, -0.0032] ▼ | 26 | 444 | 14 | 0.960→0.946 | 0.921→0.912 | 15/3 | 12/4 | 13 | 0.4366 |
| B2c | -0.1032 [-0.1136, -0.0931] ▼ | -0.0152 [-0.0179, -0.0127] ▼ | 0 | 832 | 24 | 0.960→0.926 | 0.921→0.884 | 31/1 | 36/3 | 20 | 0.1147 |
| A1 | -0.0518 [-0.0670, -0.0375] ▼ | -0.0064 [-0.0088, -0.0042] ▼ | 62 | 567 | 71 | 0.960→0.963 | 0.921→0.930 | 10/13 | 6/14 | 16 | 0.3739 |
| A2 | -0.0629 [-0.0741, -0.0526] ▼ | -0.0075 [-0.0093, -0.0058] ▼ | 28 | 617 | 76 | 0.960→0.963 | 0.921→0.922 | 12/15 | 11/12 | 13 | 0.2131 |
| A3 | -0.0253 [-0.0327, -0.0182] ▼ | -0.0042 [-0.0055, -0.0030] ▼ | 55 | 286 | 47 | 0.960→0.943 | 0.921→0.898 | 25/10 | 28/8 | 25 | 0.3683 |
| A4 | -0.0099 [-0.0168, -0.0027] ▼ | -0.0016 [-0.0027, -0.0004] ▼ | 111 | 262 | 42 | 0.960→0.941 | 0.921→0.910 | 28/11 | 20/10 | 26 | 0.5011 |
| A5 | -0.0752 [-0.0910, -0.0617] ▼ | -0.0109 [-0.0132, -0.0087] ▼ | 31 | 542 | 51 | 0.960→0.917 | 0.921→0.872 | 44/6 | 50/6 | 40 | 0.3568 |

## ΔID by family (E01 mark / E07 mark; confirmed benefit − confirmed harm count)

| Config | lowres | gaussblur | motionblur | noise | jpeg | underexposure | lowcontrast | combined | realistic2 |
|---|---|---|---|---|---|---|---|---|---|
| B1 | ≈/≈ 3−5 | ≈/≈ 0−0 | ≈/≈ 0−0 | ≈/≈ 0−0 | ≈/≈ 0−0 | ≈/≈ 0−0 | ≈/≈ 0−0 | ▼/≈ 0−0 | ▼/≈ 0−2 |
| B2 | ▼/▼ 1−51 | ▼/▼ 0−90 | ▼/▼ 0−73 | ▼/▼ 0−44 | ▼/▼ 8−30 | ▼/▼ 0−64 | ▼/▼ 0−80 | ≈/≈ 8−5 | ≈/≈ 9−7 |
| B2c | ▼/▼ 0−101 | ▼/▼ 0−109 | ▼/▼ 0−103 | ▼/▼ 0−111 | ▼/▼ 0−107 | ▼/▼ 0−79 | ▼/▼ 0−77 | ▼/▼ 0−72 | ▼/▼ 0−73 |
| A1 | ▼/≈ 11−54 | ▼/▼ 3−81 | ▼/▼ 8−65 | ▲/▼ 23−43 | ▼/▼ 1−89 | ▼/▼ 0−81 | ▼/▼ 0−73 | ▼/▼ 5−40 | ▼/▼ 11−41 |
| A2 | ▼/▼ 12−77 | ▼/▼ 2−78 | ▼/▼ 3−68 | ▲/▼ 3−44 | ▼/▼ 0−96 | ▼/▼ 0−83 | ▼/▼ 0−82 | ▼/▼ 3−44 | ▼/▼ 5−45 |
| A3 | ▼/▼ 0−93 | ▼/▼ 0−29 | ▼/▼ 0−28 | ▲/▲ 41−5 | ▼/▼ 4−18 | ▼/▼ 0−20 | ▼/▼ 0−34 | ▼/≈ 4−32 | ▼/▼ 6−27 |
| A4 | ▼/▼ 0−80 | ≈/▲ 28−19 | ▼/≈ 8−18 | ▲/▲ 54−4 | ▼/▼ 8−19 | ▼/▼ 0−33 | ▼/▼ 0−35 | ▼/≈ 3−24 | ▼/≈ 10−30 |
| A5 | ▼/▼ 0−109 | ▼/▼ 0−76 | ▼/▼ 0−62 | ▲/▼ 15−16 | ▼/▼ 8−35 | ▼/▼ 0−75 | ▼/▼ 0−75 | ▼/▼ 3−47 | ▼/▼ 5−47 |

## ΔID by severity (E01 mark / E07 mark; confirmed benefit − confirmed harm count)

| Config | mild | moderate | severe | extreme |
|---|---|---|---|---|
| B1 | ≈/≈ 0−0 | ▼/≈ 0−0 | ≈/≈ 0−2 | ≈/≈ 3−5 |
| B2 | ▼/▼ 0−144 | ▼/▼ 12−120 | ▼/▼ 6−112 | ▼/▼ 8−68 |
| B2c | ▼/▼ 0−250 | ▼/▼ 0−247 | ▼/▼ 0−221 | ▼/▼ 0−114 |
| A1 | ▼/▼ 0−237 | ▼/▼ 7−176 | ▼/▼ 19−124 | ▲/≈ 36−30 |
| A2 | ▼/▼ 0−241 | ▼/▼ 4−191 | ▼/▼ 6−142 | ≈/▼ 18−43 |
| A3 | ▼/▼ 0−78 | ▼/▼ 11−90 | ▼/▼ 22−89 | ≈/▼ 22−29 |
| A4 | ▼/▼ 1−75 | ▼/≈ 16−79 | ▼/▼ 31−81 | ▲/≈ 63−27 |
| A5 | ▼/▼ 0−174 | ▼/▼ 4−160 | ▼/▼ 9−156 | ▼/▼ 18−52 |

## Detection contract on the degraded input (non-oracle)

| Cell | OK rate |
|---|---|
| combined-mild | 1.000 |
| combined-moderate | 1.000 |
| combined-severe | 0.143 |
| gaussblur-extreme | 1.000 |
| gaussblur-mild | 1.000 |
| gaussblur-moderate | 1.000 |
| gaussblur-severe | 1.000 |
| jpeg-extreme | 1.000 |
| jpeg-mild | 1.000 |
| jpeg-moderate | 1.000 |
| jpeg-severe | 1.000 |
| lowcontrast-mild | 1.000 |
| lowcontrast-moderate | 1.000 |
| lowcontrast-severe | 0.893 |
| lowres-extreme | 0.000 |
| lowres-mild | 1.000 |
| lowres-moderate | 1.000 |
| lowres-severe | 0.286 |
| motionblur-extreme | 1.000 |
| motionblur-mild | 1.000 |
| motionblur-moderate | 1.000 |
| motionblur-severe | 1.000 |
| noise-extreme | 0.000 |
| noise-mild | 1.000 |
| noise-moderate | 0.964 |
| noise-severe | 0.857 |
| realistic2-mild | 1.000 |
| realistic2-moderate | 1.000 |
| realistic2-severe | 0.107 |
| underexposure-mild | 1.000 |
| underexposure-moderate | 1.000 |
| underexposure-severe | 1.000 |

B0 TAR by input detection: {'E01': {'detected': 0.9987, 'not_detected': 0.7368}, 'E07': {'detected': 0.9856, 'not_detected': 0.5489}}
