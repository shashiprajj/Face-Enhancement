# P4-T4 multi-evaluator results — `main` (SYNTHETIC, test split)

n = 896 pairs, 28 references. δ: E01 0.02154, E07 0.00258. FAR 1e-2 thresholds: E01 0.3133, E07 0.9336. ▲/▼/≈: 95 % cluster-bootstrap CI above / below / containing 0. Scores of the two evaluators are never pooled.

## Pooled

| Config | ΔE01 [CI] | ΔE07 [CI] | conf. benefit | conf. harm | disagree | TAR E01 B0→c | TAR E07 B0→c | flips/rescues E01 | flips/rescues E07 | closer-to-impostor (both) | κ verdicts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | -0.0006 [-0.0011, -0.0002] ▼ | -0.0001 [-0.0002, +0.0001] ≈ | 5 | 6 | 4 | 0.959→0.954 | 0.911→0.911 | 5/1 | 1/1 | 7 | 0.3016 |
| B2 | -0.0360 [-0.0393, -0.0328] ▼ | -0.0038 [-0.0050, -0.0029] ▼ | 22 | 427 | 12 | 0.959→0.950 | 0.911→0.906 | 13/5 | 9/5 | 5 | 0.356 |
| B2c | -0.1084 [-0.1206, -0.0973] ▼ | -0.0146 [-0.0162, -0.0130] ▼ | 3 | 835 | 14 | 0.959→0.921 | 0.911→0.875 | 36/2 | 33/1 | 21 | 0.1832 |
| A1 | -0.0492 [-0.0703, -0.0325] ▼ | -0.0054 [-0.0080, -0.0031] ▼ | 56 | 540 | 76 | 0.959→0.963 | 0.911→0.920 | 12/16 | 9/17 | 13 | 0.367 |
| A2 | -0.0608 [-0.0773, -0.0466] ▼ | -0.0065 [-0.0085, -0.0047] ▼ | 25 | 584 | 106 | 0.959→0.955 | 0.911→0.902 | 13/10 | 18/10 | 15 | 0.2529 |
| A3 | -0.0205 [-0.0265, -0.0141] ▼ | -0.0040 [-0.0054, -0.0028] ▼ | 58 | 301 | 49 | 0.959→0.933 | 0.911→0.889 | 38/15 | 27/8 | 23 | 0.4657 |
| A4 | -0.0078 [-0.0139, -0.0010] ▼ | -0.0013 [-0.0024, -0.0002] ▼ | 120 | 260 | 34 | 0.959→0.933 | 0.911→0.903 | 38/15 | 18/11 | 20 | 0.5393 |
| A5 | -0.0802 [-0.1189, -0.0559] ▼ | -0.0110 [-0.0171, -0.0068] ▼ | 45 | 481 | 50 | 0.959→0.891 | 0.911→0.845 | 72/11 | 65/6 | 61 | 0.4579 |

## ΔID by family (E01 mark / E07 mark; confirmed benefit − confirmed harm count)

| Config | lowres | gaussblur | motionblur | noise | jpeg | underexposure | lowcontrast | combined | realistic2 |
|---|---|---|---|---|---|---|---|---|---|
| B1 | ≈/≈ 4−4 | ≈/≈ 0−0 | ≈/≈ 0−0 | ≈/≈ 0−0 | ≈/≈ 0−0 | ≈/≈ 0−0 | ≈/≈ 0−0 | ▼/≈ 0−1 | ▼/▼ 1−1 |
| B2 | ▼/▼ 1−51 | ▼/▼ 0−93 | ▼/▼ 0−66 | ▼/▼ 1−44 | ▼/▼ 7−21 | ▼/▼ 0−58 | ▼/▼ 0−80 | ≈/≈ 6−6 | ≈/≈ 7−8 |
| B2c | ▼/▼ 1−99 | ▼/▼ 0−110 | ▼/▼ 0−105 | ▼/▼ 0−107 | ▼/▼ 0−109 | ▼/▼ 0−79 | ▼/▼ 2−77 | ▼/▼ 0−74 | ▼/▼ 0−75 |
| A1 | ▼/≈ 9−58 | ▼/▼ 2−76 | ▼/▼ 7−58 | ▲/≈ 25−37 | ▼/▼ 0−84 | ▼/▼ 0−83 | ▼/▼ 0−74 | ▼/≈ 7−36 | ▼/≈ 6−34 |
| A2 | ▼/▼ 7−70 | ▼/▼ 0−72 | ▼/▼ 4−63 | ▲/▼ 5−37 | ▼/▼ 0−88 | ▼/▼ 0−80 | ▼/▼ 0−82 | ▼/≈ 6−46 | ▼/▼ 3−46 |
| A3 | ▼/▼ 0−96 | ▼/▼ 0−27 | ▼/▼ 0−25 | ▲/▲ 46−3 | ▼/▼ 6−22 | ▼/▼ 0−27 | ▼/▼ 0−35 | ▼/▼ 3−29 | ▼/▼ 3−37 |
| A4 | ▼/▼ 0−76 | ≈/▲ 28−22 | ▼/≈ 2−13 | ▲/▲ 63−5 | ≈/≈ 12−21 | ▼/▼ 0−31 | ▼/▼ 0−33 | ▼/≈ 10−28 | ▼/≈ 5−31 |
| A5 | ▼/▼ 0−107 | ▼/▼ 0−63 | ▼/▼ 0−55 | ▲/≈ 25−10 | ▼/▼ 11−32 | ▼/▼ 0−64 | ▼/▼ 0−67 | ▼/▼ 5−41 | ▼/▼ 4−42 |

## ΔID by severity (E01 mark / E07 mark; confirmed benefit − confirmed harm count)

| Config | mild | moderate | severe | extreme |
|---|---|---|---|---|
| B1 | ▲/▼ 0−0 | ▼/≈ 0−0 | ▼/≈ 2−2 | ≈/≈ 3−4 |
| B2 | ▼/▼ 0−138 | ▼/▼ 10−119 | ▼/▼ 4−107 | ▼/▼ 8−63 |
| B2c | ▼/▼ 0−252 | ▼/▼ 0−244 | ▼/▼ 2−225 | ▼/▼ 1−114 |
| A1 | ▼/▼ 0−226 | ▼/▼ 9−163 | ▼/▼ 14−127 | ▲/≈ 33−24 |
| A2 | ▼/▼ 0−234 | ▼/▼ 3−182 | ▼/▼ 8−133 | ≈/≈ 14−35 |
| A3 | ▼/▼ 0−83 | ▼/▼ 13−91 | ▼/▼ 19−92 | ▲/▼ 26−35 |
| A4 | ▼/▼ 0−76 | ▼/▼ 19−84 | ▼/≈ 32−71 | ▲/▲ 69−29 |
| A5 | ▼/▼ 0−148 | ▼/▼ 8−145 | ▼/▼ 13−133 | ▼/▼ 24−55 |

## Detection contract on the degraded input (non-oracle)

| Cell | OK rate |
|---|---|
| combined-mild | 1.000 |
| combined-moderate | 1.000 |
| combined-severe | 0.179 |
| gaussblur-extreme | 1.000 |
| gaussblur-mild | 1.000 |
| gaussblur-moderate | 1.000 |
| gaussblur-severe | 1.000 |
| jpeg-extreme | 0.964 |
| jpeg-mild | 1.000 |
| jpeg-moderate | 1.000 |
| jpeg-severe | 1.000 |
| lowcontrast-mild | 1.000 |
| lowcontrast-moderate | 1.000 |
| lowcontrast-severe | 0.929 |
| lowres-extreme | 0.000 |
| lowres-mild | 1.000 |
| lowres-moderate | 1.000 |
| lowres-severe | 0.250 |
| motionblur-extreme | 1.000 |
| motionblur-mild | 1.000 |
| motionblur-moderate | 1.000 |
| motionblur-severe | 1.000 |
| noise-extreme | 0.000 |
| noise-mild | 1.000 |
| noise-moderate | 0.964 |
| noise-severe | 0.821 |
| realistic2-mild | 1.000 |
| realistic2-moderate | 1.000 |
| realistic2-severe | 0.143 |
| underexposure-mild | 1.000 |
| underexposure-moderate | 0.964 |
| underexposure-severe | 0.964 |

B0 TAR by input detection: {'E01': {'detected': 0.9934, 'not_detected': 0.763}, 'E07': {'detected': 0.9829, 'not_detected': 0.5037}}
