# R5-7: Guard Calibration and full_image Framing Correction — Report

Date: 2026-10-07 · Task: `Tasks/Phase-7/R5-7_Guard_Calibration_FullImage_Correction_PROMPT.txt`
Baseline: `9b9f5e3` (R5-6 YELLOW) · service implementation `9f55f33` (R5-5)

## 1. Executive summary

R5-7 found the causes of both R5-6 issues on DEV data and built a candidate correction. The unchanged R5-6 harness re-run showed **large benefit gains**, but it also showed that **the candidate returns outputs with small rainbow/dotted artifacts** that R5-6 never returned: at least **8 visually confirmed** on archive TEST. Under the task's rules this is a safety failure. **The candidate is therefore rejected.** The live service code and config are restored to the R5-6 baseline (`p5-nafnet-route-1.0.0`, seal `eb6fe101…`). The candidate is preserved as a patch for a follow-up.

**What was learned (DEV, valid):**

1. **Guard:** the R5-6 guard's chroma-p99 > 10 rule blocks about 9% of NAFNet outputs, and those are some of its best ones (DEV 10–20 band: ΔSFace +0.19 to +0.23, 1 clearly harmed in 684). Destroyed faces (large rainbow blocks) sit at p99 ≥ 22.7 and p99.9 ≥ 40.5.
2. **full_image coverage:** the black fill outside the image is the main cause, not a side issue as R5-6 concluded. 40% of face_chip-eligible DEV faces get a partly out-of-image chip, and 95% of those are lost in full_image (lapd1 ×4 from the black edges). R5-6's "96.6% of enhanced chips have no padding" was survivorship bias: padded chips almost never reached NAFNet.
3. **full_image benefit:** NAFNet internally **zero-pads** its input to a multiple of 16. On the service's 99–135-px chips this black border halves the gain. Reflect-padding first restores it (DEV ΔSFace +0.056 → +0.113, with harm and guard results no worse).

**What failed.** The DEV guard labels came from inspecting outputs with p99 > 20 plus a 10–20 band sample. They missed a **small-patch** artifact mode: a few-pixel rainbow patch with low p99 (6–15) and moderate p99.9 (26–36). That overlaps the DEV range of good outputs, so no p99/p99.9 threshold separates it. On TEST, the relaxed guard and the new full_image path let this mode through.

## 2. R5-6 frozen baseline

R5-6 is preserved unchanged as the control, in `data/manifests/phase5_rapid/r5_6/*.csv.gz` (no prefix) and `summary.json`.

| Run | Baseline result |
|---|---|
| Archive TEST face_chip | eligible 8,147 · output 7,393 · guard block 754 · ΔSFace +0.149 · policy +0.057 |
| Archive TEST full_image | eligible 5,769 · output 5,654 · guard block 115 · ΔSFace +0.053 · policy +0.016 |
| ztest | 0 NAFNet attempts |
| Real photos | 4 NAFNet · 2 A4 · 194 ORIGINAL |

HEAD was `9b9f5e3` on `main`, and `src`/`configs` were clean at the start.

## 3. DEV calibration methodology

DEV data only:
- the archive **`val` split** (19,290 pairs, never used before);
- the R5-5 DEV probe.

**Not used for any decision:** archive TEST, ztest, the 200 real photos, R5-6 results.

| Collection | Scope | Script | Output |
|---|---|---|---|
| Guard | face_chip path, all 19,290 val pairs; 8,065 NAFNet-eligible → NAFNet → every guard metric + candidate signals (chroma p50/p95/p99.9, high-chroma fractions, high-frequency chroma/luma in and out, saturation) | `scripts/r5_7_dev_collect.py --run val_guard` | `val_guard.csv.gz` |
| Framing | both paths on a fixed 3,000-pair val sample (seed 0): geometry (bbox, IOD, chip side, padding fraction, image cover), all gate features per path, NAFNet on each eligible path, output-vs-output region PSNR, identity | `--run val_frame` | `val_frame.csv.gz` |
| Padding experiment | 920 DEV chips eligible in full_image | `scripts/r5_7_frame_experiment.py` | — |
| Fill experiment | 1,091 DEV padded chips | `scripts/r5_7_fill_experiment.py` | — |

**Labels.** The pre-registered label was R5-ARCHIVE's GT chroma excess > 10. It **failed**: all 30 inspected GT-flagged outputs in the 10–20 band are clean restorations (ΔSFace up to +0.65). It was replaced by **visual DEV labels**:
- every eligible output with p99 > 20 or no face on the output: 21 inspected, 15 destroyed;
- the 30 GT-flagged plus 12 random outputs from the 10–20 band: no artifact.

## 4. Current guard analysis (DEV, 8,065 eligible)

| chroma p99 band | n | Mean ΔSFace | Clearly improved | Clearly harmed | Visually destroyed |
|---|---|---|---|---|---|
| ≤ 10 (passes) | 7,361 | +0.149 | 2,919 | 7 | 0 inspected |
| 10–15 (blocked) | 609 | **+0.193** | 300 | 1 | 0 / 42 inspected (whole 10–20 band) |
| 15–20 (blocked) | 75 | **+0.232** | 44 | 0 | |
| 20–30 | 8 | +0.125 | 2 | 0 | 3 |
| 30+ | 12 | −0.02 | 0 | 2 | 12 |

- **Current guard on DEV:** blocks 705 outputs; all 15 destroyed are among them.
- **Separating signal:** p99.9 separated the inspected sets (good ≤ 32.3, destroyed ≥ 40.5).

## 5. Guard calibration decision

**Candidate v1.1.0** (rules fixed before computing the values):
- `chroma_p999_max` = midpoint of the DEV gap = **36.4** (new rule);
- `chroma_p99_max` = 0.9 × the lowest destroyed p99 = **20.4**.

On DEV it blocks 20 outputs instead of 705, still blocks 15/15 destroyed ones (each rule catches all 15 on its own), and recovers 685 good outputs (ΔSFace +0.197, 345 clearly improved, 1 clearly harmed).

**Final decision: REJECTED after the TEST re-run (§10–12).** v1.1.0 was not tuned on TEST. It failed because the DEV labels did not cover the small-patch mode. The sealed v1.1.0 config is kept only as evidence: `Planning/R5-7_candidate/r5_nafnet_route_v1.1.0_REJECTED.json`, seal `8fb7bc39…`. The **live guard stays at v1.0.0.**

## 6. full_image framing investigation (DEV, 3,000-pair sample)

| | face_chip path | full_image path |
|---|---|---|
| NAFNet-eligible | 1,283 | 920 (eligible in both: 629; lost 654; gained 291) |
| lapd1_c among face_chip-eligible, 10 / 50 / 90 % | 0.0008 / 0.0016 / 0.0024 | 0.0008 / 0.0021 / **0.0099** |
| IOD, median | 40.1 | 39.5 (face scale ≈ unchanged; chip side 100–135, covering 61–90% of the image) |
| Blockiness, median | 1.07 | 0.98 |

**Which feature changes the decision.** lapd1 crosses the strong-blur limit in 538 cases, IOD in 126 (jitter at the 35.35-px edge), blockiness in 83.

## 7. Quantified cause analysis

1. **Padding (coverage).** 510 of 1,283 face_chip-eligible faces (40%) get a partly out-of-image chip.

   | | Padded chip | Unpadded chip |
   |---|---|---|
   | Lost in full_image | 486 (**95.3%**) | 168 (21.7%) |
   | lapd1 ratio full_image / face_chip (median) | **×4.0** | ×0.94 |
   | Main reasons | NOT_BLUR / MILD_BLUR 370, COMPRESSION 73 | mostly LOW_FACE_SIZE |

   The black edges read as sharpness and block structure. **74% of the coverage loss is caused by the black padding.**

2. **Face scale.** It changes little (IOD ratio median 0.97). It causes the LOW_FACE_SIZE losses at the threshold edge (103), a measurement-boundary effect.

3. **NAFNet input (benefit).** On faces eligible in both paths, the service's chip gives ΔSFace +0.054, while the face_chip output restricted to the same region gives +0.139. The two outputs differ inside that region (median PSNR 28.6 dB). The cause is NAFNet's **internal zero padding** to a multiple of 16 (`NAFNet_arch.check_image_size`, `F.pad`, constant 0). Only 62 of 920 chips are already a multiple of 16.

4. **Crop information.** Restricting enhancement to the chip region costs only ~0.02 ΔSFace, so the crop keeps the identity-relevant area.

## 8. Framing correction (candidate, rejected)

| Change | DEV result |
|---|---|
| **NAFNet reflect-pad16** (model wrapper): reflect-pad to a multiple of 16, crop back; /16 inputs bit-identical | ΔSFace +0.056 → **+0.113**, Δdlib +0.007 → +0.015, harmed 6 → 1, guard pass 916 → 918 (920 chips) |
| **Out-of-image reflect fill, NAFNet route only:** same chip, reflected content instead of black for the gate and NAFNet, output re-blacked (geometry and returned chip unchanged; A4 route untouched) | eligible 25 → 435 of 1,091 padded DEV chips; 434 returned (ΔSFace +0.110, 0 clearly harmed); guard blocked 1 destroyed face |

**Both changes are part of the rejected candidate** (§12). They are preserved in `Planning/R5-7_candidate/r5_7_candidate.patch` (code, config, the calibration script and 12 tests; re-applies cleanly to HEAD).

## 9. Regression results

| Build | pytest (Docker, `FES_RUN_NAFNET=1`) | UI logic (node) |
|---|---|---|
| R5-7 candidate | **263 passed, 2 skipped, 0 failed** (incl. 12 new R5-7 tests) | 21/21 |
| Restored R5-6 state (final) | **256 passed, 2 skipped, 0 failed** (identical to R5-6) | 21/21 |

Live checks on the candidate service: `/health` OK; `/ready` reported 3 replicas, seal `8fb7bc39`, 0 worker restarts; `/v1/enhance` gave NAFNet PASS and `route_mode=a4` worked; no errors in the service logs.

## 10. R5-6 re-run results (candidate build; unchanged harness, datasets and methodology)

Every request returned HTTP 200 with valid output. ORIGINAL outputs were byte-identical to the input. There were 0 A4 and 0 NAFNet worker restarts on all 3 replicas, and 0 timeouts or OOMs. One harness crash (`ENOMEM` reading an input under Docker memory pressure) was resumed by a retry wrapper; the harness itself was unchanged.

| Run | N | NAFNet out | A4 | ORIGINAL | Guard fail | Notes |
|---|---|---|---|---|---|---|
| r57_test_face_chip | 19,290 | 8,127 | 0 | 11,163 | 20 | ΔSFace +0.154 · policy +0.065 |
| r57_test_full_image | 19,290 | 8,439 | 0 | 10,851 | 31 | ΔSFace +0.113 · policy +0.049 |
| r57_ztest ×2 | 100 + 100 | 0 | 0 | 200 | 0 | 0 flips, 0 new false accepts |
| r57_real | 200 | 4 | 2 | 194 | 0 | identical routing to R5-6 |

## 11. Before / after (R5-6 vs R5-7 candidate)

| Metric | R5-6 | R5-7 candidate | Change |
|---|---|---|---|
| Guard block rate, face_chip | 9.3% (754) | 0.25% (20) | −734 |
| Guard block rate, full_image | 2.0% (115) | 0.37% (31) | −84 |
| NAFNet eligibility, face_chip / full_image | 8,147 / 5,769 | 8,147 / **8,470** | +2,701 (full_image) |
| Archive face_chip ΔSFace (enhanced) | +0.149 | +0.154 | +0.005 |
| Archive full_image ΔSFace (enhanced) | +0.053 | **+0.113** | +0.060 |
| Whole-policy ΔSFace, face_chip / full_image | +0.057 / +0.016 | +0.065 / **+0.049** | +0.008 / **+0.033** |
| Clearly improved, face_chip / full_image | 2,884 / 806 | 3,269 / 2,721 | + |
| Clearly harmed, face_chip / full_image | 6 / 28 | 7 / 20 | ≈ |
| **Artifact outputs returned (visual)** | **0** | **≥ 8** | **regression** |
| ztest NAFNet attempts | 0 | 0 | = |
| Real photos, NAFNet / A4 / ORIGINAL | 4 / 2 / 194 | 4 / 2 / 194 | = |
| Service failures / restarts | 0 / 0 | 0 / 0 | = |
| NAFNet inference median (128-px) | 0.51 s | 0.50–0.84 s | ≈ (WelOrbit load varied) |

## 12. Safety analysis

**Contained, as in R5-6:**
- ztest: 0 attempts; all 14 earlier N1 artifacts returned ORIGINAL.
- Real photos: routing identical. img154's output changed only through reflect padding and was visually crisper with no artifact.
- The large "destroyed face" blocks are all still blocked; all guard blocks reproduce (20/20, 31/31).

**Not contained.** A p99.9 scan of every returned output was run for both builds:

| Returned outputs with p99.9 > 25 / 28 / 30 / 32 | R5-6 | R5-7 candidate |
|---|---|---|
| face_chip | 1 / 1 / 0 / 0 | 13 / 5 / 3 / 1 |
| full_image | 1 / 0 / 0 / 0 | 10 / 5 / 5 / 3 |

Visual inspection of the top 16 per mode found **small rainbow/dotted patches** (on the cheek, an eye, the chin, the nose or the mouth; the faces otherwise intact):
- full_image: test/image_8964, 96161, 97266, 90856, 85126, 96475;
- face_chip: 95636 (p99.9 35.9) and **88924 (p99.9 26.8)**.

The R5-6 top outputs show none. Two of the full_image cases have p99 = 6, so they would pass even the old guard: the new full_image path (fill and padding) exposes this mode too. R5-ARCHIVE's GT chroma signal does not catch them (GT excess 1–4).

**Conclusion.** Returning these outputs is not acceptable, and no DEV-derived threshold separates them. Fixing it by choosing thresholds after seeing TEST would be test-time tuning. **The candidate is withdrawn, and the service is restored to the validated R5-6 state.**

## 13. Runtime and resources

- Candidate service: 3 replicas, ~0.95–1.2 GB each, NAFNet load 7–15 s.
- Throughput: ~4.3–9.4 requests/s, depending on CPU contention from WelOrbit, which the user stopped mid-run.
- Background shells were reaped twice for low host memory. Long jobs were moved into detached containers, which survived.
- No model downloads.

## 14. Limitations

- The DEV visual labels covered p99 > 20 and a 10–20 band sample only, not the p99.9 20–36 band. That is how the small-patch mode was missed.
- The TEST artifact count is a lower bound: only the top 16 outputs per mode by p99.9 were inspected.
- All identity results are SFace/dlib proxies on synthetic blur, not Feature-Match.
- The fill and padding experiments ran with the candidate code; re-running them requires applying the patch.

## 15. Final verdict: **RED for the R5-7 candidate**, with no unsafe change committed

**Commit contents.** The commit contains evaluation scripts, DEV and re-run manifests, the candidate patch and this report. **No service code or config change is live:** `src/`, `configs/` and `tests/` are identical to `9b9f5e3`. The service remains at the R5-6 YELLOW state, which is safe and validated but has the known coverage limits.

## 16. Recommendation for the next task (R5-8, small and DEV-only)

1. **Build a local-patch artifact detector on DEV.**
   - Candidate signals: windowed (e.g. 8×8) maximum of the chroma change, or of high-frequency colour energy, and connected blobs of high chroma with high-frequency texture.
   - Label **all** DEV `val` outputs with p99.9 > ~20 (a few hundred) plus a random sample. Use a fresh labelled DEV portion for checking.
   - Freeze the detector before any TEST run.
2. **Re-introduce the two measured framing fixes** (reflect-pad16 and out-of-image fill) behind the new detector, then re-run the unchanged R5-6 harness once.
3. **Optional and independent:** NAFNet reflect-pad16 alone (largest benefit, smallest change), validated with the new detector.

**Answers:**
1. **Did the guard improve?** On DEV, yes (705 → 20 blocks with 15/15 destroyed caught). In service, it let small-patch artifacts through, so it was not adopted.
2. **Did full_image routing improve?** In the candidate, yes (5,769 → 8,470 eligible). It was not adopted for the same safety reason.
3. **Did identity benefit improve?** In the candidate, yes: full_image ΔSFace +0.053 → +0.113, policy +0.016 → +0.049. Not live.
4. **Did safety remain intact?** For the live service, yes: it is restored to R5-6. For the candidate, no: at least 8 small-patch artifact outputs.
5. **What limitation remains?** The guard cannot detect small localized rainbow patches. full_image coverage is still reduced by black padding and NAFNet's zero padding in the live build.
6. **Are we ready to stop enhancement experimentation?** Not yet. One focused DEV-only task (R5-8: local-patch detector, then re-adopt the measured framing fixes) is justified by the size of the measured gain.
7. **Is the next task actual Feature-Match validation?** It can proceed on the R5-6 build at any time, since that build is safe. It stays blocked on Feature-Match access and ≥ 20 real identities. The preferred order is R5-8 first, so that the validated build carries the ~3× full_image benefit.

No Feature-Match benefit, production approval or commercial licensing approval is claimed; licensing remains REVIEW_REQUIRED.
