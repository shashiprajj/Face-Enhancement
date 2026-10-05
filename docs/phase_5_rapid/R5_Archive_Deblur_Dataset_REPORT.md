# R5-ARCHIVE — Paired face-deblurring dataset: evaluation report

**Date:** 2026-10-06
**Script:** `scripts/r5_archive_eval.py`
**Results:** `data/manifests/phase5_rapid/archive/` (`summary.json`, `stage1.csv.gz`, `stage2_*.csv.gz`, `realcheck_N1.*`)
**Visual sheets** (git-ignored): `experiments/phase5_rapid/outputs/archive/archive_comparison_sheet.jpg` and `realcheck_N1_sheet.jpg`

## 1. Summary

| Question | Answer |
|---|---|
| Does our service model **A4 (SwinIR-L)** fix blur? | **No.** A4 is a super-resolution model. On blurred faces its output is almost the same as the input. Identity Δ: SFace **−0.014**, dlib −0.002. It harms slightly and never helps. |
| Does the current service (gate + A4) help on this data? | **No.** The gate sends 23% of the test split to A4. On the sample, mean identity is 0.476 with the gate versus 0.480 with no enhancement. |
| Is there a model that **does** help on blur? | **Yes: NAFNet-GoPro (N1)**, a deblurring model with an MIT licence that we already have locally. On the test split: SFace **+0.162** (95% CI +0.147 to +0.177), dlib +0.021, about 0.9 s per face. 44% of faces clearly improve, **0% clearly harmed**, and it helps at every blur level (+0.12 mild, +0.15 moderate, +0.21 severe). |
| Can NAFNet be deployed as is? | **No.** It produces severe colour artifacts on inputs it wasn't trained for. On grainy blur (`ztest`) 14% of outputs have rainbow or metallic streaks. On the real CC0 photos it **destroyed 11 of 168 faces (6.5%)**, mainly pixelated, halftone and printed images. It needs an input gate (blur only) and an artifact check on its output. |
| GFPGAN / CodeFormer? | **Both harm identity.** GFPGAN: SFace −0.040, 14% clearly harmed. CodeFormer w=0.5: SFace −0.098, 38% clearly harmed. Both also have non-commercial licences. They are ruled out. |
| Is this dataset enough to decide on Feature-Match? | **No.** The blur is synthetic only, each person has one photo, and the "identity" ground truth is the sharp version of the same photo. It's good for **choosing a model on blur**, and possibly for training one. |

**Recommendation in one line:** replace A4 with a **guarded NAFNet route for blur** in the research pipeline, add an output artifact check, and validate on real paired photos with Feature-Match before claiming any benefit. See §8.

## 2. Dataset (`Dataset-and-code/archive`)

**Structure** (checked file by file):

- 386,000 JPEGs forming **192,900 + 100 pairs**: `input/<split>/<name>` (degraded) ↔ `output/<split>/<name>` (sharp).
- Every file is 128×128 RGB, and every input has a matching output with the same name.
- Splits: train 154,320 · val 19,290 · test 19,290 · ztest 100.
- Train and test share no file names, and essentially no images: 1 identical ground truth in a 3k × 20k sample check.
- No ground truth repeats within test.
- `ztest` file prefixes (`00001_…`) give 100 different identities.

**Content:**

- Celebrity face crops, CelebA-style, tightly framed and mostly frontal.
- The ground truth is good quality.

**Degradation:**

- **Synthetic blur only:** Gaussian, defocus, motion streaks and "ghost" double images.
- The blur ranges from mild to unrecognisable.
- Brightness is unchanged.
- **test:** almost noise-free (median noise σ 0.36).
- **ztest:** degraded differently, with median noise σ **1.75**, i.e. grainy blur. Mean input PSNR 23.4 (test: 21.8).

**Alignment:** median input↔ground-truth shift 0.37 px, 95th percentile 1.8 px. A few outliers reach 28 px.

**Provenance and licence: unknown.**

- The content looks like CelebA, which is licensed for non-commercial research only.
- Using it to **evaluate** internally is low-risk.
- **Training** a model for commercial use on it needs licence clearance first.

**The two notebooks are unrelated.** `nafnet-faces.ipynb` is a NAFNet-SIDD demo on other photos, and `svd-101.ipynb` is an SVD tutorial.

## 3. Method

**Stage 1 — all 19,390 test + ztest pairs (28 min):**

- Input-vs-ground-truth PSNR and SSIM.
- The **real service path**: `EnhancementPipeline`, `full_image` mode, frozen gate p5-rwgate-1.0.0. Here only the gate's decision is recorded; the A4 call is a stub.
- B0 identity: the blurred input against its own sharp ground truth.

**Stage 2 — a fixed sample:** 300 random test pairs (seed 0) plus all 100 ztest, run through each model.

| ID | Model | Licence |
|---|---|---|
| A4 | SwinIR-L ×4 GAN (current service model) | REVIEW_REQUIRED |
| N1 | **NAFNet-GoPro** width-64, stand-alone, same resolution | MIT (weights from the official release) |
| A6 | GFPGAN v1.4 | non-commercial parts |
| A7 | CodeFormer w=0.5 | S-Lab non-commercial |

GFPGAN and CodeFormer use FFHQ alignment from landmarks detected **on the blurred input**, as a deployment would. 3 of 400 inputs had no detectable face, so those two models were not run on them.

**Metrics:**

- Outputs are brought to 128 px (INTER_AREA) and compared with the ground truth on PSNR, SSIM and LPIPS (Alex).
- Identity is the SFace (E01) and dlib (E07) cosine to the sharp ground truth, using ground-truth landmarks for every condition.
- **Δ** = output minus blurred input.
- "Clearly improved" or "clearly harmed" means both evaluators moved by more than 0.02 in that direction.
- **Artifact:** the output's 99th-percentile Lab chroma distance to the ground truth exceeds the input's by more than 10 units. This catches obvious rainbow and metallic streaks.

**Stage 2b — real photos:** NAFNet on the 200 CC0 real photos (`Dataset-and-code/dataset`), using the service's face chip. There is no ground truth, so only artifacts are checked: chroma p99 against the photo's own input chip.

**Validity:**

- Identity here means similarity to the sharp version of the **same photo**, which is optimistic. The evaluators are proxies (SFace and dlib), **not Feature-Match**.
- The `ztest` verification check (100 identities, FAR 1e-2 thresholds from clean impostor pairs) has low power. B0 already reaches TAR 1.0 there, so it is reported only as a false-accept signal.

## 4. Gate behaviour on the full test split (19,290)

| Gate outcome | n | % | B0 SFace (mean) |
|---|---|---|---|
| ENHANCE_CANDIDATE → A4 (`strong_blur`) | 4,452 | 23.1 | 0.480 |
| DO_NOT_ENHANCE | 10,416 | 54.0 | 0.540 |
| UNSAFE_OR_UNCERTAIN | 4,422 | 22.9 | 0.392 |

**Main reasons:**

| Reason | Count |
|---|---|
| `degraded_no_a4_evidence` | 7,659 |
| `strong_blur` | 4,452 |
| `beyond_evidence_extreme_blur` | 2,014 |
| `degraded_below_a4_size` | 1,455 |
| `low_detection_confidence` | 953 |
| `alignment_unreliable` | 926 |
| `compression_no_a4_evidence` | 907 |
| `no_valid_face` | 529 |
| `photometric_no_a4_evidence` | 395 |

The faces routed to A4 are spread evenly across blur levels (mild 1,266 · moderate 1,555 · severe 1,631).

**Findings:**

- **The gate routes 23% here** versus about 1.5% of real photos, because synthetic blur is noise-free.
- **It sends faces to a model that does not help them** (§5).
- **It blocks the most useful cases.** It rejects 2,014 extreme-blur faces as `beyond_evidence`, and on the test sample NAFNet's largest gains are exactly on severe blur.

## 5. Results — test split sample (n = 300)

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | SFace Δ (95% CI) | dlib Δ (95% CI) | Clearly improved | Clearly harmed | Artifacts | s / face (CPU) |
|---|---|---|---|---|---|---|---|---|---|
| B0 blurred input | 21.81 | 0.600 | — | 0 (abs. 0.480) | 0 (abs. 0.928) | — | — | — | — |
| **A4 SwinIR-L** | 21.59 | 0.593 | 0.235 | **−0.014** (−0.019, −0.010) | −0.002 (−0.003, −0.001) | 0.3% | 1.7% | 0% | 30.7 |
| **N1 NAFNet-GoPro** | **22.07** | **0.633** | 0.209 | **+0.162** (+0.147, +0.177) | **+0.021** (+0.019, +0.023) | **44.3%** | **0.0%** | 0.3% | **0.9** |
| A6 GFPGAN v1.4 | 21.49 | 0.590 | **0.177** | −0.040 (−0.049, −0.031) | −0.007 (−0.009, −0.005) | 3.0% | 14.1% | 0% | 8.1 |
| A7 CodeFormer w=0.5 | 21.30 | 0.582 | 0.198 | −0.098 (−0.110, −0.086) | −0.016 (−0.018, −0.014) | 1.0% | 38.4% | 6.7% | 16.6 |

**SFace Δ by blur level** (input SSIM terciles; about 100 faces each):

| Model | Mild | Moderate | Severe |
|---|---|---|---|
| A4 | −0.018 | −0.019 | −0.006 |
| **N1** | **+0.123** | **+0.150** | **+0.210** |
| A6 | −0.058 | −0.043 | −0.019 |
| A7 | −0.148 | −0.090 | −0.058 |

**Subset the gate routes to A4** (n = 65 in the sample):

- A4: SFace −0.019, 3% clearly harmed.
- N1 on the same faces: **+0.173**, 0% harmed.

**What the scores mean in practice:**

- **GFPGAN** has the best LPIPS (it *looks* sharp) but the wrong identity: it draws a plausible face that belongs to someone else.
- **NAFNet** gives the most faithful restoration.
- **A4's** output looks like the blurred input, because super-resolution cannot undo blur.

## 6. Results — ztest (grainy blur, n = 100) and real photos

| Model | SFace Δ (95% CI) | Clearly improved | Clearly harmed | **Artifacts** | New false accepts vs B0 (SFace / dlib, 100 IDs) |
|---|---|---|---|---|---|
| A4 | +0.006 (−0.000, +0.012) | 0% | 0% | 0% | 8 / 6 |
| N1 | +0.015 (−0.017, +0.046) | 26% | **15%** | **14%** | 8 / 10 |
| A6 | −0.021 | 2% | 12% | 0% | **21** / 10 |
| A7 | −0.082 | 0% | 23% | 4% | **21** / 14 |

**Effect of grain on NAFNet:** once noise is added to the blur, NAFNet's gain disappears on average. Some faces improve a lot, but 14% get colour streaks.

**Real CC0 photos (stage 2b):**

- NAFNet visibly **destroyed 11 of 168 faces (6.5%)**, turning them into rainbow noise: pixelated, halftone, engraved or printed images.
- On the other real photos it changed little.
- The service's current output guard (face detection on the output) would **not** reliably catch this.

**False accepts:** GFPGAN and CodeFormer produce the most new false accepts (identity drifting towards someone else), which matches the earlier R5 findings.

## 7. Conclusions

1. **A4 is the wrong model for blur.**
   - The earlier evidence of an A4 gain came from synthetic Phase 4 chips that mixed noise and blur.
   - On this large paired blur set, A4 has no benefit and causes slight harm.
   - The current service routes blurred faces to A4, so on blur it adds about 30 s of CPU per face for nothing.
2. **NAFNet-GoPro is the only positive model found so far.**
   - It improves identity on clean synthetic blur at every level, never clearly harms there, and is fast (0.9 s on CPU).
   - It has an MIT licence, which suits commercial use.
3. **NAFNet is brittle outside its training distribution.**
   - It fails on grain, print and pixelation, which are common in real photos.
   - It is safe only behind a strict input gate (blur that is noise-free or low-noise, and not halftone or pixelated) plus an output artifact check.
4. **Generative face restorers (GFPGAN, CodeFormer) stay ruled out**, on identity grounds and licence grounds.
5. **The core project question is still open.** We don't know whether enhancement improves **Feature-Match** on **real** photos, and this dataset cannot answer it.

## 8. Proposed next steps (for planning)

| # | Step | Why | Effort |
|---|---|---|---|
| 1 | Add a **NAFNet blur route** to the research service. The gate selects strong, low-noise blur (and stops rejecting extreme blur for this route). An **artifact guard** checks chroma and high-frequency colour energy, falling back to ORIGINAL. | It's the only identity-positive model; the guard addresses the 6.5–14% failure mode | 1–2 days |
| 2 | Re-run this archive test and the 200-photo real check **through the service**, with the new route | Proves the gate plus guard removes the artifacts and keeps the gain | ½ day |
| 3 | Decide on A4. Keep it only for evidence-backed noise cases, or drop it | It costs about 30 s per face and gives no blur benefit | decision |
| 4 | **Fine-tuning option:** train or fine-tune NAFNet (or a smaller deblurring model) on the 154k train pairs, **with realistic grain/JPEG added to the inputs** | Fixes brittleness on real grain; the data is ideal in size and pairing. **Blocked on dataset licence clearance**; also needs a GPU | 1–2 weeks with a GPU |
| 5 | **The deciding test:** about 20+ real people (clear reference + real blurred photos) with **Feature-Match** access, comparing ORIGINAL vs NAFNet route vs A4 | The only way to answer the project question (R5-4 harness is ready) | depends on access and data |

## 9. Reproduce

```bash
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_archive_eval.py stage1 --workers 5     # ~28 min
for m in N1 A6 A7 A4; do FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_archive_eval.py stage2 --model $m --workers 4; done
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_archive_eval.py realcheck
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_archive_eval.py analyze
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_archive_eval.py sheet
```

**Wall-clock times** (16-CPU Docker VM, CPU only):

| Run | Time |
|---|---|
| N1 | 3.5 min |
| A6 | 14.5 min |
| A7 | 39 min |
| A4 | 49 min (4 workers, partly in parallel with A7) |

No model, gate or threshold was changed for this evaluation. Nothing image-like is committed: dataset images and outputs stay in git-ignored paths. The manifests contain only the public dataset file names and scores.
