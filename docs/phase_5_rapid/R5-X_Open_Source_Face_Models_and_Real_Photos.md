# R5 (interim) — GFPGAN / CodeFormer vs A4, and the frozen gate on real photos

| Field | Value |
|---|---|
| Date | 2026-10-04 |
| Trigger | User request: use the well-known open-source face restorers (GFPGAN, CodeFormer) and the real photos in `Dataset-and-code/` |
| User decisions | GFPGAN v1.4 + CodeFormer **local research only** (CodeFormer: S-Lab **non-commercial**; GFPGAN: NVIDIA NC parts → REVIEW). Identity evidence from the existing synthetic pairs; the 200 real photos have no identity labels → visual/routing evidence only. |
| Evidence | `data/manifests/phase5_rapid/{compare, realphotos, runs, metrics, metrics_e07}` |
| Scripts | `scripts/r5_compare_models.py`, `scripts/r5_realphotos_gate.py`, `scripts/r5_realphotos_visual.py`; config `configs/phase5/benchmark_p5.yaml` (= Phase 3 protocol + A6–A8) |

## 1. Bottom line

1. **On identity, GFPGAN and CodeFormer are the worst candidates tested, worse than A4 and worse than doing nothing.** Both identity evaluators agree. They also create the most **new false matches** (outputs that move towards a different person).
2. **They are the best-looking.** On the real vintage photos they produce clean, natural faces where A4 smears. They also visibly **fabricate**: glasses added (img11), whole faces invented from unrecognisable pixel blocks (img119), blue eyes and pink lips "colourised" into greyscale photos.
3. **A4 remains the only candidate with an identity gain, and only behind the gate** (noise/blur): +0.0044 SFace / +0.0005 dlib, 0 flips.
4. **The real blocker is the gate, not the model.** The frozen Phase 4 gate sends **1 of 200** real photos to A4. Its learned rules key on synthetic artefacts and do not fire on real images.

## 2. Identity comparison — synthetic test `main` (896 pairs, 28 identities, same inputs and protocol for every model)

Consistency check: the re-run untouched-input (B0) scores equal the Phase 3/4 scores exactly (max |Δ| = 0.0, both evaluators).

**Always on**

| Model | ΔID SFace | ΔID dlib | confirmed benefit | harm (any) | new false-accepts SFace/dlib | mild photos harmed |
|---|---|---|---|---|---|---|
| A4 SwinIR-L | −0.0078 ▼ | −0.0013 ▼ | **120** | **440** | **13 / 22** | **135 / 252** |
| A6 GFPGAN v1.4 | −0.0856 ▼ | −0.0097 ▼ | 64 | 734 | 53 / 16 | 243 / 252 |
| A7 CodeFormer w=0.5 (default) | **−0.2107 ▼** | **−0.0242 ▼** | 47 | **829** | **90 / 44** | 252 / 252 |
| A8 CodeFormer w=1.0 (max fidelity) | −0.1310 ▼ | −0.0152 ▼ | 79 | 767 | 62 / 35 | 251 / 252 |
| *(ref.)* A1 RestoreFormer++ / A2 VQFR / A3 Real-ESRGAN / A5 chain | −0.049 / −0.061 / −0.021 / −0.080 ▼ | all ▼ | 25–58 | 533–795 | 18–48 | 145–245 |

**Behind the frozen gate (T6) and the NOISE/BLUR route (T7 table, model swapped in)**

| Model | T6 + model: ΔID SFace / dlib | harm | T6 + NOISE/BLUR: ΔID SFace / dlib | benefit − harm | flips SFace/dlib |
|---|---|---|---|---|---|
| **A4** | **+0.0041 ▲ / +0.0006 ▲** | 12 | **+0.0044 ▲ / +0.0005 ▲** | **38 − 3** | **0 / 0** |
| A6 GFPGAN | −0.0064 ▼ / −0.0006 ▼ | 60 | −0.0023 ≈ / −0.0003 ≈ | 10 − 35 | 0 / 1 |
| A7 CodeFormer w=0.5 | −0.0231 ▼ / −0.0024 ▼ | 88 | −0.0145 ▼ / −0.0017 ▼ | 0 − 61 | 3 / 9 |
| A8 CodeFormer w=1.0 | −0.0153 ▼ / −0.0015 ▼ | 69 | −0.0087 ▼ / −0.0010 ▼ | 7 − 43 | 1 / 5 |

**Mechanism.** GFPGAN and CodeFormer reconstruct the face from a learned prior (StyleGAN2 / VQ codebook) trained on FFHQ. The result is a plausible, sharp face, but the details come from the prior and not from the person. This is the same pattern Phase 3/4 measured for RestoreFormer++ and VQFR. CodeFormer's fidelity weight does not fix it (w=1.0 is less bad than w=0.5 but still the second-worst overall).

Synthetic caveat: the degradations are synthetic and the evaluators are SFace and dlib, not Feature-Match. However, the direction is consistent across 2 evaluators, 9 degradation families and 4 severities, and it matches the visual fabrication seen on real photos.

## 3. Real photos (`Dataset-and-code/dataset`, 200 images)

- **Data.** Kaggle "Low Resolution Photographs", marked CC0 by the uploader. Mostly vintage scans, many web copies of famous historical photos, a few artworks. 61 greyscale. 72 multi-face. **No identity references**, so no identity metric is possible.
- **Frozen gate (R5-1 service, `full_image`):**

| Decision | Count | Detail |
|---|---|---|
| DO_NOT_ENHANCE | 120 | gate rules did not fire |
| UNSAFE_OR_UNCERTAIN | 75 | no face 25, face too small 22, low detector confidence 17, pose 11 |
| ENHANCE_CANDIDATE | 5 | 4 → LOWRES bucket (original), **1 → A4** |

- **Why the gate is silent.** Its rules are noise σ ≥ 9.8, extreme high-frequency loss, or *blur with noise σ ≤ 0.28*.
  - Synthetic blur is perfectly noise-free; real blurry photos always carry grain (DO_NOT_ENHANCE photos: noise σ median 1.10, p10 0.43).
  - Real noise rarely reaches σ 9.8.
  - **The learned rules detect the synthetic generator, not image quality.**
- **Visual sheets.** 16 faces × input / A4 / GFPGAN / CodeFormer ×2, in `experiments/phase5_rapid/outputs/realphotos/sheet_{1,2}.jpg` (git-ignored). Observations are in §1.
- **Cost on CPU (2 threads):** A4 37 s/face, GFPGAN 10 s, CodeFormer 21 s.

## 4. Consequences for R5-2 / R5-3

| Finding | Action |
|---|---|
| Face-prior models (A1, A2, A6, A7, A8) lose identity and add false matches, gated or not | Do not route them. They remain useful only for *viewing* (human-facing "beautify"), never before matching. |
| A4 + gate is the only identity-positive configuration | Keep A4 as the enhancement candidate |
| The gate does not transfer to real images (1/200) | **R5-3 targeted fix:** re-derive the gate's routing rules from noise-robust features. Validate on data with identity labels before trusting it. |
| No real identity-labelled data yet | Options: (a) the user's own real pairs + Feature-Match access; (b) LFW-based real degraded faces (LFW-Test from the GFPGAN paper, faces of LFW identities that have other photos), local research only |
| Greyscale inputs get colour invented | Any enhancer output for a greyscale input should be converted back to greyscale (one-line guard) |

## 5. Reproduce

```bash
# inference (resumable; parallel containers were used: A6 4 workers, A8 2→4, A7 2→3, 2 threads each)
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/p3_run_inference.py --config configs/phase5/benchmark_p5.yaml \
  --config-id A6 --population main --root data/manifests/phase5_rapid --workers 4 --threads 2      # A7, A8, B0 alike
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/p3_evaluate.py --config configs/phase5/benchmark_p5.yaml \
  --population main --root data/manifests/phase5_rapid --configs B0 A6 A7 A8
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/p4_evaluate_e07.py --population main \
  --root data/manifests/phase5_rapid --configs B0 A6 A7 A8 --out data/manifests/phase5_rapid/metrics_e07/main
PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_compare_models.py --configs A6 A7 A8
PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_realphotos_gate.py
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_realphotos_visual.py --n 16
```

Weights: `models/M01_gfpgan/GFPGANv1.4.pth` `e2cd4703…` (GitHub release v1.3.0) and `models/M02_codeformer/codeformer.pth` `1009e537…` (GitHub release v0.1.0). Both are SHA-256 verified on load, strict load with 0 missing/unexpected keys, deterministic outputs (GFPGAN noise seeded per call).
