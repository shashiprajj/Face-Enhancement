# P2-T2 — FairFace Verification & Organization

| Field | Value |
|---|---|
| Phase / Task | Phase 2 / P2-T2 |
| Status | Complete |
| Date | 2026-09-28 |
| Script | `scripts/p2_fairface_prepare.py` (stages `extract`, `analyze`) |
| Config | `configs/phase2/fairface.yaml` |
| Manifest | `data/manifests/fairface_files_v1.csv.gz` (97,698 rows, 7.8 MB) |
| Validation report | `data/manifests/fairface_validation_v1.json` |

---

## 1. Organization (raw vs derived)

| Layer | Path | Nature |
|---|---|---|
| Raw (immutable) | `datasets/raw/DS01_fairface/` | Zip + 2 label CSVs, as acquired in P1-T9. SHA-256 verified; Windows read-only attribute set. |
| Working copy | `datasets/work/DS01_fairface/v1/{train,val}/<n>.jpg` | **Byte-identical** extraction. Every file's SHA-256 was checked against the zip member bytes (plus zip CRC-32). `EXTRACTION.json` records provenance. |
| Manifest | `data/manifests/fairface_files_v1.csv.gz` (tracked) | One row per image. Paths are repo-relative. |

**How to reproduce**

```
.venv/Scripts/python.exe scripts/p2_fairface_prepare.py --stage extract   # host: fast native file writes
scripts/dock.sh python scripts/p2_fairface_prepare.py --stage analyze     # container: canonical analysis
```

- Extraction is stdlib-only and byte-level, so where it runs doesn't matter. It runs on the host because Docker bind-mount writes of ~98k small files ran at ~24 files/s.
- An aborted first attempt left 16,614 files. The rerun **verified all of them by hash**, then wrote the remaining 81,084.
- Analysis reads bytes directly from the raw zip inside the pinned container. The bytes are identical to the working copy.

## 2. Verification Results

| Check | Result |
|---|---|
| Raw zip + label SHA-256 vs Phase 1 manifest | ✅ match |
| `zipfile.testzip()` (CRC of every member) | ✅ no bad member |
| Zip members | 97,698, all `.jpg`; 0 non-image members |
| Label rows | 97,698 (train 86,744 / val 10,954); **0** missing values; **0** duplicate label rows |
| Zip ↔ labels | ✅ every image has a label, every label has an image; 0 split-directory mismatches |
| Readable (OpenCV decode) | ✅ 97,698 / 97,698 (0 corrupt) |
| Dimensions | ✅ all 224×224 |
| Grayscale (channels identical ±2) | 2,341 images |
| **Exact duplicates** (identical SHA-256) | **319 files in 158 groups**; **21 groups span train and val**; **63 groups carry conflicting labels** (same bytes, different age/gender/race) |
| **Near duplicates** (pHash-64, Hamming ≤ 3) | **671 files in 327 groups**; **63 groups span train and val** |
| JPEG encoding | **All images share one quantization table (estimated IJG Q ≈ 75)**: FairFace was uniformly re-encoded |
| Sharpness (Laplacian variance) | p1 4.0 · p10 8.5 · p50 32.4 · p90 192.9 (wide range; some sources are very soft) |

Label value sets (from the report): age groups `0-2 … more than 70`; gender `Male/Female`; race 7 groups; `service_test` True/False. **Labels were not modified.**

## 3. Findings

1. **Cross-split leakage in the official FairFace split.** 21 exact-duplicate groups and 63 near-duplicate groups cross train/val. The official split cannot be trusted as leakage-free. P2-T7 must remove these.
2. **Label inconsistencies.** 63 exact-duplicate groups have conflicting demographic labels. All exact duplicates are excluded from references. Labels are used only for aggregate strata, never corrected.
3. **Compression floor.** Every image carries Q≈75 JPEG artifacts. FairFace "clean" references are therefore **not pristine**. The floor is recorded as a limitation for P2-T3/T4 (JPEG-severity profiles above Q75 are not meaningful on this source).
4. **No identity labels.** Same-person leakage cannot be excluded by label. See P2-T7 (embedding-based screening).

## 4. Stage-1 Suitability for Restoration Benchmarking

Image-level rules, from the config:
- readable;
- min side ≥ 200 px;
- not grayscale (grayscale is kept in the working copy for a future E10 test);
- labeled;
- not an exact duplicate.

| Result | Count |
|---|---|
| Stage-1 suitable | **95,040** |
| Excluded: grayscale | 2,341 |
| Excluded: exact duplicate | 319 |

Face-level suitability (detection, alignment scale, sharpness) is decided in P2-T3.

## 5. Manifest Fields

`sample_id` (`ff-<split>-<n>`), `split_source`, `zip_member`, `work_path`, `sha256`, `size_bytes`, `readable`, `width`, `height`, `is_grayscale`, `jpeg_q_est`, `lap_var`, `mean_luma`, `rms_contrast`, `frac_dark`, `frac_bright`, `phash`, `exact_dup_group`, `near_dup_group`, `age`, `gender`, `race`, `service_test`, `license` (CC BY 4.0), `source`, `stage1_suitable`, `stage1_reasons`.

Source/sample relationships are preserved through `zip_member` ↔ `sample_id` ↔ `work_path`. FairFace supplies no identity relationships.

## 6. License / Source

FairFace: CC BY 4.0, `https://github.com/joojs/fairface` (commit `fb82308`); evidence in P1-T6 / P1-T9. Attribution is kept in `datasets/raw/DS01_fairface/LICENSE.txt`. The underlying per-photo rights (YFCC-100M) remain a noted residual risk (P1-T6). No new license claims are made.

## 7. Done-When Check

The raw source is verified: hashes, CRC, completeness, readability, labels, duplicates. A reproducible, byte-identical working representation exists, with a manifest. ✅
