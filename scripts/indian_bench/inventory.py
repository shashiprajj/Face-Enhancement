"""Phase 1 — read-only inventory of the source dataset.

    python scripts/indian_bench/inventory.py [--workers 6]

Every file under the source root is listed (nothing is skipped silently). Per file: format, size, SHA-256, decode
check, dimensions, colour mode, EXIF orientation, an estimate of the JPEG quality, the result of the service's own
face detector (YuNet, same settings as fes_service.pipeline) and cleanliness diagnostics measured at the reference
scale (config.reference.long_side). Raw measurements are appended to manifests/source_inventory_raw.jsonl (resumable);
`--finalise` (run automatically at the end) turns them into manifests/source_inventory.csv with sample ids, flags,
eligibility and exclusion reasons, and writes reports/01_inventory.md.

The source files are only ever opened for reading.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common as C

STD_LUMA_Q = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17,
              22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78,
              87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
_DET = None


def jpeg_quality(qtables) -> float | None:
    """IJG quality that would give this luminance table (estimate; a camera's own tables only approximate it)."""
    if not qtables or 0 not in qtables:
        return None
    s = 100.0 * sum(qtables[0]) / sum(STD_LUMA_Q)
    return round((200 - s) / 2 if s <= 100 else 5000 / s, 1)


def measure(args) -> dict:
    """One file → one raw record. Runs in a worker process."""
    root, rel, long_side = args
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    global _DET
    path = os.path.join(root, rel)
    rec = {"rel_path": rel.replace("\\", "/"), "expression": rel.replace("\\", "/").split("/")[0],
           "filename": os.path.basename(rel), "ext": os.path.splitext(rel)[1].lower(),
           "bytes": os.path.getsize(path), "sha256": C.sha256_file(path), "readable": False, "error": ""}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            rec.update(format=im.format, mode=im.mode, width=im.width, height=im.height,
                       exif_orientation=im.getexif().get(274, 1),
                       jpeg_quality_est=jpeg_quality(getattr(im, "quantization", None)))
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except Exception as e:  # noqa: BLE001 — recorded as the exclusion reason
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
        return rec
    rec["readable"] = True
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    rec.update(width_upright=w, height_upright=h)
    rec["dhash"] = "%016x" % int("".join("1" if b else "0" for b in (np.diff(cv2.resize(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA).astype(int), axis=1) > 0).ravel()), 2)

    # the service's detector, on the clean reference (the image the benchmark degrades)
    from fes_data import preprocess as P
    from fes_data.common import repo_path
    from fes_service import pipeline as PL
    if _DET is None:
        cv2.setNumThreads(1)
        _DET = P.YuNetDetector(repo_path(PL.YUNET), 0.7, 0.3, max_side=640)
    s = long_side / max(h, w)
    ref = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    rec.update(ref_width=ref.shape[1], ref_height=ref.shape[0], luma_mean=round(float(gray.mean()), 2),
               rms_contrast=round(float(gray.std()), 2),
               lap_var_full=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2))
    fr = P.preprocess(ref, _DET, P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
    rec.update(det_status=str(fr.status), n_faces=int(fr.n_faces), det_warnings="+".join(fr.warnings),
               face_usable=False)
    if fr.status == P.OK and fr.detection is not None:
        lm = np.asarray(fr.detection.landmarks, np.float64)
        ok = bool(fr.aligned and P.landmarks_plausible(lm))
        chip, side = PL.face_chip(ref, lm)
        cg = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        # the same face region at full resolution: focus / motion blur of the source itself shows up here
        x0, y0 = (lm.min(0) / s).astype(int)
        x1, y1 = (lm.max(0) / s).astype(int)
        fg = cv2.cvtColor(img[max(0, y0):y1 + 1, max(0, x0):x1 + 1], cv2.COLOR_BGR2GRAY)
        rec.update(face_usable=ok, det_score=round(float(fr.detection.score), 4),
                   iod_ref_px=round(float(np.linalg.norm(lm[1] - lm[0])), 2), chip_side_ref_px=int(side),
                   lap_var_chip=round(float(cv2.Laplacian(cg, cv2.CV_64F).var()), 2),
                   lap_var_face_fullres=round(float(cv2.Laplacian(fg, cv2.CV_64F).var()), 2) if fg.size else None,
                   chip_luma_mean=round(float(cg.mean()), 2))
    return rec


def finalise(cfg: dict) -> None:
    raw_path = C.bench(cfg, "manifests", "source_inventory_raw.jsonl")
    recs = {r["rel_path"]: r for r in C.read_jsonl(raw_path)}
    rows = [recs[k] for k in sorted(recs)]
    # sample ids: rank of the file among ALL files of its folder in sorted order (stable under any eligibility rule)
    rank = defaultdict(int)
    for r in rows:
        r["sample_id"] = f"{r['expression'].lower()}_{rank[r['expression']]:06d}"
        rank[r["expression"]] += 1
    first_by_hash: dict[str, str] = {}
    for r in rows:
        why = []
        if r["ext"] not in cfg["supported_extensions"]:
            why.append(f"unsupported_format:{r['ext']} (no decoder in the project environment)")
        elif not r["readable"]:
            why.append(f"unreadable:{r['error']}")
        else:
            if min(r["width"], r["height"]) < cfg["min_side_px"]:
                why.append(f"too_small:{r['width']}x{r['height']}")
            if r["sha256"] in first_by_hash:
                why.append(f"exact_duplicate_of:{first_by_hash[r['sha256']]}")
            else:
                first_by_hash[r["sha256"]] = r["rel_path"]
        r["exclusion_reason"] = "; ".join(why)
        r["eligible"] = not why

    # cleanliness flags (diagnostics): robust low outliers within the eligible set
    def robust_low(key, group, log=False):
        """median − z·MAD; Laplacian variances are roughly log-normal, so their outliers are taken on the log."""
        vals = [math.log(r[key]) if log else r[key] for r in group if r.get(key) is not None and (not log or r[key] > 0)]
        if len(vals) < 20:
            return None
        med = st.median(vals)
        mad = st.median(abs(v - med) for v in vals) * 1.4826 or 1e-9
        t = med - cfg["cleanliness_flags"]["robust_z"] * mad
        return math.exp(t) if log else t

    elig = [r for r in rows if r["eligible"]]
    usable = [r for r in elig if r.get("face_usable")]
    thr = {"lap_var_face_fullres": robust_low("lap_var_face_fullres", usable, log=True),
           "lap_var_chip": robust_low("lap_var_chip", usable, log=True),
           "lap_var_full": robust_low("lap_var_full", elig, log=True), "luma_mean": robust_low("luma_mean", elig)}
    q = sorted(r["jpeg_quality_est"] for r in elig if r.get("jpeg_quality_est") is not None)
    dh = Counter((r["expression"], r["dhash"]) for r in elig)
    for r in rows:
        flags = []
        if r["eligible"]:
            if not r.get("face_usable"):
                flags.append(f"no_usable_frontal_face:{r.get('det_status')}")
            if r.get("n_faces", 0) > 1:
                flags.append(f"multiple_faces:{r['n_faces']}")
            for key, name in (("lap_var_face_fullres", "soft_face_fullres"), ("lap_var_chip", "soft_face_ref"),
                              ("lap_var_full", "soft_image_ref"), ("luma_mean", "darker_than_typical")):
                if thr[key] is not None and r.get(key) is not None and r[key] < thr[key]:
                    flags.append(name)
            if r.get("jpeg_quality_est") is not None and r["jpeg_quality_est"] < 75:
                flags.append(f"low_jpeg_quality:{r['jpeg_quality_est']}")
            if dh[(r["expression"], r["dhash"])] > 1:
                flags.append("near_duplicate_dhash_in_folder")
        r["flags"] = "; ".join(flags)
    fields = ["sample_id", "expression", "rel_path", "filename", "ext", "format", "bytes", "sha256", "readable",
              "error", "width", "height", "mode", "exif_orientation", "width_upright", "height_upright", "ref_width",
              "ref_height", "jpeg_quality_est", "det_status", "n_faces", "face_usable", "det_score", "det_warnings",
              "iod_ref_px", "chip_side_ref_px", "lap_var_full", "lap_var_chip", "lap_var_face_fullres", "luma_mean",
              "chip_luma_mean", "rms_contrast", "dhash", "eligible", "exclusion_reason", "flags"]
    C.write_csv(C.bench(cfg, "manifests", "source_inventory.csv"), rows, fields)

    # ---- report
    exprs = sorted({r["expression"] for r in rows})
    tab = []
    for e in exprs + ["TOTAL"]:
        g = [r for r in rows if e == "TOTAL" or r["expression"] == e]
        ge = [r for r in g if r["eligible"]]
        tab.append({"expression": e, "files": len(g), "eligible": len(ge), "excluded": len(g) - len(ge),
                    "usable frontal face": sum(bool(r.get("face_usable")) for r in ge),
                    "no usable face": sum(not r.get("face_usable") for r in ge),
                    "flagged soft": sum("soft_" in r["flags"] for r in ge),
                    "flagged darker": sum("darker" in r["flags"] for r in ge)})
    excl = Counter(r["exclusion_reason"].split(":")[0] for r in rows if not r["eligible"])
    flagc = Counter(f.split(":")[0] for r in elig for f in r["flags"].split("; ") if f)
    dims = Counter((r.get("width"), r.get("height"), r.get("mode"), r.get("format"), r.get("exif_orientation"))
                   for r in rows if r["readable"])
    sides = sorted(r["chip_side_ref_px"] for r in usable)
    pct = lambda v, p: v[int(round(p * (len(v) - 1)))] if v else None  # noqa: E731
    md = ["# Phase 1 — source inventory", "",
          f"Source root (read-only): `{cfg['source_root']}`", "",
          f"Files found: **{len(rows)}** in **{len(exprs)}** expression folders: {', '.join(exprs)}.", "",
          C.md_table(tab, list(tab[0])), "",
          "## Formats, dimensions, colour modes", "",
          C.md_table([{"width": k[0], "height": k[1], "mode": k[2], "format": k[3], "exif orientation": k[4], "files": v}
                      for k, v in dims.most_common()], ["width", "height", "mode", "format", "exif orientation", "files"]),
          "", "Extensions: " + ", ".join(f"`{k}` {v}" for k, v in Counter(r["ext"] for r in rows).most_common()), "",
          "## Exclusions (every one is listed in `manifests/source_inventory.csv`, column `exclusion_reason`)", "",
          C.md_table([{"reason": k, "files": v} for k, v in excl.most_common()], ["reason", "files"]) if excl else "None.",
          "", "## Flags on eligible images (diagnostics, not exclusions)", "",
          C.md_table([{"flag": k, "files": v} for k, v in flagc.most_common()], ["flag", "files"]), "",
          f"Flag thresholds (median − {cfg['cleanliness_flags']['robust_z']}·MAD): "
          + ", ".join(f"`{k}` < {v:.1f}" for k, v in thr.items() if v is not None), "",
          f"Estimated JPEG quality of the sources: min {q[0] if q else None}, median {pct(q, .5)}, max {q[-1] if q else None}.",
          "", f"Face chip side at the reference scale (long side {cfg['reference']['long_side']} px), usable faces: "
          f"min {sides[0] if sides else None}, p5 {pct(sides, .05)}, median {pct(sides, .5)}, p95 {pct(sides, .95)}, "
          f"max {sides[-1] if sides else None} px (the service resamples chips above 224 px).", ""]
    C.bench(cfg, "reports", "01_inventory.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main(a):
    cfg = C.load_config(a.config)
    raw_path = C.bench(cfg, "manifests", "source_inventory_raw.jsonl")
    if not a.finalise:
        root = cfg["source_root"]
        files = sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs)
        done = {r["rel_path"] for r in C.read_jsonl(raw_path)}
        todo = [(root, f, cfg["reference"]["long_side"]) for f in files if f.replace("\\", "/") not in done]
        print(f"{len(files)} files, {len(todo)} to measure", flush=True)
        with open(raw_path, "a", encoding="utf-8") as fh, ProcessPoolExecutor(a.workers) as ex:
            for i, rec in enumerate(ex.map(measure, todo, chunksize=4), 1):
                fh.write(json.dumps(rec) + "\n")
                if i % 250 == 0:
                    fh.flush()
                    print(f"  {i}/{len(todo)}", flush=True)
    finalise(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--finalise", action="store_true", help="only rebuild the CSV + report from the raw records")
    main(ap.parse_args())
