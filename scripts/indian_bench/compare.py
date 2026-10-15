"""Phases 6 + 7 — three-panel comparison images and per-variant diagnostics (resumable).

    python scripts/indian_bench/compare.py [--pilot] [--workers 8]

One comparison per answered variant:   [CLEAN ORIGINAL] [DEGRADED INPUT] [FES OUTPUT]

View (identical for the three panels of a row, documented in the image footer and in comparisons.csv `view`):
  chip   the service found a face: the panels show the service's own face-chip crop. The crop box comes from the
         landmarks the service's detector (same model, same settings, run here on the same degraded bytes) finds on the
         DEGRADED image; the degraded chip cut here is verified pixel-for-pixel against the chip the service analysed
         (`chip_match`). The clean panel is the SAME box cut from the clean reference; the third panel is the PNG the
         service returned (ENHANCED) or the same box of the returned image (ORIGINAL = the input, unchanged).
         Chips are shown at `comparison.panel_px` with bilinear interpolation — the same resize for all three panels,
         no sharpening, no colour change. The chip is the FairFace/dlib framing (whole face with margin).
  full   no face was found by the service: the three full frames at their stored size, no resize.
The stored clean / degraded / enhanced files are only read.

Diagnostics (manifests/metrics.csv), all against the clean reference, in the same view:
  PSNR (RGB) and SSIM (luma) of degraded and of output; Laplacian-variance ratio; SFace cosine similarity (E01, the
  project's existing test-only evaluator, Apache-2.0) clean↔degraded and clean↔output, every image aligned with the
  landmarks found on the CLEAN reference (project convention: identity content is measured, not detector luck).
  Identity numbers are kept apart from the image-quality numbers; none of them proves identity preservation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common as C

_CTX: dict = {}
THIRD = {"enhanced": "FES ENHANCED OUTPUT"}
OUTCOME_TEXT = {"enhanced": "ENHANCED", "original_no_face": "NOT ENHANCED - no usable face found, input returned",
                "original_gate": "NOT ENHANCED - gate did not route it, input returned",
                "fallback_guard": "FALLBACK - safety guard rejected the model output, input returned",
                "fallback_model_failure": "FALLBACK - model failure, input returned"}


def ctx():
    if not _CTX:
        import cv2
        from PIL import ImageFont
        from fes_bench import metrics as M
        from fes_data import preprocess as P
        from fes_data.common import repo_path
        from fes_service import pipeline as PL
        cv2.setNumThreads(1)
        det = P.YuNetDetector(repo_path(PL.YUNET), 0.7, 0.3, max_side=640)

        def font(size):
            try:
                return ImageFont.truetype("arial.ttf", size)
            except OSError:
                return ImageFont.load_default()
        _CTX.update(P=P, PL=PL, M=M, det=det, det_cfg=P.PreprocessConfig(out_size=112), sface=M.SFace(),
                    f_head=font(20), f_label=font(18), f_small=font(14))
    return _CTX


def detect(img):
    """The service's detection contract (pipeline._process, full_image) → (landmarks | None, status text)."""
    import numpy as np
    c = ctx()
    fr = c["P"].preprocess(img, c["det"], c["det_cfg"], input_mode="face_crop")[0]
    if fr.status != c["P"].OK or fr.detection is None:
        return None, f"no_valid_face:{fr.status}"
    lm = np.asarray(fr.detection.landmarks, np.float64)
    if not fr.aligned or not c["P"].landmarks_plausible(lm):
        return None, "alignment_unreliable:landmarks"
    return lm, "OK"


def chip_box(lm):
    """The crop box of pipeline.face_chip: (x0, y0, side) in source pixels (same arithmetic; verified per variant)."""
    import numpy as np
    c = ctx()
    Mx = c["P"].umeyama_similarity(lm, c["PL"].DLIB_CHIP_TEMPLATE_150)
    scale = float(np.sqrt(abs(np.linalg.det(Mx[:, :2]))))
    cx, cy = np.linalg.inv(Mx[:, :2]) @ (np.array([75.0, 75.0]) - Mx[:, 2])
    side = int(max(8, round(150.0 / scale)))
    return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side


def triptych(panels, labels, header, footer, colour):
    """Three equally sized BGR panels → one RGB PIL image with header, panel labels and footer."""
    import cv2
    from PIL import Image, ImageDraw
    c = ctx()
    h, w = panels[0].shape[:2]
    gap, top, lab, bot = 12, 62, 30, 26
    W = max(3 * w + 4 * gap, 1180)
    x_off = (W - (3 * w + 2 * gap)) // 2
    im = Image.new("RGB", (W, top + lab + h + bot + 6), (24, 24, 24))
    d = ImageDraw.Draw(im)
    d.text((gap, 8), header[0], font=c["f_head"], fill=(255, 255, 255))
    d.text((gap, 36), header[1], font=c["f_small"], fill=colour)
    for i, (p, l) in enumerate(zip(panels, labels)):
        x = x_off + i * (w + gap)
        d.text((x, top + 4), l, font=c["f_label"], fill=(255, 255, 255) if i < 2 else colour)
        im.paste(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)), (x, top + lab))
    d.text((gap, top + lab + h + 6), footer, font=c["f_small"], fill=(170, 170, 170))
    return im


COLOURS = {"enhanced": (90, 220, 120), "original_gate": (190, 190, 190), "original_no_face": (150, 170, 230),
           "fallback_guard": (255, 180, 70), "fallback_model_failure": (255, 110, 110)}


def process_sample(args) -> list[dict]:
    cfg, sid, items = args
    import cv2
    import numpy as np
    c = ctx()
    M, PL = c["M"], c["PL"]
    root = Path(cfg["bench_root"])
    px, q = cfg["comparison"]["panel_px"], cfg["comparison"]["jpeg_quality"]
    out = []
    clean = cv2.imread(str(root / items[0][0]["clean_path"]), cv2.IMREAD_COLOR)
    lm_clean, clean_det = detect(clean)
    e_clean = c["sface"].embed(clean, lm_clean) if lm_clean is not None else None
    for v, r in items:
        rec = {"variant_id": v["variant_id"], "sample_id": sid, "expression": v["expression"], "type": v["type"],
               "category": v["category"], "level": v["level"], "outcome": r["outcome"],
               "clean_path": v["clean_path"], "degraded_path": v["degraded_path"], "enhanced_path": r["enhanced_path"],
               "clean_pixel_sha256": v["clean_pixel_sha256"], "degraded_file_sha256": v["file_sha256"],
               "output_sha256": r["output_sha256"], "clean_face_detected": lm_clean is not None, "error": ""}
        try:
            deg_bytes = (root / v["degraded_path"]).read_bytes()
            out_bytes = (root / r["enhanced_path"]).read_bytes()
            assert C.sha256_bytes(deg_bytes) == v["file_sha256"], "degraded file differs from its manifest"
            assert C.sha256_bytes(out_bytes) == r["output_sha256"], "enhanced file differs from its manifest"
            assert C.sha256_bytes(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).tobytes()) == v["clean_pixel_sha256"], \
                "clean reference differs from its manifest"
            deg = cv2.imdecode(np.frombuffer(deg_bytes, np.uint8), cv2.IMREAD_COLOR)
            res = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)
            assert deg is not None and res is not None and deg.shape == clean.shape, "unreadable or wrong-sized image"
            lm, det_status = detect(deg)
            rec.update(local_det=det_status, service_n_faces=r.get("n_faces"))
            enhanced = r["outcome"] == "enhanced"
            mask = None
            if lm is not None:
                x0, y0, side = chip_box(lm)
                d_chip, side2 = PL.face_chip(deg, lm)
                c_chip, _ = PL.face_chip(clean, lm)
                o_chip = res if enhanced else PL.face_chip(res, lm)[0]
                valid = PL.face_chip(np.full(deg.shape, 255, np.uint8), lm)[0][..., 0] >= 128
                mask = None if valid.all() else valid
                assert side == side2, "chip box arithmetic differs from pipeline.face_chip"
                match = C.sha256_bytes(d_chip.tobytes()) == r.get("service_chip_pixel_sha256")
                assert o_chip.shape == d_chip.shape, f"output chip {o_chip.shape} vs degraded chip {d_chip.shape}"
                rec.update(view="chip", chip_x0=x0, chip_y0=y0, chip_side_src=side, chip_px=d_chip.shape[0],
                           chip_match=match, oob_frac=round(float(1 - valid.mean()), 4))
                panels = [cv2.resize(p, (px, px), interpolation=cv2.INTER_LINEAR) for p in (c_chip, d_chip, o_chip)]
                footer = (f"view: service face-chip crop, box x0={x0} y0={y0} side={side} px of the 450x800 frame, "
                          f"shown at {px} px (bilinear), identical for the 3 panels"
                          + ("" if match else "  [chip differs from the one the service analysed]"))
                a, b, o = c_chip, d_chip, o_chip
                if lm_clean is not None:
                    s = d_chip.shape[0] / side
                    lm_chip = (lm_clean - np.array([x0, y0])) * s
                    e_deg = c["sface"].embed(d_chip, lm_chip)
                    e_out = c["sface"].embed(o_chip, lm_chip) if enhanced else e_deg
                    rec.update(sface_clean_vs_degraded=round(float(e_clean @ e_deg), 4),
                               sface_clean_vs_output=round(float(e_clean @ e_out), 4))
            else:
                assert not enhanced, "ENHANCED answer but no face found locally"
                assert res.shape == deg.shape
                rec.update(view="full", chip_match="")
                panels, (a, b, o) = [clean, deg, res], (clean, deg, res)
                footer = "view: full frames at their stored size (no resize, no crop)"
            rec.update(psnr_degraded=round(M.psnr_rgb(b, a, mask), 3), ssim_degraded=round(M.ssim_y(b, a, mask), 4),
                       sharpness_ratio_degraded=round(M.sharpness_ratio(b, a), 4))
            if enhanced:
                rec.update(psnr_output=round(M.psnr_rgb(o, a, mask), 3), ssim_output=round(M.ssim_y(o, a, mask), 4),
                           sharpness_ratio_output=round(M.sharpness_ratio(o, a), 4))
            else:                                   # the output IS the input: same numbers by construction
                rec.update(psnr_output=rec["psnr_degraded"], ssim_output=rec["ssim_degraded"],
                           sharpness_ratio_output=rec["sharpness_ratio_degraded"])
            why = r.get("fallback") or r.get("reason") or ""
            model = f" · model {r.get('selection_chosen') or ''} {r.get('model') or ''}" if enhanced else ""
            header = (f"{v['expression']} · {sid} · {v['type']} · level {v['level']} ({v['category']})",
                      f"{v['order']}   |   FES {r['endpoint']}: {OUTCOME_TEXT[r['outcome']]}"
                      f"{' [' + why + ']' if why and not enhanced else ''}{model}")
            labels = ["CLEAN ORIGINAL", "DEGRADED INPUT",
                      THIRD.get(r["outcome"], "FES OUTPUT = INPUT UNCHANGED (not enhanced)")]
            im = triptych(panels, labels, header, footer, COLOURS[r["outcome"]])
            rel = f"comparisons/by_expression/{v['expression']}/{sid}/{sid}__{v['type']}__L{v['level']}.jpg"
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            im.save(p, quality=q, subsampling=0)
            rel2 = f"comparisons/by_degradation/{v['type']}/level_{v['level']}/{v['expression']}/{sid}__{v['type']}__L{v['level']}.jpg"
            p2 = root / rel2
            p2.parent.mkdir(parents=True, exist_ok=True)
            if p2.exists():
                p2.unlink()
            os.link(p, p2)                          # the same file under the second grouping (hard link)
            rec.update(comparison_path=rel, comparison_path_by_degradation=rel2, comparison_width=im.width,
                       comparison_height=im.height)
        except Exception as e:  # noqa: BLE001 — recorded, the variant is reported as not compared
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        out.append(rec)
    return out


FIELDS_CMP = ["variant_id", "sample_id", "expression", "type", "category", "level", "outcome", "view", "clean_path",
              "degraded_path", "enhanced_path", "comparison_path", "comparison_path_by_degradation",
              "clean_pixel_sha256", "degraded_file_sha256", "output_sha256", "chip_x0", "chip_y0", "chip_side_src",
              "chip_px", "chip_match", "oob_frac", "comparison_width", "comparison_height", "error"]
FIELDS_MET = ["variant_id", "sample_id", "expression", "type", "category", "level", "outcome", "view",
              "clean_face_detected", "local_det", "service_n_faces", "psnr_degraded", "psnr_output", "ssim_degraded",
              "ssim_output", "sharpness_ratio_degraded", "sharpness_ratio_output", "sface_clean_vs_degraded",
              "sface_clean_vs_output"]


def finalise(cfg):
    rows = sorted({r["variant_id"]: r for r in C.read_jsonl(C.bench(cfg, "manifests", "comparisons.jsonl"))}.values(),
                  key=lambda r: r["variant_id"])
    C.write_csv(C.bench(cfg, "manifests", "comparisons.csv"), rows, FIELDS_CMP)
    C.write_csv(C.bench(cfg, "manifests", "metrics.csv"), [r for r in rows if not r["error"]], FIELDS_MET)
    return rows


def main(a):
    cfg = C.load_config(a.config)
    root = Path(cfg["bench_root"])
    variants = {v["variant_id"]: v for v in C.read_csv(C.bench(cfg, "manifests", "generated_variants.csv"))}
    results = [r for r in C.read_csv(C.bench(cfg, "manifests", "enhancement_results.csv")) if r.get("enhanced_path")]
    ids = {r["sample_id"] for r in C.scope(cfg, a.pilot)}
    results = [r for r in results if r["sample_id"] in ids]
    done = {r["variant_id"]: r for r in C.read_jsonl(C.bench(cfg, "manifests", "comparisons.jsonl"))}
    groups = defaultdict(list)
    for r in results:
        d = done.get(r["variant_id"])
        if a.redo or not (d and not d["error"] and d["output_sha256"] == r["output_sha256"]
                          and (root / d["comparison_path"]).exists()):
            groups[r["sample_id"]].append((variants[r["variant_id"]], r))
    n = sum(map(len, groups.values()))
    print(f"answered variants in scope {len(results)} · comparisons to build {n} ({len(groups)} samples)", flush=True)
    t0, k, errs = time.time(), 0, 0
    with open(C.bench(cfg, "manifests", "comparisons.jsonl"), "a", encoding="utf-8") as fh, \
            ProcessPoolExecutor(a.workers) as ex:
        for recs in ex.map(process_sample, [(cfg, sid, items) for sid, items in sorted(groups.items())], chunksize=1):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
                errs += bool(rec["error"])
            fh.flush()
            k += 1
            if k % 50 == 0:
                print(f"  {k}/{len(groups)} samples · {time.time() - t0:.0f} s", flush=True)
    rows = finalise(cfg)
    bad = [r for r in rows if r["error"]]
    nomatch = [r for r in rows if r.get("view") == "chip" and r.get("chip_match") is not True]
    print(f"done in {time.time() - t0:.0f} s · comparisons in manifest {len(rows) - len(bad)} · errors {len(bad)} · "
          f"chip views whose crop does NOT match the service's chip: {len(nomatch)}", flush=True)
    for r in bad[:10]:
        print("   ", r["variant_id"], r["error"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
