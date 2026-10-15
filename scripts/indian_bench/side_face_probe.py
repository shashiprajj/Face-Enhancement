"""Feasibility probe (offline, nothing in the service changes): can the EXISTING NAFNet models deblur a side face
when the face is cropped by its detection BOX instead of the 5-landmark alignment the service requires?

    docker run --rm -v <repo>:/workspace -w /workspace -e PYTHONPATH=src -e OMP_NUM_THREADS=4 fes-dev:cpu-p5-onnx \
        python scripts/indian_bench/side_face_probe.py [--limit N]

Input: the already generated blurred variants (TYPES x LEVEL) of the benchmark's side-face originals (no usable
frontal face). Per image:
  1. the service's detector (same model + settings) runs on the DEGRADED photo; the largest detection's box is used;
  2. square chip centred on the box, side = K * max(box w, box h) (K = median ratio of the service's own chip side to
     the box size on the frontal originals, so the framing matches what the model normally sees); cut at integer
     pixels, reflect fill outside the photo, reduced to 224 px with INTER_AREA when larger (the service's chip rule);
  3. both service models (r6 = NAFNet-GoPro, r6_ft = face fine-tune; the service's own ONNX factories, 4-run
     consensus) enhance the chip; the service's selection rule picks one;
  4. recorded: PSNR / SSIM of degraded and output against the same box of the clean reference, and whether the
     service's stability check and artifact guard WOULD pass (they are not applied here, only reported).
→ <bench>/side_face_probe/records.jsonl and one  CLEAN | DEGRADED | MODEL OUTPUT  strip per image.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TYPES = ["gaussian_blur", "motion_blur_000", "motion_blur_045", "defocus_blur", "camera_shake"]
LEVEL = "3"


def box_chip(img, cx, cy, side):
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101) if pad else img
    chip = src[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    if side > 224:
        chip = cv2.resize(chip, (224, 224), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(chip)


def main(a):
    from fes_bench import metrics as M
    from fes_data import preprocess as P
    from fes_data.common import repo_path
    from fes_service import models as MD
    from fes_service import nafnet_route as NR
    from fes_service import pipeline as PL
    cfg = C.load_config()
    root = Path(cfg["bench_root"])
    out_dir = root / "side_face_probe"
    out_dir.mkdir(exist_ok=True)
    det = P.YuNetDetector(repo_path(PL.YUNET), 0.7, 0.3, max_side=640)
    scope = C.scope(cfg)

    # K: the service's chip side relative to the detection box, measured on the frontal originals
    ratios = []
    for r in [s for s in scope if s["face_usable"] == "True"][:80]:
        img = cv2.imread(str(root / "clean_originals" / r["expression"] / f"{r['sample_id']}.png"))
        d = det.detect(img)
        if d:
            _, side = PL.face_chip(img, np.asarray(d[0].landmarks, np.float64))
            ratios.append(side / max(d[0].bbox_xywh[2], d[0].bbox_xywh[3]))
    K = round(st.median(ratios), 3)
    print(f"K = {K} (chip side / box size, median of {len(ratios)} frontal originals)", flush=True)

    route = NR.load_config(NR.R6_CONFIG)
    models = {"r6": MD.nafnet_factory_r6_onnx()[0], "r6_ft": MD.nafnet_factory_r6_ft_onnx()[0]}
    var = {(v["sample_id"], v["type"], v["level"]): v for v in C.read_csv(root / "manifests" / "generated_variants.csv")}
    side_ids = [s for s in scope if s["face_usable"] != "True"]
    todo = [(s, t) for s in side_ids for t in TYPES][:a.limit or None]
    done = {r["variant_id"] for r in C.read_jsonl(out_dir / "records.jsonl")}
    t0 = time.time()
    with open(out_dir / "records.jsonl", "a", encoding="utf-8") as fh:
        for i, (s, t) in enumerate(todo, 1):
            v = var[(s["sample_id"], t, LEVEL)]
            if v["variant_id"] in done:
                continue
            rec = {"variant_id": v["variant_id"], "sample_id": s["sample_id"], "expression": s["expression"], "type": t,
                   "level": LEVEL, "K": K}
            clean = cv2.imread(str(root / v["clean_path"]))
            deg = cv2.imread(str(root / v["degraded_path"]))
            dets = det.detect(deg)
            if not dets:
                rec["result"] = "no_detection_on_degraded"
            else:
                x, y, w, h = dets[0].bbox_xywh
                side = int(max(16, round(K * max(w, h))))
                cx, cy = x + w / 2.0, y + h / 2.0
                d_chip, c_chip = box_chip(deg, cx, cy, side), box_chip(clean, cx, cy, side)
                rec.update(det_score=round(dets[0].score, 3), box_side=side, chip_px=d_chip.shape[0])
                outs, info = {}, {}
                for name, fn in models.items():
                    try:
                        o, meta = fn(d_chip, {"s": d_chip.shape[0]})
                        stab = NR.stability_check((meta or {}).get("runs_dot_c"), route)
                        guard = NR.artifact_guard(d_chip, o, route, lambda im: len(det.detect(im)) > 0, False)
                        outs[name] = o
                        info[name] = {"stability_pass": stab.passed, "guard_pass": guard.passed,
                                      "guard_reasons": "+".join(guard.reasons),
                                      "psnr": round(M.psnr_rgb(o, c_chip), 3), "ssim": round(M.ssim_y(o, c_chip), 4),
                                      "sharpness_ratio": round(M.sharpness_ratio(o, c_chip), 4)}
                    except Exception as e:  # noqa: BLE001
                        info[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
                rec.update(models=info, psnr_degraded=round(M.psnr_rgb(d_chip, c_chip), 3),
                           ssim_degraded=round(M.ssim_y(d_chip, c_chip), 4),
                           sharpness_ratio_degraded=round(M.sharpness_ratio(d_chip, c_chip), 4))
                if outs:
                    chosen, _ = NR.select_best(d_chip, outs)
                    rec.update(result="model_ran", chosen=chosen)
                    tiles = [cv2.resize(p, (448, 448), interpolation=cv2.INTER_LINEAR) for p in (c_chip, d_chip, outs[chosen])]
                    for tile, lab in zip(tiles, ("CLEAN ORIGINAL", "DEGRADED INPUT", f"MODEL OUTPUT ({chosen}, gate bypassed)")):
                        cv2.putText(tile, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(tile, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    cv2.imwrite(str(out_dir / f"{v['variant_id']}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    rec["result"] = "model_error"
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} · {time.time() - t0:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
