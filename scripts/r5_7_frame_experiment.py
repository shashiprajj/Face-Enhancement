"""R5-7 — controlled DEV experiment for the full_image benefit loss: does NAFNet's internal ZERO padding to a multiple
of 16 (NAFNet_arch.check_image_size: F.pad(..., value 0) on the right/bottom) degrade the re-framed service chips?

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_7_frame_experiment.py [--limit N]

DEV only: archive `val` pairs from the R5-7 framing sample (val_frame records) whose full_image chip is NAFNet-eligible.
Same chip (fes_service.pipeline.face_chip), same weights, two NAFNet calls:
  zero     the service today: chip → net (internal zero pad to /16) → output
  reflect  chip reflect-padded to a multiple of 16 BEFORE the net (so the internal pad is a no-op) → cropped back
Scored exactly like R5-6 (paste back into the 128-px frame, SFace/dlib vs GT with GT landmarks, Δ vs B0) plus the
R5-7 guard (1.1.0) on each output. → data/manifests/phase5_rapid/r5_7/frame_experiment.csv.gz
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path, write_csv_gz

ARCH = "Dataset-and-code/archive"


def reflect16(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101), h, w


def main(limit: int) -> None:
    import torch
    torch.set_num_threads(4)
    from fes_bench import evaluators as EV
    from fes_bench.adapters import bgr_u8_to_rgb_t, rgb_t_to_bgr_u8
    from fes_bench.metrics import SFace
    from fes_data import preprocess as P
    from fes_service import nafnet_route as NR
    from fes_service import rwgate as RW
    from fes_service.models import load_nafnet
    import scripts_path  # noqa: F401  (sys.path helper below)
    from r5_7_dev_collect import geometry, paste_back
    det = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    sface, dlib = SFace(), EV.DlibResNet()
    net, _ = load_nafnet()
    cfg = NR.load_config()

    def run(x):
        with torch.inference_mode():
            return rgb_t_to_bgr_u8(net(bgr_u8_to_rgb_t(x)).clamp(0, 1))
    recs = []
    for line in repo_path("experiments/phase5_rapid/outputs/r5_7/val_frame/records.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("fi_eligible"):
            recs.append(r)
    recs = sorted({r["name"]: r for r in recs}.values(), key=lambda r: r["name"])[: limit or None]
    rows = []
    for r in recs:
        name = r["name"]
        inp = cv2.imread(str(repo_path(f"{ARCH}/input/val/{name}")))
        gt = cv2.imread(str(repo_path(f"{ARCH}/output/val/{name}")))
        faces = det.detect(gt)
        if not faces:
            continue
        lm = np.asarray(faces[0].landmarks, np.float64)
        eg = (sface.embed(gt, lm), dlib.embed(gt, lm))
        e0 = (sface.embed(inp, lm), dlib.embed(inp, lm))
        chip, g = geometry(inp, det)
        if chip is None:
            continue
        o_zero = run(chip)
        padded, h, w = reflect16(chip)
        o_refl = run(padded)[:h, :w]
        row = {"name": name, "chip_px": chip.shape[0], "pad16": (-chip.shape[0]) % 16, "fi_pad_frac": g["fi_pad_frac"],
               "fc_eligible": bool(r.get("fc_eligible")), "fc_d_sface": r.get("fc_d_sface"),
               "fcreg_d_sface": r.get("fcreg_d_sface")}
        for tag, o in (("zero", o_zero), ("reflect", o_refl)):
            full = paste_back(inp, o, g)
            eo = (sface.embed(full, lm), dlib.embed(full, lm))
            row[f"{tag}_d_sface"] = round(float(eo[0] @ eg[0] - e0[0] @ eg[0]), 5)
            row[f"{tag}_d_dlib"] = round(float(eo[1] @ eg[1] - e0[1] @ eg[1]), 5)
            gr = NR.artifact_guard(chip, o, cfg, lambda x: len(det.detect(x)) > 0, RW.is_gray(chip))
            row[f"{tag}_guard_pass"], row[f"{tag}_guard"] = gr.passed, "+".join(gr.reasons)
        rows.append(row)
        if len(rows) % 50 == 0:
            print(len(rows), flush=True)
    write_csv_gz(repo_path("data/manifests/phase5_rapid/r5_7/frame_experiment.csv.gz"), rows)
    print(len(rows), "rows")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(repo_path("scripts")))
    sys.modules.setdefault("scripts_path", type(sys)("scripts_path"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args().limit)
