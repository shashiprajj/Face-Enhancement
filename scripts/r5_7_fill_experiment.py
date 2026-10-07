"""R5-7 — DEV experiment for the full_image COVERAGE loss: the service fills the out-of-image part of a FairFace chip
with black (pipeline.face_chip, BORDER_CONSTANT 0); black edges inflate lapd1 (×4 median) and blockiness, so blurred
faces read as NOT_BLUR / MILD_BLUR / COMPRESSION. Candidate correction (NAFNet route only): reflect-fill that area for
the gate features and for NAFNet, then re-black it in the output.

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_7_fill_experiment.py

DEV only: every pair of the R5-7 framing sample (val, 3,000, seed 0) whose full_image chip has padding.
Per pair: gate decision with the black chip (service today) vs the reflect-filled chip; if the filled chip is eligible:
NAFNet (R5-7 wrapper: reflect pad to /16), out-of-image area re-blacked, R5-7 guard (1.1.0), identity vs GT exactly as
R5-6 (paste back), and the face_chip result for the same image for reference.
→ data/manifests/phase5_rapid/r5_7/fill_experiment.csv.gz
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from fes_data.common import repo_path, write_csv_gz

ARCH = "Dataset-and-code/archive"


def main() -> None:
    sys.path.insert(0, str(repo_path("scripts")))
    from r5_7_dev_collect import paste_back
    import torch
    torch.set_num_threads(4)
    from fes_bench import evaluators as EV
    from fes_bench.metrics import SFace
    from fes_data import preprocess as P
    from fes_service import nafnet_route as NR
    from fes_service import rwgate as RW
    from fes_service.models import nafnet_factory
    from fes_service.pipeline import face_chip
    det = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    sface, dlib = SFace(), EV.DlibResNet()
    naf, _ = nafnet_factory()
    cfg = NR.load_config()
    recs = {}
    for line in repo_path("experiments/phase5_rapid/outputs/r5_7/val_frame/records.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
            recs[r["name"]] = r
        except json.JSONDecodeError:
            pass
    todo = [r for r in sorted(recs.values(), key=lambda r: r["name"]) if (r.get("fi_pad_frac") or 0) > 0]
    rows = []
    for r in todo:
        name = r["name"]
        img = cv2.imread(str(repo_path(f"{ARCH}/input/val/{name}")))
        gt = cv2.imread(str(repo_path(f"{ARCH}/output/val/{name}")))
        fr = P.preprocess(img, det, P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
        lm = np.asarray(fr.detection.landmarks, np.float64)
        black, _ = face_chip(img, lm)
        mask = face_chip(np.full_like(img, 255), lm)[0][..., 0] < 128             # out-of-image area of the chip
        refl = cv2.copyMakeBorder(img, 300, 300, 300, 300, cv2.BORDER_REFLECT_101)
        filled, _ = face_chip(refl, lm + 300.0)
        row = {"name": name, "pad_frac": r["fi_pad_frac"], "fc_eligible": bool(r.get("fc_eligible")),
               "fc_d_sface": r.get("fc_d_sface"), "fc_d_dlib": r.get("fc_d_dlib"),
               "black_reason": r.get("fi_reason"), "mask_frac": round(float(mask.mean()), 4)}
        assert np.array_equal(filled[~mask], black[~mask]), name                  # identical where the image exists
        f = RW.rw_features(filled, det)
        f.update(NR.texture_features(filled))
        d = NR.decide(f, cfg)
        row.update(fill_reason=d.reason, fill_eligible=d.eligible, fill_lapd1=f.get("lapd1_c"),
                   black_lapd1=r.get("fi_lapd1_c"), fill_blockiness=f.get("blockiness"))
        if d.eligible:
            out, _ = naf(filled, {})
            out = out.copy()
            out[mask] = 0                                                           # re-black: geometry unchanged
            g = NR.artifact_guard(black, out, cfg, lambda x: len(det.detect(x)) > 0, RW.is_gray(filled))
            row.update(fill_guard_pass=g.passed, fill_guard=("+".join(g.reasons)), fill_chroma_p99=g.metrics.get("chroma_p99"))
            faces = det.detect(gt)
            if faces:
                gl = np.asarray(faces[0].landmarks, np.float64)
                eg, e0 = (sface.embed(gt, gl), dlib.embed(gt, gl)), (sface.embed(img, gl), dlib.embed(img, gl))
                x0 = {"fi_x0": r["fi_x0"], "fi_y0": r["fi_y0"], "fi_side": r["fi_side"]}
                full = paste_back(img, out, x0)
                eo = (sface.embed(full, gl), dlib.embed(full, gl))
                row["fill_d_sface"] = round(float(eo[0] @ eg[0] - e0[0] @ eg[0]), 5)
                row["fill_d_dlib"] = round(float(eo[1] @ eg[1] - e0[1] @ eg[1]), 5)
        rows.append(row)
        if len(rows) % 100 == 0:
            print(len(rows), "/", len(todo), flush=True)
    cols = list(dict.fromkeys(k for x in rows for k in x))
    write_csv_gz(repo_path("data/manifests/phase5_rapid/r5_7/fill_experiment.csv.gz"),
                 [{k: x.get(k, "") for k in cols} for x in rows], cols)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
