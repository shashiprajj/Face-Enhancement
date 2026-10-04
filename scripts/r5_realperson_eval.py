"""R5-2 — the user's REAL paired photos (Dataset-and-code/Real-Person-Test-Data): 1 identity, 1 clear reference,
5 degraded phone photos. Same three conditions as the LFW proxy; identity = SFace/dlib PROXY (not Feature-Match).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_realperson_eval.py

Privacy: personal photos. Nothing image-like is written to tracked paths — the manifest stores the generic file names
(A1 … A14), SHA-256 of the files, gate decisions and scores only. Chips/outputs go to the git-ignored
experiments/phase5_rapid/outputs/realperson/.
Conditions per degraded photo:
  A0 ORIGINAL (full photo)  degraded photo, own YuNet landmarks → identity (what a matcher sees without the service)
  A  ORIGINAL (service)     the service's original in the 224 E-frame (chip, bicubic)
  B  GATED                  frozen p5-rwgate-1.0.0 through the real R5-1 service (real isolated A4 if routed)
  C  FORCED A4              A4 on the same chip regardless of the gate (greyscale projection as in the service);
                            not run when the gate says UNSAFE (task rule)
Impostor gallery: the 480 LFW references of the LFW proxy (FAR-1e-2 thresholds from their clean impostor pairs).
Writes data/manifests/phase5_rapid/realperson/results.json.
"""
from __future__ import annotations

import glob
import hashlib

import cv2
import numpy as np

from fes_bench import evaluators as EV
from fes_bench import geometry as G
from fes_bench.adapters import build
from fes_bench.metrics import SFace
from fes_data import preprocess as P
from fes_data.common import read_csv_gz, repo_path, run_info, sha256_file, write_json
from fes_service import rwgate as RW
from fes_service.pipeline import EnhancementPipeline, face_chip
from fes_service.worker import IsolatedModel

DIR = "Dataset-and-code/Real-Person-Test-Data"


def main() -> None:
    import torch
    torch.set_num_threads(6)
    det = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    sf, dl = SFace(), EV.DlibResNet()

    def lm_of(img):
        d = det.detect(img)
        return np.asarray(d[0].landmarks, np.float64) if d else None

    def emb(img, lm):
        return None if lm is None else {"E01": sf.embed(img, lm), "E07": dl.embed(img, lm)}

    # gallery = LFW references (impostors) + the real person's reference
    gal = []
    for r in read_csv_gz(repo_path("data/manifests/phase5_rapid/lfw/selection.csv.gz")):
        img = cv2.imread(str(repo_path(r["ref_path"])), cv2.IMREAD_COLOR)
        e = emb(img, lm_of(img))
        if e:
            gal.append(e)
    M = {k: np.stack([g[k] for g in gal]) for k in ("E01", "E07")}
    thr = {k: float(np.quantile((M[k] @ M[k].T)[np.triu_indices(len(gal), 1)], 0.99)) for k in M}
    ref_path = sorted(glob.glob(str(repo_path(f"{DIR}/Clear image/*.jpg"))))[0]
    ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    R = emb(ref, lm_of(ref))
    assert R is not None, "no face on the reference"

    def score(e):
        if e is None:
            return {k: {"cos": None, "match": False, "rank1": False, "imp_max": None, "fa": False} for k in M}
        out = {}
        for k in M:
            g = float(R[k] @ e[k])
            imp = M[k] @ e[k]
            out[k] = {"cos": round(g, 5), "match": g >= thr[k], "rank1": bool(g > imp.max()),
                      "imp_max": round(float(imp.max()), 5), "fa": bool(imp.max() >= thr[k])}
        return out

    a4 = build("A4")
    a4.load()
    model = IsolatedModel("fes_service.models:a4_factory", timeout_s=600, threads=6)
    pipe = EnhancementPipeline(model, timeout_s=600, gate="rw")
    assert pipe.seals["rw_gate"].startswith("78837a8d")
    out_dir = repo_path("experiments/phase5_rapid/outputs/realperson")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    import os
    import re
    natural = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]  # noqa: E731
    for f in sorted(glob.glob(str(repo_path(f"{DIR}/Degraded image/*.jpg"))), key=natural):
        qid = os.path.splitext(os.path.basename(f))[0]          # generic file names (A1 … A14), not personal
        data = open(f, "rb").read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        res = pipe.process(data, "full_image")
        row = {"id": qid, "sha256": sha256_file(f), "decision": res.decision, "reason": res.reason,
               "category": res.bucket, "route": res.route, "used": res.used, "fallback": res.fallback,
               "status": res.status, "n_faces": res.n_faces, "chip_px": res.chip_px, "warnings": res.warnings,
               "quality": res.quality, "A0_full_photo": score(emb(img, lm_of(img)))}
        fr = P.preprocess(img, pipe.det, pipe.det_cfg, input_mode="face_crop")[0]
        if res.decision != "UNSAFE_OR_UNCERTAIN" and fr.status == P.OK and fr.aligned:
            chip, _ = face_chip(img, np.asarray(fr.detection.landmarks, np.float64))
            b0 = G.upsample_to_e(chip)
            c, _ = a4.enhance(chip, {"s": chip.shape[0]})
            if RW.is_gray(chip):
                c = cv2.cvtColor(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            b = cv2.imdecode(np.frombuffer(res.image, np.uint8), cv2.IMREAD_COLOR) if res.status == "ENHANCED" else b0
            lm = lm_of(b0)
            if lm is None:
                lm = lm_of(c)
            row.update(A_service_original=score(emb(b0, lm)), B_gated=score(emb(b, lm)), C_forced_a4=score(emb(c, lm)),
                       B_equals_C_pixels=bool(res.status == "ENHANCED" and np.array_equal(b, c)))
            cv2.imwrite(str(out_dir / f"{qid}_chip_orig.png"), b0)
            cv2.imwrite(str(out_dir / f"{qid}_forcedA4.png"), c)
        else:
            row.update(A_service_original=row["A0_full_photo"], B_gated=row["A0_full_photo"], C_forced_a4=None)
        rows.append(row)
        print(qid, res.decision, res.reason, res.route, res.status, flush=True)
    model.close()
    write_json(repo_path("data/manifests/phase5_rapid/realperson/results.json"), {
        "run": run_info(), "proxy": "SFace E01 + dlib E07 — NOT Feature-Match", "identities": 1,
        "reference_sha256": sha256_file(ref_path), "gallery": f"{len(gal)} LFW references (impostors) + this reference",
        "thr_far_1e-2": thr, "rows": rows,
        "privacy": "no images or crops stored; ids are the generic file names supplied by the user (A1…)"})


if __name__ == "__main__":
    main()
