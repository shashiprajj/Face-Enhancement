"""P3-T6 — SFace evaluator noise floor (TH-1 *method*, P0-T4 §10). Dev data only (112 dev_pilot references).

Identity-neutral perturbations of the clean reference; 1 − cos(ref, perturbed) shows how much similarity loss the
evaluator produces with no identity change at all. This is measured and reported; it is NOT turned into a threshold.

    scripts/dock.sh python scripts/p3_identity_noise_floor.py
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from fes_bench import geometry as G
from fes_bench import metrics as MT
from fes_data.common import read_csv_gz, repo_path, run_info, sha256_file, write_json


def perturbations(ref: np.ndarray, M: np.ndarray) -> dict[str, np.ndarray]:
    def jpeg(q):
        return cv2.imdecode(cv2.imencode(".jpg", ref, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    shift = cv2.warpAffine(ref, np.float32([[1, 0, 1], [0, 1, 1]]), (224, 224), borderMode=cv2.BORDER_REFLECT)
    f = G.warp(ref, M, (G.F_SIZE, G.F_SIZE), cv2.INTER_CUBIC)
    rt = G.warp(f, G.f_to_e(M), (G.E_SIZE, G.E_SIZE))
    cov = G.coverage_mask(M)
    rt = np.where(cov[..., None], rt, ref)
    down = cv2.resize(cv2.resize(ref, (112, 112), interpolation=cv2.INTER_AREA), (224, 224),
                      interpolation=cv2.INTER_CUBIC)
    return {"jpeg_q90": jpeg(90), "jpeg_q75": jpeg(75), "shift_1px": shift,
            "harness_roundtrip_E_F_E": rt, "down2_up_bicubic": down}


def main() -> None:
    cv2.setNumThreads(2)
    pop = read_csv_gz(repo_path("data/manifests/phase3/population_noise_floor.csv.gz"))
    sf = MT.SFace()
    res: dict[str, list[float]] = {}
    seen = set()
    for p in pop:
        if p["ref_id"] in seen:
            continue
        seen.add(p["ref_id"])
        ref = cv2.imread(str(repo_path(p["ref_path"])))
        lm = np.array(json.loads(p["det_landmarks"]))
        M = np.array(json.loads(p["ffhq512_transform_2x3"]))
        e0 = sf.embed(ref, lm)
        for k, img in perturbations(ref, M).items():
            res.setdefault(k, []).append(float(e0 @ sf.embed(img, lm)))
    out = {"run": run_info(), "evaluator": MT.SFACE, "evaluator_sha256": sha256_file(repo_path(MT.SFACE)),
           "population": "noise_floor (dev_pilot, 112 refs)", "n_refs": len(seen),
           "note": "1-cos under identity-neutral perturbations. Descriptive; no threshold is derived (TH-1 TBD).",
           "perturbations": {}}
    for k, v in res.items():
        d = 1 - np.array(v)
        out["perturbations"][k] = {"cos_mean": round(float(np.mean(v)), 5),
                                   "one_minus_cos": {q: round(float(np.quantile(d, q)), 5)
                                                     for q in (0.5, 0.9, 0.95, 0.99, 1.0)}}
        print(k, out["perturbations"][k], flush=True)
    write_json(repo_path("data/manifests/phase3/identity_noise_floor.json"), out)


if __name__ == "__main__":
    main()
