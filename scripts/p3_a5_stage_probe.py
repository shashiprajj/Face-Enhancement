"""P3-T7 — attribute the A5 chain collapse on dark inputs to a stage (diagnostic only; no protocol change).

    scripts/dock.sh python scripts/p3_a5_stage_probe.py

Runs the A5 components on selected main pairs and records, after each stage, statistics of the intermediate image
(mean/std per channel, Laplacian variance) and the deviation from the stage input. It also records the ×4 SR of each
intermediate, scored against the reference. Writes data/manifests/phase3/repro/a5_stage_probe.json. No images are
written.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch

from fes_bench import adapters as AD
from fes_bench import geometry as G
from fes_data.common import read_csv_gz, repo_path, run_info, write_json

PAIRS = ["ff-val-2699.ref-v1.gaussblur-mild", "ff-val-2699.ref-v1.jpeg-moderate", "ff-val-2699.ref-v1.motionblur-mild",
         "ff-val-1220.ref-v1.underexposure-severe", "ff-val-1594.ref-v1.gaussblur-mild"]   # last = control


def stats(t: torch.Tensor) -> dict:
    a = t[0].permute(1, 2, 0).numpy()
    g = cv2.cvtColor((a * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return {"mean_rgb": [round(float(x), 4) for x in a.mean((0, 1))], "std_rgb": [round(float(x), 4) for x in a.std((0, 1))],
            "lap_var": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2)}


def main() -> None:
    torch.set_num_threads(4)
    pop = {p["pair_id"]: p for p in read_csv_gz(repo_path("data/manifests/phase3/population_main.csv.gz"))}
    ch = AD.build("A5")
    ch.load()
    out = {"run": run_info(), "pairs": {}}
    for pid in PAIRS:
        p = pop[pid]
        deg = cv2.imread(str(repo_path(p["degraded_path"])))
        ref = cv2.imread(str(repo_path(p["ref_path"])))
        x0 = AD.bgr_u8_to_rgb_t(deg)
        with torch.inference_mode():
            x1, qf = ch.nets["M30"](x0)
            x1 = x1.clamp(0, 1)
            x2 = ch.nets[ch.nafnet_key](x1).clamp(0, 1)
            r = {"input": stats(x0), "after_fbcnn": stats(x1), "after_nafnet": stats(x2),
                 "fbcnn_pred_qf": round(float(1 - qf.item()) * 100, 1),
                 "max_abs_change_fbcnn": round(float((x1 - x0).abs().max()), 4),
                 "max_abs_change_nafnet": round(float((x2 - x1).abs().max()), 4)}
            for name, x in (("sr_of_input", x0), ("sr_after_fbcnn", x1), ("sr_after_nafnet", x2)):
                y = G.resize_to_e(AD.rgb_t_to_bgr_u8(AD._realesrgan_x4(ch.nets["M23"], x)))
                d = (y.astype(np.float64) - ref.astype(np.float64))
                r[name + "_psnr_vs_ref"] = round(float(10 * np.log10(255 ** 2 / max((d ** 2).mean(), 1e-9))), 2)
        out["pairs"][pid] = r
        print(pid, r["fbcnn_pred_qf"], r["input"]["mean_rgb"], "fbcnn lap", r["after_fbcnn"]["lap_var"],
              "nafnet lap", r["after_nafnet"]["lap_var"], "Δmax", r["max_abs_change_fbcnn"], r["max_abs_change_nafnet"],
              "psnr", r["sr_of_input_psnr_vs_ref"], r["sr_after_fbcnn_psnr_vs_ref"], r["sr_after_nafnet_psnr_vs_ref"],
              flush=True)
    write_json(repo_path("data/manifests/phase3/repro/a5_stage_probe.json"), out)


if __name__ == "__main__":
    main()
