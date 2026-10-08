"""R5-8 — automated artifact scan of every output RETURNED by the final service validation (observation only; nothing
is tuned on it). For each ENHANCED output of r58_test_{face_chip,full_image} (and, as control, the R5-6 runs):
chroma p99.9 vs its chip + every nafnet_route.local_features signal → data/manifests/phase5_rapid/r5_8/test_scan_<run>.csv.gz,
the fate of the 8 R5-7 escapes, and inspection sheets of the top outputs per signal (git-ignored).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_8_test_scan.py
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path, write_csv_gz

R56 = "data/manifests/phase5_rapid/r5_6"
IMG = "experiments/phase5_rapid/outputs/r5_6"
OUT = "data/manifests/phase5_rapid/r5_8"
R57_ESCAPES = {"face_chip": ["95636", "88924"], "full_image": ["8964", "96161", "97266", "90856", "85126", "96475"]}
SIGNALS = ("chroma_p999", "l_dot_c_max", "l_hfc_growth_max", "l_dot_l_max")


def _scan(item):
    from fes_service import nafnet_route as NR
    run, fn = item
    o = cv2.imread(os.path.join(repo_path(f"{IMG}/{run}/out"), fn))
    i = cv2.imread(os.path.join(repo_path(f"{IMG}/{run}/chip"), fn))
    if o is None or i is None or o.shape != i.shape:
        return None
    f = NR.local_features(i, o)
    return {"run": run, "file": fn, "chroma_p999": round(float(np.percentile(NR.chroma_dist(o, i), 99.9)), 3), **f}


def main():
    res = {}
    for run in ("r58_test_face_chip", "r58_test_full_image", "test_face_chip", "test_full_image"):
        d = repo_path(f"{IMG}/{run}/out")
        rec = {r["input_id"].replace("/", "__") + ".png": r for r in read_csv_gz(repo_path(f"{R56}/{run}.csv.gz"))}
        files = sorted(f for f in os.listdir(d) if rec.get(f, {}).get("status") == "ENHANCED")
        with ProcessPoolExecutor(12) as ex:
            rows = [r for r in ex.map(_scan, [(run, f) for f in files], chunksize=32) if r]
        write_csv_gz(repo_path(f"{OUT}/test_scan_{run}.csv.gz"), rows)
        res[run] = rows
        print(run, len(rows), {k: [round(float(np.quantile([r[k] for r in rows], q)), 2) for q in (0.99, 0.999, 1.0)]
                               for k in SIGNALS}, flush=True)
    # fate of the R5-7 escapes in the R5-8 candidate run
    fate = {}
    for mode, ids in R57_ESCAPES.items():
        recs = {r["input_id"].split("image_")[1][:-4]: r for r in read_csv_gz(repo_path(f"{R56}/r58_test_{mode}.csv.gz"))}
        fate[mode] = {i: {k: recs[i].get(k) for k in ("status", "fallback", "error", "reason")} for i in ids}
    json.dump(fate, open(repo_path(f"{OUT}/r57_escapes_fate.json"), "w"), indent=1)
    print(json.dumps(fate, indent=1))
    # inspection sheets: top 12 per signal per R5-8 run
    for run in ("r58_test_face_chip", "r58_test_full_image"):
        for k in SIGNALS:
            top = sorted(res[run], key=lambda r: -r[k])[:12]
            tiles = []
            for r in top:
                o = cv2.imread(os.path.join(repo_path(f"{IMG}/{run}/out"), r["file"]))
                i = cv2.imread(os.path.join(repo_path(f"{IMG}/{run}/chip"), r["file"]))
                t = np.hstack([cv2.resize(i, (256, 256), interpolation=cv2.INTER_NEAREST),
                               cv2.resize(o, (256, 256), interpolation=cv2.INTER_NEAREST)])
                cv2.putText(t, f"{r['file'].split('image_')[1][:-8]} {k}={r[k]:.1f}", (3, 14), 0, 0.42, (0, 0, 255), 1)
                tiles.append(t)
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(repo_path(f"experiments/phase5_rapid/outputs/r5_8/test_{run[4:]}_{k}.png")),
                        np.vstack([np.hstack(tiles[j:j + 3]) for j in range(0, len(tiles), 3)]))


if __name__ == "__main__":
    main()
