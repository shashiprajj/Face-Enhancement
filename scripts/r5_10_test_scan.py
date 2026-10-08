"""R5-10 — artifact scan of every output RETURNED by the frozen R5-10 service validation (observation only; nothing is
tuned on it). Same signals as scripts/r5_8_test_scan.py (chroma p99.9 vs the chip + nafnet_route.local_features), for
r510_test_{face_chip,full_image}; each output is tagged NEW (R5-8 returned ORIGINAL for that input) or SHARED.
Inspection sheets of the top NEW outputs per signal go to experiments/phase5_rapid/outputs/r5_10/ (git-ignored).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_10_test_scan.py
    → data/manifests/phase5_rapid/r5_10/test_scan_<run>.csv.gz + test_scan_summary.json
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path, write_csv_gz

sys.path.insert(0, str(repo_path("scripts")))
from r5_8_test_scan import SIGNALS, _scan  # noqa: E402  (same scan function, same signals)

R56 = "data/manifests/phase5_rapid/r5_6"
IMG = "experiments/phase5_rapid/outputs/r5_6"
OUT = "data/manifests/phase5_rapid/r5_10"
SHEETS = "experiments/phase5_rapid/outputs/r5_10"


def main():
    summary = {}
    for mode in ("face_chip", "full_image"):
        run = f"r510_test_{mode}"
        rec = {r["input_id"].replace("/", "__") + ".png": r for r in read_csv_gz(repo_path(f"{R56}/{run}.csv.gz"))}
        r58 = {r["input_id"].replace("/", "__") + ".png": r["status"]
               for r in read_csv_gz(repo_path(f"{R56}/r58_test_{mode}.csv.gz"))}
        files = sorted(f for f in os.listdir(repo_path(f"{IMG}/{run}/out")) if rec.get(f, {}).get("status") == "ENHANCED")
        with ProcessPoolExecutor(12) as ex:
            rows = [r for r in ex.map(_scan, [(run, f) for f in files], chunksize=32) if r]
        for r in rows:
            r["new"] = r58.get(r["file"]) != "ENHANCED"
            r["lapd1_c"] = rec[r["file"]].get("q_lapd1_c")
        write_csv_gz(repo_path(f"{OUT}/test_scan_{run}.csv.gz"), rows)
        q = lambda rs, k: {str(p): round(float(np.quantile([r[k] for r in rs], p)), 2) for p in (0.99, 0.999, 1.0)} if rs else {}  # noqa: E731
        new, old = [r for r in rows if r["new"]], [r for r in rows if not r["new"]]
        summary[run] = {"returned": len(rows), "new": len(new), "shared": len(old),
                        "new_quantiles": {k: q(new, k) for k in SIGNALS},
                        "shared_quantiles": {k: q(old, k) for k in SIGNALS}}
        os.makedirs(repo_path(SHEETS), exist_ok=True)
        for k in SIGNALS:
            tiles = []
            for r in sorted(new, key=lambda r: -r[k])[:12]:
                o = cv2.imread(os.path.join(repo_path(f"{IMG}/{run}/out"), r["file"]))
                i = cv2.imread(os.path.join(repo_path(f"{IMG}/{run}/chip"), r["file"]))
                t = np.hstack([cv2.resize(i, (256, 256), interpolation=cv2.INTER_NEAREST),
                               cv2.resize(o, (256, 256), interpolation=cv2.INTER_NEAREST)])
                cv2.putText(t, f"{r['file'].split('image_')[1][:-8]} {k}={r[k]:.1f}", (3, 14), 0, 0.42, (0, 0, 255), 1)
                tiles.append(t)
            while tiles and len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            if tiles:
                cv2.imwrite(str(repo_path(f"{SHEETS}/test_new_{mode}_{k}.png")),
                            np.vstack([np.hstack(tiles[j:j + 3]) for j in range(0, len(tiles), 3)]))
        print(run, json.dumps(summary[run]), flush=True)
    json.dump(summary, open(repo_path(f"{OUT}/test_scan_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
