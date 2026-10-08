"""R5-8 — local-feature study on the DEV corpus (archive val; scripts/r5_8_dev_corpus.py).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_8_local_study.py features
        → data/manifests/phase5_rapid/r5_8/local_features.csv.gz  (one row per NAFNet output: mode, name, corpus
          fields, global-guard result, every nafnet_route.local_features signal)
    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_8_local_study.py sheets
        → experiments/phase5_rapid/outputs/r5_8/label_*.png + label_queue.json (review set, see below)

Review set for visual labelling (DEV only; fixed before looking at any image):
  * every output in the top 40 of ANY local signal (per mode) and every output with global chroma p99.9 > 20
    (R5-7 lesson: cover the tails of every signal);
  * every output the global guard blocked;
  * a random sample of 60 other outputs per mode (seed 0) as the legitimate reference.
Labels (written by hand into data/manifests/phase5_rapid/r5_8/labels.json): LEGITIMATE, DESTRUCTIVE,
LOCALIZED_ARTIFACT, UNCERTAIN.
"""
from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path, write_csv_gz, write_json

ARCH = "Dataset-and-code/archive"
IMG = "experiments/phase5_rapid/outputs/r5_8"
OUT = "data/manifests/phase5_rapid/r5_8"
LOCAL_KEYS = ("l_dmean_max", "l_dmean_excess", "l_hfc_growth_max", "l_hfc_ratio_max", "l_hfc_growth_p999", "l_dot_c_max", "l_dot_l_max",
              "l_cc20_max", "l_cc30_max")


def _feat(item):
    from fes_service import nafnet_route as NR
    mode, name = item
    out = cv2.imread(str(repo_path(f"{IMG}/corpus/{mode}/{name}.png")))
    inp = cv2.imread(str(repo_path(f"{ARCH}/input/val/{name}") if mode == "fc"
                         else repo_path(f"{IMG}/corpus/fi_chip/{name}.png")))
    return {"mode": mode, "name": name, **NR.local_features(inp, out)}


def features():
    cv2.setNumThreads(1)
    rows = read_csv_gz(repo_path(f"{OUT}/corpus.csv.gz"))
    items = [(m, r["name"]) for r in rows for m in ("fc", "fi") if r.get(f"{m}_eligible") == "True"]
    by = {r["name"]: r for r in rows}
    with ProcessPoolExecutor(12) as ex:
        feats = list(ex.map(_feat, items, chunksize=64))
    out = []
    for f in feats:
        r, m = by[f["name"]], f["mode"]
        out.append({**f, "guard_pass": r[f"{m}_guard_pass"], "guard": r[f"{m}_guard"],
                    "g_chroma_p99": r.get(f"{m}_g_chroma_p99"), "g_chroma_p999": r.get(f"{m}_g_chroma_p999"),
                    "g_hf_color_ratio": r.get(f"{m}_g_hf_color_ratio"), "is_gray": r.get(f"{m}_is_gray"),
                    "oob_frac": r.get(f"{m}_oob_frac"), "chip_px": r.get(f"{m}_chip_px"),
                    "d_sface": r.get(f"{m}_d_sface"), "d_dlib": r.get(f"{m}_d_dlib")})
    write_csv_gz(repo_path(f"{OUT}/local_features.csv.gz"), out)
    print(len(out), "outputs")


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def sheets():
    rows = read_csv_gz(repo_path(f"{OUT}/local_features.csv.gz"))
    queue = {}
    for mode in ("fc", "fi"):
        R = [r for r in rows if r["mode"] == mode]
        sel = {}
        for k in LOCAL_KEYS:
            for r in sorted(R, key=lambda r: -fl(r[k]))[:40]:
                sel.setdefault(r["name"], set()).add(f"top:{k}")
        for r in R:
            if fl(r["g_chroma_p999"]) > 20:
                sel.setdefault(r["name"], set()).add("p999>20")
            if r["guard_pass"] != "True":
                sel.setdefault(r["name"], set()).add("global_blocked")
        rest = sorted(r["name"] for r in R if r["name"] not in sel)
        for n in random.Random(0).sample(rest, min(60, len(rest))):
            sel[n] = {"random"}
        queue[mode] = {n: sorted(v) for n, v in sorted(sel.items())}
    write_json(repo_path(f"{OUT}/label_queue.json"), queue)
    # sheets: 4 tiles per row, 6 rows per sheet; tile = input | output (2× nearest), index printed
    feats = {(r["mode"], r["name"]): r for r in rows}
    for mode, q in queue.items():
        names = list(q)
        for s in range(0, len(names), 24):
            tiles = []
            for i, n in enumerate(names[s:s + 24], start=s):
                out = cv2.imread(str(repo_path(f"{IMG}/corpus/{mode}/{n}.png")))
                inp = cv2.imread(str(repo_path(f"{ARCH}/input/val/{n}") if mode == "fc"
                                     else repo_path(f"{IMG}/corpus/fi_chip/{n}.png")))
                t = np.hstack([cv2.resize(inp, (160, 160), interpolation=cv2.INTER_NEAREST),
                               cv2.resize(out, (160, 160), interpolation=cv2.INTER_NEAREST)])
                f = feats[(mode, n)]
                cv2.putText(t, f"#{i} g{fl(f['l_hfc_growth_max']):.0f} d{fl(f['l_dmean_max']):.0f}", (2, 12), 0, 0.4,
                            (0, 0, 255), 1)
                tiles.append(t)
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(repo_path(f"{IMG}/label_{mode}_{s // 24:02d}.png")),
                        np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print({m: len(q) for m, q in queue.items()})


if __name__ == "__main__":
    {"features": features, "sheets": sheets}[sys.argv[1]]()
