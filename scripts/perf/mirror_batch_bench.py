"""PERF Step 2, model level: the four mirror passes one by one (current) vs as one batch of 4, same worker settings.

    scripts/dock.sh python scripts/perf/mirror_batch_bench.py [--threads 4] [--n 24]      (FES_IMAGE=fes-dev:cpu-p5)

Both variants are the real factories (models.nafnet_factory_r6 / _r6_batched and the r6_ft pair), run in ONE process
on the same faces, alternating which goes first. Reported per face size: time per 4-run consensus call, output pixel
difference, per-run tracer difference, peak resident memory. Faces: experiments/r6_ft/selector/synth (128–224 px
blurred FairFace faces, not training data), resized to the listed sides.
Output: experiments/perf/mirror_batch_model.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fes_service import models as M  # noqa: E402

SYNTH = ROOT / "experiments" / "r6_ft" / "selector" / "synth" / "input"
OUT = ROOT / "experiments" / "perf" / "mirror_batch_model.json"
SIDES = (96, 128, 160, 192, 224)
PAIRS = {"r6": (M.nafnet_factory_r6, M.nafnet_factory_r6_batched),
         "r6_ft": (M.nafnet_factory_r6_ft, M.nafnet_factory_r6_ft_batched)}


def rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def main(a):
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    files = sorted(os.listdir(SYNTH))[:a.n]
    faces = [cv2.imread(str(SYNTH / f)) for f in files]
    res = {"threads": a.threads, "torch": torch.__version__, "n_faces": len(faces), "profiles": {}}
    for prof, (seq_f, bat_f) in PAIRS.items():
        seq, _ = seq_f()
        peak_seq_loaded = rss()
        bat, info = bat_f()
        assert info["mirror_batch"]
        warm = cv2.resize(faces[0], (224, 224))
        seq(warm, {}), bat(warm, {})
        rows = []
        for side in SIDES:
            ts, tb, dmax, dmean, ddot, ident = [], [], [], [], [], 0
            for k, f in enumerate(faces):
                x = cv2.resize(f, (side, side), interpolation=cv2.INTER_AREA)
                out = {}
                for name, fn in ((("seq", seq), ("bat", bat)) if k % 2 == 0 else (("bat", bat), ("seq", seq))):
                    t0 = time.perf_counter()
                    out[name] = fn(x, {})
                    (ts if name == "seq" else tb).append((time.perf_counter() - t0) * 1e3)
                d = np.abs(out["seq"][0].astype(np.int16) - out["bat"][0].astype(np.int16))
                dmax.append(int(d.max()))
                dmean.append(float(d.mean()))
                ident += int(d.max() == 0)
                ddot.append(max(abs(p - q) for p, q in zip(out["seq"][1]["runs_dot_c"], out["bat"][1]["runs_dot_c"])))
            rows.append({"side": side, "seq_ms_p50": round(st.median(ts), 1), "batch_ms_p50": round(st.median(tb), 1),
                         "batch_vs_seq": round(st.median(tb) / st.median(ts), 3), "seq_ms_max": round(max(ts), 1),
                         "batch_ms_max": round(max(tb), 1), "px_max": max(dmax), "px_mean": round(st.mean(dmean), 5),
                         "identical_faces": ident, "runs_dot_c_max_diff": round(max(ddot), 4)})
            print(prof, rows[-1], flush=True)
        res["profiles"][prof] = {"rows": rows, "peak_rss_mb_before_batch_model": peak_seq_loaded, "peak_rss_mb_end": rss()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=24)
    main(ap.parse_args())
