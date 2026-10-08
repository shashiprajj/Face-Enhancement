"""R6-0 — per-RUN artifact tracers for the 4 NAFNet runs (identity, h, v, hv flips) of every corpus case (host).

    python scripts/r6_0/run_signals.py  ->  data/manifests/phase5_rapid/r6_0/run_signals.csv.gz

For each run vs the input chip: colour dottedness (t_dot_c_max, the R5-8 signal), luma dottedness, HF-colour growth,
chroma p99.9. Rationale: NAFNet-GoPro is unstable on some inputs; mirroring the input (which must not matter for a
deblur) then makes one or more runs explode into a dotted / grid / noise pattern. Such an explosion carries colour
even when it is later removed by the luma-only output, so the colour signals of EACH raw run are a tracer of the
unstable regime.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signals as S  # noqa: E402
from sheet import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def run_features(chip, runs, mask):
    valid = np.ones(chip.shape[:2], bool) if mask is None else ~mask
    out = {}
    for k, v in zip(("v0", "vh", "vv", "vhv"), runs):
        t = S.texture(chip, v, valid)
        p = S.photometric(chip, v, valid)
        out[k] = {"dot_c": t["t_dot_c_max"], "dot_l": t["t_dot_l_max"], "hfcg": t["t_hf_colour_growth"],
                  "c999": p["p_chroma_p999"]}
    return out


def main():
    cases = json.loads((ROOT / "data/manifests/phase5_rapid/r6_0/cases.json").read_text())
    rows = []
    for c in cases:
        chip, work, v0, fl, mask = load(c["id"])
        if v0 is None:
            continue
        for k, f in run_features(chip, [v0] + fl, mask).items():
            rows.append({"id": c["id"], "run": k, **{a: round(b, 5) for a, b in f.items()}})
    with gzip.open(ROOT / "data/manifests/phase5_rapid/r6_0/run_signals.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows))


if __name__ == "__main__":
    main()
