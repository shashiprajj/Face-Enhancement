"""R6-0 — compact labelling sheets (host). Each row: input chip | returned output v0 | flip-median + luma-only (medluma),
each upscaled to 256 px (nearest neighbour). The row index is drawn on every row; the sheet index json maps rows to cases.

    python scripts/r6_0/label_sheets.py PREFIX FILTER_EXPR [ROWS_PER_SHEET]
    FILTER_EXPR is a Python expression over case `c`, e.g. "c['prior']=='UNLABELLED' and c['src']=='p4'"
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signals as S  # noqa: E402
from sheet import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODE = "all"   # all | out (chip | v0) | mit (chip | medluma); env R6_SHEET_MODE


def row(i, cid, size=256):
    chip, work, v0, fl, mask = load(cid)
    ml = S.luma_only(work, S.flip_median(v0, fl))
    if mask is not None:
        ml[mask] = 0
    z = lambda im: cv2.resize(im, (size, size), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    sep = np.zeros((size, 4, 3), np.uint8)
    cols = [z(chip), sep, z(v0)] if MODE == "out" else [z(chip), sep, z(v0), sep, z(ml)] if MODE == "all" else [z(chip), sep, z(ml)]
    r = np.hstack(cols)
    cv2.putText(r, f"{i}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return r


def main(prefix, expr, per=8):
    global MODE
    import os
    MODE = os.environ.get("R6_SHEET_MODE", "all")
    cases = [c for c in json.loads((ROOT / "data/manifests/phase5_rapid/r6_0/cases.json").read_text()) if eval(expr)]
    idx = {}
    for s in range(0, len(cases), per):
        rows = []
        for i, c in enumerate(cases[s:s + per], start=s):
            rows.append(row(i, c["id"]))
            rows.append(np.zeros((4, rows[-1].shape[1], 3), np.uint8))
            idx[i] = c["id"]
        cv2.imwrite(f"{prefix}_{s // per:02d}.png", np.vstack(rows))
    Path(f"{prefix}_index.json").write_text(json.dumps(idx, indent=0))
    print(len(cases), "cases,", (len(cases) + per - 1) // per, "sheets")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
