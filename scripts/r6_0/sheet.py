"""R6-0 — zoomed review sheets from the variant cache (host; numpy/OpenCV only).

    python scripts/r6_0/sheet.py OUT.png CASE_ID [CASE_ID ...]
Columns: input chip | NAFNet output (v0) | flip-mean (x4) | flip-median (x4) | flip-median + luma-only | flip disagreement (x4 gain) |
|v0 - in| (x4 gain). Each case is upscaled to ~192 px with nearest neighbour, so artifacts stay visible.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signals as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "experiments/phase5_rapid/outputs/r6_0/cache"


def load(cid):
    d = CACHE / cid.replace(":", "__").replace("/", "_")
    rd = lambda k: cv2.imread(str(d / f"{k}.png"), cv2.IMREAD_COLOR)  # noqa: E731
    m = cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE) if (d / "mask.png").exists() else None
    work = rd("work") if (d / "work.png").exists() else rd("chip")
    return rd("chip"), work, rd("v0"), [rd("vh"), rd("vv"), rd("vhv")], (m > 127) if m is not None else None


def row(cid, size=256):
    chip, work, v0, fl, mask = load(cid)
    fm = S.flip_mean(v0, fl)
    md = S.flip_median(v0, fl)
    fml = S.luma_only(work, md)
    if mask is not None:
        fml[mask] = 0
    valid = np.ones(chip.shape[:2], bool) if mask is None else ~mask
    dl, dc = S.flip_maps(v0, fl, valid)
    dis = np.clip((dl + dc) * 4, 0, 255).astype(np.uint8)
    dis = cv2.applyColorMap(dis, cv2.COLORMAP_INFERNO)
    dif = np.clip(np.abs(v0.astype(int) - chip.astype(int)) * 4, 0, 255).astype(np.uint8)
    s = max(1, size // chip.shape[0])
    z = lambda im: cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)  # noqa: E731
    r = np.hstack([z(x) for x in (chip, v0, fm, md, fml, dis, dif)])
    cv2.putText(r, cid[-40:], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return r


def main(out, ids):
    rows = [row(c) for c in ids]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 3), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(out, np.vstack(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
