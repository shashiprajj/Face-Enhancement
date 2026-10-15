"""R6-BEST — whole-face synthetic blur pairs for the selector study (FairFace VAL faces not among the 300 used to pick
the fine-tuned checkpoint): the training degradations of train_nafnet_face.py (gaussian / defocus / motion, never
"none") on the whole 224 px face resized to a random 128–224 px chip.

    python scripts/r6_ft/make_synth_set.py --data DIR --out experiments/r6_ft/selector/synth [--n 160]
Writes <out>/input/NNN_<kind>.png (blurred) and <out>/sharp/NNN_<kind>.png.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_nafnet_face import ZIP, degrade  # noqa: E402


def main(a):
    with zipfile.ZipFile(os.path.join(a.data, ZIP)) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg") and n.startswith("val/"))
        step = max(1, len(names) // 300)
        used = set(names[::step][:300])                        # the checkpoint-selection faces: excluded
        pick = [n for n in names[step // 2::step] if n not in used][:a.n]
        for sub in ("input", "sharp"):
            os.makedirs(os.path.join(a.out, sub), exist_ok=True)
        for i, n in enumerate(pick):
            rng = np.random.default_rng(91_000 + i)
            img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            side = int(rng.integers(128, 225))
            sharp = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            kind = "none"
            while kind == "none":
                deg, kind = degrade(sharp, rng)
            for sub, x in (("input", deg), ("sharp", sharp)):
                cv2.imwrite(os.path.join(a.out, sub, f"{i:03d}_{kind}.png"), (x[..., ::-1] * 255 + 0.5).astype(np.uint8))
    print(f"wrote {len(pick)} pairs to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=160)
    main(ap.parse_args())
