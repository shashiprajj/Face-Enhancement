"""P2-T6 — Build deterministic edge-case example inputs from clean references (git-ignored outputs).

    scripts/dock.sh python scripts/p2_make_preprocess_examples.py
Writes datasets/derived/examples/inputs/*.png; then run scripts/p2_preprocess.py on them.
"""

from __future__ import annotations

import cv2
import numpy as np

from fes_data import common as C

OUT = C.repo_path("datasets/derived/examples/inputs")


def main() -> None:
    rows = sorted((r for r in C.read_csv_gz(C.repo_path("data/manifests/references_v1.csv.gz"))
                   if r["accepted"] == "True" and r["pool"] == "test_pool"), key=lambda r: r["ref_id"])[:2]
    a, b = (cv2.imread(str(C.repo_path(r["ref_path"]))) for r in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    grey = lambda h, w: np.full((h, w, 3), 128, np.uint8)
    ex = {}
    ex["01_one_face"] = a
    two = grey(260, 520); two[18:242, 10:234] = a; two[55:205, 330:480] = cv2.resize(b, (150, 150), interpolation=cv2.INTER_AREA)
    ex["02_two_faces"] = two
    ex["03_no_face"] = np.random.default_rng(1).integers(0, 256, (300, 300, 3), dtype=np.uint8)
    tiny = grey(200, 200); tiny[80:108, 80:108] = cv2.resize(a, (28, 28), interpolation=cv2.INTER_AREA)
    ex["04_very_small_face"] = tiny
    ex["05_partial_face"] = a[:, 70:].copy()
    occ = a.copy(); occ[56:80, :] = 0
    ex["06_occluded_eyes"] = occ
    ex["07_already_cropped"] = cv2.resize(a[30:190, 35:190], (400, 400), interpolation=cv2.INTER_LINEAR)
    for name, img in ex.items():
        cv2.imwrite(str(OUT / f"{name}.png"), img)
    print(sorted(ex), "sources:", [r["ref_id"] for r in rows])


if __name__ == "__main__":
    main()
