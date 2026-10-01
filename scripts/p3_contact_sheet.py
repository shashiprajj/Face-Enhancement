"""P3-T4/P3-T7 — contact sheets for visual review (written to the git-ignored outputs tree; no faces in git).

    python scripts/p3_contact_sheet.py --population trial --ref ff-train-13002.ref-v1 --configs B0 A3 [--cells ...]
    python scripts/p3_contact_sheet.py --population main --pairs <pair_id> ... --configs B0 A1 A2 A3 A4 A5 --name x

Columns: clean reference | configs...; one row per pair. Each tile is the 224×224 E-frame image, labelled.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path

OUT = "experiments/phase3/outputs/sheets"


def tile(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (224, 16), (0, 0, 0), -1)
    cv2.putText(img, text[:34], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--name")
    args = ap.parse_args()
    pop = read_csv_gz(repo_path(f"data/manifests/phase3/population_{args.population}.csv.gz"))
    if args.pairs:
        rows = [p for p in pop if p["pair_id"] in set(args.pairs)]
        rows.sort(key=lambda p: args.pairs.index(p["pair_id"]))
    else:
        rows = [p for p in pop if p["ref_id"] == args.ref and (not args.cells or p["cell"] in args.cells)]
    lines = []
    for p in rows:
        ref = cv2.imread(str(repo_path(p["ref_path"])))
        cols = [tile(ref, f"REF {p['cell']}")]
        for c in args.configs:
            f = repo_path(f"experiments/phase3/outputs/{args.population}/{c}/{p['cell']}/{p['ref_id']}.png")
            im = cv2.imread(str(f)) if f.exists() else np.full((224, 224, 3), 60, np.uint8)
            cols.append(tile(im, c))
        lines.append(np.hstack(cols))
    sheet = np.vstack(lines)
    name = args.name or f"{args.ref}_{'-'.join(args.configs)}"
    dst = repo_path(f"{OUT}/{args.population}/{name}.jpg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(dst)


if __name__ == "__main__":
    main()
