"""Full-photo side-by-side for the ENHANCED answers:  [CLEAN ORIGINAL] [DEGRADED INPUT] [DEGRADED + FES FACE PASTED BACK].

    python scripts/indian_bench/full_photo.py

The service returns only the enhanced face chip. Here that chip is placed back at the exact box it was cut from
(comparisons.csv: chip_x0, chip_y0, chip_side_src) in a copy of the degraded photo; everything outside the box is the
degraded input, untouched. A chip the service reduced to 224 px (box side > 224) is enlarged back to the box size with
bicubic interpolation — that enlargement is done here, not by the service (column `chip_upscaled` in the manifest).
This composite is a DISPLAY made by the benchmark, not a service output. → ENHANCED_ONLY_full_photo/ (flat folder).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common as C
import compare as CMP


def one(args) -> dict:
    cfg, r, v = args
    import cv2
    root = Path(cfg["bench_root"])
    clean = cv2.imread(str(root / r["clean_path"]), cv2.IMREAD_COLOR)
    deg = cv2.imread(str(root / r["degraded_path"]), cv2.IMREAD_COLOR)
    chip = cv2.imread(str(root / r["enhanced_path"]), cv2.IMREAD_COLOR)
    x0, y0, side = int(r["chip_x0"]), int(r["chip_y0"]), int(r["chip_side_src"])
    up = chip.shape[0] != side
    if up:
        chip = cv2.resize(chip, (side, side), interpolation=cv2.INTER_CUBIC)
    h, w = deg.shape[:2]
    out = deg.copy()
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    out[ya:yb, xa:xb] = chip[ya - y0:yb - y0, xa - x0:xb - x0]
    header = (f"{r['expression']} · {r['sample_id']} · {r['type']} · level {r['level']} ({r['category']})",
              f"{v['order']}   |   FES ENHANCED the face; shown placed back into the full photo")
    footer = (f"3rd panel = degraded photo with the FES face chip pasted at x0={x0} y0={y0} side={side} px"
              + (" (chip enlarged 224 -> box size, bicubic)" if up else " (pixel-exact)") + "; rest of the photo is the input")
    im = CMP.triptych([clean, deg, out], ["CLEAN ORIGINAL", "DEGRADED INPUT", "FES ENHANCED FACE (in full photo)"],
                      header, footer, CMP.COLOURS["enhanced"])
    name = f"{r['sample_id']}__{r['type']}__L{r['level']}.jpg"
    p = root / "ENHANCED_ONLY_full_photo" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, quality=cfg["comparison"]["jpeg_quality"], subsampling=0)
    return {"variant_id": r["variant_id"], "full_photo_path": f"ENHANCED_ONLY_full_photo/{name}", "chip_upscaled": up}


def main():
    cfg = C.load_config()
    var = {v["variant_id"]: v for v in C.read_csv(C.bench(cfg, "manifests", "generated_variants.csv"))}
    rows = [r for r in C.read_csv(C.bench(cfg, "manifests", "comparisons.csv")) if r["outcome"] == "enhanced" and not r["error"]]
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(one, [(cfg, r, var[r["variant_id"]]) for r in rows], chunksize=8))
    C.write_csv(C.bench(cfg, "manifests", "full_photo_composites.csv"), out)
    print(len(out), "full-photo comparisons ·", sum(o["chip_upscaled"] for o in out), "with an enlarged chip")


if __name__ == "__main__":
    main()
