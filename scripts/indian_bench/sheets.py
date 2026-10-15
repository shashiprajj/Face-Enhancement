"""Phase 6 — indexed contact sheets (built from the stored clean / degraded / enhanced files, not from the JPEG
comparisons, so nothing is compressed twice).

    python scripts/indian_bench/sheets.py [--pilot] [--workers 8]

Every cell is one variant:  CLEAN | DEGRADED | FES OUTPUT  as three thumbnails in the same view as its individual
comparison (the service's face-chip box when a face was found, else the full frame), reduced with INTER_AREA.
The frame around the third thumbnail tells the outcome:
  green = enhanced · grey = not routed by the gate (input returned) · blue = no usable face (input returned) ·
  orange = safety guard fallback (input returned) · red = model failure. An empty dark cell = no result for it.

comparisons/contact_sheets/
  severity_by_original/<Expression>/<sample_id>.jpg   one sheet per original: rows = degradation types, columns =
                                                      levels 1-4 (covers EVERY variant of the benchmark)
  by_expression/<Expression>/page_NNN.jpg             rows = originals (all of them, 8 per page), columns = a fixed
                                                      set of representative variants (REPRESENTATIVE below)
  by_degradation/<type>/<Expression>.jpg              rows = the first ROWS_BY_DEGRADATION selected originals of the
                                                      expression in selection order, columns = levels
  INDEX.md / index.csv                                every sheet with what it contains
"""

from __future__ import annotations

import argparse
import functools
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common as C

T = 128                                             # thumbnail side
REPRESENTATIVE = [("gaussian_blur", 2), ("motion_blur_045", 3), ("defocus_blur", 3), ("camera_shake", 3),
                  ("low_light", 2), ("motion_noise_jpeg", 2)]
ROWS_BY_DEGRADATION = 12
ROWS_BY_EXPRESSION = 8
FRAME = {"enhanced": (120, 220, 90), "original_gate": (190, 190, 190), "original_no_face": (230, 170, 150),
         "fallback_guard": (70, 180, 255), "fallback_model_failure": (110, 110, 255)}      # BGR
LEGEND = ("cell = CLEAN | DEGRADED | FES OUTPUT.  Frame of the 3rd: green enhanced, grey gate did not route "
          "(input returned), blue no usable face, orange guard fallback, red model failure, empty = no result")


@functools.lru_cache(maxsize=64)
def _read(path: str):
    import cv2
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _crop(img, x0, y0, side):
    import cv2
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)) if pad else img
    return src[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]


def _thumb(img):
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    s = T / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((T, T, 3), np.uint8)
    y, x = (T - small.shape[0]) // 2, (T - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def cell(root: Path, c: dict | None):
    import cv2
    import numpy as np
    out = np.full((T + 8, 3 * T + 16, 3), 40, np.uint8)
    if c is None:
        return out
    clean, deg = _read(str(root / c["clean_path"])), cv2.imread(str(root / c["degraded_path"]), cv2.IMREAD_COLOR)
    res = deg if c["outcome"] != "enhanced" else cv2.imread(str(root / c["enhanced_path"]), cv2.IMREAD_COLOR)
    if c["view"] == "chip":
        box = (int(c["chip_x0"]), int(c["chip_y0"]), int(c["chip_side_src"]))
        tiles = [_crop(clean, *box), _crop(deg, *box), res if c["outcome"] == "enhanced" else _crop(res, *box)]
    else:
        tiles = [clean, deg, res]
    for i, t in enumerate(tiles):
        out[4:4 + T, 4 + i * (T + 4):4 + i * (T + 4) + T] = _thumb(t)
    x = 4 + 2 * (T + 4)
    cv2.rectangle(out, (x - 2, 2), (x + T + 1, T + 5), FRAME[c["outcome"]], 2)
    return out


def build_sheet(task) -> dict:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    root = Path(task["root"])
    ch, cw, left, top = T + 8, 3 * T + 16, 250, 78
    rows, cols = task["rows"], task["cols"]
    sheet = np.full((top + len(rows) * ch + 6, left + len(cols) * cw + 6, 3), 24, np.uint8)
    for i, row in enumerate(rows):
        for j, c in enumerate(row["cells"]):
            sheet[top + i * ch:top + (i + 1) * ch, left + j * cw:left + (j + 1) * cw] = cell(root, c)
    im = Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    try:
        f1, f2 = ImageFont.truetype("arial.ttf", 20), ImageFont.truetype("arial.ttf", 13)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    d.text((8, 6), task["title"], font=f1, fill=(255, 255, 255))
    d.text((8, 32), LEGEND, font=f2, fill=(180, 180, 180))
    for j, cl in enumerate(cols):
        d.text((left + j * cw + 6, 56), cl, font=f2, fill=(255, 255, 255))
    for i, row in enumerate(rows):
        for k, line in enumerate(row["label"]):
            d.text((8, top + i * ch + 10 + k * 18), line, font=f2, fill=(255, 255, 255))
    p = root / task["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, quality=90)
    return {"sheet": task["path"], "kind": task["kind"], "expression": task.get("expression", ""),
            "type": task.get("type", ""), "rows": len(rows), "columns": len(cols),
            "cells_filled": sum(c is not None for r in rows for c in r["cells"]), "width": im.width,
            "height": im.height, "contains": task["contains"]}


def main(a):
    cfg = C.load_config(a.config)
    root = cfg["bench_root"]
    cmp_rows = [r for r in C.read_csv(C.bench(cfg, "manifests", "comparisons.csv")) if not r["error"]]
    selected = C.scope(cfg, a.pilot)
    rank = {r["sample_id"]: int(r["selection_rank"]) for r in selected}
    by = defaultdict(dict)
    for r in cmp_rows:
        if r["sample_id"] in rank:
            by[r["sample_id"]][(r["type"], int(r["level"]))] = r
    matrix = C.read_csv(C.bench(cfg, "manifests", "generated_variants.csv"))
    types = list(dict.fromkeys(r["type"] for r in matrix))
    order = {r["type"]: r["order"] for r in matrix}
    levels = {t: sorted({int(r["level"]) for r in matrix if r["type"] == t}) for t in types}
    expr_of = {r["sample_id"]: r["expression"] for r in selected}
    tasks = []
    for sid in sorted(by):
        e = expr_of[sid]
        tasks.append({"root": root, "kind": "severity_by_original", "expression": e,
                      "path": f"comparisons/contact_sheets/severity_by_original/{e}/{sid}.jpg",
                      "title": f"{e} · {sid} · every degradation type (rows) at every severity level (columns)",
                      "cols": [f"level {l}" for l in (1, 2, 3, 4)], "contains": sid,
                      "rows": [{"label": [t, *[order[t][i:i + 34] for i in range(0, min(len(order[t]), 102), 34)]],
                                "cells": [by[sid].get((t, l)) for l in (1, 2, 3, 4)]} for t in types]})
    for e in sorted(set(expr_of.values())):
        sids = sorted(s for s in by if expr_of[s] == e)
        for page, i in enumerate(range(0, len(sids), ROWS_BY_EXPRESSION), 1):
            part = sids[i:i + ROWS_BY_EXPRESSION]
            tasks.append({"root": root, "kind": "by_expression", "expression": e,
                          "path": f"comparisons/contact_sheets/by_expression/{e}/page_{page:03d}.jpg",
                          "title": f"{e} · originals {i + 1}-{i + len(part)} of {len(sids)} · representative degradations",
                          "cols": [f"{t} L{l}" for t, l in REPRESENTATIVE], "contains": " ".join(part),
                          "rows": [{"label": [s], "cells": [by[s].get(k) for k in REPRESENTATIVE]} for s in part]})
        first = sorted((s for s in by if expr_of[s] == e), key=lambda s: rank[s])[:ROWS_BY_DEGRADATION]
        for t in types:
            tasks.append({"root": root, "kind": "by_degradation", "expression": e, "type": t,
                          "path": f"comparisons/contact_sheets/by_degradation/{t}/{e}.jpg",
                          "title": f"{t} ({order[t]}) · {e} · first {len(first)} selected originals (selection order)",
                          "cols": [f"level {l}" for l in levels[t]], "contains": " ".join(first),
                          "rows": [{"label": [s], "cells": [by[s].get((t, l)) for l in levels[t]]} for s in first]})
    print(f"{len(tasks)} sheets to build", flush=True)
    t0, index = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, rec in enumerate(ex.map(build_sheet, tasks, chunksize=2), 1):
            index.append(rec)
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)} · {time.time() - t0:.0f} s", flush=True)
    C.write_csv(C.bench(cfg, "comparisons", "contact_sheets", "index.csv"), index)
    md = ["# Contact sheets", "", LEGEND, ""]
    for kind in ("severity_by_original", "by_expression", "by_degradation"):
        part = [r for r in index if r["kind"] == kind]
        md += [f"## {kind} — {len(part)} sheets", "",
               C.md_table([{"sheet": f"[{r['sheet'].split('contact_sheets/')[1]}]({r['sheet'].split('contact_sheets/')[1]})",
                            "rows": r["rows"], "columns": r["columns"], "cells filled": r["cells_filled"]} for r in part],
                          ["sheet", "rows", "columns", "cells filled"]), ""]
    C.bench(cfg, "comparisons", "contact_sheets", "INDEX.md").write_text("\n".join(md), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s · {len(index)} sheets", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
