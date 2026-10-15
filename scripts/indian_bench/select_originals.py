"""Phase 2 — seeded 50 % selection inside every expression folder.

    python scripts/indian_bench/select_originals.py

Procedure (per expression folder, independently):
  1. take the ELIGIBLE rows of manifests/source_inventory.csv, sorted by relative path;
  2. permute them with numpy PCG64 `default_rng(stable_seed(config.seed, "select", expression)).permutation(n)`;
  3. the first k = floor(n * fraction + 0.5) of the permutation are SELECTED (round half up), the rest are HOLDOUT.
Nothing about the image content or the service's likely behaviour enters the choice. The manifests are written before
any degradation is generated; an existing selection made with another seed / fraction / inventory is never overwritten
silently (use --force).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np

import common as C

FIELDS = ["sample_id", "expression", "rel_path", "filename", "sha256", "bytes", "width", "height", "mode",
          "exif_orientation", "ref_width", "ref_height", "face_usable", "flags", "selection_rank", "split"]


def main(a):
    cfg = C.load_config(a.config)
    inv = C.read_csv(C.bench(cfg, "manifests", "source_inventory.csv"))
    by = defaultdict(list)
    for r in inv:
        by[r["expression"]].append(r)
    sel, hold, table = [], [], []
    for e in sorted(by):
        elig = sorted((r for r in by[e] if r["eligible"] == "True"), key=lambda r: r["rel_path"])
        n = len(elig)
        k = math.floor(n * cfg["select_fraction"] + 0.5)
        perm = np.random.default_rng(C.stable_seed(cfg["seed"], "select", e)).permutation(n)
        for rank, idx in enumerate(perm):
            row = dict(elig[idx], selection_rank=rank, split="selected" if rank < k else "holdout")
            (sel if rank < k else hold).append(row)
        table.append({"expression": e, "total": len(by[e]), "eligible": n, "excluded": len(by[e]) - n, "selected": k,
                      "holdout": n - k, "selected with usable frontal face":
                          sum(r["face_usable"] == "True" for r in sel if r["expression"] == e)})
    table.append({"expression": "TOTAL", **{f: sum(t[f] for t in table) for f in list(table[0])[1:]}})
    assert len({r["sample_id"] for r in sel}) == len(sel) == len({r["rel_path"] for r in sel}), "duplicate selection"
    assert len({r["sha256"] for r in sel}) == len(sel), "two selected files with identical content"
    assert not {r["sample_id"] for r in sel} & {r["sample_id"] for r in hold}
    sel.sort(key=lambda r: r["sample_id"])
    hold.sort(key=lambda r: r["sample_id"])

    out = C.bench(cfg, "manifests", "selected_originals.csv")
    meta = {"seed": cfg["seed"], "fraction": cfg["select_fraction"], "rounding": cfg["select_rounding"],
            "rng": "numpy.random.default_rng (PCG64), seed = first 4 bytes of sha256('<seed>|select|<expression>')",
            "numpy": np.__version__, "selected": len(sel), "holdout": len(hold),
            "selected_ids_sha256": C.sha256_bytes("\n".join(r["sample_id"] for r in sel).encode())}
    meta_path = C.bench(cfg, "manifests", "selection_meta.json")
    if out.exists() and not a.force:
        old = json.loads(meta_path.read_text())
        if old["selected_ids_sha256"] != meta["selected_ids_sha256"]:
            raise SystemExit("a DIFFERENT selection already exists (seed / fraction / inventory changed); "
                             "use --force to replace it")
        print("existing selection reproduced exactly (same ids)")
    C.write_csv(out, sel, FIELDS)
    C.write_csv(C.bench(cfg, "manifests", "holdout_originals.csv"), hold, FIELDS)
    meta_path.write_text(json.dumps(meta, indent=1))
    md = ["# Phase 2 — selection", "", f"Seed `{cfg['seed']}`, fraction {cfg['select_fraction']}, rounding: "
          f"{cfg['select_rounding']}.", "", (__doc__ or "").split("Procedure")[1].split("Nothing")[0].join(
              ["Procedure", ""]), "", C.md_table(table, list(table[0])), "",
          f"Selected ids SHA-256: `{meta['selected_ids_sha256']}`", ""]
    C.bench(cfg, "reports", "02_selection.md").write_text("\n".join(md), encoding="utf-8")
    print(C.md_table(table, list(table[0])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
