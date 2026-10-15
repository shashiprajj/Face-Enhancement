"""Phase 3 — clean references + degraded variants (resumable, deterministic).

    python scripts/indian_bench/degrade.py --pilot            the pilot subset (see pilot_samples)
    python scripts/indian_bench/degrade.py                    every selected original
    python scripts/indian_bench/degrade.py --limit-samples N  the next N unfinished samples (batching)

Per selected original:
  1. the source file is read (never written) and its SHA-256 is compared with the selection manifest;
  2. the clean reference is built: EXIF-transposed RGB, area-downscaled so the long side is config.reference.long_side,
     stored lossless at clean_originals/<Expression>/<sample_id>.png;
  3. every variant of the matrix (tiers in config.matrix_tiers) is generated from the reference with
     seed = stable_seed(config.seed, sample_id, variant_id) and stored at
     degraded/<Expression>/<sample_id>/<type>/level_<n>.<png|jpg>.
Records are appended to manifests/generated_variants.jsonl / clean_references.jsonl as each sample finishes and are
turned into the CSV manifests at the end. A sample whose records and files all exist is skipped on a re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common as C


def load_matrix(cfg: dict) -> tuple[dict, list[dict]]:
    """→ (matrix json, flat list of variant specs in matrix order)."""
    m = json.loads((C.ROOT / cfg["matrix"]).read_text(encoding="utf-8"))
    specs = []
    for t in m["types"]:
        if t["tier"] not in cfg["matrix_tiers"]:
            continue
        for lvl, ops in t["levels"].items():
            specs.append({"type": t["type"], "category": t["category"], "level": int(lvl), "order": t["order"],
                          "ops": ops})
    assert len({(s["type"], s["level"]) for s in specs}) == len(specs)
    return m, specs


def variant_id(sample_id: str, spec: dict) -> str:
    return f"{sample_id}__{spec['type']}__L{spec['level']}"


def pilot_samples(selected: list[dict], frontal: int = 2, other: int = 1) -> list[dict]:
    """Per expression: the first `frontal` selected originals (selection order) with a usable frontal face and the
    first `other` without one. Chosen by selection rank only."""
    out = []
    for e in sorted({r["expression"] for r in selected}):
        g = sorted((r for r in selected if r["expression"] == e), key=lambda r: int(r["selection_rank"]))
        out += [r for r in g if r["face_usable"] == "True"][:frontal] + [r for r in g if r["face_usable"] != "True"][:other]
    return out


def build_reference(cfg: dict, row: dict):
    """Source file → clean reference RGB uint8 (deterministic)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    src = os.path.join(cfg["source_root"], row["rel_path"])
    got = C.sha256_file(src)
    if got != row["sha256"]:
        raise RuntimeError(f"source changed since the inventory: {row['rel_path']} ({got} != {row['sha256']})")
    with Image.open(src) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    h, w = rgb.shape[:2]
    s = cfg["reference"]["long_side"] / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def process_sample(args) -> dict:
    cfg, row, specs = args
    import cv2
    import numpy as np
    import degradations as D
    cv2.setNumThreads(1)
    sid, expr = row["sample_id"], row["expression"]
    root = Path(cfg["bench_root"])
    try:
        ref = build_reference(cfg, row)
        clean_rel = f"clean_originals/{expr}/{sid}.png"
        ok, buf = cv2.imencode(".png", cv2.cvtColor(ref, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        assert ok
        data = buf.tobytes()
        p = root / clean_rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and C.sha256_file(p) != C.sha256_bytes(data):
            raise RuntimeError(f"existing clean reference differs from the regenerated one: {clean_rel}")
        if not p.exists():
            p.write_bytes(data)
        clean = {"sample_id": sid, "expression": expr, "source_rel_path": row["rel_path"],
                 "source_sha256": row["sha256"], "source_width": row["width"], "source_height": row["height"],
                 "clean_path": clean_rel, "width": ref.shape[1], "height": ref.shape[0], "mode": "RGB",
                 "bytes": len(data), "file_sha256": C.sha256_bytes(data),
                 "pixel_sha256": C.sha256_bytes(ref.tobytes())}
        recs = []
        for spec in specs:
            vid = variant_id(sid, spec)
            seed = C.stable_seed(cfg["seed"], sid, vid)
            data, ext, pix = D.apply(ref, spec["ops"], seed)
            rel = f"degraded/{expr}/{sid}/{spec['type']}/level_{spec['level']}.{ext}"
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            recs.append({"variant_id": vid, "sample_id": sid, "expression": expr, "type": spec["type"],
                         "category": spec["category"], "level": spec["level"], "order": spec["order"],
                         "ops": json.dumps(spec["ops"], separators=(",", ":")), "seed": seed,
                         "clean_path": clean_rel, "clean_pixel_sha256": clean["pixel_sha256"], "degraded_path": rel,
                         "format": ext, "width": pix.shape[1], "height": pix.shape[0], "mode": "RGB",
                         "bytes": len(data), "file_sha256": C.sha256_bytes(data),
                         "pixel_sha256": C.sha256_bytes(pix.tobytes()),
                         "mean_abs_diff_vs_clean": round(float(np.abs(pix.astype(np.int16) - ref).mean()), 3),
                         "source_rel_path": row["rel_path"], "source_sha256": row["sha256"]})
        return {"sample_id": sid, "clean": clean, "variants": recs, "error": ""}
    except Exception as e:  # noqa: BLE001 — recorded in failures, never hidden
        return {"sample_id": sid, "clean": None, "variants": [], "error": f"{type(e).__name__}: {e}"[:300]}


def finalise(cfg: dict) -> tuple[list[dict], list[dict]]:
    clean = list({r["sample_id"]: r for r in C.read_jsonl(C.bench(cfg, "manifests", "clean_references.jsonl"))}.values())
    var = list({r["variant_id"]: r for r in C.read_jsonl(C.bench(cfg, "manifests", "generated_variants.jsonl"))}.values())
    clean.sort(key=lambda r: r["sample_id"])
    var.sort(key=lambda r: r["variant_id"])
    C.write_csv(C.bench(cfg, "manifests", "clean_references.csv"), clean)
    C.write_csv(C.bench(cfg, "manifests", "generated_variants.csv"), var)
    return clean, var


def main(a):
    cfg = C.load_config(a.config)
    matrix, specs = load_matrix(cfg)
    dst = C.bench(cfg, "manifests", "degradation_matrix.json")
    src_text = (C.ROOT / cfg["matrix"]).read_text(encoding="utf-8")
    if dst.exists() and dst.read_text(encoding="utf-8") != src_text and not a.force_matrix:
        raise SystemExit("the matrix differs from the one this benchmark was generated with "
                         "(manifests/degradation_matrix.json); use --force-matrix after removing stale variants")
    dst.write_text(src_text, encoding="utf-8")
    shutil.copyfile(a.config or C.HERE / "config.json", C.bench(cfg, "manifests", "config.json"))
    selected = C.read_csv(C.bench(cfg, "manifests", "selected_originals.csv"))
    if a.pilot:
        C.write_csv(C.bench(cfg, "manifests", "pilot_samples.csv"), pilot_samples(selected))
    todo_rows = C.scope(cfg, a.pilot)
    want = {variant_id(r["sample_id"], s) for r in todo_rows for s in specs}
    have = {r["variant_id"]: r for r in C.read_jsonl(C.bench(cfg, "manifests", "generated_variants.jsonl"))}
    root = Path(cfg["bench_root"])

    def complete(row):
        ids = [variant_id(row["sample_id"], s) for s in specs]
        return all(i in have and (root / have[i]["degraded_path"]).exists() for i in ids)

    todo = [r for r in todo_rows if not complete(r)]
    if a.limit_samples:
        todo = todo[:a.limit_samples]
    print(f"matrix: {len(specs)} variants per original · samples in scope {len(todo_rows)} "
          f"({len(want)} variants) · samples to generate now {len(todo)}", flush=True)
    t0, n, errs = time.time(), 0, 0
    with open(C.bench(cfg, "manifests", "generated_variants.jsonl"), "a", encoding="utf-8") as fv, \
            open(C.bench(cfg, "manifests", "clean_references.jsonl"), "a", encoding="utf-8") as fc, \
            open(C.bench(cfg, "manifests", "generation_failures.jsonl"), "a", encoding="utf-8") as ff, \
            ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(process_sample, [(cfg, r, specs) for r in todo], chunksize=1):
            n += 1
            if res["error"]:
                errs += 1
                ff.write(json.dumps({"stage": "generate", "sample_id": res["sample_id"], "error": res["error"]}) + "\n")
                ff.flush()
                print(f"  FAILED {res['sample_id']}: {res['error']}", flush=True)
                continue
            fc.write(json.dumps(res["clean"]) + "\n")
            for r in res["variants"]:
                fv.write(json.dumps(r) + "\n")
            fv.flush()
            fc.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} samples, {time.time() - t0:.0f} s", flush=True)
    clean, var = finalise(cfg)
    size = sum(int(r["bytes"]) for r in var)
    print(f"done in {time.time() - t0:.0f} s · failures {errs} · manifest: {len(clean)} clean references, "
          f"{len(var)} variants, {size / 2 ** 30:.2f} GiB degraded "
          f"({size / max(1, len(var)) / 1024:.0f} KiB per variant)", flush=True)
    missing = want - {r["variant_id"] for r in var}
    if missing and not a.limit_samples:
        print(f"!! {len(missing)} variants of this scope are missing (see generation_failures.jsonl)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force-matrix", action="store_true")
    main(ap.parse_args())
