"""P2-T4 — Generate paired clean/degraded images from a reference selection (deterministic).

    scripts/dock.sh python scripts/p2_degrade.py --config configs/phase2/degrade_pilot.yaml [--verify-only]

Input : a selection CSV of ref_ids (produced by config or by P2-T7) + the P2-T3 reference manifest.
Output: <out_dir>/degraded/<profile>-<severity>/<ref_id>.png, a paired manifest and generation statistics.
The reference files are opened read-only and their SHA-256 is checked before use; they are never modified.
--verify-only regenerates every pair in memory and checks the pixel hash against the manifest.
"""

from __future__ import annotations

import argparse
import collections
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from fes_data import common as C
from fes_data import degrade as D
from fes_data import preprocess as P

_CTX: dict = {}


def _init(cfg: dict) -> None:
    _CTX["cfg"] = cfg
    d = cfg["detector"]
    _CTX["det"] = P.YuNetDetector(C.repo_path(d["model"]), d["score_threshold"], d["nms_threshold"], max_side=d["max_side"])


def _load_ref(ref: dict) -> np.ndarray:
    data = C.repo_path(ref["ref_path"]).read_bytes()
    assert C.sha256_bytes(data) == ref["ref_sha256"], f"reference changed: {ref['ref_path']}"
    return C.decode_image(data)


def _gen(task: tuple[dict, str, str]) -> dict:
    ref, profile, severity = task
    cfg = _CTX["cfg"]
    img = _load_ref(ref)
    seed = D.derive_seed(cfg["seed"], ref["ref_id"], profile, severity)
    out, params = D.apply_profile(img, profile, severity, seed)
    rel = f'{cfg["out_dir"]}/degraded/{profile}-{severity}/{ref["ref_id"]}.png'
    ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    assert ok
    blob = buf.tobytes()
    dst = C.repo_path(rel)
    if dst.exists():
        assert C.sha256_file(dst) == C.sha256_bytes(blob), f"non-reproducible degradation: {rel}"
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(blob)
    # statistics (comparisons at reference size; LR outputs are bicubic-upsampled for measurement only)
    cmp = out if out.shape == img.shape else cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    g_ref, g_cmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cmp, cv2.COLOR_BGR2GRAY)
    dets = _CTX["det"].detect(cmp)
    mean_l, contrast, dark, bright = C.luma_stats(out)
    return {
        "pair_id": f'{ref["ref_id"]}.{profile}-{severity}', "ref_id": ref["ref_id"], "sample_id": ref["sample_id"],
        "pool": ref["pool"], "profile": profile, "severity": severity,
        "ref_path": ref["ref_path"], "ref_sha256": ref["ref_sha256"],
        "degraded_path": rel, "degraded_sha256": C.sha256_bytes(blob), "degraded_pixel_sha256": C.sha256_bytes(out.tobytes()),
        "out_w": out.shape[1], "out_h": out.shape[0], "seed": seed, "params_json": D.params_json(params),
        "degrade_version": D.DEGRADE_VERSION,
        "psnr_vs_ref": round(float(peak_signal_noise_ratio(img, cmp, data_range=255)), 3),
        "ssim_vs_ref": round(float(structural_similarity(g_ref, g_cmp, data_range=255)), 4),
        "deg_lap_var": round(C.laplacian_variance(out), 2), "deg_mean_luma": round(mean_l, 2),
        "deg_rms_contrast": round(contrast, 2),
        "yunet_face_found": bool(dets), "yunet_score": round(dets[0].score, 4) if dets else "",
    }


def _verify(row: dict) -> tuple[str, bool]:
    ref = {"ref_path": row["ref_path"], "ref_sha256": row["ref_sha256"]}
    img = _load_ref(ref)
    out, params = D.apply_profile(img, row["profile"], row["severity"], int(row["seed"]))
    return row["pair_id"], (C.sha256_bytes(out.tobytes()) == row["degraded_pixel_sha256"]
                            and D.params_json(params) == row["params_json"])


def select_refs(cfg: dict, refs: list[dict]) -> list[dict]:
    sel = cfg["selection"]
    if "ref_ids_csv_gz" in sel:  # explicit list (P2-T7 split selections), optionally filtered to one split
        chosen = C.read_csv_gz(C.repo_path(sel["ref_ids_csv_gz"]))
        wanted = {r["ref_id"] for r in chosen if "split" not in sel or r.get("split") == sel["split"]}
        missing = wanted - {r["ref_id"] for r in refs}
        assert not missing, f"selection references unknown/unaccepted ref_ids: {sorted(missing)[:5]}"
        return sorted((r for r in refs if r["ref_id"] in wanted), key=lambda r: r["ref_id"])
    pool = sorted((r for r in refs if r["pool"] == sel["pool"]), key=lambda r: r["ref_id"])
    return sorted(random.Random(cfg["seed"]).sample(pool, sel["n"]), key=lambda r: r["ref_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    manifest_path = C.repo_path(cfg["manifest"])

    if a.verify_only:
        rows = C.read_csv_gz(manifest_path)
        with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init, initargs=(cfg,)) as ex:
            res = list(ex.map(_verify, rows, chunksize=32))
        bad = [pid for pid, ok in res if not ok]
        print({"verified": len(res), "mismatches": len(bad), "first_mismatches": bad[:10]})
        raise SystemExit(1 if bad else 0)

    refs = [r for r in C.read_csv_gz(C.repo_path(cfg["reference_manifest"])) if r["accepted"] == "True"]
    chosen = select_refs(cfg, refs)
    cells = [(p, s) for p, s in D.all_cells() if p in cfg.get("profiles", D.PROFILES)]
    tasks = [(r, p, s) for r in chosen for p, s in cells]
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init, initargs=(cfg,)) as ex:
        rows = list(ex.map(_gen, tasks, chunksize=32))
    C.write_csv_gz(manifest_path, rows)

    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[f'{r["profile"]}-{r["severity"]}'].append(r)
    stats = {cell: {"n": len(v), "out_size": f'{v[0]["out_w"]}x{v[0]["out_h"]}',
                    "psnr_mean": round(float(np.mean([x["psnr_vs_ref"] for x in v])), 2),
                    "ssim_mean": round(float(np.mean([x["ssim_vs_ref"] for x in v])), 4),
                    "yunet_detect_rate": round(float(np.mean([x["yunet_face_found"] for x in v])), 3),
                    "deg_mean_luma": round(float(np.mean([x["deg_mean_luma"] for x in v])), 1)}
             for cell, v in sorted(by_cell.items())}
    report = {"config": cfg["_config_path"], "config_sha256": cfg["_config_sha256"], "degrade_version": D.DEGRADE_VERSION,
              "references": len(chosen), "cells": len(cells), "pairs": len(rows), "cell_stats": stats,
              "manifest": cfg["manifest"], "manifest_sha256": C.sha256_file(manifest_path), "run": C.run_info()}
    C.write_json(C.repo_path(cfg["report"]), report)
    print({k: report[k] for k in ("references", "cells", "pairs")})
    for cell, s in stats.items():
        print(f"  {cell:26s} {s}")


if __name__ == "__main__":
    main()
