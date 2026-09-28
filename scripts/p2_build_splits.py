"""P2-T7 — Leakage-safe splits and benchmark subset selection (deterministic, config-driven).

    scripts/dock.sh python scripts/p2_build_splits.py --config configs/phase2/benchmark_v1.yaml

1. Accepted references (P2-T3) → roles by pool (dev / test). No training split.
2. Leakage screening:
   a) exact / near-duplicate groups (P2-T2 pHash) must not span roles (asserted),
   b) same-person screening with the TEST-ONLY SFace evaluator: any dev reference whose cosine similarity to
      any test reference ≥ threshold is dropped from dev. Embeddings are held in memory only and never written.
3. Seeded, race×gender-stratified subsets (dev_pilot, test_core).
Outputs (in out_dir): splits.csv.gz (canonical per-reference records), selection.csv.gz, splits_report.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from fes_data import common as C

_CTX: dict = {}


def _init(cfg: dict) -> None:
    _CTX["rec"] = cv2.FaceRecognizerSF.create(str(C.repo_path(cfg["leakage"]["evaluator"])), "")


def _embed(ref: dict) -> tuple[str, np.ndarray]:
    img = cv2.imread(str(C.repo_path(ref["ref_path"])), cv2.IMREAD_COLOR)
    bbox = json.loads(ref["det_bbox_xywh"])
    lm = np.array(json.loads(ref["det_landmarks"])).reshape(-1)
    row = np.array([*bbox, *lm, float(ref["det_score"])], np.float32)[None, :]  # YuNet row layout
    aligned = _CTX["rec"].alignCrop(img, row)
    f = _CTX["rec"].feature(aligned).reshape(-1).astype(np.float64)
    return ref["ref_id"], f / np.linalg.norm(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2/benchmark_v1.yaml")
    cfg = C.load_config(ap.parse_args().config)
    out_dir = C.repo_path(cfg["out_dir"])
    refs = [r for r in C.read_csv_gz(C.repo_path(cfg["reference_manifest"])) if r["accepted"] == "True"]
    pool_to_role = {v["pool"]: k for k, v in cfg["roles"].items()}
    for r in refs:
        r["role"] = pool_to_role[r["pool"]]

    # 2a) duplicate groups must not span roles (P2-T3 excluded all near-dup members; assert it)
    ff = {r["sample_id"]: r for r in C.read_csv_gz(C.repo_path(cfg["fairface_manifest"]))}
    group_roles = collections.defaultdict(set)
    for r in refs:
        for g in (ff[r["sample_id"]]["exact_dup_group"], ff[r["sample_id"]]["near_dup_group"]):
            if g:
                group_roles[g].add(r["role"])
    dup_cross = [g for g, s in group_roles.items() if len(s) > 1]
    assert not dup_cross, f"duplicate groups spanning roles: {dup_cross[:5]}"

    # 2b) same-person screening (offline, test-only evaluator)
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init, initargs=(cfg,)) as ex:
        emb = dict(ex.map(_embed, sorted(refs, key=lambda r: r["ref_id"]), chunksize=64))
    dev_ids = sorted(r["ref_id"] for r in refs if r["role"] == "dev")
    test_ids = sorted(r["ref_id"] for r in refs if r["role"] == "test")
    D = np.stack([emb[i] for i in dev_ids])
    T = np.stack([emb[i] for i in test_ids])
    S = D @ T.T
    # 1:N screening threshold: robust impostor model from the bulk (almost all dev×test pairs are different
    # people); choose t so the expected number of false exceedances over ALL pairs is < max_expected_false
    # (Bonferroni-style). A fixed 1:1 verification threshold (e.g. 0.363) is not valid for 1:N screening.
    lk = cfg["leakage"]
    mu = float(np.median(S))
    sd = float(1.4826 * np.median(np.abs(S - mu)))
    from math import erfc, sqrt
    def expected_false(t: float) -> float:
        return S.size * 0.5 * erfc((t - mu) / (sd * sqrt(2)))
    grid = np.round(np.arange(0.30, 0.95, 0.005), 3)
    thr = float(next(t for t in grid if expected_false(t) < lk["max_expected_false"]))
    sweep = {f"{t:.3f}": {"pairs_ge": int((S >= t).sum()), "dev_flagged": int((S.max(axis=1) >= t).sum()),
                          "expected_false_pairs_gaussian": round(expected_false(t), 3)}
             for t in (0.363, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, thr)}
    dev_max = S.max(axis=1)
    dev_arg = S.argmax(axis=1)
    test_max = S.max(axis=0)
    flagged = {dev_ids[i] for i in np.where(dev_max >= thr)[0]}
    TT = T @ T.T
    np.fill_diagonal(TT, -1)
    within_test_pairs = int((np.triu(TT, 1) >= thr).sum())
    sim_hist = np.histogram(S.ravel(), bins=sorted({-1, 0, 0.1, 0.2, 0.3, 0.363, 0.5, 0.7, thr, 1.01}))

    def floor6(x: float) -> float:
        # floor (not round) so a stored similarity never exceeds the true value used for flagging
        return float(np.floor(x * 1e6) / 1e6)

    by_id = {r["ref_id"]: r for r in refs}
    for i, rid in enumerate(dev_ids):
        by_id[rid]["leak_max_cos_to_other_role"] = floor6(float(dev_max[i]))
        by_id[rid]["leak_nearest_other_role"] = test_ids[dev_arg[i]]
    for j, rid in enumerate(test_ids):
        by_id[rid]["leak_max_cos_to_other_role"] = floor6(float(test_max[j]))
        by_id[rid]["leak_nearest_other_role"] = ""
    for r in refs:
        r["leak_flag"] = r["ref_id"] in flagged
        r["split"] = "excluded_leakage" if r["leak_flag"] else r["role"]

    # 3) stratified subsets
    rng = random.Random(cfg["seed"])
    for r in refs:
        r["subset"] = ""
    subset_counts = {}
    for name, spec in cfg["subsets"].items():
        cand = sorted((r for r in refs if r["split"] == spec["role"]), key=lambda r: r["ref_id"])
        strata = collections.defaultdict(list)
        for r in cand:
            strata[tuple(r[k] for k in spec["stratify_by"])].append(r)
        chosen = []
        for k in sorted(strata):
            chosen += rng.sample(strata[k], min(spec["per_stratum"], len(strata[k])))
        for r in chosen:
            r["subset"] = name
        subset_counts[name] = {"n": len(chosen), "short_strata": {"|".join(k): len(v) for k, v in strata.items()
                                                                  if len(v) < spec["per_stratum"]}}

    # canonical per-reference records (schema: configs/phase2/schema/reference_record.schema.json)
    fields = ["ref_id", "sample_id", "split", "subset", "role", "pool", "split_source", "race", "gender", "age",
              "source_path", "source_sha256", "ref_path", "ref_sha256", "ref_pixel_sha256", "ref_size", "ref_version",
              "crop_box_xywh", "det_bbox_xywh", "det_landmarks", "det_score", "detector", "preproc_version",
              "roll_deg", "yaw_proxy", "eye_dist", "ffhq512_transform_2x3", "ffhq512_scale", "ffhq512_oob_frac",
              "lap_var", "mean_luma", "rms_contrast", "frac_dark", "frac_bright",
              "exact_dup_group", "near_dup_group", "leak_flag", "leak_max_cos_to_other_role", "leak_nearest_other_role",
              "license", "source_url", "license_status"]
    rows = []
    for r in sorted(refs, key=lambda r: r["ref_id"]):
        rows.append({**{k: r.get(k, "") for k in fields},
                     "crop_box_xywh": "[0, 0, 224, 224]",
                     "exact_dup_group": ff[r["sample_id"]]["exact_dup_group"],
                     "near_dup_group": ff[r["sample_id"]]["near_dup_group"],
                     "source_url": "https://github.com/joojs/fairface",
                     "license_status": "ELIGIBLE (evaluation; CC BY 4.0; P1-T6)"})
    C.write_csv_gz(out_dir / "splits.csv.gz", rows, fields)
    C.write_csv_gz(out_dir / "selection.csv.gz",
                   [{"ref_id": r["ref_id"], "split": r["split"], "subset": r["subset"]} for r in rows if r["subset"]],
                   ["ref_id", "split", "subset"])

    cnt = collections.Counter
    report = {
        "config": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
        "references": len(rows), "splits": dict(cnt(r["split"] for r in rows)),
        "duplicate_groups_spanning_roles": len(dup_cross),
        "leakage": {"evaluator": cfg["leakage"]["evaluator"], "cosine_threshold_resolved": thr,
                    "impostor_model": {"median": round(mu, 4), "robust_sd": round(sd, 4), "assumption": "Gaussian bulk; tails of real impostor distributions are usually heavier, so this threshold is a lower bound on strictness"},
                    "max_expected_false": lk["max_expected_false"], "threshold_sweep": sweep,
                    "dev_x_test_pairs": int(S.size), "pairs_ge_threshold": int((S >= thr).sum()),
                    "dev_refs_dropped": len(flagged), "within_test_pairs_ge_threshold": within_test_pairs,
                    "similarity_histogram": {f"[{a:.3f},{b:.3f})": int(c) for a, b, c in
                                             zip(sim_hist[1][:-1], sim_hist[1][1:], sim_hist[0])},
                    "note": "embeddings computed in memory only; not stored"},
        "subsets": subset_counts,
        "subset_strata": {s: dict(sorted(cnt(f'{r["race"]}|{r["gender"]}' for r in rows if r["subset"] == s).items()))
                          for s in cfg["subsets"]},
        "splits_sha256": C.sha256_file(out_dir / "splits.csv.gz"),
        "selection_sha256": C.sha256_file(out_dir / "selection.csv.gz"),
        "run": C.run_info(),
    }
    C.write_json(out_dir / "splits_report.json", report)
    print(json.dumps({k: report[k] for k in ("references", "splits", "leakage", "subsets")}, indent=1)[:2500])


if __name__ == "__main__":
    main()
