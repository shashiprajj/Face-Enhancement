"""P4-T3/T4 — score every configuration's outputs on a population with the validated second evaluator E07 (dlib),
using the chip route (fixed dlib geometry for every E-frame image). Complements p3_evaluate.py (SFace E01).

Impostor operating point (FAR 1e-2) comes from clean references of the SAME split:
  dev  populations → the 112 dev_pilot references (population_noise_floor)
  test populations → the 700 test_core references (population_impostor_pool)

    FES_IMAGE=fes-dev:cpu-p4 scripts/dock.sh python scripts/p4_evaluate_e07.py --population main --root data/manifests/phase3
    FES_IMAGE=fes-dev:cpu-p4 scripts/dock.sh python scripts/p4_evaluate_e07.py --population p4dev --root data/manifests/phase4

Writes data/manifests/phase4/metrics_e07/<population>/<config>.csv.gz + identity_calibration.json.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from fes_bench import evaluators as EV
from fes_bench import read_bytes_retry
from fes_data.common import read_csv_gz, repo_path, run_info, sha256_bytes, sha256_file, write_csv_gz, write_json

CONFIGS = ["B0", "B1", "B2", "B2c", "A1", "A2", "A3", "A4", "A5"]
FIELDS = ["pair_id", "ref_id", "cell", "profile", "severity", "config_id", "status", "id_cos_e07", "imp_cos_mean_e07",
          "imp_cos_max_e07", "imp_n_e07", "imp_n_ge_thr_e07", "id_ge_thr_e07"]
FAR = 1e-2
VALIDATION = "data/manifests/phase4/evaluator_E07_validation.json"


def imread(path: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(read_bytes_retry(repo_path(path)), np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--out", default=None, help="output dir (default data/manifests/phase4/metrics_e07/<population>)")
    a = ap.parse_args()
    cv2.setNumThreads(2)
    val = json.loads(repo_path(VALIDATION).read_text(encoding="utf-8"))
    assert val["validated"], "E07 not validated — its scores may not be used (P4-T3)"
    ev = EV.DlibResNet()
    pop = read_csv_gz(repo_path(f"{a.root}/population_{a.population}.csv.gz"))
    split = "dev" if a.population == "p4dev" else "test"
    pool_name = "noise_floor" if split == "dev" else "impostor_pool"

    refs = {}
    for p in pop:
        if p["ref_id"] not in refs:
            assert sha256_file(repo_path(p["ref_path"])) == p["ref_sha256"]
            refs[p["ref_id"]] = ev.embed_chip(imread(p["ref_path"]))
    ids = sorted(refs)
    R = np.stack([refs[i] for i in ids])

    seen, pool = set(), []
    for p in read_csv_gz(repo_path(f"data/manifests/phase3/population_{pool_name}.csv.gz")):
        if p["ref_id"] not in seen:
            seen.add(p["ref_id"])
            pool.append(ev.embed_chip(imread(p["ref_path"])))
    P = np.stack(pool)
    imp = (P @ P.T)[np.triu_indices(len(P), 1)]
    thr = float(np.quantile(imp, 1 - FAR))
    out_dir = repo_path(a.out or f"data/manifests/phase4/metrics_e07/{a.population}")
    write_json(out_dir / "identity_calibration.json", {
        "run": run_info({"evaluator": ev.id, "weights_sha256": ev.sha256, "dlib": EV.DLIB_VERSION}),
        "route": "chip (E-frame INTER_AREA 224→150)", "split": split, "impostor_pool": pool_name,
        "impostor_pool_refs": len(P), "impostor_pairs": int(len(imp)), "far": FAR, "threshold_at_far": round(thr, 5),
        "impostor_quantiles": {q: round(float(np.quantile(imp, q)), 5) for q in (0.5, 0.9, 0.99, 0.999)},
        "note": "Embeddings not persisted. FairFace has no identity labels; a few same-person pairs may remain."})
    print("E07 threshold@FAR1e-2", round(thr, 4), flush=True)

    for cid in a.configs:
        run_path = repo_path(f"{a.root}/runs/{a.population}/{cid}.csv.gz")
        if not run_path.exists():
            print(f"{cid}: no run manifest — skipped", flush=True)
            continue
        run = {r["pair_id"]: r for r in read_csv_gz(run_path)}
        rows = []
        for p in pop:
            r = run.get(p["pair_id"])
            rec = {k: "" for k in FIELDS}
            rec.update(pair_id=p["pair_id"], ref_id=p["ref_id"], cell=p["cell"], profile=p["profile"],
                       severity=p["severity"], config_id=cid, status=r["status"] if r else "MISSING")
            if r and r["status"] == "OK":
                data = read_bytes_retry(repo_path(r["output_path"]))
                assert sha256_bytes(data) == r["output_sha256"], r["output_path"]
                e = ev.embed_chip(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
                sims = R @ e
                i = ids.index(p["ref_id"])
                others = np.delete(sims, i)
                rec.update(id_cos_e07=round(float(sims[i]), 6), imp_cos_mean_e07=round(float(others.mean()), 6),
                           imp_cos_max_e07=round(float(others.max()), 6), imp_n_e07=len(others),
                           imp_n_ge_thr_e07=int((others >= thr).sum()), id_ge_thr_e07=bool(sims[i] >= thr))
            rows.append(rec)
        path = out_dir / f"{cid}.csv.gz"
        write_csv_gz(path, rows, FIELDS)
        print(f"{cid}: {sum(r['status'] == 'OK' for r in rows)}/{len(rows)} → {sha256_file(path)[:12]}", flush=True)


if __name__ == "__main__":
    main()
