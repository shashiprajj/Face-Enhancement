"""P4-T6/T7 prerequisite — freeze the Phase 4 SYNTHETIC dev calibration population `p4dev`.

Same selection rule as Phase 3 (seeded hash order within race × gender strata, `p3_build_populations.select_refs`),
applied to the 112 dev_pilot references minus the Phase 3 trial references. Every reference and degraded file is
SHA-256-verified. Writes data/manifests/phase4/population_p4dev.csv.gz + p4dev_population_report.json.

    scripts/dock.sh python scripts/p4_build_dev_population.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_build_populations import POP_FIELDS, select_refs  # noqa: E402

from fes_bench import geometry as G  # noqa: E402
from fes_data.common import load_config, read_csv_gz, repo_path, run_info, sha256_file, write_csv_gz, write_json  # noqa: E402

P3 = "configs/phase3/benchmark_p3.yaml"
P4 = "configs/phase4/p4_protocol.yaml"


def main() -> None:
    p3, p4 = load_config(P3), load_config(P4)
    spec = p4["populations"]["p4dev"]
    src_spec = p3["inputs"][spec["source"]]
    assert sha256_file(repo_path(src_spec["path"])) == src_spec["sha256"], "dev_pilot manifest changed"
    assert sha256_file(repo_path(p3["inputs"]["splits"]["path"])) == p3["inputs"]["splits"]["sha256"]
    splits = {r["ref_id"]: r for r in read_csv_gz(repo_path(p3["inputs"]["splits"]["path"]))}
    src = read_csv_gz(repo_path(src_spec["path"]))

    excluded = set()
    for pop in spec["exclude_populations"]:
        excluded |= {r["ref_id"] for r in read_csv_gz(repo_path(f"data/manifests/phase3/population_{pop}.csv.gz"))}
    main_refs = {r["ref_id"] for r in read_csv_gz(repo_path("data/manifests/phase3/population_main.csv.gz"))}
    candidates = sorted({r["ref_id"] for r in src} - excluded)
    ref_ids = select_refs([splits[i] for i in candidates], spec["select"], p4["seed"])
    keep = set(ref_ids)

    rows = []
    for p in src:
        if p["ref_id"] not in keep:
            continue
        s = splits[p["ref_id"]]
        M = np.array(json.loads(s["ffhq512_transform_2x3"]))
        rows.append({"population": "p4dev", "pair_id": p["pair_id"], "ref_id": p["ref_id"], "profile": p["profile"],
                     "severity": p["severity"], "cell": f"{p['profile']}-{p['severity']}",
                     "race": s["race"], "gender": s["gender"], "age": s["age"],
                     "ref_path": p["ref_path"], "ref_sha256": p["ref_sha256"],
                     "degraded_path": p["degraded_path"], "degraded_sha256": p["degraded_sha256"],
                     "out_w": p["out_w"], "seed": p["seed"], "degrade_version": p["degrade_version"],
                     "det_landmarks": s["det_landmarks"], "ffhq512_transform_2x3": s["ffhq512_transform_2x3"],
                     "ffhq512_scale": s["ffhq512_scale"],
                     "e_coverage": round(G.e_frame_coverage(M), 5),
                     "eval_mask_frac": round(float(G.coverage_mask(M, erode_px=2).mean()), 5)})
    rows.sort(key=lambda r: (r["ref_id"], r["profile"], r["severity"]))
    bad = [f for f, h in {(r["ref_path"], r["ref_sha256"]) for r in rows} | {(r["degraded_path"], r["degraded_sha256"]) for r in rows}
           if not repo_path(f).exists() or sha256_file(repo_path(f)) != h]
    assert not bad, f"{len(bad)} files missing/modified, e.g. {bad[:3]}"

    # leakage: dev only, disjoint from trial and from the (test) main population
    assert all(splits[i]["split"] == "dev" for i in keep)
    assert not keep & excluded and not keep & main_refs
    out = f"{spec['manifest_root']}/population_p4dev.csv.gz"
    write_csv_gz(repo_path(out), rows, POP_FIELDS)
    strata = Counter((splits[i]["race"], splits[i]["gender"]) for i in keep)
    rep = {"run": run_info({"config": P4, "config_sha256": p4["_config_sha256"]}),
           "population": "p4dev", "data_kind": "SYNTHETIC (fes-syn-v1 dev split) — not real-world",
           "manifest": out, "sha256": sha256_file(repo_path(out)), "source": src_spec,
           "n_refs": len(keep), "n_pairs": len(rows), "n_cells": len({r["cell"] for r in rows}),
           "files_verified": len({r["ref_path"] for r in rows}) + len({r["degraded_path"] for r in rows}),
           "strata": {f"{k[0]}|{k[1]}": v for k, v in sorted(strata.items())},
           "ref_ids": sorted(keep),
           "checks": {"all_dev_split": True, "disjoint_from_p3_trial": True, "disjoint_from_p3_main": True,
                      "files_sha256_ok": True}}
    write_json(repo_path(f"{spec['manifest_root']}/p4dev_population_report.json"), rep)
    print(rep["n_refs"], rep["n_pairs"], rep["sha256"])


if __name__ == "__main__":
    main()
