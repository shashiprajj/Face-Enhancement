"""P3-T1 — verify the Phase 2 inputs and freeze the Phase 3 evaluation populations.

Checks every input manifest against its frozen SHA-256, selects the trial/main populations deterministically
(seeded hash order within race×gender strata), verifies every selected file (reference + degraded) by SHA-256,
and writes the population manifests plus an inventory report.

    scripts/dock.sh python scripts/p3_build_populations.py            (stdlib + numpy + opencv; host venv also works)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from fes_bench import geometry as G
from fes_data.common import load_config, read_csv_gz, repo_path, run_info, sha256_file, write_csv_gz, write_json

OUT = "data/manifests/phase3"
POP_FIELDS = ["population", "pair_id", "ref_id", "profile", "severity", "cell", "race", "gender", "age",
              "ref_path", "ref_sha256", "degraded_path", "degraded_sha256", "out_w", "seed", "degrade_version",
              "det_landmarks", "ffhq512_transform_2x3", "ffhq512_scale", "e_coverage", "eval_mask_frac"]


def order_key(seed: int, salt: str, ref_id: str) -> str:
    return hashlib.sha256(f"{seed}|{salt}|{ref_id}".encode()).hexdigest()


def select_refs(refs: list[dict], sel: dict, seed: int) -> list[str]:
    strata = defaultdict(list)
    for r in refs:
        strata[tuple(r[k] for k in sel["stratify_by"])].append(r["ref_id"])
    chosen = []
    for key in sorted(strata):
        ids = sorted(strata[key], key=lambda i: order_key(seed, sel["salt"], i))
        chosen.extend(ids[:sel["per_stratum"]])
    if sel.get("max_refs"):
        chosen = sorted(chosen, key=lambda i: order_key(seed, sel["salt"] + "|cap", i))[:sel["max_refs"]]
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3/benchmark_p3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    report = {"run": run_info({"config": cfg["_config_path"], "config_sha256": cfg["_config_sha256"]}),
              "protocol_version": cfg["protocol_version"], "inputs": {}, "populations": {}}

    # 1. input manifests are exactly the P2-T8-validated ones
    data = {}
    for name, spec in cfg["inputs"].items():
        got = sha256_file(repo_path(spec["path"]))
        assert got == spec["sha256"], f"{name}: manifest hash {got} != frozen {spec['sha256']}"
        data[name] = read_csv_gz(repo_path(spec["path"]))
        report["inputs"][name] = {"path": spec["path"], "sha256": got, "rows": len(data[name])}
    splits = {r["ref_id"]: r for r in data["splits"]}

    for name in ("pairs_dev_pilot", "pairs_test_core"):
        rows = data[name]
        refs = {r["ref_id"] for r in rows}
        cells = Counter(f"{r['profile']}-{r['severity']}" for r in rows)
        assert all(v == len(refs) for v in cells.values()), f"{name}: incomplete cell coverage"
        report["inputs"][name].update(n_refs=len(refs), n_cells=len(cells),
                                      out_sizes=dict(Counter(r["out_w"] for r in rows)))

    # 2. populations
    for pop, spec in cfg["populations"].items():
        src = data[spec["source"]]
        ref_ids = sorted({r["ref_id"] for r in src})
        if "select" in spec:
            ref_ids = select_refs([splits[i] for i in ref_ids], spec["select"], cfg["seed"])
        keep = set(ref_ids)
        rows = []
        for p in src:
            if p["ref_id"] not in keep:
                continue
            s = splits[p["ref_id"]]
            M = np.array(json.loads(s["ffhq512_transform_2x3"]))
            rows.append({"population": pop, "pair_id": p["pair_id"], "ref_id": p["ref_id"], "profile": p["profile"],
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
        # 3. file integrity for everything the population will read
        bad = []
        for sha_path in {(r["ref_path"], r["ref_sha256"]) for r in rows} | \
                        {(r["degraded_path"], r["degraded_sha256"]) for r in rows}:
            if not repo_path(sha_path[0]).exists() or sha256_file(repo_path(sha_path[0])) != sha_path[1]:
                bad.append(sha_path[0])
        assert not bad, f"{pop}: {len(bad)} files missing or modified, e.g. {bad[:3]}"
        path = f"{OUT}/population_{pop}.csv.gz"
        write_csv_gz(repo_path(path), rows, POP_FIELDS)
        refs = sorted({r["ref_id"] for r in rows})
        strata = Counter((splits[i]["race"], splits[i]["gender"]) for i in refs)
        report["populations"][pop] = {
            "manifest": path, "sha256": sha256_file(repo_path(path)), "source": spec["source"],
            "n_refs": len(refs), "n_pairs": len(rows), "n_cells": len({r["cell"] for r in rows}),
            "files_verified": len({r["ref_path"] for r in rows}) + len({r["degraded_path"] for r in rows}),
            "strata": {f"{k[0]}|{k[1]}": v for k, v in sorted(strata.items())},
            "refs_with_partial_f_coverage": sum(1 for i in refs if next(r for r in rows if r["ref_id"] == i)["e_coverage"] < 1),
            "min_eval_mask_frac": min(r["eval_mask_frac"] for r in rows),
            "ref_ids": refs if len(refs) <= 50 else None}
        print(pop, report["populations"][pop]["n_refs"], report["populations"][pop]["n_pairs"], flush=True)

    # 4. disjointness: trial (dev) and main (test) share no reference; main ⊂ test split
    t, m = (set(json.loads(json.dumps(report["populations"][p]["ref_ids"]))) for p in ("trial", "main"))
    assert not t & m
    assert all(splits[i]["split"] == "test" for i in m) and all(splits[i]["split"] == "dev" for i in t)
    report["checks"] = {"trial_main_disjoint": True, "main_in_test_split": True, "trial_in_dev_split": True,
                        "input_manifests_match_p2t8": True, "population_files_sha256_ok": True}
    write_json(repo_path(f"{OUT}/populations_report.json"), report)


if __name__ == "__main__":
    main()
