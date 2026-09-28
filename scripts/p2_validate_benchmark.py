"""P2-T7/T8 — Validate the fes-syn-v1 benchmark end to end. Read-only; exit 1 if any check fails.

    scripts/dock.sh python scripts/p2_validate_benchmark.py --config configs/phase2/benchmark_v1.yaml \
        [--repro-sample 500 | --repro-all]

Checks: schema, file existence + SHA-256 (references, degraded, sources, raw), duplicate IDs/content,
broken links, leakage (sample, duplicate groups, identity screen, split provenance), metadata completeness,
counts/coverage, degradation reproducibility (seeded sample or all), license/source traceability.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fes_data import common as C
from fes_data import degrade as D

SUBSETS = {"dev_pilot": "dev", "test_core": "test"}


def _schema_errors(rows: list[dict], schema: dict, key: str) -> list[str]:
    errs = []
    props = schema["properties"]
    for r in rows:
        rid = r.get(key, "?")
        for f in schema["required"]:
            if r.get(f, "") == "" and not (f == "leak_max_cos_to_other_role" and r.get("split") == "test" and False):
                errs.append(f"{rid}: missing {f}")
        for f, spec in props.items():
            v = r.get(f, "")
            if v == "":
                continue
            t = spec.get("type")
            try:
                if t == "integer":
                    v_cast = int(v)
                elif t == "number":
                    v_cast = float(v)
                elif t == "boolean":
                    assert v in ("True", "False")
                    v_cast = v
                elif t == "json":
                    json.loads(v)
                    v_cast = v
                else:
                    v_cast = v
            except Exception:
                errs.append(f"{rid}: {f}={v!r} not {t}")
                continue
            if "enum" in spec and v_cast not in spec["enum"] and v not in [str(e) for e in spec["enum"]]:
                errs.append(f"{rid}: {f}={v!r} not in enum")
            if "pattern" in spec and not re.search(spec["pattern"], v):
                errs.append(f"{rid}: {f}={v!r} fails pattern")
    return errs


def _hash_task(item: tuple[str, str]) -> tuple[str, bool, bool]:
    path, sha = item
    p = C.repo_path(path)
    return path, p.exists(), p.exists() and C.sha256_file(p) == sha


def _repro_task(row: dict) -> tuple[str, bool]:
    img = C.decode_image(C.repo_path(row["ref_path"]).read_bytes())
    out, params = D.apply_profile(img, row["profile"], row["severity"], int(row["seed"]))
    return row["pair_id"], (C.sha256_bytes(out.tobytes()) == row["degraded_pixel_sha256"]
                            and D.params_json(params) == row["params_json"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2/benchmark_v1.yaml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--repro-sample", type=int, default=500)
    g.add_argument("--repro-all", action="store_true")
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    out_dir = C.repo_path(cfg["out_dir"])
    checks: dict[str, dict] = {}

    def check(name: str, ok: bool, **detail) -> None:
        checks[name] = {"pass": bool(ok), **detail}

    splits = C.read_csv_gz(out_dir / "splits.csv.gz")
    pairs = {s: C.read_csv_gz(out_dir / f"pairs_{s}.csv.gz") for s in SUBSETS}
    all_pairs = [p for v in pairs.values() for p in v]
    ref_schema = json.loads(C.repo_path("configs/phase2/schema/reference_record.schema.json").read_text())
    pair_schema = json.loads(C.repo_path("configs/phase2/schema/pair_record.schema.json").read_text())
    ff = {r["sample_id"]: r for r in C.read_csv_gz(C.repo_path(cfg["fairface_manifest"]))}
    report_splits = json.loads((out_dir / "splits_report.json").read_text())
    thr = report_splits["leakage"]["cosine_threshold_resolved"]

    # 1) schema / metadata completeness
    e1 = _schema_errors(splits, ref_schema, "ref_id")
    e2 = _schema_errors(all_pairs, pair_schema, "pair_id")
    check("schema_references", not e1, rows=len(splits), errors=len(e1), first=e1[:5])
    check("schema_pairs", not e2, rows=len(all_pairs), errors=len(e2), first=e2[:5])

    # 2) duplicates
    def dups(vals):
        c = collections.Counter(vals)
        return [k for k, n in c.items() if n > 1]
    check("unique_ref_id", not dups(r["ref_id"] for r in splits))
    check("unique_ref_content", not dups(r["ref_pixel_sha256"] for r in splits), note="no two references share pixels")
    check("unique_pair_id", not dups(p["pair_id"] for p in all_pairs))
    check("unique_degraded_path", not dups(p["degraded_path"] for p in all_pairs))

    # 3) file integrity (all references in splits, all degraded, all sources of used references) + raw
    items = [(r["ref_path"], r["ref_sha256"]) for r in splits]
    items += [(p["degraded_path"], p["degraded_sha256"]) for p in all_pairs]
    items += [(r["source_path"], r["source_sha256"]) for r in splits]
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
        res = list(ex.map(_hash_task, items, chunksize=128))
    missing = [p for p, ok_e, _ in res if not ok_e]
    badhash = [p for p, ok_e, ok_h in res if ok_e and not ok_h]
    check("files_exist", not missing, checked=len(res), missing=len(missing), first=missing[:5])
    check("files_sha256", not badhash, checked=len(res), mismatches=len(badhash), first=badhash[:5])
    raw_ok = C.sha256_file(C.repo_path("datasets/raw/DS01_fairface/fairface-img-margin025-trainval.zip")) == \
        "654c9b042def71b70a63b4920b05ced0a5489dcfaee75ca5ab5f5583fe451801"
    check("raw_zip_sha256", raw_ok)

    # 4) links & traceability
    by_ref = {r["ref_id"]: r for r in splits}
    broken = [p["pair_id"] for p in all_pairs if p["ref_id"] not in by_ref or by_ref[p["ref_id"]]["ref_sha256"] != p["ref_sha256"]]
    check("pair_to_reference_links", not broken, broken=len(broken), first=broken[:5])
    trace = [r["ref_id"] for r in splits if ff.get(r["sample_id"], {}).get("sha256") != r["source_sha256"]]
    check("reference_to_raw_traceability", not trace, note="source_sha256 equals the zip-member hash in the P2-T2 manifest",
          broken=len(trace))
    lic = [r["ref_id"] for r in splits if r["license"] != "CC BY 4.0" or not r["source_url"]]
    check("license_source_recorded", not lic, missing=len(lic))

    # 5) leakage
    dev = [r for r in splits if r["split"] == "dev"]
    test = [r for r in splits if r["split"] == "test"]
    check("sample_disjoint", not ({r["sample_id"] for r in dev} & {r["sample_id"] for r in test}))
    check("split_provenance", all(r["split_source"] == "train" for r in dev) and all(r["split_source"] == "val" for r in test),
          note="dev ⊂ FairFace train, test ⊂ FairFace val")
    grp = collections.defaultdict(set)
    for r in dev + test:
        for gname in ("exact_dup_group", "near_dup_group"):
            if r[gname]:
                grp[r[gname]].add(r["split"])
    check("duplicate_groups_disjoint", not [k for k, v in grp.items() if len(v) > 1])
    leaky = [r["ref_id"] for r in dev if r["leak_flag"] == "True" or float(r["leak_max_cos_to_other_role"]) >= thr]
    check("identity_screen_dev_vs_test", not leaky, threshold=thr, violations=len(leaky))
    subset_ok = all(by_ref[p["ref_id"]]["split"] == SUBSETS[s] and by_ref[p["ref_id"]]["subset"] == s
                    for s, v in pairs.items() for p in v)
    check("subset_membership", subset_ok)

    # 6) counts / coverage
    cells = D.all_cells()
    cov = {}
    for s, v in pairs.items():
        per_ref = collections.Counter(p["ref_id"] for p in v)
        selected = [r for r in splits if r["subset"] == s]
        cov[s] = {"references": len(selected), "pairs": len(v), "cells": len(cells),
                  "complete": len(v) == len(selected) * len(cells) and set(per_ref.values()) == {len(cells)}}
    check("pair_coverage", all(c["complete"] for c in cov.values()), **cov)

    # 7) degradation reproducibility
    sample = all_pairs if a.repro_all else random.Random(cfg["seed"]).sample(all_pairs, min(a.repro_sample, len(all_pairs)))
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
        rr = list(ex.map(_repro_task, sample, chunksize=32))
    bad = [pid for pid, ok in rr if not ok]
    check("degradation_reproducible", not bad, regenerated=len(rr), mismatches=len(bad), first=bad[:5])

    counts = {"references": len(splits), "splits": dict(collections.Counter(r["split"] for r in splits)),
              "subsets": {s: sum(r["subset"] == s for r in splits) for s in SUBSETS},
              "pairs": {s: len(v) for s, v in pairs.items()},
              "test_core_strata": dict(sorted(collections.Counter(f'{r["race"]}|{r["gender"]}' for r in splits
                                                                  if r["subset"] == "test_core").items()))}
    overall = all(c["pass"] for c in checks.values())
    report = {"overall_pass": overall, "checks": checks, "counts": counts,
              "manifests": {f.name: C.sha256_file(f) for f in sorted(out_dir.glob("*.csv.gz"))}, "run": C.run_info()}
    C.write_json(out_dir / "validation_report.json", report)
    for k, v in checks.items():
        print(("PASS " if v["pass"] else "FAIL ") + k, {x: y for x, y in v.items() if x not in ("pass", "first")})
    print("OVERALL:", "PASS" if overall else "FAIL")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
