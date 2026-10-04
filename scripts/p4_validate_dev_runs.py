"""P4-INFRA-RESUME Phase 7 — validate the completed p4dev inference before any scoring / T6 work.

    scripts/dock.sh python scripts/p4_validate_dev_runs.py

Per configuration (B0 B1 B2 B2c A1–A5):
  * run manifest exists; rows == population (896); unique pair ids == population ids (0 missing, 0 extra, 0 dup)
  * every row status OK; required metadata present (output_path, output_sha256, output_pixel_sha256, infer_s, meta_json)
  * output file exists; file SHA-256 == record; decodes to 224×224×3 uint8 PNG; pixel SHA-256 == record
  * checkpoint (.partial.jsonl): 0 malformed lines; every OK record agrees with the manifest row; duplicate lines
    (a pair recorded twice) reported — allowed only if every copy is identical
  * _run.json: n_ok == 896, n_failed == 0, n_missing == 0, population sha == frozen p4dev manifest
Writes data/manifests/phase4/runs/p4dev/validation.json; exit 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path, run_info, sha256_bytes, sha256_file, write_json

CONFIGS = ["B0", "B1", "B2", "B2c", "A1", "A2", "A3", "A4", "A5"]
REQ = ["output_path", "output_sha256", "output_pixel_sha256", "infer_s", "meta_json"]
RUNS = "data/manifests/phase4/runs/p4dev"


def main() -> int:
    pop_path = repo_path("data/manifests/phase4/population_p4dev.csv.gz")
    ids = {r["pair_id"] for r in read_csv_gz(pop_path)}
    pop_sha = sha256_file(pop_path)
    rep, ok_all = {"run": run_info(), "population_sha256": pop_sha, "expected_pairs": len(ids), "configs": {}}, True
    for c in CONFIGS:
        errs, r = [], {}
        man = repo_path(f"{RUNS}/{c}.csv.gz")
        if not man.exists():
            rep["configs"][c] = {"ok": False, "errors": ["run manifest missing"]}
            ok_all = False
            continue
        rows = read_csv_gz(man)
        got = [x["pair_id"] for x in rows]
        r["rows"], r["missing"], r["extra"] = len(rows), len(ids - set(got)), len(set(got) - ids)
        r["duplicate_rows"] = len(got) - len(set(got))
        r["not_ok"] = sum(x["status"] != "OK" for x in rows)
        r["missing_metadata"] = sum(any(not x.get(k) for k in REQ) for x in rows)
        bad_file = bad_dim = 0
        for x in rows:
            p = repo_path(x["output_path"])
            if not p.exists():
                bad_file += 1
                continue
            data = p.read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if sha256_bytes(data) != x["output_sha256"] or img is None:
                bad_file += 1
                continue
            if img.shape != (224, 224, 3) or img.dtype != np.uint8 or not x["output_path"].endswith(".png") \
                    or sha256_bytes(img.tobytes()) != x["output_pixel_sha256"]:
                bad_dim += 1
        r["output_file_or_sha_errors"], r["output_dim_format_or_pixel_sha_errors"] = bad_file, bad_dim
        # checkpoint consistency
        malformed, copies = 0, defaultdict(list)
        for line in repo_path(f"{RUNS}/{c}.partial.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            copies[j["pair_id"]].append(j)
        mrow = {x["pair_id"]: x for x in rows}
        r["checkpoint_malformed"] = malformed
        r["checkpoint_failed_records"] = sum(j["status"] != "OK" for v in copies.values() for j in v)
        r["checkpoint_duplicate_pairs"] = sum(len(v) > 1 for v in copies.values())
        r["checkpoint_conflicting_duplicates"] = sum(
            len({(j["status"], j["output_pixel_sha256"]) for j in v if j["status"] == "OK"}) > 1 for v in copies.values())
        r["checkpoint_vs_manifest_mismatch"] = sum(
            1 for pid, v in copies.items() if pid in mrow and
            not any(j["status"] == "OK" and j["output_pixel_sha256"] == mrow[pid]["output_pixel_sha256"] for j in v))
        run = json.loads(repo_path(f"{RUNS}/{c}_run.json").read_text(encoding="utf-8"))
        r["run_json"] = {k: run.get(k) for k in ("n_ok", "n_failed", "n_missing", "workers", "threads_per_worker")}
        if run.get("population_sha256") != pop_sha:
            errs.append("run.json population sha differs from frozen p4dev manifest")
        if not (run.get("n_ok") == len(ids) and run.get("n_failed") == 0 and run.get("n_missing") == 0):
            errs.append("run.json counts incomplete")
        for k in ("missing", "extra", "duplicate_rows", "not_ok", "missing_metadata", "output_file_or_sha_errors",
                  "output_dim_format_or_pixel_sha_errors", "checkpoint_malformed", "checkpoint_conflicting_duplicates",
                  "checkpoint_vs_manifest_mismatch"):
            if r[k]:
                errs.append(f"{k}={r[k]}")
        if r["rows"] != len(ids):
            errs.append(f"rows={r['rows']}")
        r["errors"], r["ok"] = errs, not errs
        ok_all &= not errs
        rep["configs"][c] = r
        print(c, "OK" if not errs else f"FAIL {errs}", flush=True)
    rep["all_ok"] = ok_all
    write_json(repo_path(f"{RUNS}/validation.json"), rep)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
