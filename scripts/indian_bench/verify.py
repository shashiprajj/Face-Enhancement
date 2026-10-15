"""Phase 7 — automated integrity checks. Read-only; writes reports/07_integrity.{json,md}.

    python scripts/indian_bench/verify.py [--pilot] [--repro-samples 20] [--skip-source-hash]

Checks (each PASS / FAIL with counts and the first offending ids):
  source_unchanged        every source file still has the size + SHA-256 of the inventory; no file added or removed
  selection_reproducible  re-running the seeded selection gives the same ids; no id or content selected twice
  variants_complete       scope × matrix = generated variants, each exactly once, each mapped to one clean reference
  clean_files             every clean reference exists with its manifest hash
  degraded_files          every degraded file exists, has its manifest hash, decodes to 8-bit RGB of the clean size
  reproducible            regenerating the variants of N random samples from source + manifest gives identical bytes
  results_complete        every variant has exactly one final service answer mapped to one degraded input
  enhanced_files          every stored output exists with its manifest hash; ORIGINAL outputs equal their input bytes;
                          ENHANCED outputs are readable square PNG chips
  comparisons_complete    every answered variant has one comparison referencing exactly its three images
                          (hashes equal the manifests'), and every chip view matches the chip the service analysed
  failures_listed         every failed / rejected / model-failure answer and every generation failure is in failures.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common as C
import degrade as DG

FINAL = ("enhanced", "original_no_face", "original_gate", "fallback_guard", "fallback_model_failure", "rejected")


def _hash_source(args):
    root, rel = args
    p = os.path.join(root, rel)
    return rel, (os.path.getsize(p), C.sha256_file(p)) if os.path.exists(p) else None


def _check_degraded(args):
    root, v, clean_wh = args
    import cv2
    import numpy as np
    p = Path(root) / v["degraded_path"]
    if not p.exists():
        return v["variant_id"], "missing file"
    data = p.read_bytes()
    if C.sha256_bytes(data) != v["file_sha256"]:
        return v["variant_id"], "file hash differs from the manifest"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        return v["variant_id"], "not a readable 8-bit 3-channel image"
    if (img.shape[1], img.shape[0]) != clean_wh:
        return v["variant_id"], f"size {img.shape[1]}x{img.shape[0]} != clean {clean_wh}"
    if C.sha256_bytes(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()) != v["pixel_sha256"]:
        return v["variant_id"], "decoded pixels differ from the manifest"
    return v["variant_id"], ""


def _check_enhanced(args):
    root, r, deg_sha = args
    import cv2
    import numpy as np
    p = Path(root) / r["enhanced_path"]
    if not p.exists():
        return r["variant_id"], "missing file"
    data = p.read_bytes()
    sha = C.sha256_bytes(data)
    if sha != r["output_sha256"]:
        return r["variant_id"], "file hash differs from the manifest"
    if r["outcome"] == "enhanced":
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if data[:8] != b"\x89PNG\r\n\x1a\n" or img is None or img.ndim != 3 or img.shape[0] != img.shape[1]:
            return r["variant_id"], "ENHANCED output is not a readable square PNG"
        if sha == deg_sha:
            return r["variant_id"], "ENHANCED output identical to its input"
    elif sha != deg_sha:
        return r["variant_id"], "ORIGINAL output differs from the degraded input"
    return r["variant_id"], ""


def _repro(args):
    cfg, row, specs, manifest = args
    import degradations as D
    ref = DG.build_reference(cfg, row)
    bad = []
    if C.sha256_bytes(ref.tobytes()) != manifest[next(iter(manifest))]["clean_pixel_sha256"]:
        bad.append(f"{row['sample_id']}: clean reference pixels differ")
    for s in specs:
        vid = DG.variant_id(row["sample_id"], s)
        data, _, _ = D.apply(ref, s["ops"], C.stable_seed(cfg["seed"], row["sample_id"], vid))
        m = manifest.get(vid)
        if m is None or C.sha256_bytes(data) != m["file_sha256"] or json.dumps(s["ops"], separators=(",", ":")) != m["ops"]:
            bad.append(vid)
    return bad


def main(a):
    cfg = C.load_config(a.config)
    root = cfg["bench_root"]
    t0 = time.time()
    checks: dict[str, dict] = {}

    def put(name, bad, total, note=""):
        bad = list(bad)
        checks[name] = {"status": "PASS" if not bad else "FAIL", "checked": total, "problems": len(bad),
                        "first": bad[:8], "note": note}
        print(f"{checks[name]['status']:4}  {name}: {total} checked, {len(bad)} problems {note}", flush=True)

    inv = C.read_csv(C.bench(cfg, "manifests", "source_inventory.csv"))
    selected = C.read_csv(C.bench(cfg, "manifests", "selected_originals.csv"))
    scope = C.scope(cfg, a.pilot)
    scope_ids = {r["sample_id"] for r in scope}
    _, specs = DG.load_matrix(cfg)
    variants = [v for v in C.read_csv(C.bench(cfg, "manifests", "generated_variants.csv")) if v["sample_id"] in scope_ids]
    cleans = {r["sample_id"]: r for r in C.read_csv(C.bench(cfg, "manifests", "clean_references.csv"))}
    results = [r for r in C.read_csv(C.bench(cfg, "manifests", "enhancement_results.csv")) if r["sample_id"] in scope_ids]
    comps = [r for r in C.read_csv(C.bench(cfg, "manifests", "comparisons.csv")) if r["sample_id"] in scope_ids]
    failures = C.read_csv(C.bench(cfg, "manifests", "failures.csv"))

    with ProcessPoolExecutor(a.workers) as ex:
        # 1 source unchanged
        if a.skip_source_hash:
            checks["source_unchanged"] = {"status": "SKIPPED", "checked": 0, "problems": 0, "first": [], "note": ""}
        else:
            on_disk = sorted(os.path.relpath(os.path.join(d, f), cfg["source_root"]).replace("\\", "/")
                             for d, _, fs in os.walk(cfg["source_root"]) for f in fs)
            known = {r["rel_path"]: r for r in inv}
            bad = [f"added:{f}" for f in on_disk if f not in known] + [f"removed:{f}" for f in known if f not in set(on_disk)]
            for rel, got in ex.map(_hash_source, [(cfg["source_root"], f) for f in known], chunksize=16):
                if got is None or str(got[0]) != known[rel]["bytes"] or got[1] != known[rel]["sha256"]:
                    bad.append(f"changed:{rel}")
            put("source_unchanged", bad, len(known), f"({len(on_disk)} files on disk)")

        # 2 selection
        import math
        import numpy as np
        bad = []
        want = set()
        for e in sorted({r["expression"] for r in inv}):
            elig = sorted((r for r in inv if r["expression"] == e and r["eligible"] == "True"), key=lambda r: r["rel_path"])
            perm = np.random.default_rng(C.stable_seed(cfg["seed"], "select", e)).permutation(len(elig))
            want |= {elig[i]["sample_id"] for i in perm[:math.floor(len(elig) * cfg["select_fraction"] + 0.5)]}
        got = [r["sample_id"] for r in selected]
        bad += [f"not reproduced:{s}" for s in sorted(want ^ set(got))]
        bad += [f"selected twice:{s}" for s, n in Counter(got).items() if n > 1]
        bad += [f"same content twice:{s}" for s, n in Counter(r["sha256"] for r in selected).items() if n > 1]
        hold = {r["sample_id"] for r in C.read_csv(C.bench(cfg, "manifests", "holdout_originals.csv"))}
        bad += [f"in both splits:{s}" for s in sorted(hold & set(got))]
        put("selection_reproducible", bad, len(got), f"(holdout {len(hold)})")

        # 3 variants complete
        expect = {DG.variant_id(r["sample_id"], s) for r in scope for s in specs}
        ids = [v["variant_id"] for v in variants]
        bad = [f"missing:{i}" for i in sorted(expect - set(ids))] + [f"unexpected:{i}" for i in sorted(set(ids) - expect)]
        bad += [f"duplicate id:{i}" for i, n in Counter(ids).items() if n > 1]
        bad += [f"duplicate path:{p}" for p, n in Counter(v["degraded_path"] for v in variants).items() if n > 1]
        bad += [f"no clean reference:{v['variant_id']}" for v in variants
                if v["sample_id"] not in cleans or cleans[v["sample_id"]]["clean_path"] != v["clean_path"]
                or cleans[v["sample_id"]]["pixel_sha256"] != v["clean_pixel_sha256"]]
        put("variants_complete", bad, len(expect), f"({len(scope)} originals x {len(specs)} variants)")

        # 4 clean files
        bad = [s for s in sorted(scope_ids) if s not in cleans or not (Path(root) / cleans[s]["clean_path"]).exists()
               or C.sha256_file(Path(root) / cleans[s]["clean_path"]) != cleans[s]["file_sha256"]]
        put("clean_files", bad, len(scope_ids))

        # 5 degraded files
        wh = {s: (int(c["width"]), int(c["height"])) for s, c in cleans.items()}
        bad = [f"{i}: {e}" for i, e in ex.map(_check_degraded, [(root, v, wh[v["sample_id"]]) for v in variants],
                                              chunksize=64) if e]
        put("degraded_files", bad, len(variants))

        # 6 reproducibility
        rng = random.Random(cfg["seed"])
        pick = rng.sample(sorted(scope, key=lambda r: r["sample_id"]), min(a.repro_samples, len(scope)))
        by_s = {}
        for v in variants:
            by_s.setdefault(v["sample_id"], {})[v["variant_id"]] = v
        bad = [b for part in ex.map(_repro, [(cfg, r, specs, by_s.get(r["sample_id"], {"": {"clean_pixel_sha256": ""}}))
                                             for r in pick]) for b in part]
        put("reproducible", bad, len(pick) * len(specs), f"({len(pick)} originals regenerated from source)")

        # 7 results complete
        vmap = {v["variant_id"]: v for v in variants}
        rid = [r["variant_id"] for r in results]
        bad = [f"no answer:{i}" for i in sorted(set(vmap) - set(rid))] + [f"unknown variant:{i}" for i in sorted(set(rid) - set(vmap))]
        bad += [f"duplicate:{i}" for i, n in Counter(rid).items() if n > 1]
        bad += [f"not final ({r['outcome']}):{r['variant_id']}" for r in results if r["outcome"] not in FINAL]
        bad += [f"wrong input:{r['variant_id']}" for r in results
                if r["variant_id"] in vmap and r["degraded_path"] != vmap[r["variant_id"]]["degraded_path"]]
        bad += [f"status/outcome mismatch:{r['variant_id']}" for r in results
                if (r["outcome"] == "enhanced") != (r["status"] == "ENHANCED")]
        put("results_complete", bad, len(vmap), f"{dict(Counter(r['outcome'] for r in results))}")

        # 8 enhanced files
        with_out = [r for r in results if r.get("enhanced_path")]
        bad = [f"{i}: {e}" for i, e in ex.map(_check_enhanced, [(root, r, vmap[r["variant_id"]]["file_sha256"])
                                                                for r in with_out if r["variant_id"] in vmap], chunksize=64) if e]
        bad += [f"{r['variant_id']}: answer without a stored image" for r in results
                if r["outcome"] in FINAL and r["outcome"] != "rejected" and not r.get("enhanced_path")]
        put("enhanced_files", bad, len(with_out))

    # 9 comparisons
    cmap = {c["variant_id"]: c for c in comps}
    rmap = {r["variant_id"]: r for r in results}
    bad = []
    for r in with_out:
        c = cmap.get(r["variant_id"])
        v = vmap.get(r["variant_id"])
        if c is None or v is None:
            bad.append(f"no comparison:{r['variant_id']}")
        elif c["error"]:
            bad.append(f"comparison error:{r['variant_id']}: {c['error']}")
        elif not (Path(root) / c["comparison_path"]).exists() or not (Path(root) / c["comparison_path_by_degradation"]).exists():
            bad.append(f"comparison file missing:{r['variant_id']}")
        elif (c["clean_path"], c["degraded_path"], c["enhanced_path"]) != (v["clean_path"], v["degraded_path"], r["enhanced_path"]) \
                or (c["clean_pixel_sha256"], c["degraded_file_sha256"], c["output_sha256"]) != \
                (v["clean_pixel_sha256"], v["file_sha256"], r["output_sha256"]):
            bad.append(f"comparison references other images:{r['variant_id']}")
        elif c["view"] == "chip" and c["chip_match"] != "True":
            bad.append(f"chip crop differs from the service's chip:{r['variant_id']}")
        elif (c["view"] == "full") != (r["outcome"] == "original_no_face"):
            bad.append(f"view {c['view']} inconsistent with outcome {r['outcome']}:{r['variant_id']}")
    bad += [f"comparison without a result:{i}" for i in sorted(set(cmap) - set(rmap))]
    put("comparisons_complete", bad, len(with_out), f"{dict(Counter(c['view'] for c in comps))}")

    # 10 failures listed
    listed = {f["variant_id"] for f in failures if f["stage"] == "enhance"}
    need = {r["variant_id"] for r in C.read_csv(C.bench(cfg, "manifests", "enhancement_results.csv"))
            if r["outcome"] in ("failed", "rejected", "fallback_model_failure")}
    gen = {r["sample_id"] for r in C.read_jsonl(C.bench(cfg, "manifests", "generation_failures.jsonl"))}
    bad = sorted(need - listed) + sorted(gen - {f["sample_id"] for f in failures if f["stage"] == "generate"})
    put("failures_listed", bad, len(need) + len(gen), f"(failures.csv rows: {len(failures)})")

    overall = "PASS" if all(c["status"] in ("PASS", "SKIPPED") for c in checks.values()) else "FAIL"
    out = {"scope": "pilot" if a.pilot else "full", "overall": overall, "taken": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seconds": round(time.time() - t0, 1), "checks": checks}
    name = "07_integrity_pilot" if a.pilot else "07_integrity"
    C.bench(cfg, "reports", name + ".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Integrity checks — {out['scope']} scope — {overall}", "", f"Run {out['taken']} ({out['seconds']} s).", "",
          C.md_table([{"check": k, "status": v["status"], "checked": v["checked"], "problems": v["problems"],
                       "note": v["note"], "first problems": "; ".join(map(str, v["first"][:3]))}
                      for k, v in checks.items()], ["check", "status", "checked", "problems", "note", "first problems"]), ""]
    C.bench(cfg, "reports", name + ".md").write_text("\n".join(md), encoding="utf-8")
    print(f"OVERALL {overall} ({out['seconds']} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--repro-samples", type=int, default=20)
    ap.add_argument("--skip-source-hash", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
