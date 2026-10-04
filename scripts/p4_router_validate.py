"""P4-T7 step 5 — validate the RUNTIME router (fes_bench.routing.Router) end-to-end on DEV data:
determinism, equality with the benchmark harness, resource behaviour and real fallbacks.

    scripts/dock.sh python scripts/p4_router_validate.py [--per-route 4]

Sample (seeded): up to `per_route` p4dev pairs for every route the frozen table uses, plus the same number of
DO_NOT_ENHANCE and UNSAFE_OR_UNCERTAIN pairs. Each pair is routed twice with real adapters (one candidate loaded at a
time — memory-safe); checks:
  R1 decisions/routes identical across the two passes and equal to the offline replay (p4_route_evaluate logic)
  R2 routed outputs pixel-identical across passes AND to the harness output (run-manifest output_pixel_sha256)
  R3 ORIGINAL paths return the B0 image exactly
  R4 injected TIMEOUT (timeout 1 ms) and MODEL_FAILURE (adapter replaced by one that raises) end in ORIGINAL with the right reason
  R5 latency per path and process peak RSS recorded (descriptive; shared CPU host)
Writes data/manifests/phase4/t7/router_validation.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time

import cv2
import numpy as np

from fes_bench import gate as GT
from fes_bench import geometry as G
from fes_bench import quality as Q
from fes_bench import routing as RT
from fes_bench.adapters import build
from fes_data.common import read_csv_gz, repo_path, run_info, sha256_bytes, write_json
from fes_data.preprocess import YuNetDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-route", type=int, default=4)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    g6 = json.loads(repo_path("configs/phase4/t6_gate.json").read_text(encoding="utf-8"))
    rules = GT.load_rules(g6)
    r7 = json.loads(repo_path("configs/phase4/t7_routing.json").read_text(encoding="utf-8"))
    t, table = RT.load_routing(r7, g6)
    det = YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    face_check = lambda out: len(det.detect(out)) > 0  # noqa: E731  (same rule as the harness output detection)

    pop = read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz"))
    qf = {r["pair_id"]: r for r in read_csv_gz(repo_path("data/manifests/phase4/quality/p4dev.csv.gz")) if r["item"] == "degraded"}
    runs = {}

    def run_of(c):
        if c not in runs:
            runs[c] = {r["pair_id"]: r for r in read_csv_gz(repo_path(f"data/manifests/phase4/runs/p4dev/{c}.csv.gz"))}
        return runs[c]

    def load_img(p):
        return cv2.imdecode(np.frombuffer(repo_path(p["degraded_path"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)

    # plan every dev pair from freshly computed features (not the stored ones) and group by route
    groups = {}
    for p in sorted(pop, key=lambda p: hashlib.sha256(f"p4-router|{p['pair_id']}".encode()).hexdigest()):
        img = load_img(p)
        f = Q.features(img, det)
        d, why = GT.decide(f, rules)
        route = table.get(RT.bucket_of(f, t), RT.ORIGINAL) if d == GT.ENH else RT.ORIGINAL
        key = route if route != RT.ORIGINAL else (d if d != GT.ENH else "ENHANCE_CANDIDATE->ORIGINAL")
        if len(groups.setdefault(key, [])) < a.per_route:
            groups[key].append((p, img, f))
    # R1b: the runtime plan from freshly computed features equals the offline replay from the stored manifest
    from fes_bench import p4eval as E
    stored = E.load("p4dev")
    replay_match = all(
        (GT.decide(stored[p["pair_id"]].feat, rules)[0], RT.bucket_of(stored[p["pair_id"]].feat, t)
         if GT.decide(stored[p["pair_id"]].feat, rules)[0] == GT.ENH else "")
        == (GT.decide(f, rules)[0], RT.bucket_of(f, t) if GT.decide(f, rules)[0] == GT.ENH else "")
        for v in groups.values() for p, _, f in v)

    res = {"run": run_info(), "t6_seal": g6["sha256"], "t7_seal": r7["sha256"], "table": table, "groups": {},
           "checks": {}}
    ok_r1 = ok_r2 = ok_r3 = True
    for key, items in groups.items():
        cfgs = {key} if key in RT.CANDIDATES else set()
        adapters = {}
        for c in cfgs:
            ad = build(c)
            ad.load()
            adapters[c] = ad.enhance
        router = RT.Router(rules, t, table, adapters, features_fn=lambda im: Q.features(im, det), face_check=face_check)
        rows = []
        for p, img, f in items:
            ctx = {"s": img.shape[0], "M_e2f": np.array(json.loads(p["ffhq512_transform_2x3"]), np.float64)}
            orig = G.upsample_to_e(img)
            outs, recs = [], []
            for _ in range(2):
                t0 = time.perf_counter()
                o, rec = router.route(img, ctx, orig)
                outs.append(o)
                recs.append(rec)
            same_dec = (recs[0].decision, recs[0].route, recs[0].used) == (recs[1].decision, recs[1].route, recs[1].used)
            same_pix = np.array_equal(outs[0], outs[1])
            if recs[0].used != RT.ORIGINAL:
                harness = run_of(recs[0].used)[p["pair_id"]]["output_pixel_sha256"]
                eq_harness = sha256_bytes(outs[0].tobytes()) == harness
            else:
                eq_harness = np.array_equal(outs[0], orig)
                ok_r3 &= eq_harness
            ok_r1 &= same_dec
            ok_r2 &= same_pix and eq_harness
            rows.append({"pair_id": p["pair_id"], "decision": recs[0].decision, "reason": recs[0].reason,
                         "bucket": recs[0].bucket, "route": recs[0].route, "used": recs[0].used,
                         "fallback": recs[0].fallback, "deterministic": same_dec and same_pix,
                         "equals_harness_or_original": eq_harness, "latency_s": [round(r.latency_s, 3) for r in recs]})
        # R4 fault injection on the first routed pair of this group
        inj = {}
        if adapters and items:
            p, img, f = items[0]
            ctx = {"s": img.shape[0], "M_e2f": np.array(json.loads(p["ffhq512_transform_2x3"]), np.float64)}
            orig = G.upsample_to_e(img)
            r_to = RT.Router(rules, t, table, adapters, lambda im: Q.features(im, det), face_check, timeout_s=0.001)
            o, rec = r_to.route(img, ctx, orig)
            inj["timeout"] = {"fallback": rec.fallback, "returned_original": np.array_equal(o, orig)}
            time.sleep(1)
            def broken(i, c):
                raise RuntimeError("injected model failure")
            r_fail = RT.Router(rules, t, table, {k: broken for k in adapters}, lambda im: Q.features(im, det), face_check)
            o, rec = r_fail.route(img, ctx, orig)
            inj["model_failure"] = {"fallback": rec.fallback, "error": rec.error[:120],
                                    "returned_original": np.array_equal(o, orig)}
        res["groups"][key] = {"n": len(rows), "rows": rows, "fault_injection": inj}
        del adapters
    inj_ok = all(g["fault_injection"] == {} or (g["fault_injection"]["timeout"]["fallback"] == "TIMEOUT"
                                               and g["fault_injection"]["timeout"]["returned_original"]
                                               and g["fault_injection"]["model_failure"]["fallback"] == "MODEL_FAILURE"
                                               and g["fault_injection"]["model_failure"]["returned_original"])
                 for g in res["groups"].values())
    res["checks"] = {"R1_decisions_deterministic": ok_r1, "R2_routed_outputs_deterministic_and_equal_harness": ok_r2,
                     "R3_original_paths_exact": ok_r3, "R4_fault_injection_falls_back": inj_ok,
                     "R1b_runtime_plan_equals_offline_replay": replay_match}
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    res["validated"] = all(res["checks"].values())
    write_json(repo_path("data/manifests/phase4/t7/router_validation.json"), res)
    print(json.dumps({"checks": res["checks"], "peak_rss_mb": res["peak_rss_mb"],
                      "groups": {k: v["n"] for k, v in res["groups"].items()}}, indent=1))


if __name__ == "__main__":
    main()
