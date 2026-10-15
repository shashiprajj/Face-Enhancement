"""FES finalization: latency of the single-output request on ONE running service, and a decision re-check against the
parity run (evaluation only; nothing is tuned and no threshold is defined here).

    --url URL [--sets archive100,real200] [--repeats 3] [--backend onnx]
        POST /v3/enhance for every image, `repeats` passes, one request at a time, one discarded warm-up request.
        → experiments/finalization/outputs/final/records.jsonl, experiments/finalization/final_latency.json

"Small faces" = archive100 (128 px face chips); "large faces" = the real photos on which a model ran (chips up to
224 px). Reported per group: faces, samples, median / p95 / max of the HTTP round trip, counted only where a NAFNet
model ran. Every request's decision is also compared with the PyTorch record of the same image in
experiments/finalization/outputs/parity/torch/records.jsonl, and repeated passes with each other.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "perf"))

import service_ab as AB  # noqa: E402
from service_ab import SE  # noqa: E402

OUT = ROOT / "experiments" / "finalization" / "outputs" / "final"
REP = ROOT / "experiments" / "finalization"
GROUPS = {"archive100": "small faces (128 px chips)", "real200": "large faces (real photos, chips up to 224 px)"}


def main(a):
    import httpx
    sets = set(a.sets.split(","))
    client = httpx.Client(timeout=600)
    ready = client.get(a.url + "/ready").json()
    if not ready.get("ready") or ready.get("nafnet_backend") != a.backend:
        raise SystemExit(f"not ready or not the {a.backend} backend: {ready.get('error')!r} {ready.get('nafnet_backend')!r}")
    OUT.mkdir(parents=True, exist_ok=True)
    todo = AB.items(sets)
    ref = {}
    p = ROOT / "experiments" / "finalization" / "outputs" / "parity" / "torch" / "records.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["request"] == "best":
            ref[(r["set"], r["id"])] = r
    warm = next((x for x in todo if x[0] == "archive100"), todo[0])
    SE.post(client, a.url, "best", warm[3].read_bytes(), warm[2])
    recs = []
    with (OUT / "records.jsonl").open("w", encoding="utf-8") as fh:
        for rep in range(a.repeats):
            for set_, iid, mode, path, _, _ in todo:
                http, j, rtt = SE.post(client, a.url, "best", path.read_bytes(), mode)
                rec = {"pass": rep, "set": set_, "id": iid, "http": http, "rtt_ms": rtt, **SE.flatten(j)}
                recs.append(rec)
                fh.write(json.dumps(rec) + "\n")
            print(f"pass {rep + 1}/{a.repeats} done", flush=True)
    end = client.get(a.url + "/ready").json()

    def decision(r):
        return tuple(r.get(k) for k in AB.KEYS) + (json.dumps(AB.sel(r), sort_keys=True),)

    summ = {"url": a.url, "backend": ready["nafnet_backend"], "route_mode": ready["route_mode"],
            "best_parallel": ready["best_parallel"], "r6_mirror_batch": ready["r6_mirror_batch"],
            "profiles": ready["profiles"], "repeats": a.repeats, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "worker_restarts_end": {"a4": end.get("worker_restarts"), "default_nafnet": end.get("nafnet_worker_restarts")},
            "groups": {}}
    for set_ in sorted(sets):
        rs = [r for r in recs if r["set"] == set_]
        ran = [r for r in rs if r.get("infer_s") is not None]
        v = sorted(r["rtt_ms"] for r in ran)
        first = {r["id"]: r for r in rs if r["pass"] == 0}
        summ["groups"][set_] = {
            "what": GROUPS.get(set_, set_), "images": len(first), "requests": len(rs),
            "http_errors": sum(r["http"] != 200 for r in rs),
            "status": {s: sum(r["status"] == s for r in rs) for s in sorted({str(r["status"]) for r in rs})},
            "faces_model_ran": len({r["id"] for r in ran}), "samples_model_ran": len(v),
            "http_ms_model_ran": ({"median": round(st.median(v), 1), "p95": round(v[int(round(0.95 * (len(v) - 1)))], 1),
                                   "min": v[0], "max": v[-1]} if v else {}),
            "http_ms_all_median": round(st.median(r["rtt_ms"] for r in rs), 1),
            "decision_differs_from_torch_parity_run": sorted({r["id"] for r in rs if (set_, r["id"]) in ref
                                                              and decision(r) != decision(ref[(set_, r["id"])])}),
            "decision_differs_between_passes": sorted({r["id"] for r in rs if decision(r) != decision(first[r["id"]])})}
    (REP / "final_latency.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--sets", default="archive100,real200")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--backend", default="onnx")
    main(ap.parse_args())
