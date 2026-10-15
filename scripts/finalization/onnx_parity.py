"""FES finalization: paired, interleaved parity run of the PyTorch service and the ONNX service (evaluation only).

Reuses the PERF A/B harness (scripts/perf/service_ab.py: same sets, requests, records and equivalence rule) with one
difference: the two services differ in FES_NAFNET_BACKEND, so each must report the backend its label names. Nothing is
tuned here and no threshold is defined here: a pair is equivalent when status, route, model, fallback, reasons, chosen
candidate and per-candidate fallbacks are equal and no returned pixel differs by more than 2 grey levels (the
criterion of experiments/r6_onnx/REPORT.md §4 and experiments/perf/REPORT.md §3).

    run      --a torch=URL --b onnx=URL [--sets real200,archive100,known73]
             → experiments/finalization/outputs/parity/<label>/{ready.json, records.jsonl, out/<request>/<id>.png, chip/}
    compare  → experiments/finalization/parity.json + parity.csv
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "perf"))

import service_ab as AB  # noqa: E402 — items / compare of the PERF harness
from service_ab import SE  # noqa: E402

AB.OUT = ROOT / "experiments" / "finalization" / "outputs"
AB.REP = ROOT / "experiments" / "finalization"
PAIR = "parity"
# settings that must be the same on both services (everything /ready reports except the backend and its models)
SAME = ("service_version", "seals", "routing_table", "gate", "route_mode", "nafnet_profile", "limits", "profiles",
        "default_profile", "best_parallel", "r6_mirror_batch")


def run(a):
    import httpx
    sets = set(a.sets.split(","))
    svc = [x.split("=", 1) for x in (a.a, a.b)]
    client = httpx.Client(timeout=600)
    todo = AB.items(sets)
    ready, fhs = {}, {}
    for label, url in svc:
        rd = ready[label] = client.get(url + "/ready").json()
        if not rd.get("ready") or rd.get("nafnet_backend") != label:
            raise SystemExit(f"{label}: not ready or not the {label} backend: {rd.get('error')!r} {rd.get('nafnet_backend')!r}")
        want = "onnxruntime" if label == "onnx" else None
        assert all(rd["experimental_models"][p].get("backend") == want for p in ("r6", "r6_ft")), rd["experimental_models"]
        base = AB.OUT / PAIR / label
        for kind in AB.REQUESTS:
            (base / "out" / kind).mkdir(parents=True, exist_ok=True)
        (base / "chip").mkdir(exist_ok=True)
        (base / "ready.json").write_text(json.dumps({"ready": rd, "sets": sorted(sets), "n": len(todo), "url": url,
                                                     "manifest": SE.manifest(sets & {"real200", "archive100"}),
                                                     "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        fhs[label] = (base / "records.jsonl").open("w", encoding="utf-8")
    (la, _), (lb, _) = svc
    differ = [k for k in SAME if ready[la].get(k) != ready[lb].get(k)]
    if differ:
        raise SystemExit(f"the two services are not configured alike: {differ}")
    print("settings equal on both services:", ", ".join(SAME), flush=True)
    warm = next((x for x in todo if x[0] == "archive100"), todo[0])
    for label, url in svc:                                   # warm-up, discarded: every request kind once per service
        for kind in AB.REQUESTS:
            SE.post(client, url, kind, warm[3].read_bytes(), warm[2])
    for k, (set_, iid, mode, path, _, prior) in enumerate(todo):
        data = path.read_bytes()
        for kind in AB.REQUESTS:
            for label, url in (svc if (k + AB.REQUESTS.index(kind)) % 2 == 0 else svc[::-1]):
                http, j, rtt = SE.post(client, url, kind, data, mode)
                rec = {"set": set_, "id": iid, "request": kind, "prior": prior, "http": http, "rtt_ms": rtt, **SE.flatten(j)}
                base = AB.OUT / PAIR / label
                if j.get("status") == "ENHANCED" and j.get("image_b64"):
                    (base / "out" / kind / f"{iid}.png").write_bytes(base64.b64decode(j["image_b64"]))
                if kind == "best" and j.get("chip_b64"):
                    (base / "chip" / f"{iid}.png").write_bytes(base64.b64decode(j["chip_b64"]))
                fhs[label].write(json.dumps(rec) + "\n")
        if (k + 1) % 25 == 0:
            for fh in fhs.values():
                fh.flush()
            print(f"{PAIR}: {k + 1}/{len(todo)}", flush=True)
    for label, url in svc:
        fhs[label].close()
        (AB.OUT / PAIR / label / "ready_end.json").write_text(json.dumps(client.get(url + "/ready").json(), indent=1))
    print("done", PAIR, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare"])
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--sets", default="real200,archive100,known73")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        a.pair, a.order = PAIR, "torch,onnx"
        AB.compare(a)
