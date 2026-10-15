"""PERF Step 1 / Step 2: paired, interleaved A/B benchmark of two running FES services (evaluation only).

Two services with ONE setting changed are both up; every image is sent to one, then to the other (never at the same
time; the order alternates per image), so both see the same machine load and no measurement depends on which ran
first. The services decide everything; this script records, pairs and compares.

    run      --pair NAME --a LABEL=URL --b LABEL=URL [--sets real200,archive100,known73]
             → experiments/perf/outputs/<pair>/<label>/{ready.json, records.jsonl, out/<request>/<id>.png, chip/}
    compare  --pair NAME → experiments/perf/<pair>.json + <pair>.csv

Requests per image (service defaults otherwise): best = POST /v3/enhance (the single-output page; the path Step 1
changes), r6 = POST /v1/enhance profile=r6, r6_ft = POST /v2/enhance (the paths Step 2 changes).
Sets (none used for tuning):
    real200     the 200 CC0 photos, full_image
    archive100  archive TEST, every 192nd pair, face_chip (heavy shake blur)
    known73     the 73 inputs of data/manifests/phase5_rapid/r6_0/cases.json labelled LOCALIZED_ARTIFACT / DESTRUCTIVE /
                FAINT_ARTIFACT / COLOUR_MOTTLING (dotted, grid, rainbow and mottling outputs seen in R5-6 … R5-10), each
                in the input mode it was found in
Equivalence of a pair (fixed before the runs): status, route, model, fallback, reasons, chosen candidate and
per-candidate fallbacks equal, and no returned pixel differs by more than 2 grey levels (0 expected for Step 1: the
same workers run the same code on the same input).
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "r6_onnx"))

import service_eval as SE  # noqa: E402 — items / post / flatten / manifest of the R6-ONNX service harness

OUT = ROOT / "experiments" / "perf" / "outputs"
REP = ROOT / "experiments" / "perf"
REQUESTS = ("best", "r6", "r6_ft")
CASES = ROOT / "data" / "manifests" / "phase5_rapid" / "r6_0" / "cases.json"
ARTIFACT_PRIORS = ("ARTIFACT", "DESTRUCTIVE", "MOTTLING")
MAX_DIFF = 2


def known_items():
    arch = ROOT / "Dataset-and-code" / "archive" / "input"
    out = []
    for c in json.loads(CASES.read_text(encoding="utf-8")):
        if not any(k in c["prior"] for k in ARTIFACT_PRIORS):
            continue
        path = next((arch / sp / c["name"] for sp in ("val", "test", "train", "ztest") if (arch / sp / c["name"]).exists()),
                    ROOT / "Dataset-and-code" / "dataset" / c["name"])
        out.append(("known73", c["id"].replace(":", "_").replace(".jpg", ""), "face_chip" if c["mode"] == "fc" else "full_image",
                    path, None, c["prior"]))
    return out


def items(sets):
    out = [(*x, "") for x in SE.items(sets & {"real200", "archive100"})]
    return out + (known_items() if "known73" in sets else [])


def run(a):
    import httpx
    sets = set(a.sets.split(","))
    svc = [x.split("=", 1) for x in (a.a, a.b)]
    client = httpx.Client(timeout=600)
    todo = items(sets)
    fhs = {}
    for label, url in svc:
        ready = client.get(url + "/ready").json()
        if not ready.get("ready") or ready.get("nafnet_backend") != "torch":
            raise SystemExit(f"{label}: not ready or not PyTorch: {ready.get('error')!r} {ready.get('nafnet_backend')!r}")
        base = OUT / a.pair / label
        for kind in REQUESTS:
            (base / "out" / kind).mkdir(parents=True, exist_ok=True)
        (base / "chip").mkdir(exist_ok=True)
        (base / "ready.json").write_text(json.dumps({"ready": ready, "sets": sorted(sets), "n": len(todo),
                                                     "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        fhs[label] = (base / "records.jsonl").open("w", encoding="utf-8")
        print(label, "best_parallel", ready.get("best_parallel"), "r6_mirror_batch", ready.get("r6_mirror_batch"), flush=True)
    warm = next((x for x in todo if x[0] == "archive100"), todo[0])
    for label, url in svc:                                   # warm-up, discarded: every request kind once per service
        for kind in REQUESTS:
            SE.post(client, url, kind, warm[3].read_bytes(), warm[2])
    for k, (set_, iid, mode, path, _, prior) in enumerate(todo):
        data = path.read_bytes()
        for kind in REQUESTS:
            for label, url in (svc if (k + REQUESTS.index(kind)) % 2 == 0 else svc[::-1]):
                http, j, rtt = SE.post(client, url, kind, data, mode)
                rec = {"set": set_, "id": iid, "request": kind, "prior": prior, "http": http, "rtt_ms": rtt, **SE.flatten(j)}
                base = OUT / a.pair / label
                if j.get("status") == "ENHANCED" and j.get("image_b64"):
                    (base / "out" / kind / f"{iid}.png").write_bytes(base64.b64decode(j["image_b64"]))
                if kind == "best" and j.get("chip_b64"):
                    (base / "chip" / f"{iid}.png").write_bytes(base64.b64decode(j["chip_b64"]))
                fhs[label].write(json.dumps(rec) + "\n")
        if (k + 1) % 25 == 0:
            for fh in fhs.values():
                fh.flush()
            print(f"{a.pair}: {k + 1}/{len(todo)}", flush=True)
    for label, url in svc:
        fhs[label].close()
        end = client.get(url + "/ready").json()
        (OUT / a.pair / label / "ready_end.json").write_text(json.dumps(end, indent=1))
    print("done", a.pair, flush=True)


# ================================================================== compare

KEYS = ("http", "status", "route", "used", "model", "fallback", "decision", "reason", "bucket", "gate_route",
        "nafnet_eligible", "nafnet_reason", "nafnet_profile", "artifact_check", "artifact_reasons", "stability_passed",
        "chip_px", "n_faces", "warnings", "output", "image_format", "error")


def lat(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return {"n": 0}
    return {"n": len(v), "p50": round(st.median(v), 1), "p95": round(v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))], 1),
            "max": round(v[-1], 1), "mean": round(st.mean(v), 1)}


def sel(r):
    s = r.get("selection") or {}
    return {"chosen": s.get("chosen"), "candidates": {n: (c.get("status"), c.get("fallback"), c.get("model"), c.get("score"))
                                                      for n, c in (s.get("candidates") or {}).items()}}


def compare(a):
    base = OUT / a.pair
    labels = sorted(p.name for p in base.iterdir() if (p / "records.jsonl").exists())
    A, B = (a.order.split(",") if a.order else labels)
    recs = {}
    for lab in (A, B):
        recs[lab] = {}
        for line in (base / lab / "records.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            recs[lab][(r["set"], r["id"], r["request"])] = r
    assert set(recs[A]) == set(recs[B]), "unpaired records"
    rows = []
    for key in sorted(recs[A], key=lambda k: (k[0], SE.natural(k[1]), REQUESTS.index(k[2]))):
        ra, rb = recs[A][key], recs[B][key]
        row = {"set": key[0], "id": key[1], "request": key[2], "prior": ra.get("prior", "")}
        row["decision_equal"] = all(ra.get(k) == rb.get(k) for k in KEYS)
        row["selection_equal"] = sel(ra) == sel(rb)
        row["stability_equal"] = ra.get("stability") == rb.get("stability")
        for lab, r in ((A, ra), (B, rb)):
            row[f"{lab}_status"], row[f"{lab}_used"], row[f"{lab}_fallback"] = r.get("status"), r.get("used"), r.get("fallback")
            row[f"{lab}_chosen"] = (r.get("selection") or {}).get("chosen")
            row[f"{lab}_cand_fallbacks"] = "+".join(f"{n}:{c.get('fallback') or '-'}" for n, c in ((r.get("selection") or {}).get("candidates") or {}).items())
            row[f"{lab}_rtt_ms"], row[f"{lab}_infer_s"] = r.get("rtt_ms"), r.get("infer_s")
            row[f"{lab}_server_ms"] = (r.get("timings_ms") or {}).get("total")
            row[f"{lab}_nafnet_ran"] = r.get("infer_s") is not None and r.get("used") != "A4"
        ia = cv2.imread(str(base / A / "out" / key[2] / f"{key[1]}.png"), cv2.IMREAD_COLOR)
        ib = cv2.imread(str(base / B / "out" / key[2] / f"{key[1]}.png"), cv2.IMREAD_COLOR)
        row["image_both"], row["image_one_side_only"] = ia is not None and ib is not None, (ia is None) != (ib is None)
        if row["image_both"]:
            row["shape_equal"] = ia.shape == ib.shape
            if row["shape_equal"]:
                d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
                row.update(px_max=int(d.max()), px_mean=round(float(d.mean()), 5), px_identical=bool(d.max() == 0))
        row["equivalent"] = bool(row["decision_equal"] and row["selection_equal"] and not row["image_one_side_only"]
                                 and (not row["image_both"] or (row.get("shape_equal") and row.get("px_max", 99) <= MAX_DIFF)))
        rows.append(row)

    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("set", "id", "request", "prior", "equivalent"), k))
    (REP / f"{a.pair}.csv").write_text("\n".join([",".join(cols)] + [",".join(str(r.get(c, "")) for c in cols) for r in rows]), encoding="utf-8")

    summ = {"pair": a.pair, "a": A, "b": B, "settings": {}, "resources": {}, "sets": {}}
    for lab in (A, B):
        rd = json.loads((base / lab / "ready.json").read_text())["ready"]
        end = json.loads((base / lab / "ready_end.json").read_text())
        summ["settings"][lab] = {k: rd.get(k) for k in ("best_parallel", "r6_mirror_batch", "nafnet_backend", "route_mode",
                                                        "default_profile", "profiles", "seals", "service_version")}
        summ["settings"][lab]["models"] = {n: {k: m.get(k) for k in ("model", "weights", "weights_sha256", "variant", "mirror_batch", "load_s")}
                                           for n, m in rd["experimental_models"].items()}
        summ["settings"][lab]["worker_restarts_end"] = {"a4": end.get("worker_restarts"), "default_nafnet": end.get("nafnet_worker_restarts")}
        res = {}
        for f in ("mem_loaded", "mem_end", "container_end"):
            p = base / lab / f"{f}.json"
            if p.exists():
                res[f] = json.loads(p.read_text())
        if "mem_loaded" in res and "mem_end" in res:
            cpu = res["mem_end"]["cgroup"]["cpu_usage_s"] - res["mem_loaded"]["cgroup"]["cpu_usage_s"]
            busy = sum((r.get("timings_ms") or {}).get("total") or 0 for r in recs[lab].values()) / 1e3
            ran = sum((r.get("timings_ms") or {}).get("total") or 0 for r in recs[lab].values()
                      if r.get("infer_s") is not None) / 1e3
            summ["resources"][lab] = {
                "container_mem_after_load_mb": res["mem_loaded"]["cgroup"]["current_mb"],
                "container_mem_peak_mb": res["mem_end"]["cgroup"]["peak_mb"],
                "sum_worker_peak_rss_mb": res["mem_end"]["sum_peak_rss_mb"],
                "cpu_seconds_during_run": round(cpu, 1), "request_seconds_total": round(busy, 1),
                "request_seconds_model_ran": round(ran, 1),
                "avg_cores_busy_while_serving": round(cpu / busy, 2) if busy else None,
                "container": res.get("container_end")}
    for set_ in sorted({r["set"] for r in rows}):
        S = summ["sets"][set_] = {}
        for kind in REQUESTS:
            rs = [r for r in rows if r["set"] == set_ and r["request"] == kind]
            both = [r for r in rs if r["image_both"]]
            ran = [r for r in rs if r[f"{A}_nafnet_ran"] and r[f"{B}_nafnet_ran"]]
            d = {"n": len(rs), "equivalent": sum(r["equivalent"] for r in rs),
                 "not_equivalent": [r["id"] for r in rs if not r["equivalent"]],
                 "images_compared": len(both), "px_identical": sum(bool(r.get("px_identical")) for r in both),
                 "px_max": max((r.get("px_max", 0) for r in both), default=None),
                 "stability_tracers_equal": sum(r["stability_equal"] for r in rs), "nafnet_ran": len(ran)}
            for lab in (A, B):
                stt, fb = {}, {}
                for r in rs:
                    stt[r[f"{lab}_status"]] = stt.get(r[f"{lab}_status"], 0) + 1
                    for part in filter(None, (r[f"{lab}_cand_fallbacks"] or (f"-:{r[f'{lab}_fallback']}" if r[f"{lab}_fallback"] else "")).split("+")):
                        if not part.endswith(":-"):
                            fb[part] = fb.get(part, 0) + 1
                d[lab] = {"status": stt, "fallbacks": fb, "http_errors": sum(recs[lab][(r["set"], r["id"], r["request"])]["http"] != 200 for r in rs),
                          "http_ms_all": lat([r[f"{lab}_rtt_ms"] for r in rs]),
                          "http_ms_nafnet_ran": lat([r[f"{lab}_rtt_ms"] for r in ran]),
                          "server_ms_nafnet_ran": lat([r[f"{lab}_server_ms"] for r in ran]),
                          "model_ms": lat([r[f"{lab}_infer_s"] * 1e3 for r in ran]) if kind != "best" else {"n": 0}}
            if ran:
                ratio = [r[f"{B}_rtt_ms"] / r[f"{A}_rtt_ms"] for r in ran]
                d["b_over_a_http_time"] = {"p50": round(st.median(ratio), 3), "faster_in": sum(x < 1 for x in ratio), "n": len(ratio)}
            S[kind] = d
    (REP / f"{a.pair}.json").write_text(json.dumps(summ, indent=1))
    for set_, S in summ["sets"].items():
        for kind, d in S.items():
            print(set_, kind, "equiv", d["equivalent"], "/", d["n"], "img", d["images_compared"], "ident", d["px_identical"], "pxmax", d["px_max"],
                  "| ran", d["nafnet_ran"], A, d[A]["http_ms_nafnet_ran"].get("p50"), d[A]["http_ms_nafnet_ran"].get("p95"),
                  B, d[B]["http_ms_nafnet_ran"].get("p50"), d[B]["http_ms_nafnet_ran"].get("p95"), d.get("b_over_a_http_time"), flush=True)
    print(json.dumps(summ["resources"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare"])
    ap.add_argument("--pair", required=True)
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--order", default="")
    ap.add_argument("--sets", default="real200,archive100,known73")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)
