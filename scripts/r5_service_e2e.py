"""R5-1 — end-to-end run of the standalone service with the REAL A4 worker (CPU), through the HTTP API.

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_service_e2e.py [--per-group 3]

1. Policy equivalence (no model run): every one of the 896 p4dev chips through EnhancementPipeline in face_chip mode
   with a spy model → (decision, reason, bucket, route) must equal the frozen Phase 4 decision log.
2. Representative samples over HTTP (FastAPI TestClient, real A4 in the isolated worker): `per_group` seeded p4dev
   chips per frozen outcome (A4 NOISE, A4 BLUR, ENHANCE→original, DO_NOT_ENHANCE mild, UNSAFE) + full-image cases
   (two clean faces on one canvas, a no-face image, an undecodable file). Each request is sent twice (determinism).
   A4 outputs are compared with the Phase 4 run manifest pixels.
3. Latency per request, service + worker peak RSS (psutil), cold start (worker load + warm-up).
Writes data/manifests/phase5_rapid/r5_1/{e2e.json, requests.csv.gz}; output images go to the git-ignored
experiments/phase5_rapid/outputs/r5_1/.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import threading
import time
from collections import Counter

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=3)
    a = ap.parse_args()
    import psutil
    from fastapi.testclient import TestClient

    from fes_data.common import read_csv_gz, repo_path, run_info, write_csv_gz, write_json
    from fes_service import app as A
    from fes_service import pipeline as SP

    log = {r["pair_id"]: r for r in read_csv_gz(repo_path("data/manifests/phase4/t7/decision_log_p4dev.csv.gz"))}
    pop = {r["pair_id"]: r for r in read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz"))}
    runs = {r["pair_id"]: r for r in read_csv_gz(repo_path("data/manifests/phase4/runs/p4dev/A4.csv.gz"))}
    data_of = lambda k: repo_path(pop[k]["degraded_path"]).read_bytes()  # noqa: E731

    # ---- 1. policy equivalence on all dev chips (spy model; nothing is enhanced for real)
    class Spy:
        def __call__(self, img, ctx):
            return cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC), {}
    pipe = SP.EnhancementPipeline(Spy(), timeout_s=30)
    mism, n_eq = [], 0
    for k in sorted(log):
        r = pipe.process(data_of(k), "face_chip")
        got = (r.decision, r.reason, r.bucket, r.route)
        exp = (log[k]["decision"], log[k]["reason"], log[k]["bucket"], log[k]["route"])
        n_eq += got == exp
        if got != exp:
            mism.append({"pair_id": k, "got": got, "expected": exp})
    equivalence = {"n": len(log), "identical": n_eq, "mismatches": mism[:20]}
    print("policy equivalence", n_eq, "/", len(log))

    # ---- 2. representative samples through HTTP with the real A4
    def group(k):
        r = log[k]
        if r["route"] == "A4":
            return f"A4_{r['bucket']}"
        if r["decision"] == "ENHANCE_CANDIDATE":
            return "ENHANCE_to_original"
        if r["decision"] == "DO_NOT_ENHANCE":
            return "DNE_mild" if pop[k]["severity"] == "mild" else None
        return "UNSAFE"
    picks: dict[str, list] = {}
    for k in sorted(log, key=lambda k: hashlib.sha256(f"r5-1|{k}".encode()).hexdigest()):
        g = group(k)
        if g and len(picks.setdefault(g, [])) < a.per_group:
            picks[g].append(k)
    refs = sorted((r for r in read_csv_gz(repo_path("data/manifests/references_v1.csv.gz"))
                   if r["accepted"] == "True" and r["pool"] == "test_pool"), key=lambda r: r["ref_id"])[:2]
    f1, f2 = (cv2.imread(str(repo_path(r["ref_path"]))) for r in refs)
    canvas = np.full((420, 760, 3), 110, np.uint8)
    canvas[90:314, 40:264], canvas[140:290, 480:630] = f1, cv2.resize(f2, (150, 150), interpolation=cv2.INTER_AREA)
    extra = {"full_two_faces": cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(),
             "full_no_face": cv2.imencode(".png", np.random.default_rng(3).integers(0, 255, (360, 480, 3), np.uint8))[1].tobytes(),
             "corrupt_file": b"\x89PNG\r\n\x1a\n" + b"\x00" * 200}

    out_dir = repo_path("experiments/phase5_rapid/outputs/r5_1")
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    peak = {"service_mb": 0.0, "worker_mb": 0.0}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak["service_mb"] = max(peak["service_mb"], proc.memory_info().rss / 2**20)
            for c in proc.children(recursive=True):
                try:
                    peak["worker_mb"] = max(peak["worker_mb"], c.memory_info().rss / 2**20)
                except psutil.Error:
                    pass
            stop.wait(0.2)
    threading.Thread(target=sampler, daemon=True).start()

    os.environ.setdefault("FES_A4_FACTORY", "fes_service.models:a4_factory")
    rows = []
    t_cold = time.perf_counter()
    with TestClient(A.app) as client:
        cold_s = time.perf_counter() - t_cold
        ready = client.get("/ready").json()
        jobs = [(g, k, data_of(k), "face_chip") for g, ks in sorted(picks.items()) for k in ks]
        jobs += [(name, name, d, "full_image") for name, d in extra.items()]
        for g, k, data, mode in jobs:
            res = []
            for rep in range(2):
                t0 = time.perf_counter()
                resp = client.post("/v1/enhance", files={"image": ("in", data, "application/octet-stream")},
                                   data={"input_mode": mode})
                wall = time.perf_counter() - t0
                j = resp.json()
                img = base64.b64decode(j["image_b64"]) if j.get("image_b64") else b""
                res.append((resp.status_code, j, img, wall))
            (c1, j1, i1, w1), (_, j2, i2, w2) = res
            pix = ""
            if j1["status"] == "ENHANCED":
                o = cv2.imdecode(np.frombuffer(i1, np.uint8), cv2.IMREAD_COLOR)
                pix = hashlib.sha256(o.tobytes()).hexdigest()
                cv2.imwrite(str(out_dir / f"{k}.A4.png"), o)
            strip = lambda m: {x: y for x, y in m.items() if x not in ("request_id", "timings_ms", "image_b64")}  # noqa: E731
            rows.append({"group": g, "id": k, "mode": mode, "http": c1, "status": j1["status"],
                         "decision": j1.get("decision", ""), "reason": j1.get("reason", ""),
                         "bucket": j1.get("bucket", ""), "route": j1.get("route", ""), "used": j1.get("used", ""),
                         "fallback": j1.get("fallback", ""), "escalate": j1.get("escalate", ""),
                         "warnings": "|".join(j1.get("warnings", [])), "n_faces": j1.get("n_faces", ""),
                         "chip_px": j1.get("chip_px", ""), "error": j1.get("error", ""),
                         "original_bytes_returned": j1["status"] == "ORIGINAL" and i1 == data,
                         "deterministic": strip(j1) == strip(j2) and i1 == i2,
                         "matches_phase4_decision": (mode != "face_chip") or
                         (j1["decision"], j1["route"]) == (log[k]["decision"], log[k]["route"]),
                         "a4_pixels_equal_phase4": (pix == runs[k]["output_pixel_sha256"]) if pix else "",
                         "wall_s_1": round(w1, 2), "wall_s_2": round(w2, 2),
                         "server_total_ms": j1.get("timings_ms", {}).get("total", "")})
            print(rows[-1]["group"], rows[-1]["status"], rows[-1]["used"], rows[-1]["fallback"], rows[-1]["wall_s_1"])
    stop.set()

    lat = {}
    for st in ("ENHANCED", "ORIGINAL", "REJECTED"):
        w = [x for r in rows if r["status"] == st for x in (r["wall_s_1"], r["wall_s_2"])]
        if w:
            lat[st] = {"n": len(w), "p50_s": round(float(np.median(w)), 3), "max_s": round(max(w), 3)}
    res = {"run": run_info(), "data_kind": "SYNTHETIC (Phase 4 dev chips) + 3 constructed full-image cases",
           "host_note": "shared laptop CPU; WelOrbit containers running; torch 2 threads in the worker",
           "policy_equivalence_face_chip": equivalence, "ready": ready, "cold_start_s": round(cold_s, 1),
           "groups": {g: len(v) for g, v in picks.items()},
           "status_counts": dict(Counter(r["status"] for r in rows)),
           "checks": {"all_deterministic": all(r["deterministic"] for r in rows),
                      "face_chip_matches_phase4": all(r["matches_phase4_decision"] for r in rows),
                      "a4_pixels_equal_phase4": all(r["a4_pixels_equal_phase4"] for r in rows if r["a4_pixels_equal_phase4"] != ""),
                      "originals_returned_unchanged": all(r["original_bytes_returned"] for r in rows if r["status"] == "ORIGINAL"),
                      "no_unexpected_fallbacks": not any(r["fallback"] for r in rows)},
           "latency": lat, "peak_rss_mb": {k: round(v, 1) for k, v in peak.items()}}
    out = repo_path("data/manifests/phase5_rapid/r5_1")
    write_csv_gz(out / "requests.csv.gz", rows, list(rows[0]))
    write_json(out / "e2e.json", res)
    print(json.dumps({k: res[k] for k in ("checks", "latency", "peak_rss_mb", "cold_start_s", "status_counts")}, indent=1))
    print("equivalence", equivalence["identical"], "/", equivalence["n"])


if __name__ == "__main__":
    main()
