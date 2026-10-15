"""Smoke test of a running FES service (run inside its container: docker exec <svc> python scripts/perf/smoke.py).

    --expect-parallel 1 --expect-route nafnet     settings /ready must report
    --mode normal    an enhanced face, a gate-rejected photo, a strongly noisy photo (ORIGINAL when SwinIR-L is off)
    --mode failure   for a service started with a failing r6_ft test double: the failure must stay with that candidate
                     and the other candidate's output (or the original) must be returned
Prints one JSON line per check and exits non-zero when any check fails.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "Dataset-and-code" / "archive" / "input" / "test"
REAL = ROOT / "Dataset-and-code" / "dataset"
FAILS = []


def check(name, ok, **info):
    print(json.dumps({"check": name, "ok": bool(ok), **info}), flush=True)
    if not ok:
        FAILS.append(name)


def post(c, url, path, data, mode):
    t0 = time.perf_counter()
    r = c.post(url + path, files={"image": ("x", data, "application/octet-stream")}, data={"input_mode": mode})
    return r.status_code, r.json(), round((time.perf_counter() - t0) * 1e3)


def returned_original(j, data):
    return j["status"] == "ORIGINAL" and j["image_format"] == "as_received" and base64.b64decode(j["image_b64"]) == data


def main(a):
    c = httpx.Client(timeout=300)
    rd = c.get(a.url + "/ready")
    d = rd.json()
    check("ready", rd.status_code == 200 and d["ready"] and not d["error"], http=rd.status_code, error=d["error"])
    check("settings", d["best_parallel"] is bool(a.expect_parallel) and d["r6_mirror_batch"] is False
          and d["route_mode"] == a.expect_route and d["nafnet_backend"] == "torch" and "best" in d["profiles"],
          best_parallel=d["best_parallel"], r6_mirror_batch=d["r6_mirror_batch"], route_mode=d["route_mode"],
          nafnet_backend=d["nafnet_backend"], profiles=d["profiles"], default_profile=d["default_profile"])
    check("swinir_not_loaded", (not d["model"] and not d["worker_alive"]) == (a.expect_route == "nafnet"),
          a4_model_loaded=bool(d["model"]), a4_worker_alive=d["worker_alive"])
    check("workers", d["nafnet_worker_alive"] and all(m.get("weights_sha256") or m.get("model", "").startswith("FAKE")
                                                      for m in d["experimental_models"].values()),
          models={n: [m.get("model"), (m.get("weights_sha256") or "")[:12]] for n, m in d["experimental_models"].items()})

    names = sorted(os.listdir(ARCH))
    blur = (ARCH / names[::max(1, len(names) // 100)][2]).read_bytes()          # an archive TEST face both models accept
    if a.mode == "normal":
        http, j, ms = post(c, a.url, "/v3/enhance", blur, "face_chip")
        s = j.get("selection") or {}
        out = base64.b64decode(j["image_b64"]) if j.get("image_b64") else b""
        check("enhanced", http == 200 and j["status"] == "ENHANCED" and j["used"] == "NAFNET_GOPRO" and j["image_format"] == "png"
              and out[:8] == b"\x89PNG\r\n\x1a\n" and out != blur and s.get("chosen") in ("r6", "r6_ft")
              and set(s["candidates"]) == {"r6", "r6_ft"} and j["nafnet"]["stability"]["passed"] and j["artifact_check"] == "PASS",
              status=j["status"], chosen=s.get("chosen"), candidates={n: [v["status"], v["fallback"], v["score"]] for n, v in s.get("candidates", {}).items()},
              model=j["model"], ms=ms, output_sha=hashlib.sha256(out).hexdigest()[:12])
        http2, j2, _ = post(c, a.url, "/v3/enhance", blur, "face_chip")
        check("enhanced_repeatable", j2.get("image_b64") == j.get("image_b64") and (j2.get("selection") or {}).get("chosen") == s.get("chosen"))
        # each candidate alone must give what the single-output request reports for it
        for prof, path in (("r6", "/v1/enhance"), ("r6_ft", "/v2/enhance")):
            r = c.post(a.url + path, files={"image": ("x", blur, "application/octet-stream")},
                       data={"input_mode": "face_chip", **({"profile": "r6"} if prof == "r6" else {})}).json()
            same = r["status"] == s["candidates"][prof]["status"] and (prof != s.get("chosen") or r["image_b64"] == j["image_b64"])
            check(f"candidate_{prof}_matches", same, status=r["status"], fallback=r["fallback"])
        sharp = (REAL / "img1.jpg").read_bytes()
        http, j, ms = post(c, a.url, "/v3/enhance", sharp, "full_image")
        check("gate_rejected", http == 200 and returned_original(j, sharp) and not (j.get("nafnet") or {}).get("eligible", False)
              and (j.get("nafnet") or {}).get("infer_s") is None and not j["fallback"],
              status=j["status"], reason=j["reason"], nafnet_reason=(j.get("nafnet") or {}).get("reason"), ms=ms)
        noisy = (REAL / "img72.jpg").read_bytes()
        http, j, ms = post(c, a.url, "/v3/enhance", noisy, "full_image")
        if a.expect_route == "nafnet":
            check("strong_noise_not_sent_to_swinir", http == 200 and returned_original(j, noisy) and j["used"] != "A4",
                  status=j["status"], used=j["used"], nafnet_reason=(j.get("nafnet") or {}).get("reason"), ms=ms)
        bad = c.post(a.url + "/v3/enhance", files={"image": ("x", b"not an image", "application/octet-stream")})
        check("invalid_input_rejected", bad.status_code == 422 and bad.json()["status"] == "REJECTED", http=bad.status_code, error=bad.json().get("error"))
    else:
        http, j, ms = post(c, a.url, "/v3/enhance", blur, "face_chip")
        s = j.get("selection") or {}
        cand = s.get("candidates", {})
        check("failure_stays_with_its_candidate", http == 200 and cand.get("r6_ft", {}).get("fallback") == a.expect_fallback
              and cand.get("r6", {}).get("fallback") == "" and s.get("chosen") == "r6" and j["status"] == "ENHANCED",
              status=j["status"], chosen=s.get("chosen"), candidates={n: [v["status"], v["fallback"]] for n, v in cand.items()}, ms=ms)
        r = c.post(a.url + "/v2/enhance", files={"image": ("x", blur, "application/octet-stream")}, data={"input_mode": "face_chip"}).json()
        check("failed_model_alone_returns_original", returned_original(r, blur) and r["fallback"] == a.expect_fallback,
              status=r["status"], fallback=r["fallback"], error=r["error"])
        http, j3, ms = post(c, a.url, "/v3/enhance", blur, "face_chip")
        check("still_serving_after_failure", http == 200 and j3["status"] == "ENHANCED" and (j3.get("selection") or {}).get("chosen") == "r6", ms=ms)
        d2 = c.get(a.url + "/ready").json()
        check("service_still_ready", d2["ready"] and d2["nafnet_worker_alive"])
    print(json.dumps({"failed": FAILS}))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--mode", choices=["normal", "failure"], default="normal")
    ap.add_argument("--expect-parallel", type=int, default=1)
    ap.add_argument("--expect-route", default="nafnet")
    ap.add_argument("--expect-fallback", default="MODEL_FAILURE")
    main(ap.parse_args())
