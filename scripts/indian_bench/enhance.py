"""Phase 5 — send every degraded variant to the running Face Enhancement Service (client only; resumable).

    python scripts/indian_bench/enhance.py --pilot           the pilot samples
    python scripts/indian_bench/enhance.py                   every generated variant not yet answered
    python scripts/indian_bench/enhance.py --limit 5000      one batch;  --max-hours 6  stops cleanly after that time

The request is the service's documented one: POST <config.service.endpoint>, multipart `image` = the DEGRADED file's
bytes exactly as stored, `input_mode` = full_image. One request at a time (the service serialises inference). The
service is used as it runs: nothing is configured, restarted or changed from here.

What is stored
  enhanced/<Expression>/<sample_id>/<type>/level_<n>.<ext>   the image bytes the service returned, untouched:
        ENHANCED → the service's PNG (the enhanced FACE CHIP, `output` = nafnet_face_chip_native);
        ORIGINAL → the input returned unchanged (`image_format` = as_received). The bytes are verified to equal the
                   degraded file and stored as a hard link to it (same content, no second copy on disk).
  manifests/enhancement_results.jsonl (one line per attempt) → enhancement_results.csv (latest attempt per variant) and
  failures.csv. `outcome` is derived from the service's answer only:
        enhanced | original_no_face | original_gate | fallback_guard | fallback_model_failure | rejected | failed
  A variant whose latest outcome is `failed` (timeout, connection error, 503) is retried on the next run; everything
  else is final and never requested again.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import common as C

FINAL = ("enhanced", "original_no_face", "original_gate", "fallback_guard", "fallback_model_failure", "rejected")
GUARDS = ("STABILITY_GUARD", "ARTIFACT_GUARD", "LOCAL_ARTIFACT_GUARD")


def post(url: str, data: bytes, fields: dict, timeout: float) -> tuple[int, dict, float]:
    """multipart/form-data POST with the standard library → (HTTP status, JSON body, round trip ms). 0 = no answer."""
    b = uuid.uuid4().hex
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
    body += (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="image"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read()
    except Exception as e:  # noqa: BLE001 — recorded, never hidden
        return 0, {"status": "CLIENT_ERROR", "error": f"{type(e).__name__}: {e}"[:300]}, round((time.perf_counter() - t0) * 1e3, 1)
    rtt = round((time.perf_counter() - t0) * 1e3, 1)
    try:
        return code, json.loads(raw), rtt
    except Exception as e:  # noqa: BLE001
        return code, {"status": "BAD_RESPONSE", "error": f"{type(e).__name__}: {raw[:120]!r}"}, rtt


def get_json(url: str, timeout: float = 20) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def outcome(http: int, j: dict) -> str:
    s = j.get("status")
    if http == 200 and s == "ENHANCED":
        return "enhanced"
    if http == 200 and s == "ORIGINAL":
        fb, reason = j.get("fallback") or "", j.get("reason") or ""
        if fb in GUARDS:
            return "fallback_guard"
        if fb:
            return "fallback_model_failure"
        if reason.startswith(("no_valid_face", "alignment_unreliable")):
            return "original_no_face"
        return "original_gate"
    if http == 422 and s == "REJECTED":
        return "rejected"
    return "failed"


def docker_stats(container: str) -> dict:
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}|{{.MemUsage}}", container],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        cpu, mem = out.split("|")
        used = mem.split("/")[0].strip()
        mib = float(used[:-3]) * (1024 if used.endswith("GiB") else 1)
        return {"cpu_pct": float(cpu.rstrip("%")), "mem_mib": round(mib, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:120]}


def snapshot(cfg: dict, container: str) -> dict:
    """What exactly is being benchmarked: /health, /ready, the container's image + environment, artifact hashes."""
    url = cfg["service"]["url"]
    snap = {"taken": time.strftime("%Y-%m-%d %H:%M:%S"), "service": cfg["service"],
            "health": get_json(url + "/health"), "ready": get_json(url + "/ready")}
    try:
        insp = json.loads(subprocess.run(["docker", "inspect", container], capture_output=True, text=True, timeout=30).stdout)[0]
        snap["container"] = {"name": container, "image": insp["Config"]["Image"], "image_id": insp["Image"],
                             "started_at": insp["State"]["StartedAt"], "cmd": insp["Config"]["Cmd"],
                             "env": [e for e in insp["Config"]["Env"] if e.startswith(("FES_", "OMP_", "MKL_", "PYTHONPATH"))]}
        snap["docker"] = subprocess.run(["docker", "info", "--format", "{{.NCPU}} CPU, {{.MemTotal}} bytes"],
                                        capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        snap["container"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    snap["git_head"] = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    snap["git_dirty_tracked_files"] = [l[3:] for l in subprocess.run(
        ["git", "-C", str(C.ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.splitlines()
        if not l.startswith("??")]
    files = {}
    for info in [snap["ready"].get("nafnet_model") or {}, *(snap["ready"].get("experimental_models") or {}).values()]:
        w = info.get("weights")
        if w and w not in files and (C.ROOT / w).exists():
            files[w] = {"sha256_on_disk": C.sha256_file(C.ROOT / w), "sha256_reported": info.get("weights_sha256")}
    snap["artifacts"] = files
    return snap


def flatten(j: dict) -> dict:
    nf = j.get("nafnet") or {}
    sel = j.get("selection") or {}
    r = {k: j.get(k) for k in ("status", "decision", "reason", "bucket", "route", "used", "model", "fallback", "error",
                               "escalate", "n_faces", "chip_px", "artifact_check", "output", "image_format", "profile",
                               "route_mode", "service_version")}
    r.update(warnings="+".join(j.get("warnings") or []), nafnet_eligible=nf.get("eligible"),
             nafnet_reason=nf.get("reason"), nafnet_infer_s=nf.get("infer_s"), backend=nf.get("backend"),
             stability_passed=(nf.get("stability") or {}).get("passed"),
             artifact_reasons="+".join((nf.get("artifact") or {}).get("reasons") or []),
             selection_chosen=sel.get("chosen"),
             selection_candidates=json.dumps(sel.get("candidates"), separators=(",", ":")) if sel else "",
             service_total_ms=(j.get("timings_ms") or {}).get("total"),
             quality=json.dumps(j.get("quality") or {}, separators=(",", ":")))
    return r


def finalise(cfg: dict) -> list[dict]:
    latest: dict[str, dict] = {}
    attempts: dict[str, int] = {}
    for r in C.read_jsonl(C.bench(cfg, "manifests", "enhancement_results.jsonl")):
        attempts[r["variant_id"]] = attempts.get(r["variant_id"], 0) + 1
        latest[r["variant_id"]] = r
    rows = sorted(latest.values(), key=lambda r: r["variant_id"])
    for r in rows:
        r["attempts"] = attempts[r["variant_id"]]
    C.write_csv(C.bench(cfg, "manifests", "enhancement_results.csv"), rows)
    fails = [{"stage": "generate", "variant_id": "", "sample_id": r["sample_id"], "outcome": "generation_failed",
              "http_status": "", "error": r["error"]}
             for r in C.read_jsonl(C.bench(cfg, "manifests", "generation_failures.jsonl"))]
    fails += [{"stage": "enhance", "variant_id": r["variant_id"], "sample_id": r["sample_id"], "outcome": r["outcome"],
               "http_status": r["http_status"], "error": r.get("error") or r.get("client_error") or ""}
              for r in rows if r["outcome"] in ("failed", "rejected", "fallback_model_failure")]
    C.write_csv(C.bench(cfg, "manifests", "failures.csv"), fails,
                ["stage", "variant_id", "sample_id", "outcome", "http_status", "error"])
    return rows


def main(a):
    import cv2
    import numpy as np
    cfg = C.load_config(a.config)
    svc = cfg["service"]
    root = Path(cfg["bench_root"])
    snap = snapshot(cfg, a.container)
    if not snap["ready"].get("ready"):
        raise SystemExit(f"service not ready: {snap['ready'].get('error')!r}")
    for w, h in snap["artifacts"].items():
        if h["sha256_on_disk"] != h["sha256_reported"]:
            raise SystemExit(f"artifact hash mismatch for {w}")
    snaps = C.bench(cfg, "manifests", "service_snapshots.jsonl")
    with open(snaps, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(snap) + "\n")
    C.bench(cfg, "manifests", "service_snapshot.json").write_text(json.dumps(snap, indent=1), encoding="utf-8")

    variants = C.read_csv(C.bench(cfg, "manifests", "generated_variants.csv"))
    ids = {r["sample_id"] for r in C.scope(cfg, a.pilot)}
    variants = [v for v in variants if v["sample_id"] in ids]
    res_path = C.bench(cfg, "manifests", "enhancement_results.jsonl")
    done = {r["variant_id"] for r in C.read_jsonl(res_path) if r["outcome"] in FINAL}
    todo = [v for v in variants if v["variant_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{svc['url']}{svc['endpoint']} · variants in scope {len(variants)} · already answered "
          f"{len(variants) - len([v for v in variants if v['variant_id'] not in done])} · to send now {len(todo)}", flush=True)
    url = svc["url"] + svc["endpoint"]
    t_start, streak, counts = time.time(), 0, {}
    res_log = open(C.bench(cfg, "logs", "resources.jsonl"), "a", encoding="utf-8")
    with open(res_path, "a", encoding="utf-8") as fh:
        for i, v in enumerate(todo, 1):
            src = root / v["degraded_path"]
            data = src.read_bytes()
            rec = {"variant_id": v["variant_id"], "sample_id": v["sample_id"], "expression": v["expression"],
                   "type": v["type"], "level": v["level"], "degraded_path": v["degraded_path"],
                   "endpoint": svc["endpoint"], "sent": time.strftime("%Y-%m-%d %H:%M:%S")}
            if C.sha256_bytes(data) != v["file_sha256"]:
                rec.update(outcome="failed", http_status=0, client_error="degraded file differs from its manifest hash")
            else:
                for attempt in range(svc["retries_busy"] + 1):
                    http, j, rtt = post(url, data, {"input_mode": svc["input_mode"], "return_image": "true"}, svc["timeout_s"])
                    if http != 503:
                        break
                    time.sleep(min(30, 2 * (attempt + 1)))          # BUSY / NOT_READY: wait and ask again
                oc = outcome(http, j)
                rec.update(outcome=oc, http_status=http, rtt_ms=rtt, **flatten(j))
                img = base64.b64decode(j["image_b64"]) if j.get("image_b64") else None
                chip = base64.b64decode(j["chip_b64"]) if j.get("chip_b64") else None
                if chip:        # the degraded face chip the service analysed: its pixel hash proves the comparison crop
                    c = cv2.imdecode(np.frombuffer(chip, np.uint8), cv2.IMREAD_COLOR)
                    rec["service_chip_pixel_sha256"] = C.sha256_bytes(c.tobytes())
                if oc == "enhanced" or oc.startswith(("original", "fallback")):
                    if img is None:
                        rec.update(outcome="failed", client_error="no image in a 200 answer")
                    else:
                        unchanged = img == data
                        ext = "png" if j.get("image_format") == "png" else src.suffix.lstrip(".")
                        out = root / "enhanced" / Path(v["degraded_path"]).relative_to("degraded").with_suffix("." + ext)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if out.exists():
                            out.unlink()
                        if unchanged:
                            os.link(src, out)
                        else:
                            out.write_bytes(img)
                        if oc != "enhanced" and not unchanged:
                            rec["client_note"] = "ORIGINAL answer whose bytes differ from the input (stored as returned)"
                        rec.update(enhanced_path=out.relative_to(root).as_posix(), output_bytes=len(img),
                                   output_sha256=C.sha256_bytes(img), output_equals_input=unchanged)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
            streak = streak + 1 if rec["outcome"] == "failed" else 0
            if streak >= a.max_consecutive_failures:
                print(f"!! {streak} consecutive failures — stopping (service unhealthy?). Last: "
                      f"{rec.get('error') or rec.get('client_error')}", flush=True)
                break
            if i % a.stats_every == 0 or i == len(todo):
                st = docker_stats(a.container)
                el = time.time() - t_start
                res_log.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "sent": i, "elapsed_s": round(el, 1), **st}) + "\n")
                res_log.flush()
                print(f"  {i}/{len(todo)} · {el / 60:.1f} min · {el / i:.2f} s/request · ETA {(len(todo) - i) * el / i / 3600:.1f} h"
                      f" · {counts} · {st}", flush=True)
            if a.max_hours and time.time() - t_start > a.max_hours * 3600:
                print("time budget reached — stopping cleanly (re-run to resume)", flush=True)
                break
    rows = finalise(cfg)
    end = get_json(svc["url"] + "/ready")
    print(f"done · {len(rows)} variants answered in total · this run {counts} · worker restarts at end: "
          f"default {end.get('nafnet_worker_restarts')} · ready {end.get('ready')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--container", default="fes-ui")
    ap.add_argument("--stats-every", type=int, default=250)
    ap.add_argument("--max-consecutive-failures", type=int, default=10)
    main(ap.parse_args())
