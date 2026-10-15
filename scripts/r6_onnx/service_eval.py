"""R6-ONNX-INTEGRATION: paired full-service validation, PyTorch backend vs ONNX backend (evaluation only).

Every image is POSTed to a running `uvicorn fes_service.app:app` (the service decides everything; this script only
records, pairs and scores). The same images, requests and order are used for both backends; the two runs are sequential
(one service at a time, started from the same image with one setting changed: FES_NAFNET_BACKEND).

    run      --backend torch|onnx --url http://host:8080 [--sets real200,archive100]
             → experiments/r6_onnx/outputs/service/<backend>/{ready.json, records.jsonl, out/<request>/<id>.png, chip/}
             Refuses to run when /ready does not report the expected backend (never records a substituted backend).
    compare  → experiments/r6_onnx/service/{paired_real200.csv, paired_archive100.csv, summary.json} + review sheets
             in experiments/r6_onnx/outputs/service/sheets/

Requests per image (service defaults otherwise; `route_mode` is not sent):
    r6       POST /v1/enhance profile=r6          NAFNet-GoPro, 4-run consensus
    r6_ft    POST /v2/enhance                      face fine-tune, 4-run consensus
    best     POST /v3/enhance                      both, sharper accepted output (the single-output page)
    default  POST /v1/enhance (no profile)         the live default route (PyTorch in BOTH runs: must not change)
Sets
    real200     the 200 CC0 photos in Dataset-and-code/dataset, full_image (detection → alignment → gate → model →
                guards → response). No sharp original exists: identity is measured as SFace similarity of each output
                to its own input chip and of the ONNX output to the PyTorch output. Not a Feature-Match result.
    archive100  Dataset-and-code/archive TEST, every 192nd pair, face_chip (heavy shake blur with a sharp original;
                licence unknown, evaluation only): adds blurred faces that reach the models, for latency and for
                SFace-to-sharp. Records carry an index, not the source file name.
Parity criterion (fixed before the runs, from the model-only study in REPORT.md §4: final outputs within 1 grey level
on 560 / 560 faces): a pair is EQUIVALENT when status, route, model, fallback and reasons are equal and, when an image
is returned, shapes are equal and no pixel differs by more than 2 grey levels. Every other pair is listed and reviewed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

REAL = ROOT / "Dataset-and-code" / "dataset"
ARCHIVE = ROOT / "Dataset-and-code" / "archive"
OUT = ROOT / "experiments" / "r6_onnx" / "outputs" / "service"
REP = ROOT / "experiments" / "r6_onnx" / "service"
REQUESTS = ("r6", "r6_ft", "best", "default")
BACKENDS = ("torch", "onnx")
RISKY = {"img118": "R5-10: grainy vintage print, artifact returned and missed by both guards",
         "img123": "R5-10: newly admitted moderate-blur real photo",
         "img165": "R6-0: printed / halftone stress case (all runs unstable when forced)",
         "img72": "R6-0: printed / halftone stress case (all runs unstable when forced)"}
MAX_DIFF = 2


def natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def items(sets):
    """→ (set, id, input_mode, path of the input, path of the sharp original or None)"""
    out = []
    if "real200" in sets:
        for n in sorted(os.listdir(REAL), key=natural):
            out.append(("real200", os.path.splitext(n)[0], "full_image", REAL / n, None))
    if "archive100" in sets:
        names = sorted(os.listdir(ARCHIVE / "input" / "test"))
        for i, n in enumerate(names[::max(1, len(names) // 100)][:100]):
            out.append(("archive100", f"a{i:03d}", "face_chip", ARCHIVE / "input" / "test" / n,
                        ARCHIVE / "output" / "test" / n))
    return out


def manifest(sets):
    """SHA-256 over the ordered (set, id, SHA-256 of the file) list: identifies the exact inputs of a run."""
    h, n = hashlib.sha256(), {}
    for s, i, _, p, _ in items(sets):
        h.update(f"{s}/{i}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
        n[s] = n.get(s, 0) + 1
    return {"sha256": h.hexdigest(), "counts": n}


def post(client, url, kind, data, mode):
    files = {"image": ("x", data, "application/octet-stream")}
    form = {"input_mode": mode}
    path = {"r6": "/v1/enhance", "default": "/v1/enhance", "r6_ft": "/v2/enhance", "best": "/v3/enhance"}[kind]
    if kind == "r6":
        form["profile"] = "r6"
    t0 = time.perf_counter()
    try:
        r = client.post(url + path, files=files, data=form)
        return r.status_code, r.json(), round((time.perf_counter() - t0) * 1e3, 1)
    except Exception as e:  # noqa: BLE001 — recorded, never hidden
        return 0, {"status": "CLIENT_ERROR", "error": f"{type(e).__name__}: {e}"[:200]}, round((time.perf_counter() - t0) * 1e3, 1)


def flatten(j):
    nf = j.get("nafnet") or {}
    r = {k: j.get(k) for k in ("status", "route_mode", "decision", "reason", "bucket", "route", "used", "model",
                               "fallback", "error", "escalate", "n_faces", "chip_px", "artifact_check", "gate_route",
                               "output", "image_format", "profile")}
    r["warnings"] = "+".join(j.get("warnings") or [])
    r.update(nafnet_eligible=nf.get("eligible"), nafnet_reason=nf.get("reason"), infer_s=nf.get("infer_s"),
             backend=nf.get("backend"), nafnet_profile=nf.get("profile"),
             stability=(nf.get("stability") or {}).get("metrics", {}).get("runs_dot_c"),
             stability_passed=(nf.get("stability") or {}).get("passed"),
             artifact_reasons="+".join((nf.get("artifact") or {}).get("reasons") or []),
             artifact_metrics=(nf.get("artifact") or {}).get("metrics"),
             quality=j.get("quality"), timings_ms=j.get("timings_ms"), selection=j.get("selection"))
    return r


def run(a):
    import httpx
    sets = set(a.sets.split(","))
    base = OUT / a.backend
    client = httpx.Client(timeout=600)
    ready = client.get(a.url + "/ready").json()
    if not ready.get("ready") or ready.get("nafnet_backend") != a.backend:
        raise SystemExit(f"service is not the expected backend: ready={ready.get('ready')} "
                         f"nafnet_backend={ready.get('nafnet_backend')!r} error={ready.get('error')!r}")
    exp = ready["experimental_models"]
    want = "onnxruntime" if a.backend == "onnx" else None
    assert all(exp[p].get("backend") == want for p in ("r6", "r6_ft")), exp
    for kind in REQUESTS:
        (base / "out" / kind).mkdir(parents=True, exist_ok=True)
    (base / "chip").mkdir(exist_ok=True)
    todo = items(sets)
    (base / "ready.json").write_text(json.dumps({"ready": ready, "manifest": manifest(sets), "sets": sorted(sets),
                                                 "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    # warm-up: every request kind once on a blurred face (discarded), so no measured request pays first-call costs
    warm = next((x for x in todo if x[0] == "archive100"), todo[0])
    for kind in REQUESTS:
        post(client, a.url, kind, warm[3].read_bytes(), warm[2])
    with (base / "records.jsonl").open("w", encoding="utf-8") as fh:
        for k, (set_, iid, mode, path, _) in enumerate(todo):
            data = path.read_bytes()
            for kind in REQUESTS:
                http, j, rtt = post(client, a.url, kind, data, mode)
                rec = {"set": set_, "id": iid, "request": kind, "http": http, "rtt_ms": rtt, **flatten(j)}
                if j.get("status") == "ENHANCED" and j.get("image_b64"):
                    (base / "out" / kind / f"{iid}.png").write_bytes(base64.b64decode(j["image_b64"]))
                if kind == "r6" and j.get("chip_b64"):
                    (base / "chip" / f"{iid}.png").write_bytes(base64.b64decode(j["chip_b64"]))
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (k + 1) % 25 == 0:
                print(f"{a.backend}: {k + 1}/{len(todo)}", flush=True)
    end = client.get(a.url + "/ready").json()
    (base / "ready_end.json").write_text(json.dumps({"ready": end, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print("done", a.backend, "restarts", end.get("worker_restarts"), end.get("nafnet_worker_restarts"), flush=True)


# ================================================================== compare

def load(backend):
    recs = {}
    for line in (OUT / backend / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[(r["set"], r["id"], r["request"])] = r
    return recs


def img_of(backend, kind, iid):
    p = OUT / backend / "out" / kind / f"{iid}.png"
    return cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def lat(v):
    v = [x for x in v if x is not None]
    return {"n": len(v), "p50": round(st.median(v), 1), "p95": round(pct(v, 0.95), 1), "max": round(max(v), 1)} if v else {"n": 0}


DECISION_KEYS = ("http", "status", "route", "used", "model", "fallback", "decision", "reason", "bucket", "gate_route",
                 "nafnet_eligible", "nafnet_reason", "artifact_check", "artifact_reasons", "stability_passed",
                 "chip_px", "n_faces", "warnings", "output", "image_format")


def compare(a):
    from fes_bench.metrics import YUNET, SFace
    from fes_data.common import repo_path
    from fes_data.preprocess import YuNetDetector
    det = YuNetDetector(repo_path(YUNET), score_threshold=0.7, nms_threshold=0.3, max_side=640)
    sf = SFace()
    T, O = load("torch"), load("onnx")
    meta = {b: json.loads((OUT / b / "ready.json").read_text()) for b in BACKENDS}
    assert meta["torch"]["manifest"] == meta["onnx"]["manifest"], "the two runs did not use the same inputs"
    assert set(T) == set(O), "unpaired records"
    sharp_of = {(s, i): p for s, i, _, _, p in items(set(meta["torch"]["sets"]))}
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    def embed(img, lm):
        return sf.embed(img, lm)

    def landmarks(img):
        f = det.detect(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        return np.asarray(f[0].landmarks, np.float32) / 2.0 if f else None

    pairs = []
    for key in sorted(T, key=lambda k: (k[0], natural(k[1]), REQUESTS.index(k[2]))):
        set_, iid, kind = key
        t, o = T[key], O[key]
        p = {"set": set_, "id": iid, "request": kind}
        for k in DECISION_KEYS:
            p[f"torch_{k}"], p[f"onnx_{k}"] = t.get(k), o.get(k)
        p["decision_equal"] = all(t.get(k) == o.get(k) for k in DECISION_KEYS)
        p["torch_chosen"] = (t.get("selection") or {}).get("chosen")
        p["onnx_chosen"] = (o.get("selection") or {}).get("chosen")
        p["chosen_equal"] = p["torch_chosen"] == p["onnx_chosen"]
        p["torch_backend"], p["onnx_backend"] = t.get("backend"), o.get("backend")
        for b, r in (("torch", t), ("onnx", o)):
            p[f"{b}_infer_s"], p[f"{b}_rtt_ms"] = r.get("infer_s"), r.get("rtt_ms")
            p[f"{b}_server_ms"] = (r.get("timings_ms") or {}).get("total")
            p[f"{b}_error"] = r.get("error")
            sd = r.get("stability")
            p[f"{b}_run_dot_c_max"] = max(sd) if sd else None
        a_img, b_img = img_of("torch", kind, iid), img_of("onnx", kind, iid)
        p["image_both"] = a_img is not None and b_img is not None
        p["image_one_side_only"] = (a_img is None) != (b_img is None)
        if p["image_both"]:
            p["shape_equal"] = a_img.shape == b_img.shape
            if p["shape_equal"]:
                d = np.abs(a_img.astype(np.int16) - b_img.astype(np.int16))
                p.update(px_max=int(d.max()), px_mean=round(float(d.mean()), 5), px_gt2_frac=round(float((d > 2).mean()), 6),
                         px_identical=bool(d.max() == 0))
            chip = cv2.imread(str(OUT / "torch" / "chip" / f"{iid}.png"), cv2.IMREAD_COLOR)
            sharp = cv2.imread(str(sharp_of[(set_, iid)])) if sharp_of.get((set_, iid)) else None
            ref = sharp if sharp is not None and sharp.shape == a_img.shape else chip
            if ref is not None and ref.shape == a_img.shape == b_img.shape:
                lm = landmarks(ref)
                if lm is not None:
                    e_ref, e_a, e_b = embed(ref, lm), embed(a_img, lm), embed(b_img, lm)
                    p["sface_ref"] = "sharp_original" if ref is sharp else "input_chip"
                    p["torch_sface_to_ref"] = round(float(e_ref @ e_a), 4)
                    p["onnx_sface_to_ref"] = round(float(e_ref @ e_b), 4)
                    p["sface_torch_vs_onnx"] = round(float(e_a @ e_b), 4)
                    if chip is not None and ref is sharp and chip.shape == ref.shape:
                        p["sface_input_to_ref"] = round(float(e_ref @ embed(chip, lm)), 4)
        p["equivalent"] = bool(p["decision_equal"] and p["chosen_equal"] and not p["image_one_side_only"]
                               and (not p["image_both"] or (p.get("shape_equal") and p.get("px_max", 99) <= MAX_DIFF)))
        pairs.append(p)
        show = not p["equivalent"] or (p["image_both"] and kind == "best") or (iid in RISKY and kind == "best")
        if show:
            chip = cv2.imread(str(OUT / "torch" / "chip" / f"{iid}.png"), cv2.IMREAD_COLOR)
            if chip is not None:
                tiles = []
                for img, lab in ((chip, "INPUT"), (a_img, f"TORCH {t.get('status')} {p['torch_chosen'] or ''}"),
                                 (b_img, f"ONNX {o.get('status')} {p['onnx_chosen'] or ''}")):
                    x = cv2.resize(img if img is not None else np.zeros_like(chip), (336, 336), interpolation=cv2.INTER_CUBIC)
                    cv2.putText(x, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(x)
                if p["image_both"] and p.get("shape_equal"):
                    d = np.clip(np.abs(a_img.astype(np.int16) - b_img.astype(np.int16)) * 10, 0, 255).astype(np.uint8)
                    x = cv2.resize(d, (336, 336), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(x, f"DIFF x10 (max {p['px_max']})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(x)
                cv2.imwrite(str(OUT / "sheets" / f"{set_}_{iid}_{kind}.png"), np.hstack(tiles))

    cols = sorted({k for p in pairs for k in p}, key=lambda k: (k not in ("set", "id", "request", "equivalent"), k))
    for set_ in sorted({p["set"] for p in pairs}):
        lines = [",".join(cols)]
        for p in (x for x in pairs if x["set"] == set_):
            lines.append(",".join('"' + str(p.get(c, "")).replace('"', "'") + '"' if isinstance(p.get(c), str) else str(p.get(c, "")) for c in cols))
        (REP / f"paired_{set_}.csv").write_text("\n".join(lines), encoding="utf-8")

    summary = {"parity_criterion": f"decision metadata equal and, when an image is returned, max pixel difference <= {MAX_DIFF}",
               "manifest": meta["torch"]["manifest"], "runs": {}, "sets": {}}
    for b in BACKENDS:
        rd, end = meta[b]["ready"], json.loads((OUT / b / "ready_end.json").read_text())["ready"]
        summary["runs"][b] = {
            "started": meta[b]["started"], "service_version": rd["service_version"], "nafnet_backend": rd["nafnet_backend"],
            "route_mode": rd["route_mode"], "default_profile": rd["default_profile"], "profiles": rd["profiles"], "seals": rd["seals"],
            "models": {n: {k: m.get(k) for k in ("model", "backend", "backend_version", "provider", "weights", "weights_sha256",
                                                 "source_weights_sha256", "variant", "load_s", "threads")}
                       for n, m in {**rd["experimental_models"], "default_nafnet": rd["nafnet_model"], "a4": rd["model"]}.items()},
            "worker_restarts_end": {"a4": end["worker_restarts"], "default_nafnet": end["nafnet_worker_restarts"]}}
    for set_ in sorted({p["set"] for p in pairs}):
        S = summary["sets"][set_] = {}
        for kind in REQUESTS:
            ps = [p for p in pairs if p["set"] == set_ and p["request"] == kind]
            both = [p for p in ps if p["image_both"]]
            d = {"n": len(ps), "equivalent": sum(p["equivalent"] for p in ps),
                 "decision_differs": [p["id"] for p in ps if not p["decision_equal"]],
                 "chosen_differs": [p["id"] for p in ps if not p["chosen_equal"]],
                 "image_one_side_only": [p["id"] for p in ps if p["image_one_side_only"]],
                 "pixel_over_tolerance": [p["id"] for p in both if p.get("px_max", 0) > MAX_DIFF],
                 "images_compared": len(both), "px_identical": sum(bool(p.get("px_identical")) for p in both),
                 "px_max": max((p.get("px_max", 0) for p in both), default=None),
                 "http_errors": {b: sum(p[f"{b}_http"] != 200 for p in ps) for b in BACKENDS}}
            for b in BACKENDS:
                stat, fb = {}, {}
                for p in ps:
                    stat[p[f"{b}_status"]] = stat.get(p[f"{b}_status"], 0) + 1
                    if p[f"{b}_fallback"]:
                        fb[p[f"{b}_fallback"]] = fb.get(p[f"{b}_fallback"], 0) + 1
                routes = {}
                for p in ps:
                    routes[str(p[f"{b}_used"])] = routes.get(str(p[f"{b}_used"]), 0) + 1
                d[b] = {"status": stat, "fallbacks": fb, "used": routes,
                        "backend_reported": sorted({str(p[f"{b}_backend"]) for p in ps if p[f"{b}_infer_s"] is not None}),
                        "http_ms": lat([p[f"{b}_rtt_ms"] for p in ps]),
                        "http_ms_model_ran": lat([p[f"{b}_rtt_ms"] for p in ps if p[f"{b}_infer_s"] is not None and p[f"{b}_used"] != "A4"]),
                        "server_ms_model_ran": lat([p[f"{b}_server_ms"] for p in ps if p[f"{b}_infer_s"] is not None and p[f"{b}_used"] != "A4"]),
                        "model_ms": lat([p[f"{b}_infer_s"] * 1e3 for p in ps if p[f"{b}_infer_s"] is not None and p[f"{b}_used"] != "A4"]
                                        if kind in ("r6", "r6_ft") else [])}
            sf_ps = [p for p in both if "sface_torch_vs_onnx" in p]
            if sf_ps:
                d["sface"] = {"n": len(sf_ps), "ref": sorted({p["sface_ref"] for p in sf_ps}),
                              "torch_to_ref_mean": round(st.mean(p["torch_sface_to_ref"] for p in sf_ps), 4),
                              "onnx_to_ref_mean": round(st.mean(p["onnx_sface_to_ref"] for p in sf_ps), 4),
                              "max_abs_change": round(max(abs(p["torch_sface_to_ref"] - p["onnx_sface_to_ref"]) for p in sf_ps), 4),
                              "torch_vs_onnx_min": round(min(p["sface_torch_vs_onnx"] for p in sf_ps), 4)}
                inp = [p for p in sf_ps if "sface_input_to_ref" in p]
                if inp:
                    d["sface"]["input_to_ref_mean"] = round(st.mean(p["sface_input_to_ref"] for p in inp), 4)
            S[kind] = d
    summary["risky"] = {iid: {"why": why, **{kind: {k: next((p.get(k) for p in pairs if p["set"] == "real200" and p["id"] == iid and p["request"] == kind), None)
                                                   for k in ("torch_status", "onnx_status", "torch_nafnet_reason", "onnx_nafnet_reason",
                                                             "torch_fallback", "onnx_fallback", "torch_used", "onnx_used", "px_max", "equivalent")}
                                            for kind in REQUESTS}} for iid, why in RISKY.items()}
    (REP / "summary.json").write_text(json.dumps(summary, indent=1))
    for set_, S in summary["sets"].items():
        for kind, d in S.items():
            print(set_, kind, "equivalent", d["equivalent"], "/", d["n"], "images", d["images_compared"], "px_max", d["px_max"],
                  "torch", d["torch"]["status"], "onnx", d["onnx"]["status"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare"])
    ap.add_argument("--backend", choices=BACKENDS)
    ap.add_argument("--url", default="http://fes-svc-bench:8080")
    ap.add_argument("--sets", default="real200,archive100")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)
