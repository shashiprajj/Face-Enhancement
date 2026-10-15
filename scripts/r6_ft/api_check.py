"""R6-FT — check of POST /v2/enhance (profile r6_ft) against profile r6 through the running service, on data the
fine-tuned weights never saw: archive TEST pairs (face_chip, with sharp ground truth) and the real photos (full_image).

    docker exec fes-ui python scripts/r6_ft/api_check.py [--n 100] [--out experiments/r6_ft/api_check]
Writes results.json, and per image a review picture: LEFT = face crop the service analysed, MIDDLE = r6 output,
RIGHT = r6_ft output (an output that fell back to the original is stamped ORIGINAL).
SFace cosine to the sharp ground truth (archive only) is a same-image fidelity proxy, not a cross-photo identity test.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time

import cv2
import httpx
import numpy as np

from fes_bench.metrics import YUNET, SFace
from fes_data.common import repo_path
from fes_data.preprocess import YuNetDetector

ARCHIVE = "Dataset-and-code/archive"
REAL = "Dataset-and-code/dataset"
PROFILES = ("r6", "r6_ft")


def call(client, url, data, mode, profile):
    files = {"image": ("x.png", data, "application/octet-stream")}
    t0 = time.time()
    if profile == "r6_ft":
        r = client.post(f"{url}/v2/enhance", files=files, data={"input_mode": mode})
    else:
        r = client.post(f"{url}/v1/enhance", files=files, data={"input_mode": mode, "profile": profile})
    b = r.json()
    img = b.get("image_b64") and cv2.imdecode(np.frombuffer(base64.b64decode(b["image_b64"]), np.uint8), cv2.IMREAD_COLOR)
    chip = b.get("chip_b64") and cv2.imdecode(np.frombuffer(base64.b64decode(b["chip_b64"]), np.uint8), cv2.IMREAD_COLOR)
    nf = b.get("nafnet") or {}
    rec = {"http": r.status_code, "status": b.get("status"), "fallback": b.get("fallback") or "",
           "reason": nf.get("reason") or b.get("reason") or b.get("error") or "", "model": b.get("model"),
           "server_ms": (b.get("timings_ms") or {}).get("total"), "rtt_s": round(time.time() - t0, 2),
           "lapd1_c": (b.get("quality") or {}).get("lapd1_c")}
    return rec, img, chip


def tile(img, text, side=256):
    t = cv2.resize(img, (side, side), interpolation=cv2.INTER_CUBIC)
    bar = np.zeros((22, side, 3), np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, t])


def review(path, chip, outs):
    tiles = [tile(chip, "INPUT (face crop)")]
    for p in PROFILES:
        rec, img = outs[p]
        on = rec["status"] == "ENHANCED" and img is not None
        label = f"{p}: ENHANCED" if on else f"{p}: ORIGINAL {rec['fallback']}".strip()
        tiles.append(tile(img if on else chip, label))
    cv2.imwrite(path, np.hstack(tiles))


def main(a):
    det = YuNetDetector(repo_path(YUNET), score_threshold=0.7, nms_threshold=0.3, max_side=640)
    sf = SFace()
    client = httpx.Client(timeout=300)
    res = {"archive": [], "real": []}

    names = sorted(os.listdir(repo_path(f"{ARCHIVE}/input/test")))
    names = names[::max(1, len(names) // a.n)][:a.n]
    os.makedirs(os.path.join(a.out, "archive"), exist_ok=True)
    for i, n in enumerate(names):
        data = repo_path(f"{ARCHIVE}/input/test/{n}").read_bytes()
        blur = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        sharp = cv2.imread(str(repo_path(f"{ARCHIVE}/output/test/{n}")), cv2.IMREAD_COLOR)
        row, outs = {"name": n}, {}
        for p in PROFILES:
            rec, img, _ = call(client, a.url, data, "face_chip", p)
            row[p], outs[p] = rec, (rec, img)
        faces = det.detect(cv2.resize(sharp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)) if sharp is not None else []
        if faces and sharp.shape == blur.shape:
            lm = np.asarray(faces[0].landmarks, np.float32) / 2.0
            e = sf.embed(sharp, lm)
            row["sface_input"] = float(e @ sf.embed(blur, lm))
            for p in PROFILES:
                img = outs[p][1]
                ok = row[p]["status"] == "ENHANCED" and img is not None and img.shape == blur.shape
                row[f"sface_{p}"] = float(e @ sf.embed(img, lm)) if ok else row["sface_input"]
        review(os.path.join(a.out, "archive", os.path.splitext(n)[0] + ".png"), blur, outs)
        res["archive"].append(row)
        if (i + 1) % 20 == 0:
            print(f"archive {i + 1}/{len(names)}", flush=True)

    rnames = sorted(os.listdir(repo_path(REAL)))
    os.makedirs(os.path.join(a.out, "real"), exist_ok=True)
    for i, n in enumerate(rnames):
        data = repo_path(f"{REAL}/{n}").read_bytes()
        row, outs, chip = {"name": n}, {}, None
        for p in PROFILES:
            rec, img, c = call(client, a.url, data, "full_image", p)
            row[p], outs[p] = rec, (rec, img)
            chip = c if c is not None else chip
        if chip is not None and any(row[p]["status"] == "ENHANCED" for p in PROFILES):
            review(os.path.join(a.out, "real", os.path.splitext(n)[0] + ".png"), chip, outs)
        res["real"].append(row)
        if (i + 1) % 40 == 0:
            print(f"real {i + 1}/{len(rnames)}", flush=True)

    def summ(rows, sface):
        out = {"n": len(rows)}
        for p in PROFILES:
            st = [r[p]["status"] for r in rows]
            fb = {}
            for r in rows:
                if r[p]["fallback"]:
                    fb[r[p]["fallback"]] = fb.get(r[p]["fallback"], 0) + 1
            ms = [r[p]["server_ms"] for r in rows if r[p]["status"] == "ENHANCED" and r[p]["server_ms"]]
            out[p] = {"enhanced": st.count("ENHANCED"), "original": st.count("ORIGINAL"),
                      "other": len(st) - st.count("ENHANCED") - st.count("ORIGINAL"), "fallbacks": fb,
                      "median_server_ms_enhanced": float(np.median(ms)) if ms else None}
        if sface:
            fr = [r for r in rows if "sface_input" in r]
            out["n_face"] = len(fr)
            out["sface_input"] = float(np.mean([r["sface_input"] for r in fr]))
            for p in PROFILES:
                d = np.array([r[f"sface_{p}"] - r["sface_input"] for r in fr])
                out[p].update(sface=float(np.mean([r[f"sface_{p}"] for r in fr])), better=int((d > 0.02).sum()),
                              worse=int((d < -0.02).sum()))
        return out

    res["summary"] = {"archive_test_face_chip": summ(res["archive"], True), "real_full_image": summ(res["real"], False)}
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["summary"], indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="experiments/r6_ft/api_check")
    main(ap.parse_args())
