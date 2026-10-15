"""R6-BEST — which no-reference rule picks the better of the r6 and r6_ft outputs? (run inside the service container)

For blurred faces WITH a sharp ground truth (archive VAL = heavy shake blur; synth = FairFace gaussian/defocus/motion,
make_synth_set.py) it calls both profiles through the running service (face_chip), and stores per face: status of
each, SFace cosine to the sharp target (same-image fidelity proxy) and no-reference sharpness measures of the input
and of each output. `--analyze` then compares selection rules: rule chosen on the even-indexed half, reported on the
odd-indexed half.

    docker exec fes-ui python scripts/r6_ft/selector_study.py [--n 160]
    docker exec fes-ui python scripts/r6_ft/selector_study.py --analyze
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import httpx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_check import PROFILES, call  # noqa: E402

from fes_bench.metrics import YUNET, SFace  # noqa: E402
from fes_data.common import repo_path  # noqa: E402
from fes_data.preprocess import YuNetDetector  # noqa: E402
from fes_service.rwgate import lapd1  # noqa: E402

OUT = "experiments/r6_ft/selector"
SETS = {"archive": ("Dataset-and-code/archive/input/val", "Dataset-and-code/archive/output/val"),
        "synth": (f"{OUT}/synth/input", f"{OUT}/synth/sharp")}


def sharp_feats(img):
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx, gy = cv2.Sobel(y, cv2.CV_64F, 1, 0), cv2.Sobel(y, cv2.CV_64F, 0, 1)
    return {"lapd1": lapd1(img), "lapvar": float(cv2.Laplacian(y, cv2.CV_64F).var()),
            "teng": float((gx ** 2 + gy ** 2).mean())}


def collect(a):
    det = YuNetDetector(repo_path(YUNET), score_threshold=0.7, nms_threshold=0.3, max_side=640)
    sf = SFace()
    client = httpx.Client(timeout=300)
    rows = []
    for sname, (din, dgt) in SETS.items():
        names = sorted(os.listdir(repo_path(din)))
        names = names[::max(1, len(names) // a.n)][:a.n]
        for i, n in enumerate(names):
            data = repo_path(f"{din}/{n}").read_bytes()
            blur = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            sharp = cv2.imread(str(repo_path(f"{dgt}/{n}")), cv2.IMREAD_COLOR)
            if blur is None or sharp is None or blur.shape != sharp.shape:
                continue
            faces = det.detect(cv2.resize(sharp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            if not faces:
                continue
            lm = np.asarray(faces[0].landmarks, np.float32) / 2.0
            e = sf.embed(sharp, lm)
            row = {"set": sname, "name": n, "idx": i, "kind": n.rsplit("_", 1)[-1][:-4] if sname == "synth" else "shake",
                   "input": {**sharp_feats(blur), "sface": float(e @ sf.embed(blur, lm))}}
            for p in PROFILES:
                rec, img, _ = call(client, a.url, data, "face_chip", p)
                ok = rec["status"] == "ENHANCED" and img is not None and img.shape == blur.shape
                row[p] = {"enhanced": ok, "fallback": rec["fallback"], "reason": rec["reason"],
                          "server_ms": rec["server_ms"]}
                if ok:
                    row[p].update(sharp_feats(img), sface=float(e @ sf.embed(img, lm)),
                                  mad=float(np.abs(img.astype(np.float64) - blur).mean()))
            rows.append(row)
            if (i + 1) % 40 == 0:
                print(f"{sname} {i + 1}/{len(names)}", flush=True)
    with open(repo_path(f"{OUT}/rows.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"rows {len(rows)}")


def analyze():
    rows = json.loads(repo_path(f"{OUT}/rows.json").read_text())
    score = lambda r, p: r[p]["sface"] if r[p]["enhanced"] else r["input"]["sface"]  # noqa: E731

    def pick_by(key):
        def rule(r):
            c = [p for p in PROFILES if r[p]["enhanced"]]
            return max(c, key=lambda p: r[p][key]) if c else None
        return rule

    def first(order):
        return lambda r: next((p for p in order if r[p]["enhanced"]), None)

    rules = {"always r6": lambda r: "r6" if r["r6"]["enhanced"] else None,
             "always r6_ft": lambda r: "r6_ft" if r["r6_ft"]["enhanced"] else None,
             "r6, else r6_ft": first(("r6", "r6_ft")), "r6_ft, else r6": first(("r6_ft", "r6")),
             "sharper: lapd1": pick_by("lapd1"), "sharper: lapvar": pick_by("lapvar"),
             "sharper: teng": pick_by("teng"), "bigger change: mad": pick_by("mad"),
             "ORACLE (needs the sharp original)": lambda r: max(PROFILES, key=lambda p: score(r, p))}

    def report(sel, title):
        print(f"\n== {title}: {len(sel)} faces · input SFace {np.mean([r['input']['sface'] for r in sel]):.3f}")
        both = [r for r in sel if all(r[p]["enhanced"] for p in PROFILES)]
        for name, rule in rules.items():
            got = [(rule(r), r) for r in sel]
            s = np.array([score(r, p) if p else r["input"]["sface"] for p, r in got])
            d = s - np.array([r["input"]["sface"] for r in sel])
            right = [score(r, rule(r)) >= max(score(r, p) for p in PROFILES) - 0.02 for r in both] if both else []
            print(f"{name:36s} SFace {s.mean():.3f} · enhanced {sum(p is not None for p, _ in got):3d} · worse "
                  f"{int((d < -0.02).sum()):2d} · picks the better one {int(np.sum(right)):3d}/{len(both)}")

    for half, label in ((0, "DESIGN half (even)"), (1, "TEST half (odd)")):
        sel = [r for r in rows if r["idx"] % 2 == half]
        report(sel, label)
        for s in SETS:
            report([r for r in sel if r["set"] == s], f"{label} — {s}")
    test = [r for r in rows if r["idx"] % 2 == 1 and r["set"] == "synth"]
    for k in sorted({r["kind"] for r in test}):
        report([r for r in test if r["kind"] == k], f"TEST half — synth {k}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else collect(a)
