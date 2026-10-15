"""Calibrate the admin blur scale (0-100 %) and write the sealed configs/phase5/blur_scale.json.

    python scripts/blur_scale/calibrate_blur_scale.py --probe   # 1  sharp anchor: 400 clean photos → the running service
    python scripts/blur_scale/calibrate_blur_scale.py [--write] # 2  build the scale (and write it)

The scale maps the gate's sharpness feature lapd1_c to a blur percentage, separately for the two chip framings the
service uses (landmark framing = "front", detection-box framing = "side"). It is anchored to Gaussian blur on the
benchmark's 800-px clean references: percent = 25 × σ, so 0 % = sharp and 100 % = σ 4.0 (the heaviest level).
Anchor value per σ = the MEDIAN lapd1_c of that population:
  σ 0                  clean photos sent once to the service (profile best_side, --probe); no new enhancement
  σ 0.8 / 1.5 / 2.5 / 4  gaussian_blur levels 1-4 already run in the Indian-expression benchmark:
                       front = manifests/enhancement_results.jsonl (profile best, landmark framing),
                       side  = side_profile_run/*/results*.jsonl (profile best_side, framing detection_box)
Between anchors the percentage is interpolated linearly in log(lapd1_c); it is clamped to 0 / 100 outside them.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
BENCH = ROOT / "Dataset-and-code" / "fes_indian_expression_benchmark"
PROBE = BENCH / "blur_scale" / "clean_anchor.jsonl"
SIGMAS = {"0": 0.0, "1": 0.8, "2": 1.5, "3": 2.5, "4": 4.0}     # benchmark gaussian_blur levels (sigma, 800-px refs)
PCT_PER_SIGMA = 25.0


def probe(url: str) -> None:
    """σ 0 anchor: every clean reference once through profile best_side (no image returned, nothing stored)."""
    PROBE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with PROBE.open("w", encoding="utf-8") as out:
        for i, line in enumerate((BENCH / "manifests" / "clean_references.jsonl").open(encoding="utf-8")):
            r = json.loads(line)
            data = (BENCH / r["clean_path"]).read_bytes()
            b = uuid.uuid4().hex
            body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                            for k, v in {"profile": "best_side", "return_image": "false"}.items())
            body += (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="x.png"\r\n'
                     f"Content-Type: image/png\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
            req = urllib.request.Request(url + "/v1/enhance", body,
                                         {"Content-Type": f"multipart/form-data; boundary={b}"})
            j = json.loads(urllib.request.urlopen(req, timeout=300).read())
            nf = j.get("nafnet") or {}
            out.write(json.dumps({"sample_id": r["sample_id"], "status": j["status"], "reason": j["reason"],
                                  "framing": nf.get("framing", "landmarks") if nf else "",
                                  "lapd1_c": (j.get("quality") or {}).get("lapd1_c")}) + "\n")
            if i % 100 == 0:
                print(i, f"{time.time() - t0:.0f} s", flush=True)


def lapd1(r: dict):
    v = json.loads(r["quality"]).get("lapd1_c") if r.get("quality") else None
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def populations() -> dict:
    pops = {"front": {s: [] for s in SIGMAS}, "side": {s: [] for s in SIGMAS}}
    for line in PROBE.open(encoding="utf-8"):
        r = json.loads(line)
        if r["lapd1_c"] is not None and r["framing"]:
            pops["side" if r["framing"] == "detection_box" else "front"]["0"].append(r["lapd1_c"])
    for line in (BENCH / "manifests" / "enhancement_results.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["type"] == "gaussian_blur" and r["nafnet_reason"] != "NAFNET_REJECT_UNSAFE_ALIGNMENT" \
                and not r["reason"].startswith(("no_valid_face", "alignment_unreliable")) and lapd1(r) is not None:
            pops["front"][r["level"]].append(lapd1(r))
    side = {}                                          # one record per variant (test_v11 re-ran part of test)
    for p in sorted(glob.glob(str(BENCH / "side_profile_run" / "*" / "results*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["type"] == "gaussian_blur" and r.get("framing") == "detection_box" and lapd1(r) is not None:
                side[r["variant_id"]] = r
    for r in side.values():
        pops["side"][r["level"]].append(lapd1(r))
    return pops


def main(a) -> None:
    from fes_bench import gate as GT
    from fes_service import blur_scale as BS
    from fes_service import nafnet_route as NR
    pops = populations()
    curves = {}
    for framing, by in pops.items():
        pts = []
        for lvl, vals in by.items():
            med = st.median(vals)
            pts.append({"sigma": SIGMAS[lvl], "percent": round(PCT_PER_SIGMA * SIGMAS[lvl], 3),
                        "lapd1_c": round(med, 6), "n": len(vals)})
            print(f"{framing:5} sigma {SIGMAS[lvl]:3}  n={len(vals):4}  median lapd1_c {med:.6f}")
        assert all(p["lapd1_c"] > q["lapd1_c"] for p, q in zip(pts, pts[1:])), f"{framing}: anchors not monotone"
        curves[framing] = pts
    cfg = {"scale_version": BS.SCALE_VERSION, "feature": "lapd1_c",
           "definition": "blur percent = 25 x Gaussian sigma on the 800-px clean reference (0 % sharp, 100 % = sigma "
                         "4.0); per framing, linear in log(lapd1_c) between the median-lapd1_c anchors, clamped "
                         "to 0-100",
           "curves": curves,
           "data": {"date": time.strftime("%Y-%m-%d"),
                    "sharp_anchor": "400 clean references of the Indian-expression benchmark, one request each "
                                    "(profile best_side), blur_scale/clean_anchor.jsonl",
                    "blur_anchors": "benchmark gaussian_blur levels 1-4 (front: enhancement_results.jsonl, profile "
                                    "best; side: side_profile_run, framing detection_box)",
                    "limits": "one dataset, synthetic Gaussian blur; other blur types at the same percentage look "
                              "different; grain raises lapd1_c (reads as less blurred)"}}
    sc = BS.BlurScale(cfg)
    floors = {"front": NR.load_config(NR.R6_CONFIG)["thresholds"]["blur_max"],
              "side": NR.load_side_config()["thresholds"]["blur_max"]}
    cfg["validated_floor"] = {k: {"blur_max": v, "percent": round(sc.percent(k, v), 3)} for k, v in floors.items()}
    for k, v in cfg["validated_floor"].items():
        print(f"validated limit {k}: lapd1_c {v['blur_max']} = {v['percent']} %")
    cfg["sha256"] = GT.gate_hash(cfg)
    if a.write:
        p = ROOT / BS.SCALE_CONFIG
        p.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("written", p, cfg["sha256"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="measure the sharp anchor on the running service")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    probe(args.url) if args.probe else main(args)
