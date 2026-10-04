"""R5-3 — calibrate and freeze the real-world gate (configs/phase5/r5_rwgate.json) from DEV data only.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_rwgate_calibrate.py      (numpy/OpenCV; reads tracked metrics)

Evidence used (all DEV split; the synthetic TEST population `main` and the 200 real photos are not used for fitting):
  * p4dev: 896 pairs, A4 outputs scored by SFace (E01) + dlib (E07) in Phase 4 → per-pair multi-evaluator verdict vs B0
  * r5gap: 224 pairs, the A4-favourable dev cells re-created at smaller faces (168 / 112 px chips), scored the same way
Threshold rules (each explainable; no target coverage):
  noise_min            T6 rule0 value, unchanged (per-pixel noise σ; dev-calibrated in P4-T6; not a synthetic artefact)
  blur_max             lapd1_c threshold maximising U = #CONFIRMED_BENEFIT − 3·#HARM over dev pairs that pass the
                       safety/size/photometric/compression checks and are not already noise-routed (λ = 3 as in T6/T7)
  lapd1_floor          minimum lapd1_c over dev blur pairs where A4 was measured: blurrier = outside the evidence
  iod_min_a4           smallest face-size tier whose routed dev pairs have U > 0 with harm ≤ benefit/3; threshold = 2 %
                       IOD quantile of that tier (same convention as T7's clean-reference quantiles)
  iod_min_recognizable 16 px: below the ~22 px tier no model shows any net identity benefit (harm-only at ~11 px)
  luma_mean_lo, rms_contrast_lo, blockiness_hi   copied from the frozen T7 bucket thresholds (clean dev reference
                       quantiles, no outcome data) — photometric/JPEG inputs stay original (no A4 evidence, P4-T4 P7)
Writes configs/phase5/r5_rwgate.json (sealed) + data/manifests/phase5_rapid/rwgate/calibration.json.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import cv2
import numpy as np

from fes_bench import gate as GT
from fes_bench import p4eval as E
from fes_bench import quality as Q
from fes_data import preprocess as P
from fes_data.common import read_csv_gz, repo_path, run_info, write_json
from fes_service import rwgate as RW

LAMBDA = 3.0
GAP = "data/manifests/phase5_rapid/gap"


def verdicts_gap() -> dict:
    """pair_id → (multi verdict A4 vs B0, iod tier)."""
    d = E.deltas_from_validation()
    cos = {}
    for c in ("B0", "A4"):
        for r in read_csv_gz(repo_path(f"{GAP}/metrics/r5gap/{c}.csv.gz")):
            cos[(r["pair_id"], c, "E01")] = E.fnum(r["id_cos"])
        for r in read_csv_gz(repo_path(f"{GAP}/metrics_e07/r5gap/{c}.csv.gz")):
            cos[(r["pair_id"], c, "E07")] = E.fnum(r["id_cos_e07"])
    out = {}
    for pid in {k[0] for k in cos}:
        v1 = E.verdict(cos[(pid, "A4", "E01")] - cos[(pid, "B0", "E01")], d["E01"])
        v7 = E.verdict(cos[(pid, "A4", "E07")] - cos[(pid, "B0", "E07")], d["E07"])
        out[pid] = E.multi_verdict(v1, v7)
    return out


def collect() -> list[dict]:
    """Dev rows: rw features + A4 verdict + size tier."""
    det = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    d = E.deltas_from_validation()
    rows = []
    pairs = E.load("p4dev")
    pop = {r["pair_id"]: r for r in read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz"))}
    for pid, p in pairs.items():
        img = cv2.imread(str(repo_path(pop[pid]["degraded_path"])), cv2.IMREAD_COLOR)
        f = dict(p.feat)
        f["lapd1_c"] = RW.lapd1(img)
        rows.append({"pair_id": pid, "src": "p4dev", "tier": int(pop[pid]["out_w"]), "profile": p.profile,
                     "severity": p.severity, "f": f, "v": E.multi_of(p, "A4", d)})
    gv = verdicts_gap()
    for r in read_csv_gz(repo_path(f"{GAP}/population_r5gap.csv.gz")):
        img = cv2.imread(str(repo_path(r["degraded_path"])), cv2.IMREAD_COLOR)
        f = Q.features(img, det)
        f["lapd1_c"] = RW.lapd1(img)
        rows.append({"pair_id": r["pair_id"], "src": "r5gap", "tier": int(r["out_w"]), "profile": r["profile"],
                     "severity": r["severity"], "f": f, "v": gv[r["pair_id"]]})
    return rows


def U(rows) -> tuple[float, int, int]:
    b = sum(r["v"] in GT.BENEFIT for r in rows)
    h = sum(r["v"] in GT.HARM for r in rows)
    return b - LAMBDA * h, b, h


def main() -> None:
    t7 = json.loads(repo_path("configs/phase4/t7_routing.json").read_text(encoding="utf-8"))["bucket_thresholds"]
    g6 = json.loads(repo_path("configs/phase4/t6_gate.json").read_text(encoding="utf-8"))
    noise_min = float(g6["rules"][0]["conds"][0][2])
    rows = collect()
    base = {"iod_min_recognizable": 16.0, "noise_min": noise_min, "luma_mean_lo": t7["luma_mean_lo"],
            "rms_contrast_lo": t7["rms_contrast_lo"], "blockiness_hi": t7["blockiness_hi"]}

    def eligible(r):
        f = r["f"]
        if GT.safety(f) or (f.get("iod_px") or 0) < base["iod_min_recognizable"] or math.isnan(f.get("iod_px", math.nan)):
            return False
        if f.get("luma_mean", 999) < base["luma_mean_lo"] or f.get("rms_contrast", 999) < base["rms_contrast_lo"]:
            return False
        return not (f.get("blockiness", 0) == f.get("blockiness", 0) and f.get("blockiness", 0) > base["blockiness_hi"])
    el = [r for r in rows if eligible(r)]

    # blur_max: fit on the largest tier (224, where Phase 4 evidence lives), pairs not noise-routed
    blur_pool = [r for r in el if r["tier"] == 224 and not r["f"].get("noise_sigma", 0) >= noise_min]
    cands = sorted({round(float(np.quantile([r["f"]["lapd1_c"] for r in blur_pool], q)), 6)
                    for q in np.arange(0.01, 0.51, 0.01)})
    scan = []
    for t in cands:
        u, b, h = U([r for r in blur_pool if r["f"]["lapd1_c"] <= t])
        scan.append({"t": t, "U": u, "benefit": b, "harm": h})
    best = max(scan, key=lambda s: (s["U"], -s["t"]))
    blur_max = best["t"] if best["U"] > 0 else 0.0
    # floor: below the blurriest dev blur pair A4 was measured on
    blur_rows = [r for r in rows if r["profile"] == "gaussblur"]
    lapd1_floor = round(float(min(r["f"]["lapd1_c"] for r in blur_rows)) * 0.9, 6)

    def routed(r):
        f = r["f"]
        return (f.get("noise_sigma", 0) >= noise_min) or (lapd1_floor <= f["lapd1_c"] <= blur_max)
    tiers = {}
    for tier in sorted({r["tier"] for r in el}, reverse=True):
        rr = [r for r in el if r["tier"] == tier and routed(r)]
        u, b, h = U(rr)
        iods = [r["f"]["iod_px"] for r in rr if not math.isnan(r["f"].get("iod_px", math.nan))]
        tiers[tier] = {"n_routed": len(rr), "U": u, "benefit": b, "harm": h,
                       "iod_q02": round(float(np.quantile(iods, 0.02)), 3) if iods else None,
                       "iod_median": round(float(np.median(iods)), 3) if iods else None,
                       "by_cell": dict(Counter(f"{r['profile']}-{r['severity']}:{r['v']}" for r in rr))}
    ok_tiers = [t for t, s in tiers.items() if s["n_routed"] >= 10 and s["U"] > 0 and s["harm"] * LAMBDA <= s["benefit"]]
    # the size boundary must be contiguous from the top tier down
    chosen = 224
    for t in sorted(tiers, reverse=True):
        if t in ok_tiers:
            chosen = t
        else:
            break
    iod_min_a4 = tiers[chosen]["iod_q02"]

    # clean-reference range (labels only, no routing effect): T7's clean dev-reference quantiles + lapd1 2 % quantile
    # of the same 112 clean dev references (population_noise_floor)
    nf = read_csv_gz(repo_path("data/manifests/phase3/population_noise_floor.csv.gz"))
    seen, clean = set(), []
    for r in nf:
        if r["ref_id"] not in seen:
            seen.add(r["ref_id"])
            clean.append(RW.lapd1(cv2.imread(str(repo_path(r["ref_path"])), cv2.IMREAD_COLOR)))
    th = {**base, "blur_max": blur_max, "lapd1_floor": lapd1_floor, "iod_min_a4": iod_min_a4,
          "clean_lapd1_lo": round(float(np.quantile(clean, 0.02)), 6), "clean_noise_hi": t7["noise_sigma_hi"],
          "clean_iod_lo": t7["iod_px_lo"]}
    cfg = {"name": "r5_rwgate", "gate_version": RW.RWGATE_VERSION, "feature_version": RW.RWFEAT_VERSION,
           "route_model": RW.A4, "thresholds": th,
           "decision_order": ["safety (T6, unchanged)", "insufficient_face_size", "beyond_evidence_extreme_blur",
                              "photometric_no_a4_evidence", "compression_no_a4_evidence", "strong_noise→A4",
                              "strong_blur→A4", "degraded_below_a4_size", "degraded_no_a4_evidence", "acceptable"],
           "calibration": {"data": "DEV only: p4dev (896) + r5gap (224, dev refs)", "lambda": LAMBDA,
                           "blur_scan_best": best, "size_tier_chosen": chosen},
           "validity": "Gate behaviour calibrated for real inputs (grain-robust blur, explicit face-size zones); the "
                       "A4 benefit boundary is still SYNTHETIC-evidenced. Not a production approval.",
           "frozen_phase4": {"t6_seal": g6["sha256"], "t7_seal": json.loads(
               repo_path("configs/phase4/t7_routing.json").read_text(encoding="utf-8"))["sha256"]}}
    cfg["sha256"] = GT.gate_hash(cfg)
    write_json(repo_path(RW.CONFIG), cfg)
    write_json(repo_path("data/manifests/phase5_rapid/rwgate/calibration.json"),
               {"run": run_info(), "thresholds": th, "blur_scan": scan, "size_tiers": tiers,
                "n_rows": len(rows), "n_eligible": len(el), "seal": cfg["sha256"]})
    print(json.dumps({"thresholds": th, "best": best, "tiers": {t: {k: v for k, v in s.items() if k != "by_cell"}
                                                               for t, s in tiers.items()}}, indent=1))


if __name__ == "__main__":
    main()
