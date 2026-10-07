"""R5-5 — derive and freeze the guarded NAFNet route config (configs/phase5/r5_nafnet_route.json) from DEV data only.

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_nafnet_calibrate.py [--probe-nafnet]

Data: DEV only — p4dev (896 synthetic pairs, FairFace dev refs) + r5gap (224 dev pairs at smaller faces) + the clean dev
references. NOT used: the synthetic TEST population, the archive test/ztest (R5-ARCHIVE held-out), the 200 real CC0
photos, the real-person data. No identity outcome is used for any NAFNet threshold.

Threshold rules
  iod_min, luma_mean_lo, rms_contrast_lo, blockiness_hi, blur_max, clean_lapd1_lo   reused from p5-rwgate-1.0.0 (seal
      78837a8d): iod_min = iod_min_a4 (smallest DEV face-size tier with a net enhancement benefit; NAFNet's own evidence
      is at IOD ≈ 46 px, the archive's 128-px chips), "strong blur" = the rwgate strong_blur boundary, "clean-looking"
      = 2 % lapd1 quantile of clean DEV references. NO lapd1 floor: R5-ARCHIVE §8 — the NAFNet route covers the extreme
      blur the A4 policy rejected; the T6 detection rules (score ≥ 0.85, pose) remain the recoverability limit.
  noise_max          = rwgate clean_noise_hi (98 % noise σ quantile of clean DEV references): "low noise" means no more
      grain than a clean photo. R5-ARCHIVE: grainy blur (ztest, median σ 1.75) gave 14 % colour artefacts.
  pixelation_max, halftone_peak_max, halftone_energy_max   1.1 × the maximum over the DEV inputs that pass every other
      NAFNet rule (no DEV strong-blur input is rejected by them); detection is then MEASURED on synthetic pixelated
      (nearest-neighbour ×3…×8) and halftone (dot screens, period 3…8) versions of the clean DEV references.
  guard.chroma_p99_max = 10 Lab units: the R5-ARCHIVE artefact signal (output vs input chroma p99), kept configurable.
  guard.hf_color_ratio_max = 1.5 × the 99.5 % quantile over clean DEV references (natural high-frequency colour/luma
      balance); guard.hf_color_growth_max = 3 (the output must ALSO have ≥ 3× the input's ratio — deblurring adds luma
      detail, not colour oscillation).
  guard.gray_chroma_max = 6 Lab units (≈ 2.5 just-noticeable differences); guard.channel_mean_shift_max = 25 grey
      levels (deblurring does not move a channel's mean).
--probe-nafnet runs the real NAFNet-GoPro on the DEV strong-blur inputs and on DEV grainy blur, reporting guard metrics
(diagnostic only: it does not change any threshold).
Writes the sealed config + data/manifests/phase5_rapid/nafnet_route/calibration.json.
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from fes_bench import gate as GT
from fes_data import preprocess as P
from fes_data.common import read_csv_gz, repo_path, run_info, write_json
from fes_service import nafnet_route as NR
from fes_service import rwgate as RW

OUT = "data/manifests/phase5_rapid/nafnet_route"
GAP = "data/manifests/phase5_rapid/gap"
MARGIN = 1.1


def dev_inputs():
    pop = read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz"))
    gap = read_csv_gz(repo_path(f"{GAP}/population_r5gap.csv.gz"))
    for r in pop + gap:
        yield r, cv2.imread(str(repo_path(r["degraded_path"])), cv2.IMREAD_COLOR)


def features(img, det) -> dict:
    f = RW.rw_features(img, det)
    f.update(NR.texture_features(img))
    return f


def main(probe: bool) -> None:
    rw = RW.load_config()
    t_rw = rw["thresholds"]
    det = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    base = {"iod_min": t_rw["iod_min_a4"], "luma_mean_lo": t_rw["luma_mean_lo"],
            "rms_contrast_lo": t_rw["rms_contrast_lo"], "blockiness_hi": t_rw["blockiness_hi"],
            "noise_max": t_rw["clean_noise_hi"], "blur_max": t_rw["blur_max"], "clean_lapd1_lo": t_rw["clean_lapd1_lo"]}
    # provisional config: texture rules disabled, to find the DEV inputs that pass everything else
    prov = {"thresholds": {**base, "pixelation_max": 9.0, "halftone_peak_max": 9.0, "halftone_energy_max": 9.0}}
    rows, refs = [], {}
    for r, img in dev_inputs():
        f = features(img, det)
        d = NR.decide(f, prov)
        rows.append({"pair_id": r["pair_id"], "profile": r["profile"], "severity": r["severity"], "reason": d.reason,
                     **{k: f.get(k) for k in ("pixelation", "hf_peak_frac", "hf_energy", "lapd1_c", "noise_sigma")}})
        refs[r["ref_id"]] = r["ref_path"]
    el = [r for r in rows if r["reason"] == NR.ELIGIBLE]
    th = {**base,
          "pixelation_max": round(MARGIN * max(r["pixelation"] for r in el), 6),
          "halftone_peak_max": round(MARGIN * max(r["hf_peak_frac"] for r in el), 6),
          "halftone_energy_max": round(MARGIN * max(r["hf_energy"] for r in el), 6)}
    clean = [cv2.imread(str(repo_path(p)), cv2.IMREAD_COLOR) for p in sorted(set(refs.values()))]
    hf_clean = [NR.hf_color_ratio(c) for c in clean]
    guard = {"chroma_p99_max": 10.0, "hf_color_ratio_max": round(1.5 * float(np.quantile(hf_clean, 0.995)), 6),
             "hf_color_growth_max": 3.0, "gray_chroma_max": 6.0, "channel_mean_shift_max": 25.0}

    # detection on synthetic printed-image risks (clean DEV references; reported, not fitted)
    cfg_tmp = {"thresholds": th}
    synth = {}
    for name, fn in ([(f"pixelate_x{k}", lambda i, k=k: NR.synth_pixelate(i, k)) for k in (3, 4, 6, 8)] +
                     [(f"halftone_p{p}", lambda i, p=p: NR.synth_halftone(i, p)) for p in (3, 4, 6, 8)]):
        out, tex_hits = {}, {"pixelation": 0, "halftone": 0, "either": 0, "lapd1_le_blur_max": 0}
        for c in clean:
            f = features(fn(c), det)
            why = NR.decide(f, cfg_tmp).reason
            out[why] = out.get(why, 0) + 1
            # the texture rules alone (independent of detection / blur): how sensitive are they?
            px = f["pixelation"] > th["pixelation_max"]
            ht = f["hf_peak_frac"] > th["halftone_peak_max"] and f["hf_energy"] > th["halftone_energy_max"]
            tex_hits["pixelation"] += px
            tex_hits["halftone"] += ht
            tex_hits["either"] += px or ht
            tex_hits["lapd1_le_blur_max"] += bool(f["lapd1_c"] <= th["blur_max"])
        synth[name] = {"n": len(clean), "decision": dict(sorted(out.items())), "texture_rule_hits": tex_hits}
    cnt = {}
    for r in rows:                                   # reasons with the texture rules disabled (provisional config)
        cnt[r["reason"]] = cnt.get(r["reason"], 0) + 1

    cfg = {"name": "r5_nafnet_route", "route_version": NR.ROUTE_VERSION, "feature_version": NR.FEAT_VERSION,
           "rw_feature_version": RW.RWFEAT_VERSION, "model": NR.MODEL,
           "model_asset": {"asset_id": "A06", "weights": "models/M28_nafnet/NAFNet-GoPro-width64.pth",
                           "sha256": "329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f",
                           "source": "megvii-research/NAFNet @2b4af71 (official README model table)",
                           "license": "MIT (models/M28_nafnet/LICENSE)"},
           "thresholds": th, "guard": guard,
           "decision_order": ["safety (T6, unchanged)", "missing_feature", "low_face_size", "photometric",
                              "compression", "pixelation", "halftone", "high_noise", "not_blur", "mild_blur",
                              "eligible_strong_low_noise_blur"],
           "calibration": {"data": "DEV only: p4dev (896) + r5gap (224) + clean dev references",
                           "margin": MARGIN, "n_dev": len(rows), "n_dev_eligible": len(el)},
           "rwgate_seal": rw["sha256"],
           "validity": "Research route. Input-gate boundaries reuse DEV-calibrated p5-rwgate signals; the NAFNet benefit "
                       "is evidenced only on synthetic blur (R5-ARCHIVE). Not a production approval."}
    cfg["sha256"] = GT.gate_hash(cfg)
    write_json(repo_path(NR.CONFIG), cfg)
    report = {"run": run_info(), "seal": cfg["sha256"], "thresholds": th, "guard": guard,
              "dev_reason_counts": dict(sorted(cnt.items())), "dev_eligible": el,
              "hf_color_clean_q": [round(float(q), 4) for q in np.quantile(hf_clean, [0.5, 0.99, 0.995, 1.0])],
              "synthetic_detection": synth}
    if probe:
        report["nafnet_probe"] = probe_nafnet(rows, cfg, det)
    write_json(repo_path(f"{OUT}/calibration.json"), report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("dev_eligible", "run")}, indent=1))


def probe_nafnet(rows, cfg, det) -> dict:
    """Diagnostic: real NAFNet on DEV strong-blur (eligible) inputs and DEV grainy blur (rejected HIGH_NOISE)."""
    from fes_service.models import nafnet_factory
    fn, info = nafnet_factory()
    paths = {r["pair_id"]: r["degraded_path"] for r in
             read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz")) +
             read_csv_gz(repo_path(f"{GAP}/population_r5gap.csv.gz"))}
    grp = {"eligible": [r for r in rows if r["reason"] == NR.ELIGIBLE],
           "grainy_blur": [r for r in rows if r["reason"] == NR.R_NOISE and r["lapd1_c"] <= cfg["thresholds"]["blur_max"]]}
    out = {"model": info}
    for name, rs in grp.items():
        recs = []
        for r in rs:
            img = cv2.imread(str(repo_path(paths[r["pair_id"]])), cv2.IMREAD_COLOR)
            t0 = time.perf_counter()
            o, _ = fn(img, {})
            dt = time.perf_counter() - t0
            g = NR.artifact_guard(img, o, cfg, lambda x: len(det.detect(x)) > 0, RW.is_gray(img))
            recs.append({"pair_id": r["pair_id"], "secs": round(dt, 3), "passed": g.passed, "reasons": g.reasons,
                         **g.metrics})
        out[name] = {"n": len(recs), "passed": sum(x["passed"] for x in recs),
                     "sec_per_chip": round(float(np.mean([x["secs"] for x in recs])), 3) if recs else None,
                     "rows": recs}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-nafnet", action="store_true")
    main(ap.parse_args().probe_nafnet)
