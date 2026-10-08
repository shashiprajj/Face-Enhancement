"""R6 — write the sealed research-candidate route config (configs/phase5/r6_nafnet_candidate.json).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r6_0/make_r6_config.py

= the sealed R5-8 candidate (configs/phase5/r5_8_nafnet_candidate.json: reflect-pad16, out-of-image reflect fill,
every non-blur input gate, global artifact guard, local-patch guard — unchanged) with the blur range widened to every
blurred face (blur_max = clean_lapd1_lo = 0.008, no mild-blur band) + ONE new section "stability":
NAFNet is run on the chip and its h / v / hv mirrors; the request returns ORIGINAL when any run's colour-dottedness
tracer exceeds run_dot_c_max (the DEV-GOOD 99th percentile from scripts/r6_0/analyze.py, R6-0), else the median of the
4 runs with the input's chrominance. Neither the live R5-6 config nor the R5-8 / R5-10 configs are modified.
"""
from __future__ import annotations

import hashlib
import json

from fes_bench import gate as GT
from fes_data.common import repo_path, write_json
from fes_service import nafnet_route as NR

R6 = "data/manifests/phase5_rapid/r6_0"
BLUR_MAX = 0.008


def sha(path: str) -> str:
    return hashlib.sha256(repo_path(path).read_bytes()).hexdigest()


def main() -> None:
    r58 = NR.load_config(NR.CANDIDATE_CONFIG)                       # verifies the R5-8 seal
    a = json.loads(repo_path(f"{R6}/analysis.json").read_text(encoding="utf-8"))
    limit = round(float(a["thresholds"]["run_dot_c_max"]), 4)
    cfg = {k: v for k, v in r58.items() if k != "sha256"}
    # R6 blur range: EVERY blurred face up to the point where sharp faces begin, no "mild blur" band. DEV evidence
    # (archive val, R5-10 outputs): NAFNet's ΔSFace is +0.13..+0.15 with <= 1.5 % harmed at every blur level from
    # lapd1 0.0005 to 0.012 — mild blur gains as much as strong blur. Sharp GT faces: 5th percentile lapd1 0.0073;
    # 0.008 admits 97.5 % of blurred and 6.5 % of sharp faces (on sharp faces NAFNet costs ≈ −0.02, mild).
    cfg["thresholds"] = {**r58["thresholds"], "blur_max": BLUR_MAX, "clean_lapd1_lo": BLUR_MAX}
    cfg["blur_range"] = {"old_blur_max": r58["thresholds"]["blur_max"],
                         "old_clean_lapd1_lo": r58["thresholds"]["clean_lapd1_lo"], "new_blur_max": BLUR_MAX,
                         "mild_blur_band": "removed (blur_max == clean_lapd1_lo)",
                         "evidence": "DEV: archive val dSFace by lapd1 band (R5-10 outputs); sharp GT lapd1 "
                                     "percentiles (800 val GT): p5 0.0073 -> 97.5 % blurred / 6.5 % sharp admitted"}
    cfg.update(name="r6_nafnet_candidate", route_version=NR.R6_ROUTE_VERSION,
               derived_from={**r58["derived_from"], "r5_8_candidate_seal": r58["sha256"]},
               stability={
                   "version": NR.STABILITY_VERSION,
                   "runs": ["identity", "hflip", "vflip", "hvflip"],
                   "tracer": "colour dottedness (R5-8 l_dot_c_max formula) of each run vs the input, valid pixels only",
                   "run_dot_c_max": limit, "rule": "any", "max_unstable_runs": 0,
                   "output": "per-pixel median of the 4 runs; luminance (Y) of the median + input chrominance (CrCb)",
                   "calibration": "DEV only: 99th percentile of max-over-runs tracer on the 458 R5-8 DEV LEGITIMATE "
                                  "outputs (archive val); fixed before any TEST population result was read",
                   "analysis_sha256": sha(f"{R6}/analysis.json")},
               status="RESEARCH CANDIDATE (R6) — not the live/default route",
               stage="final")
    cfg["sha256"] = GT.gate_hash(cfg)
    write_json(repo_path(NR.R6_CONFIG), cfg)
    print(cfg["sha256"], limit)


if __name__ == "__main__":
    main()
