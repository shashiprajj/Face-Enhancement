"""R5-7 — quantify why full_image routes fewer archive inputs to NAFNet than face_chip (DEV `val` sample, 3,000 pairs,
seed 0; records from scripts/r5_7_dev_collect.py --run val_frame) and summarise the padding experiment.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_7_frame_analyze.py   → data/manifests/phase5_rapid/r5_7/frame_analysis.json
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from fes_data.common import read_csv_gz, repo_path, run_info, write_json
from fes_service import nafnet_route as NR

OUT = "data/manifests/phase5_rapid/r5_7"


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def q(xs, ps=(0.1, 0.5, 0.9)):
    xs = [x for x in xs if np.isfinite(x)]
    return [round(float(np.quantile(xs, p)), 5) for p in ps] if xs else None


def main() -> None:
    rows = read_csv_gz(repo_path(f"{OUT}/val_frame.csv.gz"))
    t = NR.load_config()["thresholds"]
    fc_el = [r for r in rows if r["fc_eligible"] == "True"]
    fi_el = [r for r in rows if r.get("fi_eligible") == "True"]
    lost = [r for r in fc_el if r.get("fi_eligible") != "True"]
    padded = lambda r: fl(r.get("fi_pad_frac")) > 0  # noqa: E731
    S = {"n": len(rows), "fc_eligible": len(fc_el), "fi_eligible": len(fi_el),
         "eligible_both": sum(r.get("fi_eligible") == "True" for r in fc_el), "lost_in_fi": len(lost),
         "gained_in_fi": sum(r["fc_eligible"] != "True" for r in fi_el),
         "lost_reason": dict(Counter(r.get("fi_reason") or f"NO_CHIP:{r.get('fi_det_status')}" for r in lost).most_common())}
    # which threshold each lost input crosses (first-failing rule = reason; also: would it pass with fc's value?)
    feat = {"lapd1_c": ("NOT_BLUR/MILD", lambda v: v > t["blur_max"]), "iod_px": ("LOW_FACE_SIZE", lambda v: v < t["iod_min"]),
            "blockiness": ("COMPRESSION", lambda v: v > t["blockiness_hi"]), "noise_sigma": ("HIGH_NOISE", lambda v: v > t["noise_max"])}
    S["feature_shift_fc_eligible"] = {k: {"fc": q([fl(r[f"fc_{k}"]) for r in fc_el]), "fi": q([fl(r.get(f"fi_{k}")) for r in fc_el]),
                                          "fi_over_fc_ratio": q([fl(r.get(f"fi_{k}")) / fl(r[f"fc_{k}"]) for r in fc_el
                                                                 if fl(r[f"fc_{k}"]) not in (0,)]),
                                          "n_crossing_in_fi": sum(f(fl(r.get(f"fi_{k}"))) for r in fc_el)}
                                      for k, (_, f) in feat.items()}
    # padding vs re-framing
    S["padding"] = {"fc_eligible_with_padded_chip": sum(map(padded, fc_el)),
                    "lost_with_padded_chip": sum(map(padded, lost)), "lost_without_padding": sum(not padded(r) for r in lost),
                    "loss_rate_padded": round(sum(map(padded, lost)) / max(1, sum(map(padded, fc_el))), 3),
                    "loss_rate_unpadded": round(sum(not padded(r) for r in lost) / max(1, sum(not padded(r) for r in fc_el)), 3),
                    "lost_reason_unpadded": dict(Counter(r.get("fi_reason") for r in lost if not padded(r)).most_common()),
                    "lost_reason_padded": dict(Counter(r.get("fi_reason") for r in lost if padded(r)).most_common()),
                    "lapd1_ratio_unpadded": q([fl(r.get("fi_lapd1_c")) / fl(r["fc_lapd1_c"]) for r in fc_el if not padded(r)]),
                    "lapd1_ratio_padded": q([fl(r.get("fi_lapd1_c")) / fl(r["fc_lapd1_c"]) for r in fc_el if padded(r)])}
    S["geometry"] = {"chip_side": q([fl(r.get("fi_side")) for r in fc_el]), "img_cover_frac": q([fl(r.get("fi_img_cover_frac")) for r in fc_el]),
                     "iod_img": q([fl(r.get("fi_iod_img")) for r in fc_el]),
                     "chip_side_mod16_zero": sum(int(fl(r.get("fi_side")) or 0) % 16 == 0 for r in fc_el if r.get("fi_side"))}
    both = [r for r in fc_el if r.get("fi_eligible") == "True"]
    S["identity_both_eligible"] = {"n": len(both), "fc": q([fl(r.get("fc_d_sface")) for r in both], (0.5,)),
                                   "fc_mean": round(float(np.nanmean([fl(r.get("fc_d_sface")) for r in both])), 4),
                                   "fc_region_only_mean": round(float(np.nanmean([fl(r.get("fcreg_d_sface")) for r in both])), 4),
                                   "fi_mean": round(float(np.nanmean([fl(r.get("fi_d_sface")) for r in both])), 4),
                                   "fcfi_region_psnr": q([fl(r.get("fcfi_region_psnr")) for r in both])}
    p = repo_path(f"{OUT}/frame_experiment.csv.gz")
    if p.exists():
        e = read_csv_gz(p)
        m = lambda k, rs: round(float(np.nanmean([fl(r[k]) for r in rs])), 4)  # noqa: E731
        bo = [r for r in e if r["fc_eligible"] == "True"]
        S["padding_experiment"] = {"n": len(e), "zero_d_sface": m("zero_d_sface", e), "reflect_d_sface": m("reflect_d_sface", e),
                                   "zero_d_dlib": m("zero_d_dlib", e), "reflect_d_dlib": m("reflect_d_dlib", e),
                                   "n_pad16_zero": sum(r["pad16"] == "0" for r in e),
                                   "pad16_zero_identical": all(r["zero_d_sface"] == r["reflect_d_sface"] for r in e if r["pad16"] == "0"),
                                   "reflect_better_n": sum(fl(r["reflect_d_sface"]) > fl(r["zero_d_sface"]) for r in e),
                                   "both_eligible": {"n": len(bo), "fc_region_only": m("fcreg_d_sface", bo), "zero": m("zero_d_sface", bo),
                                                     "reflect": m("reflect_d_sface", bo)},
                                   "guard_pass_zero": sum(r["zero_guard_pass"] == "True" for r in e),
                                   "guard_pass_reflect": sum(r["reflect_guard_pass"] == "True" for r in e),
                                   "harm_both_zero": sum(fl(r["zero_d_sface"]) < -.02 and fl(r["zero_d_dlib"]) < -.02 for r in e),
                                   "harm_both_reflect": sum(fl(r["reflect_d_sface"]) < -.02 and fl(r["reflect_d_dlib"]) < -.02 for r in e)}
    write_json(repo_path(f"{OUT}/frame_analysis.json"), run_info(S))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
