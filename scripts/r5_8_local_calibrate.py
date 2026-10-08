"""R5-8 — DEV calibration of the local-patch guard (p5-nafnet-local-artifact-1.0.0). DEV (archive val) only.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_8_local_calibrate.py
    → data/manifests/phase5_rapid/r5_8/local_calibration.json   (then: scripts/r5_8_candidate_config.py final)

Inputs: local_features.csv.gz (every DEV NAFNet output of the R5-7 candidate, both modes), labels.json (visual DEV
labels; zoomed, stratified review), corpus.csv.gz (identity Δ per output).
Signal: l_dot_c_max — 8×8-window maximum of the chroma second-difference energy present in BOTH x and y, as the
output's increase over the input (dots / rainbow grids oscillate in both directions; sharpened edges do not).
Rule (stated before computing): limit = 0.9 × the smallest l_dot_c_max among DEV LOCALIZED_ARTIFACT outputs that the
signal can separate from every DEV LEGITIMATE output of the same mode (value > that mode's largest legitimate value);
artifacts it cannot separate are reported as misses, never tuned for.
"""
from __future__ import annotations

import json
import time

import numpy as np

from fes_data.common import read_csv_gz, repo_path, run_info, write_json
from fes_service import nafnet_route as NR

OUT = "data/manifests/phase5_rapid/r5_8"
KEY = "l_dot_c_max"


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    F = read_csv_gz(repo_path(f"{OUT}/local_features.csv.gz"))
    L = json.loads(repo_path(f"{OUT}/labels.json").read_text(encoding="utf-8"))
    lab = lambda r: L.get(f"{r['mode']}/{r['name']}", {}).get("label")  # noqa: E731
    P = [r for r in F if r["guard_pass"] == "True"]                        # the local guard only sees these
    sep = []
    for m in ("fc", "fi"):
        leg_max = max(fl(r[KEY]) for r in P if r["mode"] == m and lab(r) == "LEGITIMATE")
        sep += [fl(r[KEY]) for r in P if r["mode"] == m and lab(r) == "LOCALIZED_ARTIFACT" and fl(r[KEY]) > leg_max]
    limit = round(0.9 * min(sep), 2)
    blocked = lambda r: fl(r[KEY]) > limit  # noqa: E731
    S = {"signal": KEY, "limit": limit, "rule": "0.9 x min separable DEV LOCALIZED_ARTIFACT value", "modes": {}}
    for m in ("fc", "fi"):
        A = [r for r in F if r["mode"] == m]
        Pm = [r for r in P if r["mode"] == m]
        la = [r for r in Pm if lab(r) == "LOCALIZED_ARTIFACT"]
        lg = [r for r in Pm if lab(r) == "LEGITIMATE"]
        kept = [r for r in Pm if not blocked(r)]
        ds = lambda rs: round(float(np.nanmean([fl(r["d_sface"]) for r in rs])), 4)  # noqa: E731
        S["modes"][m] = {
            "nafnet_outputs": len(A), "global_guard_pass": len(Pm),
            "labels": {k: sum(lab(r) == k for r in A) for k in ("LEGITIMATE", "DESTRUCTIVE", "LOCALIZED_ARTIFACT", "UNCERTAIN")},
            "localized_recall": f"{sum(map(blocked, la))}/{len(la)}",
            "localized_missed": [{"name": r["name"], KEY: round(fl(r[KEY]), 2)} for r in la if not blocked(r)],
            "legit_labelled_blocked": f"{sum(map(blocked, lg))}/{len(lg)}",
            "uncertain_blocked": f"{sum(blocked(r) for r in Pm if lab(r) == 'UNCERTAIN')}/{sum(lab(r) == 'UNCERTAIN' for r in Pm)}",
            "population_block_rate_pct": round(100 * sum(map(blocked, Pm)) / len(Pm), 3),
            "d_sface_global_pass": ds(Pm), "d_sface_after_local": ds(kept),
            "clearly_harmed_after_local": sum(fl(r["d_sface"]) < -.02 and fl(r["d_dlib"]) < -.02 for r in kept)}
    # runtime (bounded): time the feature on 200 DEV chips
    import cv2
    t, n = 0.0, 0
    for r in P[:200]:
        out = cv2.imread(str(repo_path(f"experiments/phase5_rapid/outputs/r5_8/corpus/{r['mode']}/{r['name']}.png")))
        inp = cv2.imread(str(repo_path(f"Dataset-and-code/archive/input/val/{r['name']}") if r["mode"] == "fc" else
                             repo_path(f"experiments/phase5_rapid/outputs/r5_8/corpus/fi_chip/{r['name']}.png")))
        t0 = time.perf_counter()
        NR.local_features(inp, out)
        t += time.perf_counter() - t0
        n += 1
    S["runtime_ms_per_chip"] = round(1000 * t / n, 2)
    cfg = {"version": NR.LOCAL_VERSION, "win": NR.LOCAL_WIN, "limits": {KEY: limit},
           "calibration": {"data": "DEV only: archive val split (16,545 NAFNet outputs, both modes); visual labels",
                           "rule": S["rule"], "known_dev_misses": sum(len(v["localized_missed"]) for v in S["modes"].values())}}
    write_json(repo_path(f"{OUT}/local_calibration.json"), run_info({"summary": S, "config": cfg}))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
