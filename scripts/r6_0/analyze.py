"""R6-0 — all report tables from the corpus (host; numpy only). Deterministic.

    python scripts/r6_0/analyze.py  ->  data/manifests/phase5_rapid/r6_0/analysis.json + analysis_tables.md

Threshold rule (fixed before evaluation, research only): every single-signal threshold is the 99th percentile of
that signal over the DEV GOOD cases (R5-8 labelled archive-val LEGITIMATE outputs, n=458), oriented so that "larger
= more suspicious". Thresholds never see non-DEV data; non-DEV numbers are therefore held-out.
FA = BAD case accepted; FR = GOOD case rejected. "stress" cases (halftone prints that the live route sends to A4,
never to NAFNet) are reported separately and excluded from FA/FR.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "data/manifests/phase5_rapid/r6_0"
Q = 99.0

# family -> [(signal, direction)] ; direction +1 = larger is worse, -1 = smaller is worse
FAMILIES = {
    "geometry": [("g_score_drop", 1), ("g_lm_disp", 1), ("g_bbox_iou", -1), ("g_face_out", -1)],
    "photometric": [("p_chan_shift", 1), ("p_luma_shift", 1), ("p_chroma_p99", 1), ("p_chroma_p999", 1),
                    ("p_chroma_local_max", 1), ("p_colour_invented", 1), ("p_extreme_frac", 1)],
    "texture": [("t_hf_luma_ratio", 1), ("t_hf_colour_ratio", 1), ("t_hf_colour_growth", 1), ("t_dot_c_max", 1),
                ("t_dot_l_max", 1), ("t_spec_peak", 1)],
    "local": [("l_changed_pct", 1), ("l_largest_cc", 1), ("l_top2_conc", 1), ("l_border_share", 1),
              ("l_win_max", 1), ("l_win_max_excess", 1), ("l_luma_win_max", 1)],
    "identity": [("i_sface_io", -1), ("i_dlib_io", -1)],
    "flip": [("f_luma_win_max", 1), ("f_chroma_win_max", 1), ("f_luma_p999", 1), ("f_chroma_p999", 1),
             ("f_dis_over_change", 1), ("f_mean", 1)],
    "run_tracer": [("run_dot_c_max", 1), ("run_hfcg_max", 1), ("run_dot_l_max", 1)],
}


def load_all():
    cs = corpus.load(True)
    runs = {}
    with gzip.open(M / "run_signals.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            runs.setdefault(r["id"], []).append({k: float(v) for k, v in r.items() if k not in ("id", "run")})
    base = json.loads((M / "baselines.json").read_text())
    for c in cs:
        rr = runs[c["id"]]
        s = c["sig"]["v0"]
        s["run_dot_c_max"] = max(x["dot_c"] for x in rr)
        s["run_hfcg_max"] = max(x["hfcg"] for x in rr)
        s["run_dot_l_max"] = max(x["dot_l"] for x in rr)
        c["runs_dot_c"] = [x["dot_c"] for x in rr]
        s["g_face_out"] = float(bool(s.get("g_face_out")))
        c["base"] = base[c["id"]]
    return cs


def val(c, k, d):
    v = c["sig"]["v0"].get(k)
    return None if v is None or v == "" else d * float(v)


def auc(g, b):
    g, b = np.asarray(g), np.asarray(b)
    if not len(g) or not len(b):
        return None
    return float((b[:, None] > g[None, :]).mean() + 0.5 * (b[:, None] == g[None, :]).mean())


def thresholds(cs):
    t = {}
    dev_good = [c for c in cs if c["src"] == "dev" and c["label"] == "GOOD"]
    for fam, sigs in FAMILIES.items():
        for k, d in sigs:
            v = [val(c, k, d) for c in dev_good if val(c, k, d) is not None]
            if v:
                t[k] = (float(np.percentile(v, Q)), d)
    return t


def flags(c, t, keys):
    """True = verifier FAIL (reject) because any listed signal exceeds its DEV threshold."""
    hit = [k for k in keys if k in t and val(c, k, t[k][1]) is not None and val(c, k, t[k][1]) > t[k][0]]
    return hit


def split(cs, sel):
    good = [c for c in cs if sel(c) and c["label"] == "GOOD"]
    bad = [c for c in cs if sel(c) and c["label"] == "BAD" and c["kind"] != "stress"]
    return good, bad


def rate(good, bad, reject):
    fr = [c["id"] for c in good if reject(c)]
    fa = [c["id"] for c in bad if not reject(c)]
    return {"n_good": len(good), "n_bad": len(bad), "fr": len(fr), "fr_rate": len(fr) / max(1, len(good)),
            "fa": len(fa), "fa_rate": len(fa) / max(1, len(bad)), "fa_ids": fa, "fr_ids": fr}


def main():
    cs = load_all()
    t = thresholds(cs)
    res = {"thresholds": {k: v[0] * v[1] for k, v in t.items()}, "counts": {}, "signals": [], "combos": {},
           "challenge": []}
    from collections import Counter
    res["counts"] = {"by_group_label": {f"{a}|{b}|{k}": n for (a, b, k), n in
                                        Counter((c["group"], c["label"], c["kind"]) for c in cs).items()},
                     "by_src_label": {f"{a}|{b}": n for (a, b), n in Counter((c["src"], c["label"]) for c in cs).items()}}
    G = [c for c in cs if c["label"] == "GOOD"]
    B = [c for c in cs if c["label"] == "BAD" and c["kind"] != "stress"]
    C = [c for c in cs if c["group"] == "C"]
    nd = lambda c: c["src"] != "dev"  # noqa: E731
    for fam, sigs in FAMILIES.items():
        for k, d in sigs:
            g = [val(c, k, d) for c in G if val(c, k, d) is not None]
            b = [val(c, k, d) for c in B if val(c, k, d) is not None]
            cc = [val(c, k, d) for c in C if val(c, k, d) is not None]
            if not g or not b:
                continue
            thr = t[k][0]
            gn, bn = split(cs, nd)
            gn = [c for c in gn if val(c, k, d) is not None]
            bn = [c for c in bn if val(c, k, d) is not None]
            res["signals"].append({
                "family": fam, "signal": k, "dir": d, "n_good": len(g), "n_bad": len(b),
                "good_med": float(np.median(g)) * d, "good_p5": float(np.percentile(g, 5)) * d,
                "good_p95": float(np.percentile(g, 95)) * d,
                "bad_med": float(np.median(b)) * d, "bad_min": float(min(b)) * d, "bad_max": float(max(b)) * d,
                "border_med": float(np.median(cc)) * d if cc else None,
                "border_p95": float(np.percentile(cc, 95)) * d if cc else None,
                "auc": auc(g, b), "thr": thr * d,
                "recall_all": float(np.mean(np.array(b) > thr)),
                "fr_nondev": float(np.mean([val(c, k, d) > thr for c in gn])) if gn else None,
                "recall_nondev": float(np.mean([val(c, k, d) > thr for c in bn])) if bn else None,
                "misses": [c["id"] for c in B if val(c, k, d) is not None and val(c, k, d) <= thr]})
    fam_keys = {f: [k for k, _ in s] for f, s in FAMILIES.items()}
    best = {f: max((s for s in res["signals"] if s["family"] == f), key=lambda s: s["auc"] or 0)["signal"]
            for f in FAMILIES if any(s["family"] == f for s in res["signals"])}
    res["best_signal_per_family"] = best
    combos = {
        "R5-6 live guard": lambda c: not c["base"]["r5_6_pass"],
        "R5-8 guard (global + l_dot_c_max)": lambda c: not c["base"]["r5_8_pass"],
        "photometric + local": lambda c: bool(flags(c, t, fam_keys["photometric"] + fam_keys["local"])),
        "texture + local": lambda c: bool(flags(c, t, fam_keys["texture"] + fam_keys["local"])),
        "geometry + identity": lambda c: bool(flags(c, t, fam_keys["geometry"] + fam_keys["identity"])),
        "photometric + texture + local": lambda c: bool(flags(c, t, fam_keys["photometric"] + fam_keys["texture"]
                                                              + fam_keys["local"])),
        "all 5 requested families": lambda c: bool(flags(c, t, sum((fam_keys[f] for f in
                                                   ("geometry", "photometric", "texture", "local", "identity")), []))),
        "best signal per family (5 requested)": lambda c: bool(flags(c, t, [best[f] for f in
                                                   ("geometry", "photometric", "texture", "local", "identity") if f in best])),
        "flip consistency (family F)": lambda c: bool(flags(c, t, fam_keys["flip"])),
        "run tracer: run_dot_c_max": lambda c: bool(flags(c, t, ["run_dot_c_max"])),
        "R5-8 guard + run tracer": lambda c: (not c["base"]["r5_8_pass"]) or bool(flags(c, t, ["run_dot_c_max"])),
    }
    for name, rej in combos.items():
        res["combos"][name] = {"dev": rate(*split(cs, lambda c: c["src"] == "dev"), rej),
                               "nondev": rate(*split(cs, nd), rej), "all": rate(G, B, rej),
                               "stress_rejected": [c["id"] for c in cs if c["kind"] == "stress" and rej(c)]}
    # R6-0 proposed pipeline, judged on the image it RETURNS (median of the 4 flip runs, luma-only):
    #   accepted BAD counts as a false accept unless its returned image was reviewed CLEAN
    #   (labels_final_output.json; an unreviewed accepted BAD counts as a false accept).
    fin = json.loads((M / "labels_final_output.json").read_text())
    T = t["run_dot_c_max"][0]
    k_unstable = lambda c: sum(x > T for x in c["runs_dot_c"])  # noqa: E731
    for name, rej in (("R6 strict: any unstable run -> ORIGINAL, else medluma", lambda c: k_unstable(c) >= 1),
                      ("R6 2-of-4: >=2 unstable runs -> ORIGINAL, else medluma", lambda c: k_unstable(c) >= 2)):
        e = {}
        for part, sel in (("dev", lambda c: c["src"] == "dev"), ("nondev", nd), ("all", lambda c: True)):
            good, bad = split(cs, sel)
            acc_bad = [c for c in bad if not rej(c)]
            vis = [c["id"] for c in acc_bad if fin.get(c["id"], {}).get("medluma") != "CLEAN"]
            fr = [c["id"] for c in good if rej(c)]
            e[part] = {"n_good": len(good), "n_bad": len(bad), "fr": len(fr), "fr_rate": len(fr) / max(1, len(good)),
                       "fr_ids": fr, "bad_accepted": len(acc_bad), "bad_accepted_ids": [c["id"] for c in acc_bad],
                       "fa": len(vis), "fa_rate": len(vis) / max(1, len(bad)), "fa_ids": vis,
                       "fa_detail": {i: fin.get(i, {}).get("medluma", "UNREVIEWED") for i in vis}}
        e["stress_rejected"] = [c["id"] for c in cs if c["kind"] == "stress" and rej(c)]
        res["combos"][name] = e
    named = ["test_r57", "test_r58", "test_r510"]
    for c in sorted(cs, key=lambda c: c["id"]):
        if c["label"] == "BAD" and (c["src"] in named or c["src"].startswith("real") or c["src"] == "p4"
                                    or c["src"] == "test_r56" or c["src"] == "dev"):
            res["challenge"].append({"id": c["id"], "kind": c["kind"], "profile": c["profile"],
                                     "r5_8": "DETECTED" if not c["base"]["r5_8_pass"] else "MISSED",
                                     "run_tracer": "DETECTED" if flags(c, t, ["run_dot_c_max"]) else "MISSED",
                                     "v0_dot_c": c["sig"]["v0"]["t_dot_c_max"],
                                     "run_dot_c_max": c["sig"]["v0"]["run_dot_c_max"],
                                     "n_unstable_runs": k_unstable(c),
                                     "r6_strict": ("REJECTED (original returned)" if k_unstable(c) >= 1 else
                                                   "ACCEPTED, returned image " + fin.get(c["id"], {}).get("medluma", "UNREVIEWED"))})
    (M / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: {kk: {x: v[kk][x] for x in ("fr", "fr_rate", "fa", "fa_rate")} for kk in ("dev", "nondev")}
                      for k, v in res["combos"].items()}, indent=0))


if __name__ == "__main__":
    main()
