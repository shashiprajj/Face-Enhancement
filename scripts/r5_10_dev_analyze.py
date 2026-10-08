"""R5-10 — DEV analysis + boundary selection (applies the pre-registered data/manifests/phase5_rapid/r5_10/protocol.json).

    .venv/Scripts/python.exe scripts/r5_10_dev_analyze.py          (numpy only)

Inputs (DEV only): r5_8/corpus.csv.gz + r5_8/local_features.csv.gz (R5-8-eligible archive val outcomes),
r5_7/val_guard.csv.gz (face-chip features, image IOD), r5_10/dev_{archive,clean,p4}.csv.gz (scripts/r5_10_dev_collect.py).
Writes data/manifests/phase5_rapid/r5_10/{dev_analysis.json, dev_tables.md}.
"""
from __future__ import annotations

import json
import math

import numpy as np

from fes_data.common import read_csv_gz, repo_path, write_json

OUT = "data/manifests/phase5_rapid/r5_10"
PROTO = json.loads(repo_path(f"{OUT}/protocol.json").read_text(encoding="utf-8"))
GRID = PROTO["grid"]
OLD, CLEAN = GRID[0], GRID[-1]
LOCAL_LIMIT = 24.45
IOD_BUCKETS = [(35.347, 40.0), (40.0, 45.0), (45.0, 71.44), (71.44, 1e9)]
NAMED = [("A_VERY_STRONG", 0.0, 0.0015), ("B_STRONG", 0.0015, OLD), ("C_MODERATE", OLD, 0.0040),
         ("D_MILD", 0.0040, CLEAN), ("E_NOT_BLUR", CLEAN, 1.0)]
NON_BLUR = ("jpeg", "noise", "lowres", "lowcontrast", "underexposure")


def fl(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def tb(v):
    return str(v) in ("True", "true", "1")


def boot_lo(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5))


def attempt(name, mode, lapd1, iod, ds, dd, gpass, lpass, src, **extra):
    returned = gpass and lpass
    return {"name": name, "mode": mode, "lapd1": lapd1, "iod": iod, "ds_raw": ds, "dd_raw": dd,
            "g_block": not gpass, "l_block": gpass and not lpass, "returned": returned,
            "ds": ds if returned else 0.0, "dd": dd if returned else 0.0, "src": src, **extra}


def load_archive():
    vg = {r["name"]: r for r in read_csv_gz(repo_path("data/manifests/phase5_rapid/r5_7/val_guard.csv.gz"))}
    loc = {(r["mode"], r["name"]): fl(r["l_dot_c_max"])
           for r in read_csv_gz(repo_path("data/manifests/phase5_rapid/r5_8/local_features.csv.gz"))}
    rows, inputs = [], {"fc": 0, "fi": 0}
    for r in read_csv_gz(repo_path("data/manifests/phase5_rapid/r5_8/corpus.csv.gz")):
        v = vg[r["name"]]
        iod = fl(v["fi_iod_img"]) if v["fi_iod_img"] else fl(v["fc_iod_px"])
        for m in ("fc", "fi"):
            if r.get(f"{m}_reason"):
                inputs[m] += 1
            if not tb(r.get(f"{m}_eligible")) or r.get(f"{m}_guard_pass", "") == "":
                continue
            ld = loc.get((m, r["name"]), float("nan"))
            rows.append(attempt(r["name"], m, fl(r[f"{m}_lapd1"]), iod, fl(r.get(f"{m}_d_sface")),
                                fl(r.get(f"{m}_d_dlib")), tb(r[f"{m}_guard_pass"]),
                                bool(ld <= LOCAL_LIMIT) if math.isfinite(ld) else False, "r5_8_corpus",
                                noise=fl(v["fc_noise_sigma"]), blockiness=fl(v["fc_blockiness"]),
                                luma=fl(v["fc_luma_mean"]), is_gray=tb(v["fc_is_gray"])))
    for r in read_csv_gz(repo_path(f"{OUT}/dev_archive.csv.gz")):
        v = vg[r["name"]]
        iod = fl(v["fi_iod_img"]) if v["fi_iod_img"] else fl(v["fc_iod_px"])
        for m in ("fc", "fi"):
            if r.get(f"{m}_guard_pass", "") == "":
                continue
            rows.append(attempt(r["name"], m, fl(r[f"{m}_lapd1_c"]), iod, fl(r.get(f"{m}_d_sface")),
                                fl(r.get(f"{m}_d_dlib")), tb(r[f"{m}_guard_pass"]), tb(r[f"{m}_local_pass"]),
                                "r5_10_dev", noise=fl(r[f"{m}_noise_sigma"]), blockiness=fl(r[f"{m}_blockiness"]),
                                luma=fl(r[f"{m}_luma_mean"]), is_gray=tb(r[f"{m}_is_gray"])))
    return rows, inputs


def stats(rows):
    ok = [r for r in rows if math.isfinite(r["ds_raw"]) and math.isfinite(r["dd_raw"])]
    n = len(rows)
    ds, dd = np.array([r["ds"] for r in ok]), np.array([r["dd"] for r in ok])
    imp = int(sum((a > .02) and (b > .02) for a, b in zip(ds, dd)))
    harm = int(sum((a < -.02) and (b < -.02) for a, b in zip(ds, dd)))
    k = len(ok)
    out = {"attempts": n, "with_identity": k, "returned": sum(r["returned"] for r in rows),
           "global_blocks": sum(r["g_block"] for r in rows), "local_blocks": sum(r["l_block"] for r in rows)}
    if k == 0:
        return out
    out.update({
        "mean_dsface": round(float(ds.mean()), 4), "median_dsface": round(float(np.median(ds)), 4),
        "dsface_lo95": round(boot_lo(ds), 4),
        "mean_ddlib": round(float(dd.mean()), 4), "median_ddlib": round(float(np.median(dd)), 4),
        "improved": imp, "harmed": harm, "benefit_rate_pct": round(100 * imp / k, 2),
        "harm_rate_pct": round(100 * harm / k, 2), "zero_or_neg_pct": round(100 * float((ds <= 0).mean()), 2),
        "block_rate_pct": round(100 * (out["global_blocks"] + out["local_blocks"]) / max(n, 1), 2)})
    return out


def band(rows, lo, hi, closed_hi=True):
    return [r for r in rows if lo < r["lapd1"] <= hi] if closed_hi else [r for r in rows if lo < r["lapd1"] < hi]


def main():
    rows, inputs = load_archive()
    res = {"protocol": PROTO["selection"], "inputs_per_mode": inputs, "modes": {}}
    md = ["# R5-10 DEV tables (archive val; outcomes after both guards)\n"]
    verdicts = {}
    for m in ("fc", "fi"):
        R = [r for r in rows if r["mode"] == m]
        ref = stats(band(R, 0.0, OLD))
        mres = {"reference_current_zone": ref, "named_bands": {}, "subbands": [], "cumulative": [], "iod": {}}
        for nm, lo, hi in NAMED:
            mres["named_bands"][nm] = stats(band(R, lo, hi, hi != CLEAN))
        ok_so_far = True
        for lo, hi in zip(GRID[:-1], GRID[1:]):
            s = stats(band(R, lo, hi, hi != CLEAN))
            chk = {"R1": s["attempts"] >= 100,
                   "R2": s.get("dsface_lo95", -1) > 0 and s.get("mean_ddlib", -1) > 0,
                   "R3": s.get("harm_rate_pct", 999) <= ref["harm_rate_pct"] + 1.0,
                   "R4": s.get("improved", 0) >= 3 * s.get("harmed", 0),
                   "R5": s.get("block_rate_pct", 999) <= ref["block_rate_pct"] + 2.0}
            ok_so_far = ok_so_far and all(chk.values())
            mres["subbands"].append({"lo": lo, "hi": hi, **s, "checks": chk, "pass": all(chk.values()),
                                     "contiguous_pass": ok_so_far})
            cum = stats(band(R, 0.0, hi, hi != CLEAN))
            cum["eligible_pct_of_inputs"] = round(100 * cum["attempts"] / inputs[m], 2)
            mres["cumulative"].append({"t": hi, **cum})
        mres["cumulative"].insert(0, {"t": OLD, **ref, "eligible_pct_of_inputs":
                                      round(100 * ref["attempts"] / inputs[m], 2)})
        beyond = stats([r for r in R if r["lapd1"] >= CLEAN])
        mres["beyond_clean_lapd1_lo (diagnostic only)"] = beyond
        for lo, hi in [(OLD, 0.0040), (0.0040, CLEAN)]:
            Z = band(R, lo, hi, hi != CLEAN)
            mres["iod"][f"{lo}-{hi}"] = {f"{a}-{b}": stats([r for r in Z if a <= r["iod"] < b]) for a, b in IOD_BUCKETS}
        mres["iod"][f"0-{OLD}"] = {f"{a}-{b}": stats([r for r in band(R, 0, OLD) if a <= r["iod"] < b])
                                   for a, b in IOD_BUCKETS}
        res["modes"][m] = mres
        verdicts[m] = [s["hi"] for s in mres["subbands"] if s["contiguous_pass"]]
    # ---------------------------------------------------------------- clean protection (sharp GT)
    clean = read_csv_gz(repo_path(f"{OUT}/dev_clean.csv.gz"))
    pass_other = [r for r in clean if tb(r["fc_relaxed_eligible"])]
    cres = {"gt_images": len(clean), "pass_every_other_gate": len(pass_other), "admission": []}
    for t in GRID:
        adm = [r for r in pass_other if fl(r["fc_lapd1_c"]) <= t]
        ds = [fl(r["fc_d_sface"]) for r in adm if r.get("fc_d_sface") not in ("", None)]
        ret = [r for r in adm if tb(r.get("fc_guard_pass")) and tb(r.get("fc_local_pass"))]
        cres["admission"].append({"t": t, "admitted": len(adm), "pct": round(100 * len(adm) / max(len(pass_other), 1), 2),
                                  "returned": len(ret),
                                  "mean_dsface_raw": round(float(np.mean(ds)), 4) if ds else None,
                                  "R6_pass": 100 * len(adm) / max(len(pass_other), 1) <= 2.0})
    lap = np.array([fl(r["fc_lapd1_c"]) for r in pass_other])
    cres["gt_lapd1_quantiles"] = {q: round(float(np.nanpercentile(lap, q)), 6) for q in (0.5, 1, 2, 5, 10, 25, 50)}
    res["clean"] = cres
    # ---------------------------------------------------------------- p4dev / r5gap
    p4 = read_csv_gz(repo_path(f"{OUT}/dev_p4.csv.gz"))
    pres = {"pairs": len(p4), "by_t": []}
    for t in GRID:
        new = [r for r in p4 if r["fc_reason"] == "NAFNET_REJECT_MILD_BLUR" and fl(r["fc_lapd1_c"]) <= t]
        cur = [r for r in p4 if r["fc_reason"] == "NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR"]
        ent = {"t": t, "current_eligible": len(cur), "newly_admitted": len(new), "by_profile": {}}
        for prof in sorted({r["profile"] for r in new}):
            A = [attempt(r["name"], "fc", fl(r["fc_lapd1_c"]), fl(r["fc_iod_px"]), fl(r.get("fc_d_sface")),
                         fl(r.get("fc_d_dlib")), tb(r["fc_guard_pass"]), tb(r["fc_local_pass"]), "p4")
                 for r in new if r["profile"] == prof]
            ent["by_profile"][prof] = stats(A)
        nb = [attempt(r["name"], "fc", fl(r["fc_lapd1_c"]), fl(r["fc_iod_px"]), fl(r.get("fc_d_sface")),
                      fl(r.get("fc_d_dlib")), tb(r["fc_guard_pass"]), tb(r["fc_local_pass"]), "p4")
              for r in new if r["profile"] in NON_BLUR]
        s = stats(nb)
        ent["non_blur_newly_admitted"] = s
        ent["R7_pass"] = (s["attempts"] == 0 or (s.get("mean_dsface", 0) >= 0 and s.get("harmed", 0) <= s.get("improved", 0)))
        pres["by_t"].append(ent)
    cur = [attempt(r["name"], "fc", fl(r["fc_lapd1_c"]), fl(r["fc_iod_px"]), fl(r.get("fc_d_sface")),
                   fl(r.get("fc_d_dlib")), tb(r["fc_guard_pass"]), tb(r["fc_local_pass"]), "p4")
           for r in p4 if r["fc_reason"] == "NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR"]
    pres["current_eligible_stats"] = stats(cur)
    import collections
    pres["reasons"] = dict(collections.Counter(r["fc_reason"] for r in p4).most_common())
    res["p4"] = pres
    # ---------------------------------------------------------------- selection
    cand = [t for t in GRID[1:] if t in verdicts["fc"] and t in verdicts["fi"]]
    sel = OLD
    for t in GRID[1:]:
        if t not in cand:
            break
        c6 = next(x for x in cres["admission"] if x["t"] == t)["R6_pass"]
        c7 = next(x for x in pres["by_t"] if x["t"] == t)["R7_pass"]
        if not (c6 and c7):
            break
        sel = t
    res["selected_blur_max"] = sel
    res["decision"] = "NO CHANGE" if sel == OLD else f"widen blur_max {OLD} -> {sel}"
    write_json(repo_path(f"{OUT}/dev_analysis.json"), res)
    # ---------------------------------------------------------------- markdown
    H = ("| band | attempts | returned | mean ΔSFace (lo95) | median ΔSFace | mean Δdlib | median Δdlib | improved | "
         "harmed | benefit % | harm % | ≤0 % | global blk | local blk |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def line(lbl, s):
        return (f"| {lbl} | {s['attempts']} | {s['returned']} | {s.get('mean_dsface')} ({s.get('dsface_lo95')}) | "
                f"{s.get('median_dsface')} | {s.get('mean_ddlib')} | {s.get('median_ddlib')} | {s.get('improved')} | "
                f"{s.get('harmed')} | {s.get('benefit_rate_pct')} | {s.get('harm_rate_pct')} | {s.get('zero_or_neg_pct')} | "
                f"{s['global_blocks']} | {s['local_blocks']} |")
    for m, mres in res["modes"].items():
        md.append(f"\n## {m} — named bands\n\n{H}")
        md += [line(k, v) for k, v in mres["named_bands"].items()]
        md.append(line("beyond clean_lapd1_lo (diag.)", mres["beyond_clean_lapd1_lo (diagnostic only)"]))
        md.append(f"\n## {m} — marginal sub-bands (pre-registered checks)\n\n{H[:-1]}| checks |\n" + "|---" * 15 + "|")
        for s in mres["subbands"]:
            md.append(line(f"({s['lo']}, {s['hi']}]", s)[:-1] + f"| {' '.join(k + ('✓' if v else '✗') for k, v in s['checks'].items())} |")
        md.append(f"\n## {m} — cumulative boundary t\n\n| t | eligible % of inputs | attempts | mean ΔSFace | mean Δdlib | improved | harmed | harm % |\n|---|---|---|---|---|---|---|---|")
        for c in mres["cumulative"]:
            md.append(f"| {c['t']} | {c['eligible_pct_of_inputs']} | {c['attempts']} | {c.get('mean_dsface')} | "
                      f"{c.get('mean_ddlib')} | {c.get('improved')} | {c.get('harmed')} | {c.get('harm_rate_pct')} |")
        md.append(f"\n## {m} — face IOD × blur zone\n\n{H}")
        for z, bk in mres["iod"].items():
            md += [line(f"{z} · IOD {b}", s) for b, s in bk.items() if s["attempts"]]
    md.append("\n## Clean protection (sharp archive val GT, face_chip)\n\n| t | admitted | % of clean faces passing other gates | returned | mean ΔSFace (raw) | R6 |\n|---|---|---|---|---|---|")
    for c in cres["admission"]:
        md.append(f"| {c['t']} | {c['admitted']} | {c['pct']} | {c['returned']} | {c['mean_dsface_raw']} | {'✓' if c['R6_pass'] else '✗'} |")
    md.append(f"\nclean GT lapd1 quantiles: {cres['gt_lapd1_quantiles']}")
    md.append("\n## p4dev + r5gap (newly admitted, by family)\n")
    for e in pres["by_t"][1:]:
        md.append(f"- t={e['t']}: newly admitted {e['newly_admitted']} · non-blur {e['non_blur_newly_admitted']} · R7 {'✓' if e['R7_pass'] else '✗'}")
        for prof, s in e["by_profile"].items():
            md.append(f"  - {prof}: n={s['attempts']} ret={s['returned']} ΔSFace={s.get('mean_dsface')} Δdlib={s.get('mean_ddlib')} imp/harm={s.get('improved')}/{s.get('harmed')}")
    md.append(f"\n**Selected blur_max: {sel}** ({res['decision']})")
    repo_path(f"{OUT}/dev_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"selected blur_max: {sel} ({res['decision']}); tables: {OUT}/dev_tables.md")


if __name__ == "__main__":
    main()
