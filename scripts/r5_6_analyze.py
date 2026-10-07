"""R5-6 — analysis of the service-level runs (records from scripts/r5_6_service_eval.py). Evaluation only.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_6_analyze.py          (numpy only; reads tracked manifests)

Identity methodology = R5-ARCHIVE (scripts/r5_archive_eval.py analyze): per-pair cosine to the sharp GT with GT
landmarks (SFace E01 / dlib E07), Δ vs B0 (input vs GT, stage-1), bootstrap 95 % CI (2,000 resamples, seed 0),
"clearly improved" / "clearly harmed" = both evaluators Δ > +0.02 / < −0.02, artefact = output chroma p99 to GT exceeds
the input's by > 10. Whole-policy Δ: ENHANCED pairs use the service output, every other pair keeps B0 (Δ = 0).
Matched comparisons: the R5-ARCHIVE stage-2 sample (300 test pairs, seed 0) with its always-on N1 (NAFNet) and A4
outputs, restricted to the pairs the service routed to NAFNet.
ztest verification: FAR-1e-2 thresholds from clean GT-vs-GT impostor pairs (100 identities), TAR and false accepts.
Writes data/manifests/phase5_rapid/r5_6/{summary.json, tables.md}.
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import Counter

import numpy as np

from fes_data.common import read_csv_gz, repo_path, run_info, write_json

OUT = "data/manifests/phase5_rapid/r5_6"
A_OUT = "data/manifests/phase5_rapid/archive"
A_IMG = "experiments/phase5_rapid/outputs/archive"
IMG = "experiments/phase5_rapid/outputs/r5_6"
RUNS = ("test_face_chip", "test_full_image", "ztest_face_chip", "ztest_full_image", "real_full_image")
NAF, A4 = "NAFNET_GOPRO", "A4"


def f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def mean(xs, d=4):
    xs = [x for x in xs if not math.isnan(x)]
    return round(float(np.mean(xs)), d) if xs else None


def med(xs, d=3):
    xs = [x for x in xs if not math.isnan(x)]
    return round(float(np.median(xs)), d) if xs else None


def q(xs, p, d=3):
    xs = [x for x in xs if not math.isnan(x)]
    return round(float(np.quantile(xs, p)), d) if xs else None


def boot_ci(d, n=2000, seed=0):
    d = np.array([x for x in d if not math.isnan(x)])
    if len(d) < 5:
        return None
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def load(name):
    p = repo_path(f"{OUT}/{name}.csv.gz")
    return read_csv_gz(p) if p.exists() else None


def routing(rows) -> dict:
    st = Counter(r["status"] for r in rows)
    model = Counter(r["model"] or "none" for r in rows)
    used = Counter(r["used"] for r in rows)
    fb = Counter(r["fallback"] for r in rows if r["fallback"])
    unsafe = sum(r["decision"] == "UNSAFE_OR_UNCERTAIN" for r in rows)
    elig = sum(r["nafnet_eligible"] == "True" for r in rows)
    naf_exec = sum(r["model"] == NAF for r in rows)
    lat_all = [f(r["client_s"]) for r in rows]
    clean = [r for r in rows if r.get("client_run", "") != "overlap_window"]
    return {"n": len(rows), "http": dict(Counter(r["http"] for r in rows)), "status": dict(st),
            "success_rate_pct": round(100 * sum(r["http"] == "200" for r in rows) / len(rows), 2),
            "output_valid_pct": round(100 * sum(r["output_valid"] == "True" for r in rows) / len(rows), 2),
            "output_check": dict(Counter(r["output_check"] for r in rows)),
            "nafnet_eligible": elig, "nafnet_executed": naf_exec,
            "nafnet_enhanced": used.get(NAF, 0), "a4_executed": model.get("A4_SWINIR_L", 0),
            "a4_enhanced": used.get(A4, 0), "original": st.get("ORIGINAL", 0), "unsafe_uncertain": unsafe,
            "rejected": st.get("REJECTED", 0), "artifact_fail": sum(r["artifact_check"] == "FAIL" for r in rows),
            "artifact_pass": sum(r["artifact_check"] == "PASS" for r in rows), "fallback": dict(fb),
            "reason": dict(Counter(r["reason"].split(":")[0] for r in rows).most_common()),
            "a4_reasons": dict(Counter(r["reason"] for r in rows if r["model"] == "A4_SWINIR_L")),
            "latency_s": {"client_median_all": med([f(r["client_s"]) for r in clean]),
                          "client_p95_all": q([f(r["client_s"]) for r in clean], 0.95),
                          "client_median_original": med([f(r["client_s"]) for r in clean if r["status"] == "ORIGINAL"]),
                          "server_ms_median_original": med([f(r["ms_total"]) for r in clean if r["status"] == "ORIGINAL"], 1),
                          "nafnet_infer_median": med([f(r["nafnet_infer_s"]) for r in clean if r["model"] == NAF]),
                          "nafnet_infer_p95": q([f(r["nafnet_infer_s"]) for r in clean if r["model"] == NAF], 0.95),
                          "nafnet_infer_max": q([f(r["nafnet_infer_s"]) for r in clean if r["model"] == NAF], 1.0),
                          "server_ms_median_nafnet": med([f(r["ms_total"]) for r in clean if r["used"] == NAF], 1),
                          "a4_request_s": [f(r["client_s"]) for r in rows if r["model"] == "A4_SWINIR_L"],
                          "n_clean": len(clean), "n_overlap_window": len(rows) - len(clean),
                          "client_s_max_all": q(lat_all, 1.0)},
            "artifact_metrics_pass": {"chroma_p99_median": med([f(r.get("art_chroma_p99")) for r in rows
                                                                if r["artifact_check"] == "PASS"]),
                                      "chroma_p99_max": q([f(r.get("art_chroma_p99")) for r in rows
                                                           if r["artifact_check"] == "PASS"], 1.0),
                                      "hf_color_ratio_max": q([f(r.get("art_hf_color_ratio")) for r in rows
                                                               if r["artifact_check"] == "PASS"], 1.0)},
            "artifact_fail_reasons": dict(Counter(r["artifact_reasons"] for r in rows if r["artifact_check"] == "FAIL"))}


def identity(name, rows, base) -> dict | None:
    p = repo_path(f"{OUT}/{name}_identity.csv.gz")
    ids = {r["input_id"]: r for r in read_csv_gz(p)} if p.exists() else {}
    enh = [r for r in rows if r["status"] == "ENHANCED"]
    if not enh:
        return {"n_enhanced": 0}
    ok = [r["input_id"] for r in enh if ids.get(r["input_id"], {}).get("id_status") == "OK"]
    d_s = [f(ids[i]["id_sface"]) - f(base[i]["id_sface_B0"]) for i in ok]
    d_d = [f(ids[i]["id_dlib"]) - f(base[i]["id_dlib_B0"]) for i in ok]
    allids = [r["input_id"] for r in rows]
    okset = set(ok)
    pos = {i: k for k, i in enumerate(ok)}
    pol_s = [d_s[pos[i]] if i in okset else 0.0 for i in allids]
    pol_d = [d_d[pos[i]] if i in okset else 0.0 for i in allids]
    by_used = {}
    for u in (NAF, A4):
        sel = [k for k, i in enumerate(ok) if ids[i]["used"] == u]
        if sel:
            ds, dd = [d_s[k] for k in sel], [d_d[k] for k in sel]
            by_used[u] = {"n": len(sel), "d_sface": mean(ds), "d_sface_ci95": boot_ci(ds), "d_dlib": mean(dd),
                          "d_dlib_ci95": boot_ci(dd),
                          "B0_sface": mean([f(base[ok[k]]["id_sface_B0"]) for k in sel]),
                          "out_sface": mean([f(ids[ok[k]]["id_sface"]) for k in sel]),
                          "clearly_improved_pct": round(100 * np.mean([a > 0.02 and b > 0.02 for a, b in zip(ds, dd)]), 1),
                          "clearly_harmed_pct": round(100 * np.mean([a < -0.02 and b < -0.02 for a, b in zip(ds, dd)]), 1),
                          "clearly_improved_n": int(sum(a > 0.02 and b > 0.02 for a, b in zip(ds, dd))),
                          "clearly_harmed_n": int(sum(a < -0.02 and b < -0.02 for a, b in zip(ds, dd))),
                          "artefact_pct": round(100 * np.mean([f(ids[ok[k]]["chroma_excess_vs_input"]) > 10 for k in sel]), 2),
                          "out_face_found_pct": round(100 * np.mean([ids[ok[k]]["out_face_found"] == "True" for k in sel]), 1),
                          "psnr_out": mean([f(ids[ok[k]]["psnr"]) for k in sel], 3),
                          "psnr_in": mean([f(base[ok[k]]["psnr_in"]) for k in sel], 3)}
    geo = [ids[i].get("geometry_verified") for i in ok if ids[i].get("geometry_verified", "") != ""]
    return {"n_enhanced": len(enh), "n_scored": len(ok),
            "not_scored": dict(Counter(ids.get(r["input_id"], {}).get("id_status", "missing") for r in enh
                                       if r["input_id"] not in okset)),
            "by_model": by_used,
            "policy_vs_original": {"n": len(allids), "d_sface": mean(pol_s, 5), "d_sface_ci95": boot_ci(pol_s),
                                   "d_dlib": mean(pol_d, 5), "d_dlib_ci95": boot_ci(pol_d),
                                   "B0_sface": mean([f(base[i]["id_sface_B0"]) for i in allids]),
                                   "policy_sface": mean([f(base[i]["id_sface_B0"]) + pol_s[k] for k, i in enumerate(allids)])},
            "full_image_geometry_verified": dict(Counter(geo)) if geo else None}


def matched(name, rows, base) -> dict:
    """Stage-2 sample (300 test, seed 0): N1 (always-on NAFNet) and A4 outputs from R5-ARCHIVE, on the pairs that the
    service routed to NAFNet vs everything else. Same sample definition as R5-ARCHIVE."""
    test = sorted(os.listdir(repo_path("Dataset-and-code/archive/input/test")))
    pick = sorted(random.Random(0).sample(test, 300))
    svc = {r["input_id"]: r for r in rows}
    out = {"sample_n": len(pick)}
    for m in ("N1", "A4"):
        p = repo_path(f"{A_OUT}/stage2_{m}.csv.gz")
        if not p.exists():
            continue
        R = {r["name"]: r for r in read_csv_gz(p) if r["split"] == "test" and r["status"] == "OK"}
        for grp, sel in (("service_nafnet", lambda r: r["used"] == NAF),
                         ("service_eligible_guard_fail", lambda r: r["nafnet_eligible"] == "True" and r["used"] != NAF),
                         ("service_not_eligible", lambda r: r["nafnet_eligible"] != "True")):
            its = [n for n in pick if f"test/{n}" in svc and sel(svc[f"test/{n}"]) and n in R]
            ds = [f(R[n]["id_sface"]) - f(base[f"test/{n}"]["id_sface_B0"]) for n in its]
            dd = [f(R[n]["id_dlib"]) - f(base[f"test/{n}"]["id_dlib_B0"]) for n in its]
            out[f"{m}_on_{grp}"] = {"n": len(its), "d_sface": mean(ds), "d_sface_ci95": boot_ci(ds),
                                    "d_dlib": mean(dd), "d_dlib_ci95": boot_ci(dd),
                                    "clearly_improved_n": int(sum(a > 0.02 and b > 0.02 for a, b in zip(ds, dd))),
                                    "clearly_harmed_n": int(sum(a < -0.02 and b < -0.02 for a, b in zip(ds, dd)))}
    # the service's own outputs on the same sample's NAFNet pairs (should equal N1 there: same model, same chip)
    p = repo_path(f"{OUT}/{name}_identity.csv.gz")
    if p.exists():
        ids = {r["input_id"]: r for r in read_csv_gz(p)}
        its = [n for n in pick if ids.get(f"test/{n}", {}).get("id_status") == "OK"
               and svc.get(f"test/{n}", {}).get("used") == NAF]
        ds = [f(ids[f"test/{n}"]["id_sface"]) - f(base[f"test/{n}"]["id_sface_B0"]) for n in its]
        out["service_on_service_nafnet"] = {"n": len(its), "d_sface": mean(ds), "d_sface_ci95": boot_ci(ds)}
    return out


def ztest_verification(rows, name) -> dict:
    """R5-ARCHIVE ztest protocol; policy = service output where ENHANCED, else B0."""
    z1 = np.load(repo_path(f"{A_IMG}/stage1_ztest_emb.npy"), allow_pickle=True).item()
    zs_p = repo_path(f"{IMG}/{name}/ztest_emb.npy")
    zs = np.load(zs_p, allow_pickle=True).item() if zs_p.exists() else {}
    names = sorted(n for n, e in z1.items() if e["gt"][0] is not None)
    out = {}
    for k, ev in ((0, "sface"), (1, "dlib")):
        Gm = np.stack([z1[n]["gt"][k] for n in names])
        thr = float(np.quantile((Gm @ Gm.T)[~np.eye(len(names), dtype=bool)], 0.99))
        res = {"threshold_far_1e-2": round(thr, 4)}
        b0_acc = {}
        for c in ("B0", "service"):
            E = {n: (zs[n]["out"][k] if c == "service" and n in zs else z1[n]["B0"][k]) for n in names}
            E = {n: e for n, e in E.items() if e is not None}
            acc = {n: float(E[n] @ Gm[i]) >= thr for i, n in enumerate(names) if n in E}
            fa = {n: int(max(E[n] @ Gm[j] for j, o in enumerate(names) if o != n) >= thr) for n in E}
            res[c] = {"TAR": round(float(np.mean(list(acc.values()))), 3), "n": len(acc),
                      "false_accept_any": int(sum(fa.values()))}
            if c == "B0":
                b0_acc, b0_fa = acc, fa
            else:
                res["flips_accept_to_reject"] = int(sum(b0_acc[n] and not acc[n] for n in acc))
                res["flips_reject_to_accept"] = int(sum(not b0_acc[n] and acc[n] for n in acc))
                res["new_false_accepts_vs_B0"] = int(sum(v and not b0_fa.get(n, 0) for n, v in fa.items()))
        out[ev] = res
    out["n_enhanced"] = len(zs)
    return out


def n1_artefacts_vs_service(rows, split: str) -> dict:
    """R5-ARCHIVE always-on NAFNet (N1) artefacts on <split> (chroma p99 to GT exceeds the input's by > 10, the
    established signal) → what did the service do with exactly those inputs?"""
    import cv2

    def cp99(a, b):
        A = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
        B = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
        return float(np.percentile(np.hypot(A[..., 1] - B[..., 1], A[..., 2] - B[..., 2]), 99))
    svc = {r["input_id"]: r for r in rows}
    d = repo_path(f"{A_IMG}/N1/{split}")
    if not d.exists():
        return {}
    art = []
    for p in sorted(d.glob("*.png")):
        n = p.name.replace(".png", ".jpg")
        inp = cv2.imread(str(repo_path(f"Dataset-and-code/archive/input/{split}/{n}")))
        gt = cv2.imread(str(repo_path(f"Dataset-and-code/archive/output/{split}/{n}")))
        ex = cp99(cv2.imread(str(p)), gt) - cp99(inp, gt)
        if ex > 10:
            art.append(f"{split}/{n}")
    return {"n1_outputs": len(list(d.glob("*.png"))), "n1_artefacts": len(art),
            "service_on_those": dict(Counter(f"{svc[i]['status']}:{svc[i]['reason'].split(':')[0]}" for i in art if i in svc)),
            "service_sent_to_nafnet": sum(svc[i]["model"] == NAF for i in art if i in svc), "ids": art}


def md_table(S) -> str:
    L = ["# R5-6 tables (generated by scripts/r5_6_analyze.py)\n"]
    L.append("| Run | N | HTTP 200 | NAFNet eligible | NAFNet executed | NAFNet output | A4 | ORIGINAL | UNSAFE | "
             "Artifact FAIL | Fallback | Output valid |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, R in S["routing"].items():
        L.append(f"| {n} | {R['n']} | {R['success_rate_pct']}% | {R['nafnet_eligible']} | {R['nafnet_executed']} | "
                 f"{R['nafnet_enhanced']} | {R['a4_executed']} | {R['original']} | {R['unsafe_uncertain']} | "
                 f"{R['artifact_fail']} | {sum(R['fallback'].values())} | {R['output_valid_pct']}% |")
    L.append("")
    for n, I in S.get("identity", {}).items():
        if not I or not I.get("by_model"):
            continue
        L.append(f"## Identity — {n}\n")
        L.append("| Model | n | B0 SFace | out SFace | ΔSFace (95% CI) | Δdlib (95% CI) | clearly improved | clearly harmed |"
                 " artefact | out face |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for m, b in I["by_model"].items():
            L.append(f"| {m} | {b['n']} | {b['B0_sface']} | {b['out_sface']} | {b['d_sface']} {b['d_sface_ci95']} | "
                     f"{b['d_dlib']} {b['d_dlib_ci95']} | {b['clearly_improved_n']} ({b['clearly_improved_pct']}%) | "
                     f"{b['clearly_harmed_n']} ({b['clearly_harmed_pct']}%) | {b['artefact_pct']}% | "
                     f"{b['out_face_found_pct']}% |")
        p = I["policy_vs_original"]
        L.append(f"\nWhole policy vs ORIGINAL (n={p['n']}): ΔSFace {p['d_sface']} {p['d_sface_ci95']} · Δdlib "
                 f"{p['d_dlib']} {p['d_dlib_ci95']} · SFace B0 {p['B0_sface']} → policy {p['policy_sface']}\n")
    return "\n".join(L) + "\n"


def main():
    s1 = read_csv_gz(repo_path(f"{A_OUT}/stage1.csv.gz"))
    base = {f"{r['split']}/{r['name']}": r for r in s1}
    S = {"routing": {}, "identity": {}}
    data = {n: load(n) for n in RUNS}
    for n, rows in data.items():
        if rows:
            S["routing"][n] = routing(rows)
            if n.startswith(("test", "ztest")):
                S["identity"][n] = identity(n, rows, base)
    for n in ("test_face_chip", "test_full_image"):
        if data.get(n):
            S[f"matched_{n}"] = matched(n, data[n], base)
    if data.get("test_face_chip") and data.get("test_full_image"):
        a = {r["input_id"]: r for r in data["test_face_chip"]}
        b = {r["input_id"]: r for r in data["test_full_image"]}
        both = sorted(set(a) & set(b))
        S["mode_transition"] = {"n": len(both),
                                "face_chip_reason_to_full_image_reason_for_fc_eligible": dict(Counter(
                                    b[i]["reason"].split(":")[0] for i in both if a[i]["nafnet_eligible"] == "True")
                                    .most_common()),
                                "eligible_both": sum(a[i]["nafnet_eligible"] == "True" and b[i]["nafnet_eligible"] == "True"
                                                     for i in both),
                                "eligible_fc_only": sum(a[i]["nafnet_eligible"] == "True" and b[i]["nafnet_eligible"] != "True"
                                                        for i in both),
                                "eligible_fi_only": sum(a[i]["nafnet_eligible"] != "True" and b[i]["nafnet_eligible"] == "True"
                                                        for i in both)}
        p = repo_path(f"{OUT}/test_full_image_identity.csv.gz")
        if p.exists():
            pads = [int(r["chip_pad_px"]) for r in read_csv_gz(p) if r.get("chip_pad_px", "") != ""]
            S["mode_transition"]["full_image_enhanced_chip_pad_px"] = {"n": len(pads), "zero": sum(x == 0 for x in pads),
                                                                       "median": med([float(x) for x in pads])}
    for n in ("ztest_face_chip", "ztest_full_image"):
        if data.get(n):
            S[f"verification_{n}"] = ztest_verification(data[n], n)
            S[f"n1_artefacts_{n}"] = n1_artefacts_vs_service(data[n], "ztest")
    for n in ("test_face_chip", "test_full_image"):
        if data.get(n):
            S[f"n1_artefacts_{n}"] = n1_artefacts_vs_service(data[n], "test")
    for n in RUNS:
        p = repo_path(f"{OUT}/{n}_guardcheck.csv.gz")
        if p.exists():
            g = read_csv_gz(p)
            S[f"guardcheck_{n}"] = {"n": len(g), "reproduced": sum(r["reproduced"] == "True" for r in g)}
    write_json(repo_path(f"{OUT}/summary.json"), run_info({"summary": S}))
    repo_path(f"{OUT}/tables.md").write_text(md_table(S), encoding="utf-8")
    print(json.dumps(S, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
