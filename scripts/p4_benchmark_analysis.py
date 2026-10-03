"""P4-T4 — multi-evaluator benchmark analysis of a frozen population (Phase 3 harness outputs, unchanged).

    python scripts/p4_benchmark_analysis.py --population main      (SYNTHETIC test; Phase 3 outputs)
    python scripts/p4_benchmark_analysis.py --population p4dev     (SYNTHETIC dev)

Per configuration vs B0 (paired, same pairs): ΔID per evaluator (cluster-bootstrap CI over references) pooled / by
family / by severity; per-pair multi-evaluator verdicts; TAR at the split's FAR 1e-2 point, flips and rescues per
evaluator; outputs closer to another identity; evaluator agreement (Cohen's κ on verdicts, Spearman of Δ);
detection-contract failures on inputs (non-oracle) and YuNet detection on outputs. No ranking, no thresholds beyond
the measured δ and FAR operating points.

Writes data/manifests/phase4/results/<population>/{summary.json, verdicts.csv.gz, tables.md}.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from fes_bench import p4eval as E
from fes_data.common import repo_path, run_info, sha256_file, write_csv_gz, write_json

MULTI = ["CONFIRMED_BENEFIT", "PARTIAL_BENEFIT", "NEUTRAL", "PARTIAL_HARM", "CONFIRMED_HARM", "DISAGREE"]


def kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def spearman(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True, choices=list(E.POPS))
    a = ap.parse_args()
    pairs = E.load(a.population)
    configs = E.available_configs(pairs)
    delta, thr = E.deltas_from_validation(), E.thresholds(a.population)
    P = list(pairs.values())
    summary = {"population": a.population, "data_kind": "SYNTHETIC", "split": E.POPS[a.population]["split"],
               "delta": delta, "thr_far_1e-2": thr, "n_pairs": len(P), "n_refs": len({p.ref_id for p in P}),
               "configs": configs, "by_config": {}}
    vrows = []
    groups = {"all": lambda p: True}
    groups |= {f"family={f}": (lambda f: lambda p: p.profile == f)(f) for f in E.FAMILIES}
    groups |= {f"severity={s}": (lambda s: lambda p: p.severity == s)(s) for s in E.SEV_ORDER}

    for c in configs:
        if c == "B0":
            continue
        res = {}
        for gname, g in groups.items():
            sel = [p for p in P if g(p)]
            row = {"n": len(sel)}
            for e in E.EVALUATORS:
                by_ref = defaultdict(list)
                for p in sel:
                    by_ref[p.ref_id].append(p.cos[(c, e)] - p.cos[("B0", e)])
                est, lo, hi = E.cluster_boot(by_ref)
                row[f"d_{e}"] = [round(est, 5), round(lo, 5), round(hi, 5), E.tie_mark(lo, hi)]
            mv = Counter(E.multi_verdict(E.verdict(p.cos[(c, "E01")] - p.cos[("B0", "E01")], delta["E01"]),
                                         E.verdict(p.cos[(c, "E07")] - p.cos[("B0", "E07")], delta["E07"])) for p in sel)
            row["verdicts"] = {k: mv.get(k, 0) for k in MULTI}
            for e in E.EVALUATORS:
                b0 = [p.cos[("B0", e)] >= thr[e] for p in sel]
                cc = [p.cos[(c, e)] >= thr[e] for p in sel]
                row[f"TAR_{e}"] = [round(float(np.mean(b0)), 4), round(float(np.mean(cc)), 4)]
                row[f"flips_{e}"] = int(sum(x and not y for x, y in zip(b0, cc)))
                row[f"rescues_{e}"] = int(sum(y and not x for x, y in zip(b0, cc)))
                row[f"closer_to_impostor_{e}"] = int(sum(p.imp_max[(c, e)] > p.cos[(c, e)] for p in sel))
            row["closer_to_impostor_both"] = int(sum(all(p.imp_max[(c, e)] > p.cos[(c, e)] for e in E.EVALUATORS) for p in sel))
            res[gname] = row
        v1 = [E.verdict(p.cos[(c, "E01")] - p.cos[("B0", "E01")], delta["E01"]) for p in P]
        v7 = [E.verdict(p.cos[(c, "E07")] - p.cos[("B0", "E07")], delta["E07"]) for p in P]
        res["agreement"] = {"kappa_verdicts": round(kappa(v1, v7), 4),
                            "spearman_delta": round(spearman([p.cos[(c, "E01")] - p.cos[("B0", "E01")] for p in P],
                                                             [p.cos[(c, "E07")] - p.cos[("B0", "E07")] for p in P]), 4),
                            "verdict_crosstab": {f"{x}|{y}": n for (x, y), n in sorted(Counter(zip(v1, v7)).items())}}
        res["output_detected_rate"] = round(float(np.mean([p.det_out.get(c, False) for p in P])), 4)
        summary["by_config"][c] = res
        for p, x, y in zip(P, v1, v7):
            vrows.append({"pair_id": p.pair_id, "ref_id": p.ref_id, "cell": p.cell, "config_id": c,
                          "d_E01": round(p.cos[(c, "E01")] - p.cos[("B0", "E01")], 5),
                          "d_E07": round(p.cos[(c, "E07")] - p.cos[("B0", "E07")], 5),
                          "v_E01": x, "v_E07": y, "multi": E.multi_verdict(x, y)})

    # non-oracle detection contract on inputs (what a production pipeline would face)
    det = defaultdict(list)
    for p in P:
        det[p.cell].append(p.feat.get("det_status") == "OK")
    summary["input_detection_ok_rate_by_cell"] = {k: round(float(np.mean(v)), 4) for k, v in sorted(det.items())}
    summary["b0_tar_by_input_detection"] = {
        e: {s: round(float(np.mean([p.cos[("B0", e)] >= thr[e] for p in P if (p.feat.get("det_status") == "OK") == ok])), 4)
            for s, ok in (("detected", True), ("not_detected", False))} for e in E.EVALUATORS}

    out = repo_path(f"data/manifests/phase4/results/{a.population}")
    write_csv_gz(out / "verdicts.csv.gz", vrows, list(vrows[0]))
    summary["verdicts_sha256"] = sha256_file(out / "verdicts.csv.gz")
    summary["run"] = run_info()
    write_json(out / "summary.json", summary)
    (out / "tables.md").write_text(render(summary), encoding="utf-8")
    print(f"{a.population}: {len(configs)} configs, {len(P)} pairs → {out}")


def render(s: dict) -> str:
    L = [f"# P4-T4 multi-evaluator results — `{s['population']}` ({s['data_kind']}, {s['split']} split)", "",
         f"n = {s['n_pairs']} pairs, {s['n_refs']} references. δ: E01 {s['delta']['E01']}, E07 {s['delta']['E07']}. "
         f"FAR 1e-2 thresholds: E01 {s['thr_far_1e-2']['E01']:.4f}, E07 {s['thr_far_1e-2']['E07']:.4f}. "
         "▲/▼/≈: 95 % cluster-bootstrap CI above / below / containing 0. Scores of the two evaluators are never pooled.", ""]
    cs = list(s["by_config"])
    L += ["## Pooled", "", "| Config | ΔE01 [CI] | ΔE07 [CI] | conf. benefit | conf. harm | disagree | TAR E01 B0→c | TAR E07 B0→c | flips/rescues E01 | flips/rescues E07 | closer-to-impostor (both) | κ verdicts |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cs:
        r, ag = s["by_config"][c]["all"], s["by_config"][c]["agreement"]
        f = lambda d: f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] {d[3]}"  # noqa: E731
        L.append(f"| {c} | {f(r['d_E01'])} | {f(r['d_E07'])} | {r['verdicts']['CONFIRMED_BENEFIT']} | "
                 f"{r['verdicts']['CONFIRMED_HARM']} | {r['verdicts']['DISAGREE']} | {r['TAR_E01'][0]:.3f}→{r['TAR_E01'][1]:.3f} | "
                 f"{r['TAR_E07'][0]:.3f}→{r['TAR_E07'][1]:.3f} | {r['flips_E01']}/{r['rescues_E01']} | "
                 f"{r['flips_E07']}/{r['rescues_E07']} | {r['closer_to_impostor_both']} | {ag['kappa_verdicts']} |")
    for kind, keys in (("family", E.FAMILIES), ("severity", E.SEV_ORDER)):
        L += ["", f"## ΔID by {kind} (E01 mark / E07 mark; confirmed benefit − confirmed harm count)", "",
              "| Config | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for c in cs:
            cells = []
            for k in keys:
                r = s["by_config"][c][f"{kind}={k}"]
                cells.append(f"{r['d_E01'][3]}/{r['d_E07'][3]} {r['verdicts']['CONFIRMED_BENEFIT']}−{r['verdicts']['CONFIRMED_HARM']}")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## Detection contract on the degraded input (non-oracle)", "",
          "| Cell | OK rate |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in s["input_detection_ok_rate_by_cell"].items()]
    L += ["", f"B0 TAR by input detection: {s['b0_tar_by_input_detection']}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
