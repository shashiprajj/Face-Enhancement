"""Summary tables of the benchmark from the manifests only (nothing is recomputed, nothing is estimated).

    python scripts/indian_bench/report.py [--pilot]      → reports/08_results[_pilot].md + .json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import common as C

OUTCOMES = ["enhanced", "original_gate", "original_no_face", "fallback_guard", "fallback_model_failure", "rejected", "failed"]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(vals, nd=3):
    vals = [v for v in vals if v is not None]
    return round(st.fmean(vals), nd) if vals else ""


def pctl(vals, p):
    vals = sorted(vals)
    return vals[int(round(p * (len(vals) - 1)))] if vals else ""


def main(a):
    cfg = C.load_config(a.config)
    root = Path(cfg["bench_root"])
    scope = C.scope(cfg, a.pilot)
    ids = {r["sample_id"] for r in scope}
    variants = [v for v in C.read_csv(C.bench(cfg, "manifests", "generated_variants.csv")) if v["sample_id"] in ids]
    results = [r for r in C.read_csv(C.bench(cfg, "manifests", "enhancement_results.csv")) if r["sample_id"] in ids]
    metrics = [r for r in C.read_csv(C.bench(cfg, "manifests", "metrics.csv")) if r["sample_id"] in ids]
    comps = [r for r in C.read_csv(C.bench(cfg, "manifests", "comparisons.csv")) if r["sample_id"] in ids and not r["error"]]
    types = list(dict.fromkeys(v["type"] for v in variants))
    cat = {v["type"]: v["category"] for v in variants}
    out: dict = {"scope": "pilot" if a.pilot else "full"}
    md = [f"# Results — {out['scope']} scope", ""]

    # ---- counts
    oc = Counter(r["outcome"] for r in results)
    counts = {"selected originals in scope": len(scope), "clean references": len({v["sample_id"] for v in variants}),
              "degraded variants": len(variants), "service answers": len(results),
              "enhanced images stored (service PNG chips)": oc["enhanced"],
              "outputs stored that are the unchanged input": sum(oc[k] for k in OUTCOMES[1:5]),
              "comparison images": len(comps),
              "degraded GiB": round(sum(int(v["bytes"]) for v in variants) / 2 ** 30, 2)}
    out["counts"], out["outcomes"] = counts, dict(oc)
    md += ["## Counts", "", C.md_table([{"item": k, "value": v} for k, v in counts.items()], ["item", "value"]), "",
           "## Service outcomes", "",
           C.md_table([{"outcome": k, "requests": oc[k], "share": f"{100 * oc[k] / max(1, len(results)):.1f} %"}
                       for k in OUTCOMES], ["outcome", "requests", "share"]), ""]
    reasons = Counter((r["outcome"], r["fallback"] or r["reason"] or r["error"]) for r in results if r["outcome"] != "enhanced")
    md += ["Reasons given by the service for not enhancing:", "",
           C.md_table([{"outcome": k[0], "reason / fallback": k[1], "requests": v} for k, v in reasons.most_common(25)],
                      ["outcome", "reason / fallback", "requests"]), ""]
    out["reasons"] = [{"outcome": k[0], "reason": k[1], "requests": v} for k, v in reasons.most_common()]

    # ---- by type × level (faces the service found only: the gate saw them)
    md += ["## Enhancement rate by degradation type and level", "",
           "Share of requests answered ENHANCED among the requests in which the service found a usable face "
           "(`n` = those requests; originals without a usable frontal face are counted in the outcome table above).", ""]
    rows = []
    for t in types:
        row = {"type": t, "category": cat[t]}
        for l in "1234":
            g = [r for r in results if r["type"] == t and r["level"] == l and r["outcome"] != "original_no_face"]
            row[f"L{l}"] = f"{100 * sum(r['outcome'] == 'enhanced' for r in g) / len(g):.0f} % (n={len(g)})" if g else "-"
        g = [r for r in results if r["type"] == t]
        row["guard fallbacks"] = sum(r["outcome"] == "fallback_guard" for r in g)
        row["no face"] = sum(r["outcome"] == "original_no_face" for r in g)
        rows.append(row)
    out["by_type"] = rows
    md += [C.md_table(rows, list(rows[0])), ""]

    rows = []
    for e in sorted({r["expression"] for r in results}):
        g = [r for r in results if r["expression"] == e]
        c = Counter(r["outcome"] for r in g)
        rows.append({"expression": e, "originals": len({r["sample_id"] for r in g}), "requests": len(g),
                     **{k: c[k] for k in OUTCOMES}})
    out["by_expression"] = rows
    md += ["## Outcomes by expression", "", C.md_table(rows, list(rows[0])), ""]

    # ---- time and resources
    rtt_all = [f(r["rtt_ms"]) for r in results if f(r.get("rtt_ms")) is not None]
    rtt_enh = [f(r["rtt_ms"]) for r in results if r["outcome"] in ("enhanced", "fallback_guard") and f(r.get("rtt_ms"))]
    res_log = [r for r in C.read_jsonl(C.bench(cfg, "logs", "resources.jsonl")) if "mem_mib" in r]
    timing = {"requests": len(rtt_all), "sum of round trips (h)": round(sum(rtt_all) / 3.6e6, 2),
              "median ms, all": pctl(rtt_all, .5), "p95 ms, all": pctl(rtt_all, .95),
              "median ms, model ran": pctl(rtt_enh, .5), "p95 ms, model ran": pctl(rtt_enh, .95),
              "max ms": max(rtt_all) if rtt_all else "",
              "peak service memory MiB (docker stats samples)": max((r["mem_mib"] for r in res_log), default=""),
              "memory samples": len(res_log)}
    out["timing"] = timing
    md += ["## Time and resources (HTTP round trip, one request at a time)", "",
           C.md_table([{"item": k, "value": v} for k, v in timing.items()], ["item", "value"]), ""]

    # ---- diagnostics
    enh = [m for m in metrics if m["outcome"] == "enhanced"]
    md += ["## Image-quality diagnostics on ENHANCED answers (face chip, against the clean reference)", "",
           "Mean over the enhanced answers of each type. `Δ` = output − degraded input. PSNR / SSIM measure closeness "
           "to the clean pixels; they are diagnostics, not a statement about identity.", ""]
    rows = []
    for t in [*types, "ALL"]:
        g = [m for m in enh if t == "ALL" or m["type"] == t]
        if not g:
            continue
        dp = [f(m["psnr_output"]) - f(m["psnr_degraded"]) for m in g]
        ds = [f(m["ssim_output"]) - f(m["ssim_degraded"]) for m in g]
        rows.append({"type": t, "enhanced n": len(g), "PSNR degraded": mean(f(m["psnr_degraded"]) for m in g),
                     "PSNR output": mean(f(m["psnr_output"]) for m in g), "Δ PSNR": mean(dp),
                     "PSNR better in": f"{100 * sum(d > 0 for d in dp) / len(dp):.0f} %",
                     "SSIM degraded": mean((f(m["ssim_degraded"]) for m in g), 4),
                     "SSIM output": mean((f(m["ssim_output"]) for m in g), 4), "Δ SSIM": mean(ds, 4),
                     "sharpness ratio degraded": mean(f(m["sharpness_ratio_degraded"]) for m in g),
                     "sharpness ratio output": mean(f(m["sharpness_ratio_output"]) for m in g)})
    out["quality_enhanced"] = rows
    md += [C.md_table(rows, list(rows[0])) if rows else "No enhanced answers.", ""]

    md += ["## Identity diagnostic on ENHANCED answers (kept separate from image quality)", "",
           "SFace (E01) cosine similarity to the clean reference; `Δ` = (clean↔output) − (clean↔degraded). One "
           "evaluator, synthetic degradations, same-image pairs only: this does not measure a matching system's "
           "accuracy and is not proof that identity is preserved.", ""]
    rows = []
    for t in [*types, "ALL"]:
        g = [m for m in enh if (t == "ALL" or m["type"] == t) and f(m["sface_clean_vs_output"]) is not None]
        if not g:
            continue
        d = [f(m["sface_clean_vs_output"]) - f(m["sface_clean_vs_degraded"]) for m in g]
        rows.append({"type": t, "n": len(g), "clean↔degraded": mean((f(m["sface_clean_vs_degraded"]) for m in g), 4),
                     "clean↔output": mean((f(m["sface_clean_vs_output"]) for m in g), 4), "Δ mean": mean(d, 4),
                     "Δ median": round(st.median(d), 4), "Δ p5": round(pctl(d, .05), 4),
                     "higher in": f"{100 * sum(x > 0 for x in d) / len(d):.0f} %",
                     "lower by > 0.05 in": f"{100 * sum(x < -0.05 for x in d) / len(d):.1f} %"})
    out["identity_enhanced"] = rows
    md += [C.md_table(rows, list(rows[0])) if rows else "No enhanced answers with a clean-reference face.", ""]
    lv = defaultdict(list)
    for m in enh:
        if f(m["sface_clean_vs_output"]) is not None:
            lv[m["level"]].append(f(m["sface_clean_vs_output"]) - f(m["sface_clean_vs_degraded"]))
    md += ["By severity level (all types): " + " · ".join(
        f"L{l}: n={len(v)}, Δ mean {st.fmean(v):+.4f}" for l, v in sorted(lv.items())), ""]
    out["identity_by_level"] = {l: {"n": len(v), "delta_mean": round(st.fmean(v), 4)} for l, v in lv.items()}

    name = "08_results_pilot" if a.pilot else "08_results"
    C.bench(cfg, "reports", name + ".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    C.bench(cfg, "reports", name + ".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--pilot", action="store_true")
    main(ap.parse_args())
