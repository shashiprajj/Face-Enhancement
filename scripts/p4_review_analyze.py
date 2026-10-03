"""P4-T5 — analyse returned human-review files (unblinded with the key). Rules fixed before any review exists.

    scripts/dock.sh python scripts/p4_review_analyze.py --version rv1 --responses <dir with review_rv1_*.json>

Pre-registered rules
  * Reviewer qualification: control accuracy ≥ QUALIFY (same-controls rated 4–5, different-controls rated 1–2).
    Unqualified reviewers are reported but excluded from the evidence.
  * Item-level outcome = mean over qualified reviewers (same-person 1–5); binary outcomes by majority (ties → 'unsure').
  * Rates are computed on RANDOM items only (TARGETED items are biased by construction and reported separately).
  * CIs: cluster bootstrap over references (B=2000, seed 20260928), paired vs B0 on the same pairs.
  * Agreement: Krippendorff's α (ordinal for same-person, nominal for substitution / invented / artefacts);
    intra-rater consistency on REPEAT items. Disagreements (same-person range ≥ 2 or split substitution) are listed.
  * No pass/fail threshold (TH-5 TBD with product); attractiveness is not a dimension.
Evidence status requires ≥ 2 qualified reviewers (G-5); otherwise the report is marked INSUFFICIENT.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fes_data.common import read_csv_gz, repo_path, run_info, sha256_file, write_json

QUALIFY = 0.85
MIN_REVIEWERS = 2
B, SEED = 2000, 20260928
ORD = {"no": 0, "unsure": 1, "yes": 2}


def kripp_alpha(units: list[list[float]], metric: str = "nominal") -> float | None:
    """Krippendorff's α from per-unit value lists (missing values simply absent). Ordinal uses rank-based δ²."""
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    vals = sorted({v for u in units for v in u})
    ix = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    o = np.zeros((k, k))
    for u in units:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[ix[u[a]], ix[u[b]]] += 1.0 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if metric == "nominal":
        d = 1.0 - np.eye(k)
    elif metric == "ordinal":
        d = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                lo, hi = min(i, j), max(i, j)
                d[i, j] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2.0) ** 2
    else:
        v = np.array(vals, float)
        d = (v[:, None] - v[None, :]) ** 2
    de = (np.outer(n_c, n_c) * d).sum() / (n * (n - 1))
    do = (o * d).sum() / n
    return None if de == 0 else float(1 - do / de)


def majority(xs: list[str]) -> str:
    c = defaultdict(int)
    for x in xs:
        c[x] += 1
    top = max(c.values())
    w = [k for k, v in c.items() if v == top]
    return w[0] if len(w) == 1 else "unsure"


def cluster_ci(values: dict[str, list[float]], rng: np.random.Generator) -> tuple[float, float, float]:
    refs = sorted(values)
    arrs = [np.asarray(values[r], float) for r in refs]
    means = []
    for _ in range(B):
        pick = rng.integers(0, len(refs), len(refs))
        means.append(np.concatenate([arrs[i] for i in pick]).mean())
    return float(np.concatenate(arrs).mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyse(key: list[dict], files: list[dict]) -> dict:
    kmap = {k["item_id"]: k for k in key}
    out = {"reviewers": {}, "qualified": []}
    for f in files:
        rid, resp = f["reviewer"], f["responses"]
        ctrl = [(kmap[i]["role"], int(r["same"])) for i, r in resp.items() if kmap[i]["role"].startswith("CONTROL")]
        correct = sum((role == "CONTROL_SAME" and s >= 4) or (role == "CONTROL_DIFFERENT" and s <= 2) for role, s in ctrl)
        acc = correct / len(ctrl) if ctrl else 0.0
        rep = [(int(r["same"]), int(resp[kmap[i]["repeat_of"]]["same"]), r["subst"] == resp[kmap[i]["repeat_of"]]["subst"])
               for i, r in resp.items() if kmap[i]["role"] == "REPEAT" and kmap[i]["repeat_of"] in resp]
        out["reviewers"][rid] = {"n_answered": len(resp), "n_items": f.get("n_items"), "control_accuracy": round(acc, 4),
                                 "n_controls": len(ctrl), "qualified": acc >= QUALIFY,
                                 "repeat_same_within_1": round(np.mean([abs(a - b) <= 1 for a, b, _ in rep]), 4) if rep else None,
                                 "repeat_subst_identical": round(np.mean([c for *_, c in rep]), 4) if rep else None}
        if acc >= QUALIFY:
            out["qualified"].append(rid)
    q = [f for f in files if f["reviewer"] in out["qualified"]]
    out["status"] = "EVIDENCE" if len(q) >= MIN_REVIEWERS else "INSUFFICIENT (G-5: < 2 qualified reviewers)"

    per_item = defaultdict(list)
    for f in q:
        for i, r in f["responses"].items():
            if kmap[i]["role"] == "TEST":
                per_item[i].append(r)
    out["agreement"] = {
        "alpha_same_ordinal": kripp_alpha([[int(r["same"]) for r in rs] for rs in per_item.values()], "ordinal"),
        "alpha_subst_nominal": kripp_alpha([[ORD[r["subst"]] for r in rs] for rs in per_item.values()], "nominal"),
        "alpha_invented_ordinal": kripp_alpha([[["none", "minor", "major"].index(r["invented"]) for r in rs]
                                               for rs in per_item.values()], "ordinal"),
        "alpha_artef_ordinal": kripp_alpha([[["none", "minor", "major"].index(r["artef"]) for r in rs]
                                            for rs in per_item.values()], "ordinal")}
    out["disagreements"] = sorted(i for i, rs in per_item.items() if len(rs) >= 2 and (
        max(int(r["same"]) for r in rs) - min(int(r["same"]) for r in rs) >= 2 or len({r["subst"] for r in rs}) > 1))

    items = {}
    for i, rs in per_item.items():
        k = kmap[i]
        items[i] = {"config": k["config_id"], "pair": k["pair_id"], "ref": k["ref_id"], "sel": k["selection"],
                    "severity": k["severity"], "same": float(np.mean([int(r["same"]) for r in rs])),
                    "subst_yes": majority([r["subst"] for r in rs]) == "yes",
                    "invented_any": majority([r["invented"] for r in rs]) != "none",
                    "invented_major": majority([r["invented"] for r in rs]) == "major",
                    "attr_change": majority(["none" if r["attr"] == ["none"] else "changed" for r in rs]) == "changed",
                    "appear_change": majority(["none" if r["appear"] == ["none"] else "changed" for r in rs]) == "changed",
                    "artef_major": majority([r["artef"] for r in rs]) == "major"}
    rng = np.random.default_rng(SEED)
    res = {}
    b0 = {v["pair"]: v for v in items.values() if v["config"] == "B0" and v["sel"] == "RANDOM"}
    for c in sorted({v["config"] for v in items.values()}):
        rnd = [v for v in items.values() if v["config"] == c and v["sel"] == "RANDOM"]
        if not rnd:
            continue
        row = {"n_random": len(rnd)}
        for m in ("same", "subst_yes", "invented_any", "invented_major", "attr_change", "appear_change", "artef_major"):
            by_ref = defaultdict(list)
            for v in rnd:
                by_ref[v["ref"]].append(float(v[m]))
            row[m] = [round(x, 4) for x in cluster_ci(by_ref, rng)]
            if c != "B0" and b0:
                dref = defaultdict(list)
                for v in rnd:
                    if v["pair"] in b0:
                        dref[v["ref"]].append(float(v[m]) - float(b0[v["pair"]][m]))
                if dref:
                    row[f"d_{m}_vs_B0"] = [round(x, 4) for x in cluster_ci(dref, rng)]
        res[c] = row
    out["random_by_config"] = res
    out["targeted"] = {c: {"n": len(v), "same_mean": round(float(np.mean([x["same"] for x in v])), 3),
                           "subst_yes": int(sum(x["subst_yes"] for x in v))}
                       for c in sorted({x["config"] for x in items.values()})
                       if (v := [x for x in items.values() if x["config"] == c and x["sel"].startswith("TARGETED")])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="rv1")
    ap.add_argument("--responses", required=True)
    a = ap.parse_args()
    key_path = repo_path(f"data/manifests/phase4/review/key_{a.version}.csv.gz")
    key = read_csv_gz(key_path)
    files = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(a.responses).glob(f"review_{a.version}_*.json"))]
    files = [f for f in files if f.get("kit") == a.version]
    res = analyse(key, files)
    res["run"] = run_info({"key_sha256": sha256_file(key_path), "n_files": len(files)})
    write_json(repo_path(f"data/manifests/phase4/review/analysis_{a.version}.json"), res)
    print(res["status"], res["reviewers"])


if __name__ == "__main__":
    main()
