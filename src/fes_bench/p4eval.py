"""Phase 4 shared evaluation layer (P4-T4/T6/T7): join per-pair metrics of BOTH identity evaluators with the
reference-free input-quality features, and the per-evaluator decision rules.

Per-evaluator rules (P4-T3 §4: scores are never pooled across evaluators)
  δ_e        identity-change threshold = max over identity-neutral perturbations of the dev p99 of 1−cos
             (evaluator_E07_validation.json; E01 value equals the Phase 3 1-px-shift p99 used in P3-T6)
  thr_e      FAR 1e-2 operating point from clean references of the SAME split as the population
  verdict    per pair and candidate vs B0: BENEFIT (Δ > δ), HARM (Δ < −δ), NEUTRAL otherwise
  multi      CONFIRMED_BENEFIT (both BENEFIT) · CONFIRMED_HARM (both HARM) · DISAGREE (opposite) ·
             PARTIAL_BENEFIT / PARTIAL_HARM (one evaluator, other NEUTRAL) · NEUTRAL
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from fes_data.common import read_csv_gz, repo_path

EVALUATORS = ("E01", "E07")
CANDIDATES = ("A1", "A2", "A3", "A4", "A5")
BASELINES = ("B0", "B1", "B2", "B2c")
SEV_ORDER = ["mild", "moderate", "severe", "extreme"]
FAMILIES = ["lowres", "gaussblur", "motionblur", "noise", "jpeg", "underexposure", "lowcontrast", "combined", "realistic2"]
VALIDATION = "data/manifests/phase4/evaluator_E07_validation.json"
POPS = {"main": {"root": "data/manifests/phase3", "split": "test"},
        "p4dev": {"root": "data/manifests/phase4", "split": "dev"}}


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def deltas_from_validation() -> dict[str, float]:
    v = json.loads(repo_path(VALIDATION).read_text(encoding="utf-8"))
    nf = v["noise_floor_1_minus_cos_p99"]
    return {e: max(p[e] for p in nf.values()) for e in EVALUATORS}


def thresholds(pop: str) -> dict[str, float]:
    """FAR 1e-2 operating points from clean references of the population's own split."""
    e07 = json.loads(repo_path(f"data/manifests/phase4/metrics_e07/{pop}/identity_calibration.json").read_text())
    if POPS[pop]["split"] == "test":
        e01 = json.loads(repo_path("data/manifests/phase3/metrics/main/identity_calibration.json").read_text())
        t01 = e01["threshold_at_far"]
    else:
        t01 = json.loads(repo_path(VALIDATION).read_text())["dev_impostor"]["E01"]["thr_far_1e-2"]
    return {"E01": float(t01), "E07": float(e07["threshold_at_far"])}


@dataclass
class Pair:
    pair_id: str
    ref_id: str
    cell: str
    profile: str
    severity: str
    race: str
    gender: str
    feat: dict = field(default_factory=dict)           # p4-quality features of the degraded input
    cos: dict = field(default_factory=dict)            # (config, evaluator) → cos to reference
    imp_max: dict = field(default_factory=dict)        # (config, evaluator) → max cos to another identity
    det_out: dict = field(default_factory=dict)        # config → YuNet found a face on the output (E01 metrics file)


def load(pop: str, configs=BASELINES + CANDIDATES) -> dict[str, Pair]:
    root = POPS[pop]["root"]
    pairs: dict[str, Pair] = {}
    for p in read_csv_gz(repo_path(f"{root}/population_{pop}.csv.gz")):
        pairs[p["pair_id"]] = Pair(p["pair_id"], p["ref_id"], p["cell"], p["profile"], p["severity"],
                                   p["race"], p["gender"])
    for r in read_csv_gz(repo_path(f"data/manifests/phase4/quality/{pop}.csv.gz")):
        if r["item"] == "degraded":
            pairs[r["pair_id"]].feat = {k: (fnum(v) if k not in ("det_status", "families", "label_source",
                                                                 "data_kind", "split", "cell", "profile", "severity",
                                                                 "item", "pair_id", "ref_id") else v)
                                        for k, v in r.items()}
    for c in configs:
        e01 = repo_path(f"{root}/metrics/{pop}/{c}.csv.gz")
        e07 = repo_path(f"data/manifests/phase4/metrics_e07/{pop}/{c}.csv.gz")
        if not (e01.exists() and e07.exists()):
            continue
        for r in read_csv_gz(e01):
            P = pairs[r["pair_id"]]
            P.cos[(c, "E01")] = fnum(r["id_cos"]) if r["status"] == "OK" else float("nan")
            P.imp_max[(c, "E01")] = fnum(r["imp_cos_max"])
            P.det_out[c] = r["det_found"] == "True"
        for r in read_csv_gz(e07):
            P = pairs[r["pair_id"]]
            P.cos[(c, "E07")] = fnum(r["id_cos_e07"]) if r["status"] == "OK" else float("nan")
            P.imp_max[(c, "E07")] = fnum(r["imp_cos_max_e07"])
    return pairs


def available_configs(pairs: dict[str, Pair]) -> list[str]:
    any_p = next(iter(pairs.values()))
    return [c for c in BASELINES + CANDIDATES if (c, "E01") in any_p.cos and (c, "E07") in any_p.cos]


def verdict(d: float, delta: float) -> str:
    return "BENEFIT" if d > delta else ("HARM" if d < -delta else "NEUTRAL")


def multi_verdict(v1: str, v7: str) -> str:
    if v1 == v7:
        return {"BENEFIT": "CONFIRMED_BENEFIT", "HARM": "CONFIRMED_HARM", "NEUTRAL": "NEUTRAL"}[v1]
    if {v1, v7} == {"BENEFIT", "HARM"}:
        return "DISAGREE"
    return "PARTIAL_BENEFIT" if "BENEFIT" in (v1, v7) else "PARTIAL_HARM"


def cluster_boot(by_ref: dict[str, list[float]], B: int = 2000, seed: int = 20260928) -> tuple[float, float, float]:
    """Mean over all pairs, 95 % percentile CI resampling references (P3-T1 §8)."""
    refs = sorted(by_ref)
    s = np.array([np.nansum(by_ref[r]) for r in refs], float)
    c = np.array([np.sum(~np.isnan(np.asarray(by_ref[r], float))) for r in refs], float)
    est = float(s.sum() / max(c.sum(), 1))
    idx = np.random.default_rng(seed).integers(0, len(refs), (B, len(refs)))
    bs = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def tie_mark(lo: float, hi: float) -> str:
    return "▲" if lo > 0 else ("▼" if hi < 0 else "≈")


def multi_of(p: Pair, c: str, delta: dict) -> str:
    return multi_verdict(verdict(p.cos[(c, "E01")] - p.cos[("B0", "E01")], delta["E01"]),
                         verdict(p.cos[(c, "E07")] - p.cos[("B0", "E07")], delta["E07"]))


def policy_metrics(pairs: list[Pair], used: dict[str, str], delta: dict, thr: dict) -> dict:
    """Outcome of a policy that returns config `used[pair_id]` (B0 = original) for each pair. Per evaluator,
    never pooled: mean ΔID vs original (cluster CI), TAR, flips/rescues; multi-evaluator: confirmed benefit, harm
    (any evaluator), confirmed harm, disagreements; enhancement rate."""
    from collections import Counter, defaultdict
    harm_set = {"CONFIRMED_HARM", "PARTIAL_HARM", "DISAGREE"}
    row = {"n": len(pairs), "enhanced": sum(used[p.pair_id] != "B0" for p in pairs)}
    row["enhancement_rate"] = round(row["enhanced"] / max(1, len(pairs)), 4)
    for e in EVALUATORS:
        by_ref = defaultdict(list)
        for p in pairs:
            c = used[p.pair_id]
            by_ref[p.ref_id].append(p.cos[(c, e)] - p.cos[("B0", e)])
        est, lo, hi = cluster_boot(by_ref)
        row[f"d_{e}"] = [round(est, 5), round(lo, 5), round(hi, 5), tie_mark(lo, hi)]
        row[f"TAR_{e}"] = round(float(np.mean([p.cos[(used[p.pair_id], e)] >= thr[e] for p in pairs])), 4)
        row[f"flips_{e}"] = int(sum(p.cos[("B0", e)] >= thr[e] > p.cos[(used[p.pair_id], e)] for p in pairs))
        row[f"rescues_{e}"] = int(sum(p.cos[(used[p.pair_id], e)] >= thr[e] > p.cos[("B0", e)] for p in pairs))
        row[f"benefit_{e}"] = int(sum(used[p.pair_id] != "B0" and verdict(p.cos[(used[p.pair_id], e)] - p.cos[("B0", e)], delta[e]) == "BENEFIT" for p in pairs))
        row[f"harm_{e}"] = int(sum(used[p.pair_id] != "B0" and verdict(p.cos[(used[p.pair_id], e)] - p.cos[("B0", e)], delta[e]) == "HARM" for p in pairs))
    v = Counter(multi_of(p, used[p.pair_id], delta) for p in pairs if used[p.pair_id] != "B0")
    row["confirmed_benefit"] = v.get("CONFIRMED_BENEFIT", 0)
    row["harm_any"] = sum(v.get(k, 0) for k in harm_set)
    row["confirmed_harm"] = v.get("CONFIRMED_HARM", 0)
    row["disagree"] = v.get("DISAGREE", 0)
    return row
