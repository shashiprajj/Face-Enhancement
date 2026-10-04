"""P4-T6 — deterministic input-quality gate: should this face be enhanced at all?

Decisions (from reference-free features, fes_bench.quality; candidate-agnostic):
  UNSAFE_OR_UNCERTAIN  fixed safety rules, NOT calibrated (P0 C2/C6; P2-T6 constants defined before Phase 4):
                         * detection contract did not return an aligned face (NO_FACE_DETECTED, FACE_TOO_SMALL
                           < 12 px IOD, LANDMARK_FAILURE …)                         → 'no_valid_face'
                         * detector confidence < 0.85 (P2-T6 low_conf_warn)        → 'low_detection_confidence'
                         * |yaw proxy| > 0.25 (P2-T6 yaw_warn) or implausible landmarks (NaN pose)
                                                                                     → 'alignment_unreliable'
                       The service returns the ORIGINAL and flags/escalates; it never enhances.
  ENHANCE_CANDIDATE    the first frozen rule whose conditions all hold (T7 then picks original or a candidate)
  DO_NOT_ENHANCE       otherwise — the default; the original (B0) is returned untouched.

A rule = conjunction of ≤ 2 conditions (feature ≤ t | feature ≥ t). A condition on a NaN feature is False.

Calibration objective — PRE-REGISTERED 2026-10-03 before any p4dev candidate output was scored (A3 dev outputs were
scored by neither evaluator before this file was frozen):
  per DEV pair, over the five benchmarked candidates (multi-evaluator verdicts vs B0, P4-T4 rules):
    WORTHY   at least one candidate is CONFIRMED_BENEFIT (both evaluators gain beyond their δ)
    UNSAFE   no candidate is CONFIRMED_BENEFIT and EVERY candidate shows harm (any evaluator beyond δ)
    else     neutral
  U = #WORTHY(enhance) − LAMBDA · #UNSAFE(enhance), LAMBDA = 3 (a false enhancement costs three missed rescues;
  identity is a hard gate, P0 master §7). Greedy rule list; the number of rules K ≤ K_MAX is chosen by 4-fold CV
  grouped by reference (DEV only); refit on all DEV; frozen with a SHA-256 seal before any test evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

GATE_VERSION = "p4-gate-1.1.0"
LAMBDA = 3.0
K_MAX = 4
N_FOLDS = 4
LOW_CONF = 0.85          # fes_data.preprocess.PreprocessConfig.low_conf_warn (P2-T6)
YAW_WARN = 0.25          # fes_data.preprocess.PreprocessConfig.yaw_warn (P2-T6)
FEATURES = ["noise_sigma", "lap_norm_c", "hf_ratio_c", "blockiness", "luma_mean", "rms_contrast", "dyn_range",
            "iod_px", "in_min_side", "det_score", "dark_clip"]
QUANTILES = [round(q, 2) for q in np.arange(0.05, 0.96, 0.05)]
HARM = {"CONFIRMED_HARM", "PARTIAL_HARM", "DISAGREE"}
BENEFIT = {"CONFIRMED_BENEFIT"}
DNE, ENH, UNSAFE = "DO_NOT_ENHANCE", "ENHANCE_CANDIDATE", "UNSAFE_OR_UNCERTAIN"


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


@dataclass(frozen=True)
class Cond:
    feature: str
    op: str          # "<=" | ">="
    t: float

    def __call__(self, f: dict) -> bool:
        v = f.get(self.feature)
        if _nan(v):
            return False
        return v <= self.t if self.op == "<=" else v >= self.t


@dataclass(frozen=True)
class Rule:
    conds: tuple
    label: str = ENH

    def __call__(self, f: dict) -> bool:
        return all(c(f) for c in self.conds)

    def to_json(self) -> dict:
        return {"conds": [[c.feature, c.op, c.t] for c in self.conds]}


def safety(f: dict) -> str | None:
    """Fixed UNSAFE_OR_UNCERTAIN reasons (None = safe to consider)."""
    if f.get("det_status") != "OK":
        return f"no_valid_face:{f.get('det_status')}"
    if _nan(f.get("det_score")) or f["det_score"] < LOW_CONF:
        return "low_detection_confidence"
    if _nan(f.get("yaw_proxy")) or abs(f["yaw_proxy"]) > YAW_WARN:
        return "alignment_unreliable"
    return None


def decide(f: dict, rules: list[Rule]) -> tuple[str, str]:
    """→ (decision, reason)."""
    why = safety(f)
    if why:
        return UNSAFE, why
    for i, r in enumerate(rules):
        if r(f):
            return ENH, f"rule{i}"
    return DNE, "default"


def gate_hash(gate: dict) -> str:
    body = {k: v for k, v in gate.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_rules(gate: dict) -> list[Rule]:
    assert gate["sha256"] == gate_hash(gate), "frozen file modified after freeze"
    return [Rule(tuple(Cond(f, op, float(t)) for f, op, t in r["conds"])) for r in gate["rules"]]


def pair_label(verdicts: list[str]) -> str:
    """Candidate-agnostic T6 label from the five candidates' multi-evaluator verdicts on one pair."""
    if any(v in BENEFIT for v in verdicts):
        return "WORTHY"
    if verdicts and all(v in HARM for v in verdicts):
        return "UNSAFE"
    return "NEUTRAL"


# ------------------------------------------------------------------ calibration (DEV only)

def candidate_conditions(feats: list[dict]) -> list[Cond]:
    out = []
    for name in FEATURES:
        v = np.array([f.get(name, np.nan) for f in feats], float)
        v = v[~np.isnan(v)]
        if not len(v):
            continue
        for t in sorted({round(float(np.quantile(v, q)), 6) for q in QUANTILES}):
            out += [Cond(name, "<=", t), Cond(name, ">=", t)]
    return out


def _score(mask: np.ndarray, good: np.ndarray, bad: np.ndarray) -> float:
    return float(good[mask].sum() - LAMBDA * bad[mask].sum())


def fit_rules(feats: list[dict], labels: list[str], k_max: int, conds: list[Cond] | None = None) -> list[Rule]:
    """Greedy rule list maximising U on not-yet-covered pairs. labels ∈ {WORTHY, UNSAFE, NEUTRAL}."""
    conds = conds or candidate_conditions(feats)
    cm = {c: np.array([c(f) for f in feats]) for c in conds}
    good = np.array([l == "WORTHY" for l in labels])
    bad = np.array([l == "UNSAFE" for l in labels])
    covered = np.zeros(len(feats), bool)
    rules: list[Rule] = []
    for _ in range(k_max):
        scored = sorted(((_score(cm[c] & ~covered, good, bad), i, c) for i, c in enumerate(conds)),
                        key=lambda x: (-x[0], x[1]))
        top = [c for _, _, c in scored[:25]]
        options = [((c,), cm[c]) for c in top]
        options += [((a, b), cm[a] & cm[b]) for i, a in enumerate(top) for b in top[i + 1:] if a.feature != b.feature]
        best, best_u = None, 0.0
        for cs, m in options:
            u = _score(m & ~covered, good, bad)
            if u > best_u + 1e-9:
                best, best_u = (cs, m), u
        if best is None:
            break
        rules.append(Rule(tuple(best[0])))
        covered |= best[1]
    return rules


def utility(feats: list[dict], labels: list[str], rules: list[Rule]) -> float:
    u = 0.0
    for f, l in zip(feats, labels):
        if any(r(f) for r in rules):
            u += (l == "WORTHY") - LAMBDA * (l == "UNSAFE")
    return u


def choose_k(feats: list[dict], labels: list[str], groups: list[str]) -> tuple[int, dict]:
    """Grouped K-fold CV (by reference) over K = 0..K_MAX; smallest K reaching the best mean CV utility."""
    refs = sorted(set(groups))
    fold_of = {r: int(hashlib.sha256(f"p4-gate-cv|{r}".encode()).hexdigest(), 16) % N_FOLDS for r in refs}
    conds = candidate_conditions(feats)
    cv = {k: [] for k in range(K_MAX + 1)}
    for fold in range(N_FOLDS):
        tr = [i for i, g in enumerate(groups) if fold_of[g] != fold]
        te = [i for i, g in enumerate(groups) if fold_of[g] == fold]
        rules = fit_rules([feats[i] for i in tr], [labels[i] for i in tr], K_MAX, conds)
        for k in range(K_MAX + 1):
            cv[k].append(utility([feats[i] for i in te], [labels[i] for i in te], rules[:k]))
    means = {k: float(np.mean(v)) for k, v in cv.items()}
    best = max(means.values())
    k = min(k for k, m in means.items() if m >= best - 1e-9)
    return k, {"cv_utility_mean_by_k": means, "cv_utility_by_fold": cv}
