"""P4-T6 — calibrate and FREEZE the Do-Not-Enhance gate on the SYNTHETIC DEV population `p4dev` only.

    scripts/dock.sh python scripts/p4_gate_calibrate.py

Labels (fes_bench.gate.pair_label) come from all five candidates' multi-evaluator verdicts (SFace E01 + dlib E07,
each against its own δ). Inputs that fail the fixed safety rules are UNSAFE_OR_UNCERTAIN by rule and are excluded
from fitting. Main (test) data is never read here.

Writes configs/phase4/t6_gate.json (SHA-256 sealed) + data/manifests/phase4/t6/calibration_report.json.
"""

from __future__ import annotations

import json
from collections import Counter

from fes_bench import gate as GT
from fes_bench import p4eval as E
from fes_data.common import repo_path, run_info, sha256_file, write_json


def main() -> None:
    pairs = E.load("p4dev")
    configs = E.available_configs(pairs)
    missing = [c for c in E.CANDIDATES if c not in configs]
    assert not missing, f"dev outputs/metrics incomplete for {missing} — finish inference first"
    delta = E.deltas_from_validation()
    allp = sorted(pairs.values(), key=lambda p: p.pair_id)
    P = [p for p in allp if GT.safety(p.feat) is None]
    feats, groups = [p.feat for p in P], [p.ref_id for p in P]
    labels = [GT.pair_label([E.multi_of(p, c, delta) for c in E.CANDIDATES]) for p in P]
    k, cv = GT.choose_k(feats, labels, groups)
    rules = GT.fit_rules(feats, labels, k)

    inputs = {"population_p4dev_sha256": sha256_file(repo_path("data/manifests/phase4/population_p4dev.csv.gz")),
              "quality_p4dev_sha256": sha256_file(repo_path("data/manifests/phase4/quality/p4dev.csv.gz")),
              "metrics_e01": {c: sha256_file(repo_path(f"data/manifests/phase4/metrics/p4dev/{c}.csv.gz")) for c in ("B0",) + E.CANDIDATES},
              "metrics_e07": {c: sha256_file(repo_path(f"data/manifests/phase4/metrics_e07/p4dev/{c}.csv.gz")) for c in ("B0",) + E.CANDIDATES},
              "evaluator_validation_sha256": sha256_file(repo_path(E.VALIDATION))}
    gate = {"name": "t6_gate", "gate_version": GT.GATE_VERSION, "decisions": [GT.UNSAFE, GT.ENH, GT.DNE],
            "safety_rules": {"det_status": "OK required", "det_score_min": GT.LOW_CONF, "abs_yaw_proxy_max": GT.YAW_WARN},
            "features": {"module": "src/fes_bench/quality.py", "version": "p4-quality-1.0.0", "used": GT.FEATURES},
            "objective": {"lambda": GT.LAMBDA, "labels": "WORTHY / UNSAFE / NEUTRAL over A1–A5, E01+E07 verdicts"},
            "k_chosen": k, "rules": [r.to_json() for r in rules],
            "calibration": {"population": "p4dev", "split": "dev", "data_kind": "SYNTHETIC", **inputs},
            "validity": "SYNTHETIC dev calibration. NOT valid for real-world inputs until re-calibrated on the rw dev split."}
    gate["sha256"] = GT.gate_hash(gate)
    write_json(repo_path("configs/phase4/t6_gate.json"), gate)

    dec = {p.pair_id: GT.decide(p.feat, rules)[0] for p in allp}
    insample = {d: Counter(GT.pair_label([E.multi_of(p, c, delta) for c in E.CANDIDATES])
                           for p in allp if dec[p.pair_id] == d) for d in (GT.ENH, GT.DNE, GT.UNSAFE)}
    write_json(repo_path("data/manifests/phase4/t6/calibration_report.json"), {
        "run": run_info(), "gate_sha256": gate["sha256"], "k": k, "rules": gate["rules"], **cv,
        "n_dev_pairs": len(allp), "n_fit_pairs": len(P), "label_counts_fit": dict(Counter(labels)),
        "dev_in_sample_utility": GT.utility(feats, labels, rules),
        "dev_in_sample_label_by_decision": {d: dict(v) for d, v in insample.items()},
        "note": "In-sample (dev) figures describe the fit only; the single test evaluation is p4_gate_evaluate.py."})
    print("T6 frozen:", gate["sha256"][:12], "K =", k, json.dumps(gate["rules"]))


if __name__ == "__main__":
    main()
