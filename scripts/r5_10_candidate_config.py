"""R5-10 — write the sealed research-candidate route config (configs/phase5/r5_10_nafnet_candidate.json).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_10_candidate_config.py

= the sealed R5-8 candidate (configs/phase5/r5_8_nafnet_candidate.json) with exactly ONE change: thresholds.blur_max
is the boundary selected on DEV by scripts/r5_10_dev_analyze.py under the pre-registered protocol
(data/manifests/phase5_rapid/r5_10/protocol.json). Model, padding, framing, every other input gate, the global artifact
guard and the local-patch guard are copied unchanged. Neither the live R5-6 config nor the R5-8 config is modified;
the candidate is used only with FES_NAFNET_PROFILE=r5_10 or per request (FES_EXPERIMENTAL_PROFILES=r5_10).
"""
from __future__ import annotations

import hashlib
import json

from fes_bench import gate as GT
from fes_data.common import repo_path, write_json
from fes_service import nafnet_route as NR

R10 = "data/manifests/phase5_rapid/r5_10"


def sha(path: str) -> str:
    return hashlib.sha256(repo_path(path).read_bytes()).hexdigest()


def main() -> None:
    r58 = NR.load_config(NR.CANDIDATE_CONFIG)                       # verifies the R5-8 seal
    dev = json.loads(repo_path(f"{R10}/dev_analysis.json").read_text(encoding="utf-8"))
    new = float(dev["selected_blur_max"])
    old = r58["thresholds"]["blur_max"]
    assert old < new < r58["thresholds"]["clean_lapd1_lo"], (old, new)
    cfg = {k: v for k, v in r58.items() if k != "sha256"}
    cfg.update(name="r5_10_nafnet_candidate", route_version=NR.R5_10_ROUTE_VERSION,
               thresholds={**r58["thresholds"], "blur_max": new},
               derived_from={**r58["derived_from"], "r5_8_candidate_seal": r58["sha256"]},
               blur_recalibration={
                   "old_blur_max": old, "new_blur_max": new,
                   "unchanged": "clean_lapd1_lo (NOT_BLUR zone), every other gate, guard, local_artifact, framing, model",
                   "protocol_sha256": sha(f"{R10}/protocol.json"),
                   "dev_analysis_sha256": sha(f"{R10}/dev_analysis.json"),
                   "data": "DEV only: archive val (both modes), archive val sharp GT, p4dev + r5gap"},
               status="RESEARCH CANDIDATE (R5-10) — not the live/default route",
               stage="final")
    cfg["sha256"] = GT.gate_hash(cfg)
    write_json(repo_path(NR.R5_10_CONFIG), cfg)
    print(cfg["sha256"], old, "->", new)


if __name__ == "__main__":
    main()
