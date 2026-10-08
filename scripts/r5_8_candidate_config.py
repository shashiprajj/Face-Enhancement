"""R5-8 — write the sealed research-candidate route config (configs/phase5/r5_8_nafnet_candidate.json).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_8_candidate_config.py corpus
    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_8_candidate_config.py final   (after DEV calibration)

The live config (configs/phase5/r5_nafnet_route.json, p5-nafnet-route-1.0.0) is NOT modified; the candidate is used only
with FES_NAFNET_PROFILE=r5_8.
  corpus  = the R5-7 candidate exactly (rejected on its own in R5-7): input gate unchanged, guard chroma_p99_max 20.4 +
            chroma_p999_max 36.4 (R5-7 DEV calibration, data/manifests/phase5_rapid/r5_7/guard_calibration.json),
            out-of-image reflect fill on. Used to generate the DEV artifact corpus.
  final   = corpus + "local_artifact" (thresholds from data/manifests/phase5_rapid/r5_8/local_calibration.json, DEV only).
"""
from __future__ import annotations

import json
import sys

from fes_bench import gate as GT
from fes_data.common import repo_path, write_json
from fes_service import nafnet_route as NR


def main(stage: str) -> None:
    live = json.loads(repo_path(NR.CONFIG).read_text(encoding="utf-8"))
    r57 = json.loads(repo_path("data/manifests/phase5_rapid/r5_7/guard_calibration.json").read_text(encoding="utf-8"))
    cfg = {k: v for k, v in live.items() if k not in ("sha256", "calibration")}
    cfg.update(name="r5_8_nafnet_candidate", route_version=NR.CANDIDATE_ROUTE_VERSION,
               guard={**live["guard"], "chroma_p99_max": r57["t99"], "chroma_p999_max": r57["t999"]},
               framing={"oob_reflect_fill": True, "nafnet_reflect_pad16": True,
                        "note": "pad16 lives in the model factory: use fes_service.models:nafnet_factory_pad16"},
               derived_from={"live_route_seal": live["sha256"], "r5_7_guard_calibration_seal": r57["seal"]},
               status="RESEARCH CANDIDATE (R5-8) — not the live/default route",
               stage=stage)
    if stage == "final":
        lc = json.loads(repo_path("data/manifests/phase5_rapid/r5_8/local_calibration.json").read_text(encoding="utf-8"))
        cfg["local_artifact"] = lc["config"]
    cfg["sha256"] = GT.gate_hash(cfg)
    write_json(repo_path(NR.CANDIDATE_CONFIG), cfg)
    print(stage, cfg["sha256"], json.dumps(cfg["guard"]), json.dumps(cfg.get("local_artifact")))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "corpus")
