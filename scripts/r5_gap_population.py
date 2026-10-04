"""R5-3 — small DEV-only gap population: does A4's benefit on strong blur / strong noise survive at smaller face sizes?

    .venv/Scripts/python.exe scripts/r5_gap_population.py        (numpy/OpenCV only)

Phase 4 evidence for A4 exists only at ~80 px inter-ocular distance (224-px chips); real photos sit mostly at 20–80 px.
For the 28 p4dev references (DEV split; the test population is not touched) and the four cells where A4 helped
(gaussblur severe/extreme, noise moderate/severe), the clean 224-px reference is reduced to s ∈ {168, 112} px
(INTER_AREA, the fes-syn-v1 rule) and the SAME degradation is applied at that size:
  gaussian blur  sigma × s/224   (same blur relative to the face)
  noise          sigma unchanged (per-pixel sensor/scan grain)
Seed: fes_data.degrade.derive_seed(…, profile, f"{severity}@{s}"). Images → datasets/benchmark/r5gap/ (git-ignored).
Manifest → data/manifests/phase5_rapid/gap/population_r5gap.csv.gz (columns of population_p4dev; cell "<cell>@<s>").
"""
from __future__ import annotations

import cv2
import numpy as np

from fes_data import degrade as D
from fes_data.common import read_csv_gz, repo_path, sha256_bytes, write_csv_gz, write_json, run_info

CELLS = [("gaussblur", "severe"), ("gaussblur", "extreme"), ("noise", "moderate"), ("noise", "severe")]
SIZES = [168, 112]
SEED = 20261004


def main() -> None:
    pop = read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz"))
    refs = {}
    for p in pop:
        refs.setdefault(p["ref_id"], p)
    rows = []
    for rid, p in sorted(refs.items()):
        ref = cv2.imread(str(repo_path(p["ref_path"])), cv2.IMREAD_COLOR)
        for s in SIZES:
            small = cv2.resize(ref, (s, s), interpolation=cv2.INTER_AREA)
            for prof, sev in CELLS:
                spec = D.PROFILES[prof][sev]
                seed = D.derive_seed(SEED, rid, prof, f"{sev}@{s}")
                if prof == "gaussblur":
                    out = D.gaussian_blur(small, spec["sigma"] * s / 224.0)
                else:
                    out = D.add_gaussian_noise(small, spec["sigma"], np.random.default_rng(seed))
                ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                cell = f"{prof}-{sev}@{s}"
                path = f"datasets/benchmark/r5gap/{cell}/{rid}.png"
                dst = repo_path(path)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(buf.tobytes())
                r = dict(p)
                r.update(population="r5gap", pair_id=f"{rid}.{cell}", profile=prof, severity=sev, cell=cell,
                         degraded_path=path, degraded_sha256=sha256_bytes(buf.tobytes()), out_w=str(s), seed=str(seed),
                         degrade_version=f"{D.DEGRADE_VERSION}+r5gap")
                rows.append(r)
    out = repo_path("data/manifests/phase5_rapid/gap")
    write_csv_gz(out / "population_r5gap.csv.gz", rows, list(pop[0].keys()))
    write_json(out / "population_r5gap_report.json", {"run": run_info(), "n": len(rows), "refs": len(refs),
                                                      "cells": [f"{a}-{b}" for a, b in CELLS], "sizes": SIZES,
                                                      "split": "dev", "seed": SEED})
    print(len(rows), "pairs")


if __name__ == "__main__":
    main()
