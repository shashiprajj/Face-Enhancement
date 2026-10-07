"""R5-6 preparation — run a folder of images through the REAL service pipeline (EnhancementPipeline with the real
process-isolated A4 / NAFNet workers) and write one structured record per input (Result.validation_record).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_service_validate.py <image_dir> --name <run> \
        [--route-mode auto|nafnet|a4] [--input-mode full_image|face_chip] [--limit N] [--save-outputs]

Writes data/manifests/phase5_rapid/service_validation/<run>.csv.gz (+ <run>_summary.json). With --save-outputs the
returned images go to the git-ignored experiments/phase5_rapid/outputs/service_validation/<run>/. Records hold no
pixels, landmarks or embeddings; input_id is the file name relative to <image_dir>.
R5-5 only smoke-tests this runner on DEV images; the archive test/ztest and the 200 real photos are R5-6.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from fes_data.common import repo_path, run_info, write_csv_gz, write_json
from fes_service import pipeline as SP
from fes_service.worker import IsolatedModel

OUT = "data/manifests/phase5_rapid/service_validation"
IMG_OUT = "experiments/phase5_rapid/outputs/service_validation"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--route-mode", default="auto", choices=SP.ROUTE_MODES)
    ap.add_argument("--input-mode", default="full_image", choices=("full_image", "face_chip"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-outputs", action="store_true")
    a = ap.parse_args()
    root = Path(a.image_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXT)[: a.limit or None]
    a4 = IsolatedModel("fes_service.models:a4_factory", timeout_s=300, threads=2)
    nafnet = IsolatedModel("fes_service.models:nafnet_factory", timeout_s=120, threads=4)
    pipe = SP.EnhancementPipeline(a4, timeout_s=300, gate="rw", nafnet=nafnet, route_mode=a.route_mode)
    rows, t0 = [], time.time()
    img_dir = repo_path(f"{IMG_OUT}/{a.name}")
    try:
        for i, p in enumerate(files):
            rid = p.relative_to(root).as_posix()
            res = pipe.process(p.read_bytes(), a.input_mode)
            rows.append(res.validation_record(rid))
            if a.save_outputs and res.status == SP.ENHANCED:
                img_dir.mkdir(parents=True, exist_ok=True)
                (img_dir / (rid.replace("/", "__") + ".out.png")).write_bytes(res.image)
            if i % 25 == 0:
                print(f"{a.name} {i}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    finally:
        a4.close()
        nafnet.close()
    cols = list(dict.fromkeys(k for r in rows for k in r))      # artifact_* metrics exist only where the guard ran
    write_csv_gz(repo_path(f"{OUT}/{a.name}.csv.gz"), [{k: r.get(k, "") for k in cols} for r in rows], cols)
    summ = {"name": a.name, "image_dir": str(root), "route_mode": a.route_mode, "input_mode": a.input_mode,
            "n": len(rows), "seconds": round(time.time() - t0, 1), "seals": pipe.seals,
            "status": dict(Counter(r["status"] for r in rows)), "model": dict(Counter(r["model"] for r in rows)),
            "reason": dict(Counter(r["reason"].split(":")[0] for r in rows)),
            "fallback": dict(Counter(r["fallback"] for r in rows if r["fallback"])),
            "artifact_check": dict(Counter(r["artifact_check"] for r in rows)),
            "a4_restarts": a4.restarts, "nafnet_restarts": nafnet.restarts}
    write_json(repo_path(f"{OUT}/{a.name}_summary.json"), run_info(summ))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
