"""P2-T6 — Run the reusable preprocessing pipeline on image files and emit crops + metadata.

    scripts/dock.sh python scripts/p2_preprocess.py --inputs a.jpg b.png --mode full_image \
        --out-dir datasets/derived/examples/preproc --metadata data/manifests/preprocess_examples_v1.json

Metadata per face: status, warnings, detector id, score, bbox, landmarks, similarity transform, scale,
pose proxies, out-of-source fraction, preprocessing version, input path and SHA-256.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from fes_data import common as C
from fes_data import preprocess as P


def build(cfg: dict) -> tuple[P.YuNetDetector, P.PreprocessConfig]:
    d = cfg["detector"]
    det = P.YuNetDetector(C.repo_path(d["model"]), d["score_threshold"], d["nms_threshold"], max_side=d["max_side"])
    return det, P.PreprocessConfig(**cfg["preprocess"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2/preprocess.yaml")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--mode", choices=["full_image", "face_crop"], default="full_image")
    ap.add_argument("--out-dir", default="datasets/derived/examples/preproc")
    ap.add_argument("--metadata", required=True)
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    det, pcfg = build(cfg)
    records = []
    for inp in a.inputs:
        data = Path(inp).read_bytes()
        img = C.decode_image(data)
        for res in P.preprocess(img, det, pcfg, input_mode=a.mode):
            rec = {"input": Path(inp).as_posix(), "input_sha256": C.sha256_bytes(data), **res.record()}
            if res.crop is not None:
                out = Path(a.out_dir) / f"{Path(inp).stem}_face{res.face_index}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), res.crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                rec["crop_path"] = out.as_posix()
                rec["crop_pixel_sha256"] = C.sha256_bytes(res.crop.tobytes())
            records.append(rec)
    C.write_json(Path(a.metadata), {"config": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
                                    "results": records, "run": C.run_info()})
    for r in records:
        print(r["input"], r["status"], r["warnings"], r.get("n_faces"))


if __name__ == "__main__":
    main()
