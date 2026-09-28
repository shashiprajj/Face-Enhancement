"""P2-T3 design probe — choose the canonical reference size from data, not by assumption.

Measures YuNet detections on a seeded sample of FairFace images (read from the raw zip) and reports,
for candidate output sizes, the fraction of faces that would need upsampling (scale > 1) when aligned
to the FFHQ template. Output: data/manifests/reference_size_probe_v1.json

    scripts/dock.sh python scripts/p2_reference_size_probe.py
"""

from __future__ import annotations

import random
import zipfile

import numpy as np

from fes_data import common as C
from fes_data.preprocess import TEMPLATE_EYE_DIST_512, YuNetDetector, landmarks_plausible

ZIP = "datasets/raw/DS01_fairface/fairface-img-margin025-trainval.zip"
SEED, N = 20260928, 2000
SIZES = [192, 224, 256, 320, 384, 512]


def main() -> None:
    det = YuNetDetector(C.repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), score_threshold=0.7)
    with zipfile.ZipFile(C.repo_path(ZIP)) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
        sample = random.Random(SEED).sample(names, N)
        eye, order_ok, plausible, n_faces, found = [], 0, 0, [], 0
        for n in sample:
            img = C.decode_image(z.read(n))
            ds = det.detect(img)
            n_faces.append(len(ds))
            if not ds:
                continue
            found += 1
            lm = np.asarray(ds[0].landmarks)
            order_ok += int(lm[0, 0] < lm[1, 0])
            plausible += int(landmarks_plausible(lm))
            eye.append(ds[0].eye_dist)
    eye = np.array(eye)
    out = {
        "sample": N, "seed": SEED, "detector": det.id,
        "detected": found, "detection_rate": round(found / N, 4),
        "n_faces_hist": {str(k): int(v) for k, v in zip(*np.unique(n_faces, return_counts=True))},
        "landmark_order_image_left_eye_first": round(order_ok / max(found, 1), 4),
        "landmarks_plausible": round(plausible / max(found, 1), 4),
        "eye_dist_src_px": {q: round(float(np.percentile(eye, p)), 2) for q, p in
                            [("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]},
        "fraction_needing_upsampling": {
            str(s): round(float((TEMPLATE_EYE_DIST_512 * s / 512 / eye > 1.0).mean()), 4) for s in SIZES},
        "fraction_upsampling_over_5pct": {
            str(s): round(float((TEMPLATE_EYE_DIST_512 * s / 512 / eye > 1.05).mean()), 4) for s in SIZES},
        "run": C.run_info(),
    }
    C.write_json(C.repo_path("data/manifests/reference_size_probe_v1.json"), out)
    print({k: out[k] for k in ["detection_rate", "n_faces_hist", "landmark_order_image_left_eye_first",
                               "landmarks_plausible", "eye_dist_src_px", "fraction_needing_upsampling",
                               "fraction_upsampling_over_5pct"]})


if __name__ == "__main__":
    main()
