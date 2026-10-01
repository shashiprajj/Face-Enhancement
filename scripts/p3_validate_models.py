"""P3-T2 — model-load validation report for every acquired candidate network (and every adapter configuration).

    scripts/dock.sh python scripts/p3_validate_models.py

Per network: weights SHA-256 vs P1-T9, vendored-source SHA-256, safe load (weights_only=True), missing/unexpected
keys, parameter count, one forward on a real degraded trial input, finite/range checks.
Per adapter configuration: enhance() twice on the same input → output contract (224×224×3 uint8) and
byte-identical repeat (determinism at a fixed thread count).
Writes data/manifests/phase3/model_validation_report.json.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import torch

from fes_bench import adapters as AD
from fes_bench import archs
from fes_data.common import load_config, read_csv_gz, repo_path, run_info, sha256_bytes, sha256_file, write_json

FORWARD = {  # key → (input builder, forward) mirroring the adapter call
    "M03": ("f512", lambda n, x: n(x)[0]), "M05": ("f512", lambda n, x: n(x)["main_dec"]),
    "M23": ("d", lambda n, x: n(x)), "M24": ("d", lambda n, x: AD._swinir_x4(n, x)),
    "M30": ("d", lambda n, x: n(x)[0]), "M28-SIDD": ("d", lambda n, x: n(x)), "M28-GoPro": ("d", lambda n, x: n(x)),
}


def main() -> None:
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cfg = load_config("configs/phase3/benchmark_p3.yaml")
    pop = read_csv_gz(repo_path("data/manifests/phase3/population_trial.csv.gz"))
    p56 = next(p for p in pop if p["cell"] == "lowres-moderate")
    deg = cv2.imread(str(repo_path(p56["degraded_path"])))
    M = np.array(json.loads(p56["ffhq512_transform_2x3"]))
    from fes_bench import geometry as G
    f512 = G.warp(deg, G.d_to_f(56, M), (512, 512))
    rep = {"run": run_info({"torch": torch.__version__}), "input_pair": p56["pair_id"], "networks": {},
           "vendored_sources": {}, "adapters": {}}
    for rel, sha in archs.VENDORED_SHA256.items():
        rep["vendored_sources"][rel] = {"sha256_ok": sha256_file(archs.THIRD_PARTY / rel) == sha}

    for key in AD.WEIGHTS:
        r = {"asset_id": AD.WEIGHTS[key][0], "weights": AD.WEIGHTS[key][1]}
        try:
            net, lr = AD.load_net(key)            # raises on checksum mismatch or missing keys
            r.update(lr, sha256_ok=True)
            kind, fwd = FORWARD[key]
            x = AD.bgr_u8_to_rgb_t(f512) * 2 - 1 if kind == "f512" else AD.bgr_u8_to_rgb_t(deg)
            t = time.perf_counter()
            with torch.inference_mode():
                y = fwd(net, x)
            r.update(forward_ok=True, forward_s=round(time.perf_counter() - t, 3), in_shape=list(x.shape),
                     out_shape=list(y.shape), out_finite=bool(torch.isfinite(y).all()),
                     out_range=[round(float(y.min()), 4), round(float(y.max()), 4)], status="VALIDATED")
            del net
        except Exception as e:
            r.update(status="FAILED", error=f"{type(e).__name__}: {e}")
        rep["networks"][key] = r
        print(key, r.get("status"), r.get("n_model_params"), r.get("missing_keys"), r.get("unexpected_keys"),
              r.get("out_shape"), r.get("out_range"), r.get("error", ""), flush=True)

    for cid in cfg["configurations"]:
        a = AD.build(cid)
        r = {"describe": a.describe()}
        try:
            a.load()
            outs = [a.enhance(deg, {"s": 56, "M_e2f": M})[0] for _ in range(2)]
            r.update(contract_ok=bool(outs[0].shape == (224, 224, 3) and outs[0].dtype == np.uint8),
                     repeat_identical=sha256_bytes(outs[0].tobytes()) == sha256_bytes(outs[1].tobytes()),
                     status="VALIDATED")
        except Exception as e:
            r.update(status="FAILED", error=f"{type(e).__name__}: {e}")
        rep["adapters"][cid] = r
        print(cid, r["status"], r.get("contract_ok"), r.get("repeat_identical"), flush=True)
        del a
    write_json(repo_path("data/manifests/phase3/model_validation_report.json"), rep)


if __name__ == "__main__":
    main()
