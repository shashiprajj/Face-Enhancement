"""Quick latency check of the running service's single-output endpoint (docker exec <svc> python scripts/perf/quick_timing.py).
Small faces: 30 archive TEST chips (128 px, face_chip). Large faces: the real photos that reach the models (full_image).
Each image is sent twice; the second time is reported. Prints medians for requests where a model ran."""

import json
import os
import statistics as st
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8080"
c = httpx.Client(timeout=300)


def run(data, mode):
    out = None
    for _ in range(2):
        t0 = time.perf_counter()
        j = c.post(URL + "/v3/enhance", files={"image": ("x", data, "application/octet-stream")},
                   data={"input_mode": mode, "return_image": "true"}).json()
        out = (round((time.perf_counter() - t0) * 1e3), j)
    return out


rd = c.get(URL + "/ready").json()
print(json.dumps({k: rd[k] for k in ("best_parallel", "r6_mirror_batch", "route_mode", "nafnet_backend")}))
names = sorted(os.listdir(ROOT / "Dataset-and-code/archive/input/test"))
small = []
for n in names[::max(1, len(names) // 100)][:30]:
    ms, j = run((ROOT / "Dataset-and-code/archive/input/test" / n).read_bytes(), "face_chip")
    if (j.get("nafnet") or {}).get("infer_s") is not None:
        small.append(ms)
large = []
for n in ("img86", "img118", "img123", "img147", "img148", "img153", "img154", "img155", "img166"):
    ms, j = run((ROOT / "Dataset-and-code/dataset" / f"{n}.jpg").read_bytes(), "full_image")
    if (j.get("nafnet") or {}).get("infer_s") is not None:
        large.append((ms, j.get("chip_px")))
rej, _ = run((ROOT / "Dataset-and-code/dataset/img2.jpg").read_bytes(), "full_image")
print(json.dumps({"small_128px": {"n": len(small), "p50_ms": st.median(small), "min": min(small), "max": max(small)},
                  "large": {"n": len(large), "p50_ms": st.median(x[0] for x in large), "min": min(x[0] for x in large),
                            "max": max(x[0] for x in large), "chip_px": sorted({x[1] for x in large})},
                  "gate_rejected_ms": rej}))
