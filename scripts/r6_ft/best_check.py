"""R6-BEST — live check of POST /v3/enhance against the selector study rows (run inside the service container).

For study faces (selector_study.py rows.json) it calls /v3/enhance and checks that the single output returned is the
candidate the study's rule picks (higher lapd1), ORIGINAL when neither candidate was accepted.

    docker exec fes-ui python scripts/r6_ft/best_check.py [--n 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector_study import OUT, PROFILES, SETS  # noqa: E402

from fes_data.common import repo_path  # noqa: E402


def main(a):
    rows = json.loads(repo_path(f"{OUT}/rows.json").read_text())
    rows = rows[::max(1, len(rows) // a.n)][:a.n]
    client = httpx.Client(timeout=300)
    agree, n_enh, chosen, ms = 0, 0, {}, []
    for r in rows:
        data = repo_path(f"{SETS[r['set']][0]}/{r['name']}").read_bytes()
        t0 = time.time()
        b = client.post(f"{a.url}/v3/enhance", files={"image": ("x.png", data, "application/octet-stream")},
                        data={"input_mode": "face_chip", "return_image": "false"}).json()
        ok = [p for p in PROFILES if r[p]["enhanced"]]
        want = max(ok, key=lambda p: r[p]["lapd1"]) if ok else None
        got = (b.get("selection") or {}).get("chosen")
        agree += got == want and (b["status"] == "ENHANCED") == (want is not None)
        if b["status"] == "ENHANCED":
            n_enh += 1
            ms.append(time.time() - t0)
        chosen[f"{r['set']}:{got}"] = chosen.get(f"{r['set']}:{got}", 0) + 1
    print(json.dumps({"faces": len(rows), "enhanced": n_enh, "agrees_with_study_rule": agree, "chosen": chosen,
                      "median_s_enhanced": round(sorted(ms)[len(ms) // 2], 2) if ms else None}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=60)
    main(ap.parse_args())
