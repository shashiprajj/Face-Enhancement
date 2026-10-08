"""R6-0 — visual tail review of the population outputs (host). Builds sheets from the saved strips
(chip | v0 | medluma | meanluma) for:
  acc_tail   accepted by the run tracer, highest luma-dottedness / flip disagreement of the medluma output
  acc_near   accepted with run_dot_c_max closest to the threshold
  acc_rand   random accepted (seed 0)
  rej        every rejected case (what the tracer throws away)
  r58_only   rejected by R5-8 but accepted by the tracer (disagreements)

    python scripts/r6_0/pop_review.py SPLIT OUTDIR
"""
import gzip
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "data/manifests/phase5_rapid/r6_0"


def main(split, outdir):
    T = json.loads((M / "analysis.json").read_text())["thresholds"]["run_dot_c_max"]
    rows = {}
    for line in gzip.open(M / f"population_{split}.jsonl.gz", "rt").read().splitlines():
        r = json.loads(line)
        rows.setdefault((r["name"], r["mode"]), {})[r["variant"]] = r
    cs = [(k, v) for k, v in rows.items() if "v0" in v and "medluma" in v]
    acc = [x for x in cs if x[1]["v0"]["run_dot_c_max"] <= T]
    rej = [x for x in cs if x[1]["v0"]["run_dot_c_max"] > T]
    sets = {
        "acc_tail": sorted(acc, key=lambda x: -x[1]["medluma"]["t_dot_l_max"])[:21]
        + sorted(acc, key=lambda x: -x[1]["medluma"]["f_luma_win_max"])[:14],
        "acc_near": sorted(acc, key=lambda x: -x[1]["v0"]["run_dot_c_max"])[:14],
        "acc_rand": random.Random(0).sample(acc, min(28, len(acc))),
        "rej": rej,
        "r58_only": [x for x in acc if not (x[1]["v0"]["guard_pass"] and x[1]["v0"]["r58_local_pass"])],
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, items in sets.items():
        seen, uniq = set(), []
        for (n, m), v in items:
            if (n, m) not in seen:
                seen.add((n, m))
                uniq.append((n, m))
        index[name] = [f"{m}/{n}" for n, m in uniq]
        for s in range(0, len(uniq), 7):
            strips = []
            for i, (n, m) in enumerate(uniq[s:s + 7], start=s):
                im = cv2.imread(str(ROOT / f"experiments/phase5_rapid/outputs/r6_0/pop_{split}/{m}/{n}.png"))
                if im is None:
                    continue
                h = im.shape[0]
                im = cv2.resize(im, (im.shape[1] * 192 // h, 192), interpolation=cv2.INTER_NEAREST)
                cv2.putText(im, f"{i}", (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                strips.append(np.pad(im, ((0, 4), (0, 0), (0, 0))))
            if strips:
                w = max(x.shape[1] for x in strips)
                cv2.imwrite(str(out / f"{split}_{name}_{s // 7:02d}.png"),
                            np.vstack([np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0))) for x in strips]))
    (out / f"{split}_index.json").write_text(json.dumps(index, indent=0))
    print({k: len(v) for k, v in index.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
