"""R6-FT — local check of the fine-tuned NAFNet on the 300 fixed FairFace VAL pairs used during training.

Rebuilds exactly the validation pairs of train_nafnet_face.py (same file list, seed 7, fixed degradations) and
compares: blurred input · original NAFNet-GoPro · fine-tuned, against the sharp target.
  PSNR (same definition as training) and SFace cosine to the sharp target (landmarks from the sharp image, YuNet
  on a 2x upscale because the pairs are 128 px crops). SFace here is a same-image fidelity proxy, not a
  cross-photo identity test.

    python scripts/r6_ft/eval_val300.py --data DIR --ft models/M28_nafnet/nafnet_face_ft.pth --out DIR
DIR (data) holds fairface-img-margin025-trainval.zip and NAFNet-GoPro-width64.pth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_nafnet_face import WEIGHTS, ZIP, FaceBlur, NAFNet  # noqa: E402

from fes_bench.metrics import YUNET, SFace  # noqa: E402
from fes_data.common import repo_path  # noqa: E402
from fes_data.preprocess import YuNetDetector  # noqa: E402

KINDS = ("gauss", "disk", "motion", "none")


def staged_load(path):
    """Copy to local tmp first: large torch.load reads through the bind mount fail with ENOMEM (P3-T2 F5)."""
    tmp = os.path.join(tempfile.gettempdir(), os.path.basename(path))
    shutil.copyfile(path, tmp)
    sha = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
    net = NAFNet()
    net.load_state_dict(torch.load(tmp, map_location="cpu", weights_only=True)["params"], strict=True)
    os.remove(tmp)
    return net.eval(), sha


def psnr1(a, b):
    mse = max(float(((a.clamp(0, 1) - b) ** 2).mean()), 1e-10)
    return 10 * np.log10(1 / mse)


def to_bgr(t):
    return (t.clamp(0, 1).permute(1, 2, 0).numpy()[..., ::-1] * 255 + 0.5).astype(np.uint8)


def main(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    zp = os.path.join(a.data, ZIP)
    with zipfile.ZipFile(zp) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
    vnames = [n for n in names if n.startswith("val/")]
    val = vnames[::max(1, len(vnames) // 300)][:300]
    ds = FaceBlur(zp, val, 7, fixed=True)
    pairs = [ds[i] for i in range(len(ds))]
    print(f"val {len(pairs)}", flush=True)

    orig, sha_o = staged_load(os.path.join(a.data, WEIGHTS))
    ft, sha_f = staged_load(a.ft)
    print(f"orig {sha_o[:16]} · ft {sha_f[:16]}", flush=True)

    d = torch.stack([p[0] for p in pairs])
    s = torch.stack([p[1] for p in pairs])
    kinds = [p[2] for p in pairs]
    outs = {}
    for tag, net in (("orig", orig), ("ft", ft)):
        t0 = time.time()
        with torch.inference_mode():
            o = torch.cat([net(d[i:i + a.bs]) for i in range(0, len(d), a.bs)])
        outs[tag] = torch.nan_to_num(o.float(), nan=0.0, posinf=1.0, neginf=0.0)
        print(f"{tag} {(time.time() - t0) / len(d):.2f}s/img", flush=True)

    det = YuNetDetector(repo_path(YUNET), score_threshold=0.7, nms_threshold=0.3, max_side=640)
    sf = SFace()
    rows = []
    for i in range(len(d)):
        r = {"name": val[i], "kind": kinds[i], "psnr_input": psnr1(d[i], s[i]),
             "psnr_orig": psnr1(outs["orig"][i], s[i]), "psnr_ft": psnr1(outs["ft"][i], s[i])}
        sharp = to_bgr(s[i])
        faces = det.detect(cv2.resize(sharp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        if faces:
            lm = np.asarray(faces[0].landmarks, np.float32) / 2.0
            e = sf.embed(sharp, lm)
            for tag, img in (("input", d[i]), ("orig", outs["orig"][i]), ("ft", outs["ft"][i])):
                r[f"sface_{tag}"] = float(e @ sf.embed(to_bgr(img), lm))
        rows.append(r)

    def agg(sel):
        out = {"n": len(sel), "n_face": sum("sface_ft" in r for r in sel)}
        for k in ("psnr_input", "psnr_orig", "psnr_ft"):
            out[k] = float(np.mean([r[k] for r in sel]))
        fr = [r for r in sel if "sface_ft" in r]
        if fr:
            for k in ("sface_input", "sface_orig", "sface_ft"):
                out[k] = float(np.mean([r[k] for r in fr]))
            dlt = np.array([r["sface_ft"] - r["sface_input"] for r in fr])
            out["sface_ft_better"], out["sface_ft_worse"] = int((dlt > 0.02).sum()), int((dlt < -0.02).sum())
        dp = np.array([r["psnr_ft"] - r["psnr_input"] for r in sel])
        out["psnr_ft_worse_than_input"] = int((dp < 0).sum())
        return out

    res = {"ft_sha256": sha_f, "orig_sha256": sha_o, "all": agg(rows),
           "by_kind": {k: agg([r for r in rows if r["kind"] == k]) for k in KINDS}, "rows": rows}
    with open(os.path.join(a.out, "val300_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    # sheets: input | original NAFNet | fine-tuned | sharp — first 8 of each kind, and the 16 worst fine-tuned
    def sheet(idx, path):
        tiles = [np.hstack([to_bgr(x[i]) for x in (d, outs["orig"], outs["ft"], s)]) for i in idx]
        cv2.imwrite(path, cv2.resize(np.vstack(tiles), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

    for k in KINDS:
        idx = [i for i, kk in enumerate(kinds) if kk == k][:8]
        if idx:
            sheet(idx, os.path.join(a.out, f"sheet_{k}.png"))
    worst = sorted(range(len(rows)), key=lambda i: rows[i]["psnr_ft"] - rows[i]["psnr_input"])[:16]
    sheet(worst, os.path.join(a.out, "sheet_worst16.png"))
    # review set: one image per face, LEFT = blurred input, RIGHT = fine-tuned output (3x, captioned), by blur kind
    label = {"gauss": "gaussian_blur", "disk": "defocus_blur", "motion": "motion_blur", "none": "already_sharp"}
    sc, bar = 3, 22
    csv = ["file,blur_kind,source,psnr_input,psnr_enhanced,psnr_gain,sface_input,sface_enhanced,sface_gain"]
    html = ["<!doctype html><meta charset='utf-8'><title>R6-FT val300 review</title><style>body{font:14px sans-serif;"
            "background:#111;color:#ddd;margin:16px}figure{display:inline-block;margin:6px}img{display:block}"
            "figcaption{font-size:12px;color:#aaa}</style><h1>Fine-tuned NAFNet: blurred input (left) vs enhanced "
            "(right)</h1>"]
    for k in KINDS:
        os.makedirs(os.path.join(a.out, "pairs", label[k]), exist_ok=True)
        idx = [i for i, kk in enumerate(kinds) if kk == k]
        html.append(f"<h2>{label[k]} ({len(idx)} faces)</h2>")
        for i in idx:
            r = rows[i]
            up = [cv2.resize(to_bgr(x[i]), None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) for x in (d, outs["ft"])]
            w = up[0].shape[1]
            img = np.vstack([np.zeros((bar, 2 * w + 4, 3), np.uint8),
                             np.hstack([up[0], np.full((up[0].shape[0], 4, 3), 255, np.uint8), up[1]])])
            for j, txt in enumerate(("BLURRED INPUT", "ENHANCED (fine-tuned)")):
                cv2.putText(img, txt, (j * (w + 4) + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
            rel = f"pairs/{label[k]}/{i:03d}_{label[k]}.png"
            cv2.imwrite(os.path.join(a.out, rel), img)
            si, sf_ = r.get("sface_input"), r.get("sface_ft")
            f3 = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
            csv.append(f"{rel},{label[k]},{r['name']},{r['psnr_input']:.2f},{r['psnr_ft']:.2f},"
                       f"{r['psnr_ft'] - r['psnr_input']:.2f},{f3(si)},{f3(sf_)},"
                       f"{'' if si is None else format(sf_ - si, '.3f')}")
            cap = f"#{i:03d} · PSNR {r['psnr_input']:.1f} → {r['psnr_ft']:.1f} dB"
            if si is not None:
                cap += f" · SFace {si:.2f} → {sf_:.2f}"
            html.append(f"<figure><img src='{rel}' loading='lazy'><figcaption>{cap}</figcaption></figure>")
    with open(os.path.join(a.out, "pairs_index.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")
    with open(os.path.join(a.out, "review.html"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))

    res["worst16"] = [{"name": rows[i]["name"], "kind": rows[i]["kind"],
                       "dpsnr": rows[i]["psnr_ft"] - rows[i]["psnr_input"]} for i in worst]
    with open(os.path.join(a.out, "val300_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"all": res["all"], "by_kind": res["by_kind"]}, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    main(ap.parse_args())
