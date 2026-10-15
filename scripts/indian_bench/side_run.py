"""Validation of the SIDE-FACE profile `best_side` against the finished /v3 (`best`) benchmark. Client only.

    python scripts/indian_bench/side_run.py --url http://127.0.0.1:8082 [--regression 2000] [--limit N]
    python scripts/indian_bench/side_run.py --summary

Sent to POST /v1/enhance with profile=best_side (same degraded bytes as the benchmark):
  turned   every variant `best` returned unchanged because the face was turned: outcome original_no_face, or the gate
           reason NAFNET_REJECT_UNSAFE_ALIGNMENT. Only these can change under best_side.
  control  a seeded random sample of the other variants: best_side must give the SAME status, reason and output
           bytes as the benchmark's `best` answer (regression check).
Stored under <bench>/side_profile_run/: results.jsonl (resumable), enhanced/<variant_id>.png for the newly ENHANCED,
side_by_side/<variant_id>.jpg (CLEAN | DEGRADED | FES OUTPUT in the service's box crop, the crop verified against the
chip the service analysed) and summary.{md,json}. Diagnostics: PSNR / SSIM against the clean reference in that crop.
No identity metric: SFace is not reliable on profile faces.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import statistics as st
import time
from collections import Counter
from pathlib import Path

import common as C
import enhance as E

TYPES_BLUR = ("gaussian_blur", "motion_blur_000", "motion_blur_045", "motion_blur_090", "defocus_blur", "box_blur",
              "camera_shake", "defocus_motion", "down_up", "resample")


def read_all(out_dir) -> list[dict]:
    return [r for p in sorted(out_dir.glob("results_*.jsonl")) for r in C.read_jsonl(p)]


def run(a, cfg, root, out_dir):
    import cv2
    import numpy as np
    import compare as CMP
    c = CMP.ctx()
    PL, M = c["PL"], c["M"]
    ready = E.get_json(a.url + "/ready")
    if not ready.get("ready") or "best_side" not in ready.get("profiles", []):
        raise SystemExit(f"service not ready or without profile best_side: {ready.get('error')!r} {ready.get('profiles')}")
    (out_dir / "service_ready.json").write_text(json.dumps(ready, indent=1), encoding="utf-8")
    ids = {r["sample_id"] for r in C.scope(cfg)}
    var = {v["variant_id"]: v for v in C.read_csv(root / "manifests" / "generated_variants.csv") if v["sample_id"] in ids}
    base = {r["variant_id"]: r for r in C.read_csv(root / "manifests" / "enhancement_results.csv") if r["variant_id"] in var}
    turned = sorted(k for k, r in base.items() if r["outcome"] == "original_no_face"
                    or r["reason"] == "NAFNET_REJECT_UNSAFE_ALIGNMENT")
    rest = sorted(set(base) - set(turned))
    control = sorted(random.Random(cfg["seed"]).sample(rest, min(a.regression, len(rest))))
    todo = [("control", k) for k in control] + [("turned", k) for k in turned]
    if a.expressions:
        todo = [t for t in todo if var[t[1]]["expression"] in a.expressions.split(",")]
    if a.types:
        todo = [t for t in todo if var[t[1]]["type"] in a.types.split(",")]
    k, n = map(int, a.shard.split("/"))
    todo = [t for t in todo if C.stable_seed("shard", var[t[1]]["sample_id"]) % n == k]      # whole originals per shard
    done = {r["variant_id"] for r in read_all(out_dir) if r["outcome"] != "failed"}
    todo = [t for t in todo if t[1] not in done][:a.limit or None]
    print(f"turned {len(turned)} · control {len(control)} · to send now {len(todo)}", flush=True)
    (out_dir / "enhanced").mkdir(exist_ok=True)
    (out_dir / "side_by_side").mkdir(exist_ok=True)
    t0, counts = time.time(), Counter()
    with open(out_dir / f"results_{k}of{n}.jsonl", "a", encoding="utf-8") as fh:
        for i, (group, vid) in enumerate(todo, 1):
            v, b = var[vid], base[vid]
            data = (root / v["degraded_path"]).read_bytes()
            http, j, rtt = E.post(a.url + "/v1/enhance", data, {"input_mode": "full_image", "return_image": "true",
                                                                "profile": "best_side"}, 300)
            oc = E.outcome(http, j)
            nf = j.get("nafnet") or {}
            rec = {"variant_id": vid, "group": group, "sample_id": v["sample_id"], "expression": v["expression"],
                   "type": v["type"], "level": v["level"], "outcome": oc, "http_status": http, "rtt_ms": rtt,
                   "framing": nf.get("framing", "landmarks" if nf else ""), "landmark_framing": nf.get("landmark_framing", ""),
                   "baseline_outcome": b["outcome"], "baseline_reason": b["reason"], **E.flatten(j)}
            img = base64.b64decode(j["image_b64"]) if j.get("image_b64") else None
            if img is not None:
                rec["output_sha256"] = C.sha256_bytes(img)
                rec["same_as_baseline"] = (oc == b["outcome"] and (j.get("reason") or "") == b["reason"]
                                           and rec["output_sha256"] == b["output_sha256"])
            if oc == "enhanced" and rec["framing"] == "detection_box":
                try:
                    (out_dir / "enhanced" / f"{vid}.png").write_bytes(img)
                    deg = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    clean = cv2.imread(str(root / v["clean_path"]), cv2.IMREAD_COLOR)
                    o = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                    s_chip = cv2.imdecode(np.frombuffer(base64.b64decode(j["chip_b64"]), np.uint8), cv2.IMREAD_COLOR)
                    fr = c["P"].preprocess(deg, c["det"], c["det_cfg"], input_mode="face_crop")[0]
                    box = fr.detection.bbox_xywh
                    d_chip, side = PL.face_chip_box(deg, box)
                    c_chip, _ = PL.face_chip_box(clean, box)
                    valid = PL.face_chip_box(np.full(deg.shape, 255, np.uint8), box)[0][..., 0] >= 128
                    mask = None if valid.all() else valid
                    match = bool(np.array_equal(d_chip, s_chip))
                    rec.update(chip_match=match, box_side=side, chip_px=d_chip.shape[0],
                               psnr_degraded=round(M.psnr_rgb(d_chip, c_chip, mask), 3),
                               psnr_output=round(M.psnr_rgb(o, c_chip, mask), 3),
                               ssim_degraded=round(M.ssim_y(d_chip, c_chip, mask), 4),
                               ssim_output=round(M.ssim_y(o, c_chip, mask), 4),
                               sharpness_ratio_degraded=round(M.sharpness_ratio(d_chip, c_chip), 4),
                               sharpness_ratio_output=round(M.sharpness_ratio(o, c_chip), 4))
                    panels = [cv2.resize(p, (448, 448), interpolation=cv2.INTER_LINEAR) for p in (c_chip, d_chip, o)]
                    header = (f"{v['expression']} · {v['sample_id']} · {v['type']} · level {v['level']} ({v['category']})",
                              f"{v['order']}   |   FES profile best_side: ENHANCED (side-face box framing) · model "
                              f"{rec.get('selection_chosen') or ''}   ·   /v3 `best` had returned the input: {b['reason']}")
                    footer = (f"view: the service's detection-box crop (side {side} px of the 450x800 frame), shown at 448 px "
                              f"(bilinear), identical for the 3 panels" + ("" if match else "  [crop differs from the service's chip]"))
                    CMP.triptych(panels, ["CLEAN ORIGINAL", "DEGRADED INPUT", "FES ENHANCED OUTPUT (best_side)"], header,
                                 footer, CMP.COLOURS["enhanced"]).save(out_dir / "side_by_side" / f"{vid}.jpg",
                                                                       quality=92, subsampling=0)
                except Exception as e:  # noqa: BLE001 — recorded
                    rec["client_error"] = f"{type(e).__name__}: {e}"[:200]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            counts[(group, oc)] += 1
            if i % 1000 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} · {el / 60:.1f} min · {el / i:.2f} s/request · ETA {(len(todo) - i) * el / i / 3600:.1f} h · "
                      f"{dict(counts)}", flush=True)


def summary(cfg, root, out_dir):
    recs = list({r["variant_id"]: r for r in read_all(out_dir)}.values())
    scope = {r["sample_id"]: r for r in C.scope(cfg)}
    ctrl = [r for r in recs if r["group"] == "control"]
    turned = [r for r in recs if r["group"] == "turned"]
    md = ["# Side-face profile `best_side` — validation against the `/v3` benchmark", "",
          f"Requests: {len(recs)} (turned {len(turned)}, control {len(ctrl)}); failed: "
          f"{sum(r['outcome'] == 'failed' for r in recs)}.", "",
          "## Regression control (variants `best` did not stop on a turned face)", "",
          f"Same status, reason and output bytes as the `/v3` answer: **{sum(bool(r.get('same_as_baseline')) for r in ctrl)} "
          f"of {len(ctrl)}** (of which ENHANCED in both: {sum(r['outcome'] == 'enhanced' for r in ctrl)}).", ""]
    diff = [r for r in ctrl if not r.get("same_as_baseline")]
    if diff:
        md += ["Differences:", "", C.md_table([{"variant": r["variant_id"], "best": f"{r['baseline_outcome']} {r['baseline_reason']}",
                                               "best_side": f"{r['outcome']} {r['reason']}"} for r in diff[:20]],
                                             ["variant", "best", "best_side"]), ""]
    md += ["## Turned faces (`best` had returned the input unchanged)", ""]
    rows = []
    for name, g in (("side / profile originals (no usable frontal face)", [r for r in turned if scope[r["sample_id"]]["face_usable"] != "True"]),
                    ("frontal-usable originals, turned or uncertain pose", [r for r in turned if scope[r["sample_id"]]["face_usable"] == "True"]),
                    ("all", turned)):
        c = Counter(r["outcome"] for r in g)
        rows.append({"group": name, "requests": len(g), "originals": len({r["sample_id"] for r in g}),
                     "now ENHANCED": c["enhanced"], "originals with ≥1 enhanced": len({r["sample_id"] for r in g if r["outcome"] == "enhanced"}),
                     "gate: not routed": c["original_gate"], "no face": c["original_no_face"], "guard fallback": c["fallback_guard"],
                     "failed": c["failed"]})
    md += [C.md_table(rows, list(rows[0])), "",
           "Reasons for the ones still not enhanced: " + ", ".join(
               f"{k} {v}" for k, v in Counter(r["fallback"] or r["reason"] for r in turned if r["outcome"] != "enhanced").most_common(8)), ""]
    enh = [r for r in turned if r["outcome"] == "enhanced" and "psnr_output" in r]
    md += ["## Enhancement rate and image-quality diagnostics on the turned faces, by type and level", "",
           "`rate` = ENHANCED / turned-face requests of that cell. PSNR / SSIM against the clean reference in the "
           "service's box crop; diagnostics only, no identity metric (SFace is not reliable on profile faces).", ""]
    rows = []
    types = list(dict.fromkeys(r["type"] for r in sorted(turned, key=lambda r: r["type"])))
    for t in [*types, "ALL"]:
        g = [r for r in turned if t == "ALL" or r["type"] == t]
        e = [r for r in enh if t == "ALL" or r["type"] == t]
        row = {"type": t}
        for l in "1234":
            gl = [r for r in g if r["level"] == l]
            row[f"rate L{l}"] = f"{100 * sum(r['outcome'] == 'enhanced' for r in gl) / len(gl):.0f} % (n={len(gl)})" if gl else "-"
        if e:
            d = [r["psnr_output"] - r["psnr_degraded"] for r in e]
            row.update({"enhanced": len(e), "Δ PSNR": round(st.fmean(d), 2), "PSNR better in": f"{100 * sum(x > 0 for x in d) / len(d):.0f} %",
                        "Δ SSIM": round(st.fmean(r["ssim_output"] - r["ssim_degraded"] for r in e), 4)})
        rows.append(row)
    md += [C.md_table(rows, ["type", "rate L1", "rate L2", "rate L3", "rate L4", "enhanced", "Δ PSNR", "PSNR better in", "Δ SSIM"]), "",
           f"Box crops that match the chip the service analysed: {sum(bool(r.get('chip_match')) for r in enh)} of {len(enh)}.",
           f"Client-side errors while building comparisons: {sum(bool(r.get('client_error')) for r in recs)}.", ""]
    rt = sorted(r["rtt_ms"] for r in recs)
    md += [f"Round trip: median {rt[len(rt) // 2]} ms, p95 {rt[int(.95 * (len(rt) - 1))]} ms, sum {sum(rt) / 3.6e6:.2f} h.", ""]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8082")
    ap.add_argument("--regression", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--name", default="full", help="sub-folder of side_profile_run for this run")
    ap.add_argument("--expressions", default="", help="comma list: only these expression folders")
    ap.add_argument("--types", default="", help="comma list: only these degradation types")
    ap.add_argument("--shard", default="0/1", help="k/n: this process takes shard k of n (whole originals)")
    a = ap.parse_args()
    cfg = C.load_config()
    root = Path(cfg["bench_root"])
    out = root / "side_profile_run" / a.name
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary:
        run(a, cfg, root, out)
    summary(cfg, root, out)
