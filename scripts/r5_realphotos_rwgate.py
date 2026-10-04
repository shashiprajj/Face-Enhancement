"""R5-3 step 7 — the 200 real photos through detector → gate → router → A4/original, frozen T6/T7 vs R5-3 rw gate.

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_realphotos_rwgate.py [--no-a4]

Behaviour analysis only: the photos have no same-person references, so nothing here measures identity.
Both policies run through the R5-1 service pipeline (full_image mode). The rw run uses the REAL A4 in the isolated
worker for every routed photo (fallbacks recorded); --no-a4 replaces it with a stub (routing counts only).
Per photo: decision, reason, category, route, used, fallback, the gate's features, is_gray, and
`human_recognizable` = NOT_REVIEWED (the protocol does not accept the model's own visual judgement as human evidence;
the contact sheets are provided for a human reviewer).
Writes data/manifests/phase5_rapid/rwgate/{realphotos.csv.gz, realphotos_summary.json} and contact sheets in
experiments/phase5_rapid/outputs/rwgate/ (git-ignored).
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import Counter

import cv2
import numpy as np

from fes_data import preprocess as P
from fes_data.common import repo_path, run_info, sha256_file, write_csv_gz, write_json
from fes_service.pipeline import EnhancementPipeline, face_chip

KEYS = ("iod_px", "det_score", "noise_sigma", "lap_norm_c", "lapd1_c", "blockiness", "luma_mean", "rms_contrast",
        "is_gray")


class Stub:
    def __call__(self, img, ctx):
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC), {}


def tile(img, label, w=224):
    h0, w0 = img.shape[:2]
    s = w / max(h0, w0)
    im = cv2.resize(img, (max(1, round(w0 * s)), max(1, round(h0 * s))),
                    interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    t = np.full((w + 30, w, 3), 255, np.uint8)
    t[:im.shape[0], :im.shape[1]] = im
    for i, line in enumerate(label.split("\n")[:2]):
        cv2.putText(t, line[:34], (2, w + 12 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1, cv2.LINE_AA)
    return t


def sheet(tiles, path, cols=6):
    if not tiles:
        return
    blank = np.full_like(tiles[0], 255)
    rows = [np.hstack(tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-a4", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(str(repo_path("Dataset-and-code/dataset/*.jpg"))), key=lambda f: int(os.path.basename(f)[3:-4]))
    old = EnhancementPipeline(Stub(), timeout_s=30, gate="t6")
    if a.no_a4:
        model = Stub()
    else:
        from fes_service.worker import IsolatedModel
        model = IsolatedModel("fes_service.models:a4_factory", timeout_s=300, threads=4)
    new = EnhancementPipeline(model, timeout_s=300, gate="rw")
    rows, chips, outs = [], {}, {}
    for f in files:
        name, data = os.path.basename(f), open(f, "rb").read()
        r0 = old.process(data, "full_image")
        r1 = new.process(data, "full_image")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        fr = P.preprocess(img, new.det, new.det_cfg, input_mode="face_crop")[0]
        if fr.status == P.OK and fr.aligned:
            chips[name] = face_chip(img, np.asarray(fr.detection.landmarks, np.float64))[0]
        if r1.status == "ENHANCED":
            outs[name] = cv2.imdecode(np.frombuffer(r1.image, np.uint8), cv2.IMREAD_COLOR)
        rows.append({"image": name, "sha256": sha256_file(f), "n_faces": r1.n_faces, "chip_px": r1.chip_px,
                     "t6_decision": r0.decision, "t6_reason": r0.reason, "t6_route": r0.route,
                     "rw_decision": r1.decision, "rw_reason": r1.reason, "rw_category": r1.bucket,
                     "rw_route": r1.route, "rw_used": r1.used, "rw_fallback": r1.fallback, "rw_status": r1.status,
                     "rw_ms": r1.timings_ms.get("total"), "warnings": "|".join(r1.warnings),
                     "human_recognizable": "NOT_REVIEWED",
                     # only features the gate actually computed for THIS photo (empty when rejected before the gate)
                     **{f"q_{k}": (r1.quality.get(k, "") if r1.quality else "") for k in KEYS}})
        print(name, r0.route, "->", r1.route, r1.reason, r1.status, r1.fallback, flush=True)
    if not a.no_a4:
        model.close()
    out = repo_path("data/manifests/phase5_rapid/rwgate")
    write_csv_gz(out / "realphotos.csv.gz", rows, list(rows[0]))
    cat = lambda k: dict(sorted(Counter(r[k] for r in rows).items()))  # noqa: E731
    summ = {"run": run_info(), "n": len(rows), "a4_real": not a.no_a4, "seals": new.seals,
            "before_t6": {"routes": cat("t6_route"), "decisions": cat("t6_decision")},
            "after_rw": {"routes": cat("rw_route"), "used": cat("rw_used"), "decisions": cat("rw_decision"),
                         "categories": cat("rw_category"), "reasons": cat("rw_reason"),
                         "fallbacks": dict(Counter(r["rw_fallback"] for r in rows if r["rw_fallback"])),
                         "grayscale_preserved": sum("grayscale_preserved" in r["warnings"] for r in rows)},
            "transitions": dict(Counter(f"{r['t6_route']}->{r['rw_route']}" for r in rows)),
            "identity_measured": False,
            "note": "No same-person references: routing behaviour only. human_recognizable not reviewed."}
    write_json(out / "realphotos_summary.json", summ)

    sd = repo_path("experiments/phase5_rapid/outputs/rwgate")
    sd.mkdir(parents=True, exist_ok=True)
    by = lambda pred: [r for r in rows if pred(r)]  # noqa: E731
    admitted = by(lambda r: r["rw_route"] == "A4")
    t = []
    for r in admitted:
        if r["image"] in chips:
            t.append(tile(chips[r["image"]], f"{r['image']} IN\n{r['rw_reason']} iod {float(r['q_iod_px']):.0f}"))
            if r["image"] in outs:
                t.append(tile(outs[r["image"]], f"{r['image']} A4 OUT\n{r['warnings']}"))
            else:
                t.append(tile(np.full((224, 224, 3), 200, np.uint8), f"{r['image']} -> ORIGINAL\nfallback {r['rw_fallback']}"))
    sheet(t, sd / "admitted_A4_in_out.jpg", cols=6)
    for name, pred in (("left_original_acceptable", lambda r: r["rw_reason"] == "acceptable"),
                       ("blocked_degraded_below_a4_size", lambda r: r["rw_reason"].startswith("degraded_below")),
                       ("left_original_degraded_no_a4_evidence", lambda r: r["rw_category"] == "degraded_no_a4_evidence"),
                       ("blocked_tiny_or_unrecognizable", lambda r: r["rw_category"] in ("tiny", "unrecognizable")),
                       ("blocked_photometric_or_compression", lambda r: r["rw_category"] in ("photometric", "compression")),
                       ("unsafe_uncertain_or_no_face", lambda r: r["rw_decision"] == "UNSAFE_OR_UNCERTAIN" and r["rw_category"] not in ("tiny", "unrecognizable"))):
        sel = by(pred)[:24]
        sheet([tile(chips.get(r["image"], cv2.imread(str(repo_path("Dataset-and-code/dataset/" + r["image"])))),
                    f"{r['image']}\n{r['rw_reason'][:30]}") for r in sel], sd / f"{name}.jpg")
    print({k: v for k, v in summ.items() if k != "run"})


if __name__ == "__main__":
    main()
