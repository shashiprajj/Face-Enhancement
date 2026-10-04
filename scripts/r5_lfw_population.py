"""R5-2 — LFW identity-labelled RESEARCH-PROXY population (small, deterministic).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/r5_lfw_population.py [--ids 80]

Source: LFW (datasets/raw/lfw/lfw.tgz, sha256 055f7d9c…, 13 233 images / 5 749 identities; figshare mirror used by
scikit-learn). Phase 1 status DS09 REVIEW_REQUIRED (no licence grant; news photos) → local research only, never
training, never production evidence (user instruction 2026-10-04).

Selection: identities with ≥ 2 images, ordered by sha256("r5-lfw|name"); kept if YuNet finds a face with score ≥ 0.85
on both the reference (the identity's first image) and the query source (its second image) — two DIFFERENT photos, so
genuine matching is cross-photo, as in real use. First `--ids` identities that pass.
Impostor pool: the first image of 400 further identities (same hash order, single-image identities allowed) + the
selected references; used only for FAR thresholds and false-accept / rank-1 checks.

Query degradations (applied to the full 250×250 query photo; seed = sha256(name|cell)); LFW faces are ~40 px IOD, so
blur σ here ≈ 2× the same blur relative to the face on 224-px FairFace chips:
  clean · grain-mild σ5 · noise-strong σ15 · noise-severe σ25 · blur-moderate σ1 · blur-strong σ2 ·
  blur-strong+grain (σ2 then grain σ4) · motion (length 9, random angle) · lowres-x0.5 (whole photo 125 px) ·
  jpeg-q15 · vintage-mix (blur σ1 → grain σ6 → JPEG q40)
Writes datasets/work/r5lfw/queries/<cell>/<name>.png (git-ignored) and
data/manifests/phase5_rapid/lfw/{selection.csv.gz, queries.csv.gz, population_report.json}.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from fes_data import degrade as D
from fes_data import preprocess as P
from fes_data.common import repo_path, run_info, sha256_bytes, sha256_file, write_csv_gz, write_json

LFW = "datasets/raw/lfw/lfw"
TGZ_SHA = "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"
CELLS = ["clean", "grain-mild", "noise-strong", "noise-severe", "blur-moderate", "blur-strong", "blur-strong+grain",
         "motion", "lowres-x0.5", "jpeg-q15", "vintage-mix"]


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def degrade(img: np.ndarray, cell: str, rng: np.random.Generator) -> np.ndarray:
    g = lambda x, s: D.add_gaussian_noise(x, s, rng)  # noqa: E731
    if cell == "clean":
        return img.copy()
    if cell == "grain-mild":
        return g(img, 5)
    if cell == "noise-strong":
        return g(img, 15)
    if cell == "noise-severe":
        return g(img, 25)
    if cell == "blur-moderate":
        return D.gaussian_blur(img, 1.0)
    if cell == "blur-strong":
        return D.gaussian_blur(img, 2.0)
    if cell == "blur-strong+grain":
        return g(D.gaussian_blur(img, 2.0), 4)
    if cell == "motion":
        return D.motion_blur(img, 9, float(rng.uniform(0, 180)))
    if cell == "lowres-x0.5":
        return cv2.resize(img, (125, 125), interpolation=cv2.INTER_AREA)
    if cell == "jpeg-q15":
        return D.jpeg(img, 15)
    if cell == "vintage-mix":
        return D.jpeg(g(D.gaussian_blur(img, 1.0), 6), 40)
    raise ValueError(cell)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, default=80)
    a = ap.parse_args()
    assert sha256_file(repo_path("datasets/raw/lfw/lfw.tgz")) == TGZ_SHA
    root = repo_path(LFW)
    people = {p.name: sorted(p.glob("*.jpg")) for p in root.iterdir() if p.is_dir()}
    order = sorted(people, key=lambda n: h(f"r5-lfw|{n}"))
    det = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)

    def ok(path: Path) -> bool:
        d = det.detect(cv2.imread(str(path), cv2.IMREAD_COLOR))
        return bool(d) and d[0].score >= 0.85

    sel, pool = [], []
    for n in order:
        if len(sel) < a.ids and len(people[n]) >= 2 and ok(people[n][0]) and ok(people[n][1]):
            sel.append(n)
        elif len(sel) >= a.ids or len(people[n]) < 2:
            if len(pool) < 400 and n not in sel and ok(people[n][0]):
                pool.append(n)
        if len(sel) >= a.ids and len(pool) >= 400:
            break
    rel = lambda p: p.relative_to(repo_path(".")).as_posix()  # noqa: E731
    selection = [{"identity": n, "role": "selected", "ref_path": rel(people[n][0]), "query_src": rel(people[n][1]),
                  "ref_sha256": sha256_file(people[n][0]), "n_images": len(people[n])} for n in sel]
    selection += [{"identity": n, "role": "impostor_pool", "ref_path": rel(people[n][0]), "query_src": "",
                   "ref_sha256": sha256_file(people[n][0]), "n_images": len(people[n])} for n in pool]
    rows = []
    for n in sel:
        src = cv2.imread(str(people[n][1]), cv2.IMREAD_COLOR)
        for cell in CELLS:
            rng = np.random.default_rng(int(h(f"{n}|{cell}")[:16], 16))
            q = degrade(src, cell, rng)
            ok_, buf = cv2.imencode(".png", q, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            path = f"datasets/work/r5lfw/queries/{cell}/{n}.png"
            repo_path(path).parent.mkdir(parents=True, exist_ok=True)
            repo_path(path).write_bytes(buf.tobytes())
            rows.append({"query_id": f"{n}.{cell}", "identity": n, "cell": cell, "query_path": path,
                         "query_sha256": sha256_bytes(buf.tobytes()), "query_w": q.shape[1], "query_h": q.shape[0],
                         "ref_path": rel(people[n][0])})
    out = repo_path("data/manifests/phase5_rapid/lfw")
    write_csv_gz(out / "selection.csv.gz", selection, list(selection[0]))
    write_csv_gz(out / "queries.csv.gz", rows, list(rows[0]))
    write_json(out / "population_report.json", {
        "run": run_info(), "source": "LFW (figshare 5976018 = scikit-learn mirror)", "tgz_sha256": TGZ_SHA,
        "licence_status": "DS09 REVIEW_REQUIRED (no grant) — local research proxy only",
        "identities": len(sel), "impostor_pool": len(pool), "cells": CELLS, "queries": len(rows),
        "reference_vs_query": "different photos of the same person"})
    print(len(sel), "identities,", len(pool), "pool,", len(rows), "queries")


if __name__ == "__main__":
    main()
