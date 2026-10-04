"""P3-T5/P3-T6 — per-pair metrics for every configuration run on a population.

    scripts/dock.sh python scripts/p3_evaluate.py --population main [--configs B0 A1 ...]

Image quality, geometry and identity are computed in the E-frame against the clean reference, under the same
per-reference evaluation mask for every configuration (P3-T1 §4.4, §7). Outputs are verified against the run
manifest's SHA-256 before scoring.

Identity: SFace (E01, test-only). Face embeddings are held **in memory only** and never written to disk (same rule as
P2-T7). Impostor scores: each output vs the clean references of all *other* references of the population.
The TAR/FAR operating point is taken from clean-reference impostor scores over the 700-ref impostor pool (FAR 1e-2).

Writes data/manifests/phase3/metrics/<population>/<config>.csv.gz and identity_calibration.json.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from fes_bench import geometry as G
from fes_bench import metrics as MT
from fes_bench import read_bytes_retry
from fes_data.common import (load_config, read_csv_gz, repo_path, run_info, sha256_bytes, sha256_file, write_csv_gz,
                             write_json)

FIELDS = ["pair_id", "ref_id", "cell", "profile", "severity", "in_size", "config_id", "status",
          "psnr", "ssim", "lpips", "sharp_ratio", "color_de", "det_found", "det_score", "lm_nme",
          "id_cos", "imp_cos_mean", "imp_cos_max", "imp_n", "imp_n_ge_thr", "id_ge_thr", "race", "gender"]
FAR = 1e-2


def imread(path: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(read_bytes_retry(repo_path(path)), np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3/benchmark_p3.yaml")
    ap.add_argument("--population", required=True, help="trial | main (Phase 3) | p4dev (Phase 4, with --root)")
    ap.add_argument("--root", default="data/manifests/phase3", help="manifest root (Phase 4: data/manifests/phase4)")
    ap.add_argument("--configs", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    MT.set_torch_home()
    configs = args.configs or list(cfg["configurations"])

    pop = read_csv_gz(repo_path(f"{args.root}/population_{args.population}.csv.gz"))
    refs = {}
    for p in pop:
        if p["ref_id"] not in refs:
            M = np.array(json.loads(p["ffhq512_transform_2x3"]))
            refs[p["ref_id"]] = {"img": imread(p["ref_path"]), "lm": np.array(json.loads(p["det_landmarks"])),
                                 "mask": G.coverage_mask(M, erode_px=cfg["geometry"]["eval_mask"]["erode_px"]),
                                 "race": p["race"], "gender": p["gender"]}
            assert sha256_file(repo_path(p["ref_path"])) == p["ref_sha256"]
    lp, sf, geo = MT.LPIPS(), MT.SFace(), MT.Geometry()
    ref_emb = {r: sf.embed(v["img"], v["lm"]) for r, v in refs.items()}
    ref_ids = sorted(refs)
    E_ref = np.stack([ref_emb[r] for r in ref_ids])

    # operating point from clean-reference impostors (impostor pool, test split)
    pool = read_csv_gz(repo_path("data/manifests/phase3/population_impostor_pool.csv.gz"))
    seen, pool_emb = set(), []
    for p in pool:
        if p["ref_id"] in seen:
            continue
        seen.add(p["ref_id"])
        pool_emb.append(sf.embed(imread(p["ref_path"]), np.array(json.loads(p["det_landmarks"]))))
    P = np.stack(pool_emb)
    S = P @ P.T
    iu = np.triu_indices(len(P), 1)
    imp = S[iu]
    thr = float(np.quantile(imp, 1 - FAR))
    calib = {"run": run_info(), "evaluator": MT.SFACE, "evaluator_sha256": sha256_file(repo_path(MT.SFACE)),
             "alignment": "reference landmarks (fixed) → FaceRecognizerSF.alignCrop → 128-d, L2-normalised",
             "impostor_pool_refs": len(P), "impostor_pairs": int(len(imp)), "far": FAR,
             "threshold_at_far": round(thr, 5),
             "impostor_quantiles": {q: round(float(np.quantile(imp, q)), 5) for q in (0.5, 0.9, 0.99, 0.999)},
             "pairs_above_0.59": int((imp >= 0.59).sum()),
             "note": "Clean-vs-clean impostors on the test split; FairFace has no identity labels, so a small "
                     "number of same-person pairs (P2-T7: 1,060 within-test high-similarity pairs) can remain in "
                     "this distribution and raise the threshold slightly. Embeddings not persisted."}
    out_dir = repo_path(f"{args.root}/metrics/{args.population}")
    write_json(out_dir / "identity_calibration.json", calib)
    print("threshold@FAR1e-2", round(thr, 4), flush=True)

    for cid in configs:
        run_path = repo_path(f"{args.root}/runs/{args.population}/{cid}.csv.gz")
        if not run_path.exists():
            print(f"{cid}: no run manifest — skipped (recorded as not run)", flush=True)
            continue
        run = {r["pair_id"]: r for r in read_csv_gz(run_path)}
        rows = []
        for p in pop:
            r = run.get(p["pair_id"])
            ref = refs[p["ref_id"]]
            rec = {k: "" for k in FIELDS}
            rec.update(pair_id=p["pair_id"], ref_id=p["ref_id"], cell=p["cell"], profile=p["profile"],
                       severity=p["severity"], in_size=p["out_w"], config_id=cid, race=ref["race"],
                       gender=ref["gender"], status=r["status"] if r else "MISSING")
            if r and r["status"] == "OK":
                data = read_bytes_retry(repo_path(r["output_path"]))
                assert sha256_bytes(data) == r["output_sha256"], r["output_path"]
                out = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                m = ref["mask"]
                found, score, nme = geo(out, ref["lm"])
                e = sf.embed(out, ref["lm"])
                sims = E_ref @ e
                i = ref_ids.index(p["ref_id"])
                others = np.delete(sims, i)
                rec.update(psnr=round(MT.psnr_rgb(out, ref["img"], m), 4), ssim=round(MT.ssim_y(out, ref["img"], m), 5),
                           lpips=round(lp([out], ref["img"], m)[0], 5),
                           sharp_ratio=round(MT.sharpness_ratio(out, ref["img"]), 4),
                           color_de=round(MT.color_shift(out, ref["img"]), 4),
                           det_found=found, det_score=score if score is not None else "",
                           lm_nme=round(nme, 5) if nme is not None else "",
                           id_cos=round(float(sims[i]), 5), imp_cos_mean=round(float(others.mean()), 5),
                           imp_cos_max=round(float(others.max()), 5), imp_n=len(others),
                           imp_n_ge_thr=int((others >= thr).sum()), id_ge_thr=bool(sims[i] >= thr))
            rows.append(rec)
        path = out_dir / f"{cid}.csv.gz"
        write_csv_gz(path, rows, FIELDS)
        n_ok = sum(r["status"] == "OK" for r in rows)
        print(f"{cid}: {n_ok}/{len(rows)} scored → {path.name} {sha256_file(path)[:12]}", flush=True)


if __name__ == "__main__":
    main()
