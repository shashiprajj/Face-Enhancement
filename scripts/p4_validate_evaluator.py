"""P4-T3 — validate the second identity evaluator E07 (dlib ResNet) on DEV data only, before any of its scores is
used as evidence. Acceptance criteria are fixed below BEFORE the run (P4-T3 §4) and are not tuned afterwards.

Data: the 112 dev_pilot clean references (population_noise_floor) and the p4dev B0 inputs (dev, 28 refs × 32 cells).
SFace (E01) is scored on the same items for the agreement/independence analysis. Embeddings stay in memory.

    FES_IMAGE=fes-dev:cpu-p4 scripts/dock.sh python scripts/p4_validate_evaluator.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

import cv2
import numpy as np

from fes_bench import evaluators as EV
from fes_bench import metrics as MT
from fes_bench import read_bytes_retry
from fes_data.common import read_csv_gz, repo_path, run_info, sha256_file, write_json

CRITERIA = {   # frozen 2026-10-03 before the first validation run
    "V1_weights_and_runtime_pinned": "weights SHA-256 == DLIB_MODEL_SHA256 and dlib == 20.0.1",
    "V2_output_contract": "128-d, all finite, raw descriptor norm in [0.5, 2.0] for every dev reference",
    "V3_determinism": "bit-identical descriptors on repeat (all 112 dev references)",
    "V4_route_agreement": "median cos(chip route, template route with reference landmarks) >= 0.95",
    "V5_noise_floor": "p99 of 1-cos under jpeg_q90 / shift_1px / down2_up < 0.05 (identity-neutral changes)",
    "V6_discrimination_mild": "AUC(genuine = ref vs B0 mild cells, impostor = ref vs other dev refs) >= 0.95",
    "V7_independence_reported": "Spearman(E07, E01) on ref-vs-B0 cos over p4dev reported (no pass/fail)",
}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def imread(p: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(read_bytes_retry(repo_path(p)), np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    cv2.setNumThreads(2)
    res, ok = {"criteria": CRITERIA}, {}
    dl = EV.DlibResNet()                                         # V1 asserted inside
    sf = MT.SFace()
    ok["V1_weights_and_runtime_pinned"] = True

    nf = read_csv_gz(repo_path("data/manifests/phase3/population_noise_floor.csv.gz"))
    refs = {}
    for r in nf:
        if r["ref_id"] not in refs:
            img = imread(r["ref_path"])
            assert sha256_file(repo_path(r["ref_path"])) == r["ref_sha256"]
            refs[r["ref_id"]] = (img, np.array(json.loads(r["det_landmarks"])))
    ids = sorted(refs)

    # V2/V3/V4 + timing
    t0 = time.perf_counter()
    E = {i: dl.embed_chip(refs[i][0]) for i in ids}
    t_chip = (time.perf_counter() - t0) / len(ids)
    E2 = {i: dl.embed_chip(refs[i][0]) for i in ids}
    norms = [dl.raw_norm(refs[i][0]) for i in ids]
    ok["V2_output_contract"] = all(E[i].shape == (128,) and np.isfinite(E[i]).all() for i in ids) and \
        0.5 <= min(norms) and max(norms) <= 2.0
    ok["V3_determinism"] = all(np.array_equal(E[i], E2[i]) for i in ids)
    route = np.array([float(E[i] @ dl.embed(*refs[i])) for i in ids])
    ok["V4_route_agreement"] = float(np.median(route)) >= 0.95
    res["raw_norm"] = {"min": round(min(norms), 4), "max": round(max(norms), 4)}
    res["route_agreement_cos"] = {q: round(float(np.quantile(route, q)), 4) for q in (0.0, 0.05, 0.5)}
    res["latency_ms_per_embed_chip"] = round(1000 * t_chip, 1)

    # V5 noise floor (same perturbations as P3-T6)
    def perturb(img):
        jq = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
        sh = cv2.warpAffine(img, np.float32([[1, 0, 1], [0, 1, 1]]), (224, 224), borderMode=cv2.BORDER_REFLECT)
        du = cv2.resize(cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA), (224, 224),
                        interpolation=cv2.INTER_CUBIC)
        return {"jpeg_q90": jq, "shift_1px": sh, "down2_up_bicubic": du}
    nfl = defaultdict(lambda: {"E07": [], "E01": []})
    for i in ids:
        img, lm = refs[i]
        e1 = sf.embed(img, lm)
        for k, p in perturb(img).items():
            nfl[k]["E07"].append(1 - float(E[i] @ dl.embed_chip(p)))
            nfl[k]["E01"].append(1 - float(e1 @ sf.embed(p, lm)))
    res["noise_floor_1_minus_cos_p99"] = {k: {e: round(float(np.quantile(v[e], 0.99)), 5) for e in v} for k, v in nfl.items()}
    ok["V5_noise_floor"] = all(v["E07"] < 0.05 for v in res["noise_floor_1_minus_cos_p99"].values())

    # impostor distributions (dev refs)
    A = np.stack([E[i] for i in ids])
    S1 = np.stack([sf.embed(*refs[i]) for i in ids])
    iu = np.triu_indices(len(ids), 1)
    imp7, imp1 = (A @ A.T)[iu], (S1 @ S1.T)[iu]
    res["dev_impostor"] = {e: {"n": int(len(v)), "q50": round(float(np.median(v)), 4), "q99": round(float(np.quantile(v, 0.99)), 4),
                               "thr_far_1e-2": round(float(np.quantile(v, 0.99)), 4)}
                           for e, v in (("E07", imp7), ("E01", imp1))}

    # V6/V7 on p4dev B0 (dev inputs as seen by the evaluators: bicubic to E-frame = B0 rule)
    pop = read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz"))
    run = {r["pair_id"]: r for r in read_csv_gz(repo_path("data/manifests/phase4/runs/p4dev/B0.csv.gz"))}
    gen = defaultdict(lambda: {"E07": [], "E01": []})
    c7, c1 = [], []
    pref = {}
    for p in pop:
        if p["ref_id"] not in pref:
            img = imread(p["ref_path"])
            lm = np.array(json.loads(p["det_landmarks"]))
            pref[p["ref_id"]] = (dl.embed_chip(img), sf.embed(img, lm), lm)
        e7r, e1r, lm = pref[p["ref_id"]]
        out = imread(run[p["pair_id"]]["output_path"])
        a, b = float(e7r @ dl.embed_chip(out)), float(e1r @ sf.embed(out, lm))
        gen[p["severity"]]["E07"].append(a)
        gen[p["severity"]]["E01"].append(b)
        c7.append(a)
        c1.append(b)
    res["auc_genuine_vs_dev_impostor"] = {s: {"E07": round(auc(np.array(v["E07"]), imp7), 4),
                                              "E01": round(auc(np.array(v["E01"]), imp1), 4), "n": len(v["E07"])}
                                          for s, v in sorted(gen.items())}
    ok["V6_discrimination_mild"] = res["auc_genuine_vs_dev_impostor"]["mild"]["E07"] >= 0.95
    c7, c1 = np.array(c7), np.array(c1)
    res["independence"] = {"spearman_ref_vs_B0_cos": round(spearman(c7, c1), 4),
                           "spearman_dev_impostor": round(spearman(imp7, imp1), 4), "n_pairs": len(c7)}
    ok["V7_independence_reported"] = True
    res["pass"] = ok
    res["validated"] = all(ok.values())
    res["run"] = run_info({"evaluator": EV.DlibResNet.id, "weights": EV.DLIB_MODEL, "weights_sha256": EV.DLIB_MODEL_SHA256,
                           "dlib": EV.DLIB_VERSION, "sface_sha256": sha256_file(repo_path(MT.SFACE))})
    res["data"] = "DEV only: 112 dev_pilot references + p4dev B0 (896 pairs). SYNTHETIC."
    write_json(repo_path("data/manifests/phase4/evaluator_E07_validation.json"), res)
    print(json.dumps({"pass": ok, "validated": res["validated"]}, indent=1))


if __name__ == "__main__":
    main()
