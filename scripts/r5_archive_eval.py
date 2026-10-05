"""R5-ARCHIVE — evaluation on the paired face-deblurring archive (Dataset-and-code/archive).

Dataset: 192,900 + 100 pairs of 128×128 face crops, input/<split>/<name> (degraded: mostly Gaussian / motion / defocus
blur) ↔ output/<split>/<name> (sharp ground truth, pixel-aligned). Splits train 154,320 · val 19,290 · test 19,290 ·
ztest 100 (file prefix = 100 distinct identities). Provenance/licence unknown (celebrity crops, CelebA-like).
Only `test` and `ztest` are used here; train/val are untouched (reserved for a possible fine-tuning step).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_archive_eval.py stage1 [--workers 4]
    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_archive_eval.py stage2 --model A4 [--workers 2]
    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_archive_eval.py analyze

stage1  every test + ztest pair: input-vs-GT PSNR/SSIM, quality features, the REAL service path (EnhancementPipeline,
        full_image mode, frozen p5-rwgate-1.0.0; the A4 call is stubbed — only the decision is recorded here),
        and B0 identity (input vs its own sharp GT) with SFace E01 + dlib E07.
stage2  a fixed sample (300 random test pairs, seed 0, + all 100 ztest) through one model per run, outputs compared
        with the GT at 128 px: PSNR, SSIM, LPIPS, SFace/dlib cosine to the GT, face found on the output.
        Models: A4 SwinIR-L (the service model) · N1 NAFNet-GoPro (motion-deblur, same resolution; never used
        stand-alone before) · A6 GFPGAN v1.4 · A7 CodeFormer w=0.5 (research-only licences).
analyze tables + plots data → data/manifests/phase5_rapid/archive/{summary.json, tables.md}

Identity here = similarity of the output to the SHARP VERSION OF THE SAME PHOTO (pixel-aligned; GT landmarks for all
conditions). It measures whether a model restores identity cues, not cross-photo recognition, and it is a proxy
(SFace/dlib), not Feature-Match. ztest (100 identities) adds an impostor set: FAR-1e-2 thresholds from clean GT-vs-GT
impostor pairs, TAR and new false accepts per model.
Images written only to the git-ignored experiments/phase5_rapid/outputs/archive/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path, run_info, write_csv_gz, write_json

ARCH = "Dataset-and-code/archive"
OUT = "data/manifests/phase5_rapid/archive"
IMG_OUT = "experiments/phase5_rapid/outputs/archive"
SPLITS = ("test", "ztest")
N_SAMPLE_TEST, SEED = 300, 0
MODELS = {"A4": "SwinIR-L ×4 (A4, service model)", "N1": "NAFNet-GoPro (deblur)", "A6": "GFPGAN v1.4",
          "A7": "CodeFormer w=0.5"}
W = {}                                                    # per-process state


def pair(split: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    a = cv2.imread(str(repo_path(f"{ARCH}/input/{split}/{name}")), cv2.IMREAD_COLOR)
    b = cv2.imread(str(repo_path(f"{ARCH}/output/{split}/{name}")), cv2.IMREAD_COLOR)
    return a, b


def nan() -> float:
    return float("nan")


def _common_init(threads: int) -> None:
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(threads)
    from fes_bench import evaluators as EV
    from fes_bench.metrics import SFace
    from fes_data import preprocess as P
    W["det"] = P.YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    W["sface"], W["dlib"] = SFace(), EV.DlibResNet()


def gt_landmarks(gt: np.ndarray):
    faces = W["det"].detect(gt)
    return (np.asarray(faces[0].landmarks, np.float64), faces[0].score) if faces else (None, nan())


def ident(img: np.ndarray, lm) -> tuple[np.ndarray | None, np.ndarray | None]:
    if lm is None:
        return None, None
    return W["sface"].embed(img, lm), W["dlib"].embed(img, lm)


def cos(a, b) -> float:
    return float(np.dot(a, b)) if a is not None and b is not None else nan()


# ------------------------------------------------------------------ stage 1

class _Stub:
    """Stands in for A4 in stage 1: the gate's decision is what is measured (A4 itself runs in stage 2)."""

    def __call__(self, img, ctx):
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC), {}


def _init1():
    _common_init(1)
    from fes_service.pipeline import EnhancementPipeline
    W["pipe"] = EnhancementPipeline(_Stub(), timeout_s=30, gate="rw")


def _job1(item):
    from fes_bench import metrics as M
    split, name = item
    inp, gt = pair(split, name)
    r = {"split": split, "name": name, "psnr_in": round(M.psnr_rgb(inp, gt), 3), "ssim_in": round(M.ssim_y(inp, gt), 4)}
    data = cv2.imencode(".png", inp)[1].tobytes()          # lossless: the service sees exactly the input pixels
    res = W["pipe"].process(data, "full_image")
    r.update(gate_decision=res.decision, gate_reason=res.reason, gate_route=res.route, n_faces=res.n_faces,
             chip_px=res.chip_px, **{f"q_{k}": v for k, v in res.quality.items()})
    lm, gs = gt_landmarks(gt)
    r["gt_det_score"] = round(gs, 4) if not math.isnan(gs) else None
    r["in_face_found"] = bool(W["det"].detect(inp))
    eg, dg = ident(gt, lm)
    ei, di = ident(inp, lm)
    r["id_sface_B0"], r["id_dlib_B0"] = cos(ei, eg), cos(di, dg)
    emb = {"gt": (eg, dg), "B0": (ei, di)} if split == "ztest" else None
    return r, emb, (lm.tolist() if lm is not None else None)


def stage1(workers: int, limit: int = 0) -> None:
    items = [(s, n) for s in SPLITS for n in sorted(os.listdir(repo_path(f"{ARCH}/input/{s}")))]
    if limit:
        items = items[:limit] + [i for i in items if i[0] == "ztest"]
    rows, embs, lms = [], {}, {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init1) as ex:
        for i, (r, e, lm) in enumerate(ex.map(_job1, items, chunksize=32)):
            rows.append(r)
            lms[f"{r['split']}/{r['name']}"] = lm
            if e:
                embs[r["name"]] = e
            if i % 2000 == 0:
                print(f"stage1 {i}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    write_csv_gz(repo_path(f"{OUT}/stage1.csv.gz"), rows)
    np.save(repo_path(f"{IMG_OUT}/stage1_ztest_emb.npy"), embs, allow_pickle=True)
    np.save(repo_path(f"{IMG_OUT}/gt_landmarks.npy"), lms, allow_pickle=True)
    write_json(repo_path(f"{OUT}/stage1_run.json"), run_info({"n": len(rows), "seconds": round(time.time() - t0, 1),
                                                              "workers": workers}))
    print("stage1 done", len(rows), f"{time.time() - t0:.0f}s")


# ------------------------------------------------------------------ stage 2

def sample() -> list[tuple[str, str]]:
    test = sorted(os.listdir(repo_path(f"{ARCH}/input/test")))
    pick = random.Random(SEED).sample(test, N_SAMPLE_TEST)
    return [("test", n) for n in sorted(pick)] + [("ztest", n) for n in sorted(os.listdir(repo_path(f"{ARCH}/input/ztest")))]


class NAFNetGoPro:
    """Stand-alone NAFNet-GoPro (MIT): same-resolution deblurring; output stays 128 px."""

    def load(self):
        from fes_bench.adapters import load_net
        self.net, rep = load_net("M28-GoPro")
        self.load_reports = {"M28-GoPro": rep}

    def enhance(self, img, ctx):
        import torch
        from fes_bench.adapters import bgr_u8_to_rgb_t, rgb_t_to_bgr_u8
        with torch.inference_mode():
            return rgb_t_to_bgr_u8(self.net(bgr_u8_to_rgb_t(img)).clamp(0, 1)), {}


def _init2(model: str, threads: int):
    _common_init(threads)
    from fes_bench.metrics import LPIPS, set_torch_home
    set_torch_home()
    W["lpips"] = LPIPS()
    if model == "N1":
        ad = NAFNetGoPro()
    else:
        from fes_bench.adapters import build
        ad = build(model)
    ad.load()
    W["model"], W["name"] = ad, model
    W["lms"] = np.load(repo_path(f"{IMG_OUT}/gt_landmarks.npy"), allow_pickle=True).item()


def _job2(item):
    from fes_bench import metrics as M
    from fes_data import preprocess as P
    split, name = item
    inp, gt = pair(split, name)
    lm_l = W["lms"].get(f"{split}/{name}")
    lm = np.asarray(lm_l, np.float64) if lm_l is not None else None
    ctx = {"s": inp.shape[0]}
    if W["name"] in ("A6", "A7"):
        # FFHQ alignment from landmarks detected on the DEGRADED input (what a deployment would have)
        faces = W["det"].detect(inp)
        if not faces:
            return {"split": split, "name": name, "model": W["name"], "status": "NO_INPUT_FACE"}, None
        lm_in_e = np.asarray(faces[0].landmarks, np.float64) * (224.0 / inp.shape[0])
        ctx["M_e2f"] = P.umeyama_similarity(lm_in_e, P.FFHQ_TEMPLATE_512)
    t0 = time.perf_counter()
    try:
        out, _ = W["model"].enhance(inp, ctx)
    except Exception as e:  # noqa: BLE001
        return {"split": split, "name": name, "model": W["name"], "status": f"ERROR {type(e).__name__}"}, None
    secs = time.perf_counter() - t0
    out128 = out if out.shape[0] == gt.shape[0] else cv2.resize(out, gt.shape[1::-1], interpolation=cv2.INTER_AREA)
    d = os.path.join(repo_path(IMG_OUT), W["name"], split)
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, name.replace(".jpg", ".png")), out128)
    eg, dg = ident(gt, lm)
    eo, do = ident(out128, lm)
    r = {"split": split, "name": name, "model": W["name"], "status": "OK", "secs": round(secs, 3),
         "psnr": round(M.psnr_rgb(out128, gt), 3), "ssim": round(M.ssim_y(out128, gt), 4),
         "lpips": round(W["lpips"]([out128], gt)[0], 4), "id_sface": cos(eo, eg), "id_dlib": cos(do, dg),
         "out_face_found": bool(W["det"].detect(out128))}
    return r, ({"out": (eo, do)} if split == "ztest" else None)


def stage2(model: str, workers: int, threads: int) -> None:
    done, lock = repo_path(f"{OUT}/stage2_{model}.csv.gz"), repo_path(f"{IMG_OUT}/stage2_{model}.running")
    if done.exists() or lock.exists():
        print(f"stage2 {model} skipped: already {'done' if done.exists() else 'running'}")
        return
    lock.write_text(str(os.getpid()))
    try:
        _stage2(model, workers, threads)
    finally:
        lock.unlink(missing_ok=True)


def _stage2(model: str, workers: int, threads: int) -> None:
    items = sample()
    rows, embs = [], {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init2, initargs=(model, threads)) as ex:
        for i, (r, e) in enumerate(ex.map(_job2, items)):
            rows.append(r)
            if e:
                embs[r["name"]] = e
            if i % 25 == 0:
                print(f"stage2 {model} {i}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    write_csv_gz(repo_path(f"{OUT}/stage2_{model}.csv.gz"), rows)
    np.save(repo_path(f"{IMG_OUT}/stage2_{model}_ztest_emb.npy"), embs, allow_pickle=True)
    write_json(repo_path(f"{OUT}/stage2_{model}_run.json"),
               run_info({"model": model, "label": MODELS[model], "n": len(rows), "seconds": round(time.time() - t0, 1),
                         "workers": workers, "threads": threads}))
    print(f"stage2 {model} done {len(rows)} in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ analyze

def f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else nan()
    except (TypeError, ValueError):
        return nan()


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return round(float(np.mean(xs)), 4) if xs else None


def boot_ci(d, n=2000, seed=0):
    d = np.array([x for x in d if not math.isnan(x)])
    if len(d) < 5:
        return None
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def chroma_p99(img: np.ndarray, gt: np.ndarray) -> float:
    """99th percentile of the Lab chroma distance to the GT (colour artefacts: rainbow / metallic streaks)."""
    a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(gt, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.percentile(np.hypot(a[..., 1] - b[..., 1], a[..., 2] - b[..., 2]), 99))


ARTEFACT_EXCESS = 10.0          # output chroma p99 exceeds the blurred input's by > 10 Lab units (visually obvious)


def artefacts(m: str, items) -> dict:
    out = {}
    for split, name in items:
        p = repo_path(f"{IMG_OUT}/{m}/{split}/{name.replace('.jpg', '.png')}")
        if p.exists():
            inp, gt = pair(split, name)
            out[(split, name)] = chroma_p99(cv2.imread(str(p)), gt) - chroma_p99(inp, gt)
    return out


def analyze() -> None:
    s1 = read_csv_gz(repo_path(f"{OUT}/stage1.csv.gz"))
    S = {}
    test = [r for r in s1 if r["split"] == "test"]
    # ---- dataset characterisation + gate on the full test split
    from collections import Counter
    dec = Counter(r["gate_decision"] or "NONE" for r in test)
    reason = Counter((r["gate_reason"] or "").split(":")[0] for r in test)
    routed = [r for r in test if r["gate_route"] == "A4"]
    ssim_in = np.array([f(r["ssim_in"]) for r in test])
    q = np.nanquantile(ssim_in, [1 / 3, 2 / 3])
    sev = lambda r: "severe" if f(r["ssim_in"]) < q[0] else ("moderate" if f(r["ssim_in"]) < q[1] else "mild")  # noqa: E731
    S["dataset"] = {"n_test": len(test), "n_ztest": sum(r["split"] == "ztest" for r in s1),
                    "psnr_in_mean": mean([f(r["psnr_in"]) for r in test]), "ssim_in_mean": mean(list(ssim_in)),
                    "ssim_tercile_cuts": [round(float(x), 4) for x in q],
                    "gt_face_found": round(sum(r["gt_det_score"] not in ("", None) for r in test) / len(test), 4),
                    "input_face_found": round(sum(r["in_face_found"] == "True" for r in test) / len(test), 4),
                    "q_noise_sigma_median": mean([float(np.nanmedian([f(r["q_noise_sigma"]) for r in test]))]),
                    "q_lapd1_c_median": mean([float(np.nanmedian([f(r["q_lapd1_c"]) for r in test]))]),
                    "q_blockiness_median": mean([float(np.nanmedian([f(r["q_blockiness"]) for r in test]))])}
    S["gate_full_test"] = {"decision": dict(dec), "reason": dict(reason.most_common()),
                           "routed_to_A4": len(routed), "routed_pct": round(100 * len(routed) / len(test), 2),
                           "routed_by_severity": dict(Counter(sev(r) for r in routed)),
                           "B0_id_sface_mean_by_decision": {d: mean([f(r["id_sface_B0"]) for r in test
                                                                     if r["gate_decision"] == d]) for d in dec}}
    # ---- models on the sample
    key = lambda r: (r["split"], r["name"])  # noqa: E731
    base = {key(r): r for r in s1}
    items = sample()
    res = {}
    for m in MODELS:
        p = repo_path(f"{OUT}/stage2_{m}.csv.gz")
        if p.exists():
            res[m] = {key(r): r for r in read_csv_gz(p)}
    groups = {"all": items, "test": [i for i in items if i[0] == "test"], "ztest": [i for i in items if i[0] == "ztest"]}
    for sv in ("mild", "moderate", "severe"):
        groups[f"test_{sv}"] = [i for i in groups["test"] if sev(base[i]) == sv]
    groups["test_gate_A4"] = [i for i in groups["test"] if base[i]["gate_route"] == "A4"]
    tab = {}
    art = {m: artefacts(m, items) for m in res}
    for g, its in groups.items():
        row = {"n": len(its), "B0": {"psnr": mean([f(base[i]["psnr_in"]) for i in its]),
                                     "ssim": mean([f(base[i]["ssim_in"]) for i in its]),
                                     "id_sface": mean([f(base[i]["id_sface_B0"]) for i in its]),
                                     "id_dlib": mean([f(base[i]["id_dlib_B0"]) for i in its])}}
        for m, R in res.items():
            ok = [i for i in its if i in R and R[i]["status"] == "OK"]
            d_s = [f(R[i]["id_sface"]) - f(base[i]["id_sface_B0"]) for i in ok]
            d_d = [f(R[i]["id_dlib"]) - f(base[i]["id_dlib_B0"]) for i in ok]
            row[m] = {"n_ok": len(ok), "psnr": mean([f(R[i]["psnr"]) for i in ok]),
                      "ssim": mean([f(R[i]["ssim"]) for i in ok]), "lpips": mean([f(R[i]["lpips"]) for i in ok]),
                      "id_sface": mean([f(R[i]["id_sface"]) for i in ok]), "id_dlib": mean([f(R[i]["id_dlib"]) for i in ok]),
                      "d_sface": mean(d_s), "d_sface_ci95": boot_ci(d_s), "d_dlib": mean(d_d), "d_dlib_ci95": boot_ci(d_d),
                      "both_improve_pct": round(100 * np.mean([(a > 0.02 and b > 0.02) for a, b in zip(d_s, d_d)]), 1) if ok else None,
                      "both_harm_pct": round(100 * np.mean([(a < -0.02 and b < -0.02) for a, b in zip(d_s, d_d)]), 1) if ok else None,
                      "out_face_found_pct": round(100 * np.mean([R[i]["out_face_found"] == "True" for i in ok]), 1) if ok else None,
                      "sec_per_img": mean([f(R[i]["secs"]) for i in ok]),
                      "artefact_pct": round(100 * np.mean([art[m].get(i, 0) > ARTEFACT_EXCESS for i in ok]), 1) if ok else None}
        tab[g] = row
    S["sample_tables"] = tab
    # ---- gated policy (service decision + model-track A4 output) vs always-on, on the test sample
    if "A4" in res:
        its = groups["test"]
        pol = [f(res["A4"][i]["id_sface"]) if base[i]["gate_route"] == "A4" and i in res["A4"] else f(base[i]["id_sface_B0"])
               for i in its]
        S["gated_policy_test_sample"] = {"id_sface": mean(pol), "B0": tab["test"]["B0"]["id_sface"],
                                         "n_routed": len(groups["test_gate_A4"])}
    # ---- ztest verification (100 identities): FAR-1e-2 thresholds from clean GT impostor pairs
    z1 = np.load(repo_path(f"{IMG_OUT}/stage1_ztest_emb.npy"), allow_pickle=True).item()
    names = sorted(n for n, e in z1.items() if e["gt"][0] is not None)
    ver = {}
    for k, ev in ((0, "sface"), (1, "dlib")):
        Gm = np.stack([z1[n]["gt"][k] for n in names])
        imp = (Gm @ Gm.T)[~np.eye(len(names), dtype=bool)]
        thr = float(np.quantile(imp, 0.99))
        conds = {"B0": {n: z1[n]["B0"][k] for n in names}}
        for m in res:
            p = repo_path(f"{IMG_OUT}/stage2_{m}_ztest_emb.npy")
            if p.exists():
                zm = np.load(p, allow_pickle=True).item()
                conds[m] = {n: zm[n]["out"][k] for n in names if n in zm and zm[n]["out"][k] is not None}
        out = {"threshold_far_1e-2": round(thr, 4), "n_ids": len(names)}
        b0_fa = {}
        for c, E in conds.items():
            gen = [float(E[n] @ Gm[i]) for i, n in enumerate(names) if n in E]
            fa = {n: int(np.max([E[n] @ Gm[j] for j, o in enumerate(names) if o != n]) >= thr) for n in names if n in E}
            if c == "B0":
                b0_fa = fa
            out[c] = {"TAR": round(float(np.mean([g >= thr for g in gen])), 3), "n": len(gen),
                      "false_accept_any": int(sum(fa.values())),
                      "new_false_accepts_vs_B0": int(sum(v and not b0_fa.get(n, 0) for n, v in fa.items()))}
        ver[ev] = out
    S["ztest_verification"] = ver
    S["split_character"] = {sp: {"psnr_in": mean([f(r["psnr_in"]) for r in s1 if r["split"] == sp]),
                                 "noise_sigma_median": round(float(np.nanmedian([f(r["q_noise_sigma"]) for r in s1
                                                                                  if r["split"] == sp])), 3)}
                            for sp in SPLITS}
    write_json(repo_path(f"{OUT}/summary.json"), run_info({"summary": S}))
    print(json.dumps(S, indent=1)[:6000])


def realcheck() -> None:
    """NAFNet-GoPro on the 200 real CC0 photos (Dataset-and-code/dataset, no GT): the service's face chip → N1.
    Artefact signal = chroma p99 of the output vs its own input chip (> ARTEFACT_EXCESS). Sheet of the 12 worst + 12
    random goes to the git-ignored outputs. Answers: does the archive-ztest artefact appear on real grainy photos?"""
    _common_init(4)
    from fes_data import preprocess as P
    from fes_service.pipeline import face_chip
    ad = NAFNetGoPro()
    ad.load()
    rows, tiles = [], {}
    for path in sorted(repo_path("Dataset-and-code/dataset").glob("*.jpg")):
        img = cv2.imread(str(path))
        fr = P.preprocess(img, W["det"], P.PreprocessConfig(out_size=112), input_mode="face_crop")[0]
        if fr.status != P.OK or fr.detection is None:
            rows.append({"image": path.name, "status": "NO_FACE"})
            continue
        chip, side = face_chip(img, np.asarray(fr.detection.landmarks, np.float64))
        t0 = time.perf_counter()
        out, _ = ad.enhance(chip, {})
        ex = chroma_p99(out, chip)
        rows.append({"image": path.name, "status": "OK", "chip_px": side, "secs": round(time.perf_counter() - t0, 3),
                     "chroma_excess_vs_input": round(ex, 2), "artefact": ex > ARTEFACT_EXCESS})
        tiles[path.name] = (chip, out, ex)
    write_csv_gz(repo_path(f"{OUT}/realcheck_N1.csv.gz"), rows)
    ok = [r for r in rows if r["status"] == "OK"]
    summ = {"n": len(rows), "n_face": len(ok), "artefact_n": sum(r["artefact"] for r in ok),
            "artefact_pct": round(100 * np.mean([r["artefact"] for r in ok]), 1),
            "sec_per_chip": mean([r["secs"] for r in ok])}
    write_json(repo_path(f"{OUT}/realcheck_N1.json"), run_info(summ))
    worst = sorted(tiles, key=lambda k: -tiles[k][2])[:12]
    rnd = random.Random(0).sample(sorted(set(tiles) - set(worst)), 12)
    T = 128
    grid = []
    for grp in (worst, rnd):
        for half in (grp[:6], grp[6:]):
            grid.append(np.hstack([cv2.resize(tiles[k][0], (T, T)) for k in half]))
            grid.append(np.hstack([cv2.resize(tiles[k][1], (T, T)) for k in half]))
            grid.append(np.full((6, 6 * T, 3), 255, np.uint8))
    cv2.imwrite(str(repo_path(f"{IMG_OUT}/realcheck_N1_sheet.jpg")), np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(json.dumps(summ))


def sheet() -> None:
    """Visual sheet (git-ignored): blurred input | A4 | N1 | A6 | A7 | sharp GT, SFace cos-to-GT under each tile.
    Rows: 3 random sample pairs per blur tercile + the 2 largest NAFNet gains + the 2 largest NAFNet losses."""
    s1 = {(r["split"], r["name"]): r for r in read_csv_gz(repo_path(f"{OUT}/stage1.csv.gz"))}
    res = {m: {(r["split"], r["name"]): r for r in read_csv_gz(repo_path(f"{OUT}/stage2_{m}.csv.gz"))} for m in MODELS
           if repo_path(f"{OUT}/stage2_{m}.csv.gz").exists()}
    test = [i for i in sample() if i[0] == "test"]
    q = np.nanquantile([f(s1[i]["ssim_in"]) for i in s1 if i[0] == "test"], [1 / 3, 2 / 3])
    rng = random.Random(1)
    rows = []
    for lo, hi, lab in ((-1, q[0], "severe"), (q[0], q[1], "moderate"), (q[1], 2, "mild")):
        rows += [(i, lab) for i in rng.sample([i for i in test if lo <= f(s1[i]["ssim_in"]) < hi], 3)]
    if "N1" in res:
        d = sorted(test, key=lambda i: f(res["N1"][i]["id_sface"]) - f(s1[i]["id_sface_B0"]))
        rows += [(i, "N1 best gain") for i in d[-2:]] + [(i, "N1 worst") for i in d[:2]]
    cols = ["B0"] + list(res) + ["GT"]
    T, H = 160, 22
    canvas = np.full((H + len(rows) * (T + H), 110 + len(cols) * T, 3), 255, np.uint8)
    put = lambda img, txt, x, y, c=(0, 0, 0): cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1,  # noqa: E731
                                                          cv2.LINE_AA)
    for j, c in enumerate(cols):
        put(canvas, {"B0": "blurred input", "GT": "sharp GT"}.get(c, c), 110 + j * T + 4, 15)
    for k, ((split, name), lab) in enumerate(rows):
        y = H + k * (T + H)
        put(canvas, lab, 4, y + T // 2)
        inp, gt = pair(split, name)
        for j, c in enumerate(cols):
            if c == "B0":
                img, sc = inp, f(s1[(split, name)]["id_sface_B0"])
            elif c == "GT":
                img, sc = gt, nan()
            else:
                p = repo_path(f"{IMG_OUT}/{c}/{split}/{name.replace('.jpg', '.png')}")
                img = cv2.imread(str(p)) if p.exists() else np.zeros_like(gt)
                sc = f(res[c].get((split, name), {}).get("id_sface"))
            canvas[y:y + T, 110 + j * T:110 + (j + 1) * T] = cv2.resize(img, (T, T), interpolation=cv2.INTER_CUBIC)
            if not math.isnan(sc):
                put(canvas, f"SFace {sc:.2f}", 110 + j * T + 4, y + T + 15)
    path = repo_path(f"{IMG_OUT}/archive_comparison_sheet.jpg")
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("sheet", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "analyze", "sheet", "realcheck"])
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="stage1 smoke run on the first N test pairs")
    a = ap.parse_args()
    os.makedirs(repo_path(IMG_OUT), exist_ok=True)
    os.makedirs(repo_path(OUT), exist_ok=True)
    if a.stage == "stage1":
        stage1(a.workers, a.limit)
    elif a.stage == "stage2":
        stage2(a.model, a.workers, a.threads)
    elif a.stage == "sheet":
        sheet()
    elif a.stage == "realcheck":
        realcheck()
    else:
        analyze()
