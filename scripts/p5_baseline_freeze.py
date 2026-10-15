"""P5-T1 — freeze the exact Phase 4 behaviour as the Phase 5 baseline (machine-readable manifest + verification).

    FES_IMAGE=fes-dev:cpu-p4 scripts/dock.sh python scripts/p5_baseline_freeze.py --write  [--a4-check 2] [--image-id ID]
    FES_IMAGE=fes-dev:cpu-p4 scripts/dock.sh python scripts/p5_baseline_freeze.py --check

Nothing here fits, tunes or routes anything new. The tool:
  1. records the frozen Phase 4 configuration: T6 gate + T7 routing (file SHA-256 and seals), protocol/taxonomy,
     the source of every module on the runtime/evaluation path, A4 weights + architecture + settings, the YuNet
     detection contract, both identity evaluators and their decision constants;
  2. verifies the large assets against the P1-T9 acquisition SHA-256 (A4 weights, YuNet, SFace, dlib);
  3. replays T6 decisions and T7 decision logs for `main` (synthetic test) and `p4dev` (synthetic dev) from the tracked
     quality features and metrics, writes them with the same deterministic writer to a scratch directory, and requires
     the bytes to equal the committed Phase 4 files;
  4. (optional, --a4-check N) re-runs A4 on N A4-routed dev pairs and requires the output pixels to equal the
     Phase 4 run manifest (`output_pixel_sha256`);
  5. writes configs/phase5/baseline_manifest.json with a SHA-256 seal (same rule as the T6/T7 seals).
`--check` recomputes everything except the A4 re-run and compares it to the sealed manifest (exit 1 on any drift).
Code hashes are over LF-normalised bytes (the Windows checkout uses core.autocrlf).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from fes_bench import gate as GT
from fes_bench import p4eval as E
from fes_bench import quality as Q
from fes_bench import routing as RT
from fes_data import preprocess as P
from fes_data.common import read_csv_gz, repo_path, run_info, sha256_file, write_csv_gz, write_json

MANIFEST = "configs/phase5/baseline_manifest.json"
MANIFEST_VERSION = "p5-baseline-1.0.0"

PHASE4_COMMITS = {
    "phase4_final_T8": "7aa8b2905bf347a1a6e5375f976617d5fb11a0fe",
    "T7_evaluation": "ad79455f92e27b6bc1741573480b6bbd114396a0",
    "T7_freeze": "785a4b213020ad75aee94048b415181358dcb724",
    "T6_evaluation": "bc5ef801d861a1c734a6c0eba64e0dcbaab71ea6",
    "T6_freeze": "e504dfa12700414a4ffcc75bb2cfaa54ed850460",
    "phase3_baseline": "fdfce67a0e01f35dcb775d146b0a68938dd345c5",
}
FROZEN_CONFIGS = ["configs/phase4/t6_gate.json", "configs/phase4/t7_routing.json",
                  "configs/phase4/p4_protocol.yaml", "configs/phase4/taxonomy_v1.yaml",
                  "configs/phase3/benchmark_p3.yaml"]
CODE = ["src/fes_bench/gate.py", "src/fes_bench/routing.py", "src/fes_bench/quality.py", "src/fes_bench/geometry.py",
        "src/fes_bench/adapters.py", "src/fes_bench/archs.py", "src/fes_bench/evaluators.py",
        "src/fes_bench/metrics.py", "src/fes_bench/p4eval.py", "src/fes_data/preprocess.py", "src/fes_data/common.py",
        "scripts/p4_gate_evaluate.py", "scripts/p4_route_evaluate.py", "scripts/p4_router_validate.py"]
# P1-T9 acquisition manifest (docs/phase_1/model_acquisition_manifest.csv)
ASSETS = {
    "A4_weights": ("models/M24_swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
                   "99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a"),
    "yunet_detector": ("models/D01_yunet/face_detection_yunet_2026may.onnx",
                       "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"),
    "E01_sface": ("models/E01_sface/face_recognition_sface_2021dec.onnx",
                  "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    "E07_dlib": ("models/E07_dlib_resnet/dlib_face_recognition_resnet_model_v1.dat",
                 "55533b28a95800a551ba546ba62fe69625c7e95a7061c338adffead08719da30"),
}
# committed Phase 4 outputs the replay must reproduce byte-for-byte
REPLAY_FILES = {("T6", "main"): "data/manifests/phase4/t6/decisions_main.csv.gz",
                ("T6", "p4dev"): "data/manifests/phase4/t6/decisions_p4dev.csv.gz",
                ("T7", "main"): "data/manifests/phase4/t7/decision_log_main.csv.gz",
                ("T7", "p4dev"): "data/manifests/phase4/t7/decision_log_p4dev.csv.gz"}
NOT_ROUTABLE = ("A1", "A2", "A5")      # Phase 4 gate / Phase 5 scope: must never be routed


def text_sha256(rel: str) -> str:
    return hashlib.sha256(repo_path(rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def seal(d: dict) -> str:
    return GT.gate_hash(d)


def load_frozen() -> tuple[dict, list, dict, dict, dict]:
    g6 = json.loads(repo_path("configs/phase4/t6_gate.json").read_text(encoding="utf-8"))
    rules = GT.load_rules(g6)                       # asserts the T6 seal
    r7 = json.loads(repo_path("configs/phase4/t7_routing.json").read_text(encoding="utf-8"))
    t, table = RT.load_routing(r7, g6)             # asserts the T7 seal and the T6 link
    return g6, rules, r7, t, table


# ------------------------------------------------------------------ replay (same row logic as p4_gate_evaluate /
# p4_route_evaluate; only the output directory differs)

def replay_rows(pop: str, rules, t, table) -> tuple[list[dict], list[dict], Counter]:
    pairs = E.load(pop)
    delta = E.deltas_from_validation()
    P_ = sorted(pairs.values(), key=lambda p: p.pair_id)
    t6, t7 = [], []
    mild = Counter()
    for p in P_:
        d, why = GT.decide(p.feat, rules)
        lab = GT.pair_label([E.multi_of(p, c, delta) for c in E.CANDIDATES])
        t6.append({"pair_id": p.pair_id, "ref_id": p.ref_id, "cell": p.cell, "decision": d, "reason": why, "label": lab})
        b = RT.bucket_of(p.feat, t) if d == GT.ENH else ""
        route = table.get(b, RT.ORIGINAL) if d == GT.ENH else RT.ORIGINAL
        fb, used = "", route
        if route != RT.ORIGINAL and not p.det_out.get(route, False):
            fb, used = "INVALID_OUTPUT", RT.ORIGINAL
        t7.append({"pair_id": p.pair_id, "ref_id": p.ref_id, "cell": p.cell, "decision": d, "reason": why,
                   "bucket": b, "route": route, "used": used, "fallback": fb, "escalate": d == GT.UNSAFE,
                   "d_E01": round(p.cos[(used, "E01")] - p.cos[("B0", "E01")], 5),
                   "d_E07": round(p.cos[(used, "E07")] - p.cos[("B0", "E07")], 5),
                   "verdict": E.multi_of(p, used, delta) if used != RT.ORIGINAL else ""})
        if p.severity == "mild":
            mild[(d, used)] += 1
    return t6, t7, mild


def reproduce(rules, t, table) -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for pop in ("main", "p4dev"):
            t6, t7, mild = replay_rows(pop, rules, t, table)
            for task, rows in (("T6", t6), ("T7", t7)):
                fp = Path(tmp) / f"{task}_{pop}.csv.gz"
                write_csv_gz(fp, rows, list(rows[0]))
                committed = sha256_file(repo_path(REPLAY_FILES[(task, pop)]))
                out[f"{task}_{pop}"] = {"committed_file": REPLAY_FILES[(task, pop)], "committed_sha256": committed,
                                        "replayed_sha256": sha256_file(fp), "identical": sha256_file(fp) == committed,
                                        "n": len(rows)}
            md = Counter()
            for (d, _), n in mild.items():
                md[d] += n
            out[f"T7_{pop}"].update(
                decisions=dict(sorted(Counter(r["decision"] for r in t7).items())),
                used=dict(sorted(Counter(r["used"] for r in t7).items())),
                fallbacks=dict(sorted(Counter(r["fallback"] for r in t7 if r["fallback"]).items())),
                escalated=sum(r["escalate"] for r in t7),
                mild={"n": sum(mild.values()), "enhanced": sum(n for (_, u), n in mild.items() if u != RT.ORIGINAL),
                      "decisions": dict(sorted(md.items()))})
    return out


# ------------------------------------------------------------------ A4 artefact

def a4_record() -> dict:
    from fes_bench import adapters as AD
    from fes_bench import archs
    import torch
    ad = AD.build("A4")
    asset, rel, sha = AD.WEIGHTS[ad.key]
    return {"config_id": ad.config_id, "label": ad.label, "kind": ad.kind, "weights_key": ad.key,
            "acquisition_asset_id": asset, "weights": rel, "weights_sha256": sha,
            "weights_size_bytes": repo_path(rel).stat().st_size,
            "upstream": {"repo": "https://github.com/JingyunLiang/SwinIR", "commit": "6545850fbf8df298df73d81f3e8cba638787c8bd",
                         "release": "v0.0", "checkpoint_key": "params_ema", "strict_load": True},
            "architecture": {"builder": "fes_bench.archs.build_swinir_realsr_l_x4",
                             "vendored_source": "third_party/swinir/models/network_swinir.py",
                             "vendored_source_sha256": archs.VENDORED_SHA256["swinir/models/network_swinir.py"],
                             "upscale": 4, "window_size": 8, "img_size": 64, "variant": "SwinIR-L real-SR x4 GAN"},
            "settings": ad.settings(),
            "input_contract": "native degraded face chip (D-frame), BGR uint8 → RGB [0,1]; uses no alignment transform "
                              "(ctx['M_e2f'] is ignored by GeneralSR)",
            "output_contract": "×4 → E-frame 224×224 BGR uint8 (INTER_AREA if 4s>224, INTER_LANCZOS4 if 4s<224)",
            "runtime": {"torch": torch.__version__, "threads": 2, "device": "cpu", "precision": "fp32",
                        "timeout_s": 120.0},
            "status": "RESEARCH CANDIDATE — not production-approved; licence REVIEW_REQUIRED (FFHQ/DIV2K-type training "
                      "data, R4); code Apache-2.0"}


def a4_rerun(n: int, table: dict) -> dict:
    """Re-run A4 on `n` A4-routed p4dev pairs; output pixels must equal the Phase 4 run manifest."""
    import cv2
    import numpy as np
    import torch
    from fes_bench.adapters import build
    from fes_data.common import sha256_bytes
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    log = {r["pair_id"]: r for r in read_csv_gz(repo_path(REPLAY_FILES[("T7", "p4dev")]))}
    pop = {r["pair_id"]: r for r in read_csv_gz(repo_path("data/manifests/phase4/population_p4dev.csv.gz"))}
    runs = {r["pair_id"]: r for r in read_csv_gz(repo_path("data/manifests/phase4/runs/p4dev/A4.csv.gz"))}
    ids = sorted((k for k, r in log.items() if r["used"] == "A4"),
                 key=lambda k: hashlib.sha256(f"p5-baseline|{k}".encode()).hexdigest())[:n]
    ad = build("A4")
    ad.load()
    rows = []
    for k in ids:
        img = cv2.imdecode(np.frombuffer(repo_path(pop[k]["degraded_path"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        out, _ = ad.enhance(img, {"s": img.shape[0]})
        rows.append({"pair_id": k, "bucket": log[k]["bucket"], "infer_s": round(time.perf_counter() - t0, 1),
                     "pixel_sha256": sha256_bytes(out.tobytes()), "phase4_pixel_sha256": runs[k]["output_pixel_sha256"],
                     "identical": sha256_bytes(out.tobytes()) == runs[k]["output_pixel_sha256"]})
    return {"n": len(rows), "all_identical": all(r["identical"] for r in rows), "rows": rows,
            "load_report": {k: {kk: v for kk, v in r.items() if kk in ("sha256", "strict", "load_s", "missing_keys")}
                            for k, r in ad.load_reports.items()}}


# ------------------------------------------------------------------ manifest

def build_manifest(image_id: str | None) -> dict:
    g6, rules, r7, t, table = load_frozen()
    assets = {}
    for k, (rel, sha) in ASSETS.items():
        got = sha256_file(repo_path(rel))
        assets[k] = {"path": rel, "sha256": sha, "verified": got == sha}
    det_cfg = P.PreprocessConfig(out_size=112)
    m = {
        "name": "phase5_baseline", "manifest_version": MANIFEST_VERSION,
        "purpose": "Exact Phase 4 behaviour frozen as the Phase 5 baseline. Phase 5 changes are compared against it; "
                   "nothing in this file may be edited — a changed baseline is a new manifest version.",
        "data_kind": "SYNTHETIC", "phase4_gate": "YELLOW",
        "validity": r7["validity"],
        "phase4_commits": PHASE4_COMMITS,
        "frozen_configs": {p: {"sha256_text": text_sha256(p)} for p in FROZEN_CONFIGS},
        "code_sha256_text": {p: text_sha256(p) for p in CODE},
        "versions": {"gate": GT.GATE_VERSION, "routing": RT.ROUTING_VERSION, "quality": Q.QUALITY_VERSION,
                     "preprocess": P.PREPROC_VERSION, "protocol": "p4-protocol-1.0.0",
                     "baseline_protocol": "p3-protocol-1.0.1"},
        "gate": {"file": "configs/phase4/t6_gate.json", "seal": g6["sha256"], "k": g6["k_chosen"],
                 "rules": g6["rules"], "decisions": [GT.DNE, GT.ENH, GT.UNSAFE],
                 "safety": {"det_status": "OK required", "det_score_min": GT.LOW_CONF, "abs_yaw_proxy_max": GT.YAW_WARN,
                            "nan_feature_condition": "False (missing evidence never enhances)"},
                 "default": GT.DNE},
        "routing": {"file": "configs/phase4/t7_routing.json", "seal": r7["sha256"], "t6_gate_seal": r7["t6_gate_seal"],
                    "table": table, "bucket_thresholds": t, "bucket_rule": r7["bucket_rule"],
                    "fallbacks": r7["fallbacks"], "timeout_s": r7["timeout_s"],
                    "routed_candidates": sorted(set(table.values()) - {RT.ORIGINAL}),
                    "must_not_route": list(NOT_ROUTABLE), "original_id": RT.ORIGINAL,
                    "output_guard": "INVALID_OUTPUT if shape/dtype ≠ 224×224×3 uint8, std < 1, or YuNet finds no face "
                                    "on the output (score ≥ 0.7)",
                    "timeout_mechanism": "thread future timeout (cannot kill a hung model; process isolation is P5 work)"},
        "a4": a4_record(),
        "detection_alignment": {
            "detector": {"model": ASSETS["yunet_detector"][0], "sha256": ASSETS["yunet_detector"][1],
                         "score_threshold": 0.7, "nms_threshold": 0.3, "top_k": 50, "max_side": 640,
                         "licence": "MIT code; weights REVIEW_REQUIRED (WIDER FACE, R4)"},
            "quality_features": {"input_mode": "face_crop", "out_size": 112, "crop_pad_frac": det_cfg.crop_pad_frac,
                                 "crop_relaxed_threshold": det_cfg.crop_relaxed_threshold,
                                 "min_eye_dist_px": det_cfg.min_eye_dist_px, "c_frame": Q.C_SIZE},
            "candidate_input": "A4 receives the native degraded chip; no alignment transform is applied to it",
            "evaluation_frame": "E-frame = FairFace dlib chip (224 px) with the RECORDED per-reference alignment "
                                "(ffhq512_transform_2x3, P2-T3). This is oracle alignment for evaluation; Phase 5 "
                                "task 5.3 replaces it for real-world inputs.",
            "router_original_rule": "ORIGINAL returned in the E-frame via geometry.upsample_to_e (bicubic if s < 224)"},
        "evaluators": {
            "E01": {"id": "SFace (OpenCV Zoo 2021dec)", "model": ASSETS["E01_sface"][0], "sha256": ASSETS["E01_sface"][1],
                    "role": "TEST-ONLY", "thr_far_1e-2": {p: E.thresholds(p)["E01"] for p in E.POPS}},
            "E07": {"id": "dlib ResNet v1", "model": ASSETS["E07_dlib"][0], "sha256": ASSETS["E07_dlib"][1],
                    "dlib_version": "20.0.1", "image": "fes-dev:cpu-p4", "role": "TEST-ONLY",
                    "thr_far_1e-2": {p: E.thresholds(p)["E07"] for p in E.POPS}},
            "identity_change_delta": E.deltas_from_validation(),
            "verdict_rules": "per evaluator BENEFIT Δ>δ / HARM Δ<−δ; multi: CONFIRMED_* both, PARTIAL_* one, DISAGREE opposite",
            "note": "Neither evaluator is the Feature-Match system. Feature-Match is not integrated (Phase 0 boundary)."},
        "runtime": {"image_tag": "fes-dev:cpu-p4", "image_id": image_id, "torch_threads": 2, "cv2_threads": 1,
                    "device": "cpu"},
        "assets": assets,
        "reproduction": reproduce(rules, t, table),
    }
    return m


def compare(old: dict, new: dict, path: str = "") -> list[str]:
    diffs = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if k in ("run", "sha256", "a4_rerun") and path == "":
                continue
            diffs += compare(old.get(k), new.get(k), f"{path}.{k}")
    elif old != new:
        diffs.append(f"{path}: {str(old)[:80]} != {str(new)[:80]}")
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--a4-check", type=int, default=2)
    ap.add_argument("--image-id", default=None)
    a = ap.parse_args()
    if a.check:
        old = json.loads(repo_path(MANIFEST).read_text(encoding="utf-8"))
        ok_seal = old["sha256"] == seal(old)
        new = build_manifest(old["runtime"]["image_id"])
        diffs = compare(old, new)
        ok_repro = all(v["identical"] for v in new["reproduction"].values())
        ok_assets = all(v["verified"] for v in new["assets"].values())
        print(json.dumps({"seal_ok": ok_seal, "drift": diffs, "reproduction_identical": ok_repro,
                          "assets_verified": ok_assets}, indent=1))
        sys.exit(0 if ok_seal and not diffs and ok_repro and ok_assets else 1)
    m = build_manifest(a.image_id)
    if a.a4_check:
        _, _, _, _, table = load_frozen()
        m["a4_rerun"] = a4_rerun(a.a4_check, table)
    m["run"] = run_info()
    m["sha256"] = seal(m)
    write_json(repo_path(MANIFEST), m)
    print(json.dumps({"reproduction": {k: v["identical"] for k, v in m["reproduction"].items()},
                      "assets": {k: v["verified"] for k, v in m["assets"].items()},
                      "a4_rerun": {k: v for k, v in m.get("a4_rerun", {}).items() if k != "load_report"},
                      "seal": m["sha256"]}, indent=1))


if __name__ == "__main__":
    main()
