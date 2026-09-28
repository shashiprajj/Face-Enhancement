"""P2-T3 — Build clean reference faces from verified FairFace chips.

    scripts/dock.sh python scripts/p2_build_references.py --config configs/phase2/references.yaml

Pass 1 (all candidates): source bytes (SHA-256 checked against the P2-T2 manifest) → decode → YuNet
detection → landmarks, pose proxies, FFHQ-512 transform (recorded, not applied), quality statistics.
Pool medians of the chip geometry are then computed and the objective acceptance rules applied.
Pass 2 (accepted only): the decoded source pixels are written unchanged as lossless PNG.

No enhancement, sharpening, resampling or geometric warping is applied to references.
Every candidate gets a manifest row with its acceptance decision and reasons.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from fes_data import common as C
from fes_data import preprocess as P

_CTX: dict = {}


def _init(cfg: dict) -> None:
    d = cfg["detector"]
    _CTX["det"] = P.YuNetDetector(C.repo_path(d["model"]), d["score_threshold"], d["nms_threshold"], max_side=d["max_side"])
    _CTX["cfg"] = cfg


def _measure(item: dict) -> dict:
    """Pass 1: detection + measurements for one candidate (no writes)."""
    cfg = _CTX["cfg"]
    data = C.repo_path(item["work_path"]).read_bytes()
    row = {"sample_id": item["sample_id"], "pool": item["pool"], "split_source": item["split_source"],
           "source_path": item["work_path"], "source_sha256": item["sha256"],
           "source_hash_ok": C.sha256_bytes(data) == item["sha256"],
           "detector": _CTX["det"].id, "preproc_version": P.PREPROC_VERSION,
           "stage1_suitable": item["stage1_suitable"], "stage1_reasons": item["stage1_reasons"],
           "near_dup_group": item["near_dup_group"], "source_lap_var": item["lap_var"],
           "race": item["race"], "gender": item["gender"], "age": item["age"], "license": item["license"]}
    img = C.decode_image(data)
    dets = _CTX["det"].detect(img) if img is not None else []
    row["n_faces"] = len(dets)
    if img is not None:
        m, c, dark, bright = C.luma_stats(img)
        row.update(ref_h=img.shape[0], ref_w=img.shape[1], mean_luma=round(m, 2), rms_contrast=round(c, 2),
                   frac_dark=round(dark, 4), frac_bright=round(bright, 4), lap_var=round(C.laplacian_variance(img), 2))
    if dets:
        d = dets[0]
        lm = np.asarray(d.landmarks)
        roll, yaw = P.pose_proxies(lm)
        M, scale, oob = P.canonical_transform(lm, cfg["reference"]["ffhq_record_size"], img.shape)
        row.update(det_score=round(d.score, 4), det_bbox_xywh=json.dumps([round(x, 2) for x in d.bbox_xywh]),
                   det_landmarks=json.dumps([[round(x, 2) for x in p] for p in d.landmarks]),
                   landmarks_plausible=P.landmarks_plausible(lm), eye_dist=round(d.eye_dist, 3),
                   eye_mid_x=round(float((lm[0, 0] + lm[1, 0]) / 2), 3), eye_mid_y=round(float((lm[0, 1] + lm[1, 1]) / 2), 3),
                   roll_deg=round(roll, 3), yaw_proxy=round(yaw, 4),
                   ffhq512_transform_2x3=json.dumps(np.round(M, 8).tolist()), ffhq512_scale=round(scale, 5),
                   ffhq512_oob_frac=round(oob, 5))
    return row


def _write(row: dict) -> dict:
    """Pass 2: write decoded source pixels losslessly (reference == source chip)."""
    cfg = _CTX["cfg"]
    data = C.repo_path(row["source_path"]).read_bytes()
    assert C.sha256_bytes(data) == row["source_sha256"]
    img = C.decode_image(data)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, cfg["png_compression"]])
    assert ok
    blob = buf.tobytes()
    rel = f'{cfg["out_dir"]}/{row["split_source"]}/{row["sample_id"]}.png'
    out = C.repo_path(rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        assert C.sha256_file(out) == C.sha256_bytes(blob), f"non-reproducible reference: {rel}"
    else:
        out.write_bytes(blob)
    # lossless check: PNG decodes to exactly the source pixels
    assert np.array_equal(cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR), img)
    return {"sample_id": row["sample_id"], "ref_id": f'{row["sample_id"]}.{cfg["version"]}', "ref_path": rel,
            "ref_sha256": C.sha256_bytes(blob), "ref_pixel_sha256": C.sha256_bytes(img.tobytes())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2/references.yaml")
    cfg = C.load_config(ap.parse_args().config)
    rows = C.read_csv_gz(C.repo_path(cfg["source_manifest"]))
    acc = cfg["acceptance"]

    # candidate pools: all val; seeded race×gender-stratified sample of train
    rng = random.Random(cfg["seed"])
    items = []
    for name, spec in cfg["pools"].items():
        cand = sorted((r for r in rows if r["split_source"] == spec["split_source"]), key=lambda r: r["sample_id"])
        if spec["sample"] != "all":
            strata = collections.defaultdict(list)
            for r in cand:
                strata[tuple(r[k] for k in spec["stratify_by"])].append(r)
            per = spec["sample"] // len(strata)
            chosen = []
            for k in sorted(strata):
                chosen += rng.sample(strata[k], min(per, len(strata[k])))
            cand = sorted(chosen, key=lambda r: r["sample_id"])
        items += [{**r, "pool": name} for r in cand]

    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init, initargs=(cfg,)) as ex:
        meas = list(ex.map(_measure, items, chunksize=64))

    # resolved thresholds
    suitable = [r for r in rows if r["stage1_suitable"] == "True"]
    min_lv = float(np.percentile([float(r["lap_var"]) for r in suitable], acc["min_lap_var_percentile"]))
    geo = [r for r in meas if r.get("det_score") != None and r.get("landmarks_plausible") and r["n_faces"] == 1]
    med_eye = float(np.median([r["eye_dist"] for r in geo]))
    med_mid = np.median([[r["eye_mid_x"], r["eye_mid_y"]] for r in geo], axis=0)
    resolved = {"min_source_lap_var": round(min_lv, 3), "median_eye_dist": round(med_eye, 3),
                "median_eye_mid_xy": [round(float(x), 3) for x in med_mid],
                "eye_dist_range_px": [round(acc["eye_dist_ratio_range"][0] * med_eye, 3), round(acc["eye_dist_ratio_range"][1] * med_eye, 3)],
                "max_eye_mid_offset_px": round(acc["max_eye_mid_offset_ratio"] * med_eye, 3)}

    for r in meas:
        reasons = []
        if not r["source_hash_ok"]:
            reasons.append("source_hash_mismatch")
        if acc["require_stage1_suitable"] and r["stage1_suitable"] != "True":
            reasons.append("stage1:" + r["stage1_reasons"])
        if acc["exclude_near_duplicates"] and r["near_dup_group"]:
            reasons.append("near_duplicate")
        if r["n_faces"] == 0:
            reasons.append("no_face_detected")
        else:
            if acc["require_single_face"] and r["n_faces"] != 1:
                reasons.append("multiple_faces")
            if r["det_score"] < acc["min_det_score"]:
                reasons.append("low_det_score")
            if acc["require_plausible_landmarks"] and not r["landmarks_plausible"]:
                reasons.append("implausible_landmarks")
            ratio = r["eye_dist"] / med_eye
            if not acc["eye_dist_ratio_range"][0] <= ratio <= acc["eye_dist_ratio_range"][1]:
                reasons.append("nonstandard_scale")
            if np.hypot(r["eye_mid_x"] - med_mid[0], r["eye_mid_y"] - med_mid[1]) > acc["max_eye_mid_offset_ratio"] * med_eye:
                reasons.append("nonstandard_position")
            if abs(r["roll_deg"]) > acc["max_abs_roll_deg"]:
                reasons.append("nonstandard_roll")
        if float(r.get("lap_var", 0)) < min_lv:
            reasons.append("source_blurry")
        if r.get("frac_dark", 1) > acc["max_clip_frac"] or r.get("frac_bright", 1) > acc["max_clip_frac"]:
            reasons.append("clipped_exposure")
        r["accepted"] = not reasons
        r["reject_reasons"] = ";".join(reasons)

    accepted = [r for r in meas if r["accepted"]]
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init, initargs=(cfg,)) as ex:
        written = {w["sample_id"]: w for w in ex.map(_write, accepted, chunksize=64)}
    for r in meas:
        r.update(written.get(r["sample_id"], {"ref_id": "", "ref_path": "", "ref_sha256": "", "ref_pixel_sha256": ""}))
        r["ref_size"] = cfg["reference"]["size"] if r["accepted"] else ""
        r["ref_version"] = cfg["version"]

    lead = ["sample_id", "ref_id", "pool", "split_source", "accepted", "reject_reasons"]
    fields = lead + sorted({k for r in meas for k in r} - set(lead))
    C.write_csv_gz(C.repo_path(cfg["manifest"]), meas, fields)

    cnt = collections.Counter
    def pct(key):
        v = [float(r[key]) for r in accepted if r.get(key) not in (None, "")]
        return {p: round(float(np.percentile(v, p)), 3) for p in (5, 50, 95)} if v else None
    report = {
        "config": cfg["_config_path"], "config_sha256": cfg["_config_sha256"], "resolved_thresholds": resolved,
        "candidates": len(meas), "accepted": len(accepted),
        "by_pool": {p: {"candidates": sum(r["pool"] == p for r in meas), "accepted": sum(r["pool"] == p and r["accepted"] for r in meas)}
                    for p in cfg["pools"]},
        "reject_reasons": dict(cnt(x for r in meas for x in r["reject_reasons"].split(";") if x)),
        "accepted_quality": {k: pct(k) for k in ("det_score", "eye_dist", "roll_deg", "yaw_proxy", "lap_var", "mean_luma",
                                                  "rms_contrast", "ffhq512_scale", "ffhq512_oob_frac")},
        "accepted_strata": {k: dict(sorted(cnt(r[k] for r in accepted).items())) for k in ("race", "gender", "age")},
        "accepted_strata_by_pool": {p: dict(sorted(cnt(f'{r["race"]}|{r["gender"]}' for r in accepted if r["pool"] == p).items()))
                                    for p in cfg["pools"]},
        "manifest": cfg["manifest"], "manifest_sha256": C.sha256_file(C.repo_path(cfg["manifest"])),
        "run": C.run_info(),
    }
    C.write_json(C.repo_path(cfg["report"]), report)
    print({k: report[k] for k in ("resolved_thresholds", "candidates", "accepted", "by_pool", "reject_reasons")})


if __name__ == "__main__":
    main()
