"""P2-T2 — Verify FairFace (raw, read-only) and build a byte-identical working copy + manifest.

Two stages (the raw zip is the single source of truth for both):
  extract  — byte-identical extraction to datasets/work (stdlib only; run on the host because
             Docker bind-mount writes of ~98k small files are very slow):
             .venv/Scripts/python.exe scripts/p2_fairface_prepare.py --stage extract
  analyze  — verification, statistics, duplicates, manifest (reads bytes from the zip; canonical
             run inside the pinned container):
             scripts/dock.sh python scripts/p2_fairface_prepare.py --stage analyze

Never writes to datasets/raw. Extraction is idempotent: existing files are hash-verified, not overwritten.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fes_data import common as C

LABEL_FIELDS = ["age", "gender", "race", "service_test"]


_ZIP = None


def _init_worker(zip_path: str) -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(zip_path)


def _analyze(member: str):
    """Worker: hash + decode + image statistics for one zip member (bytes identical to the working copy)."""
    rel = member
    data = _ZIP.read(member)
    row = {"sha256": C.sha256_bytes(data), "size_bytes": len(data)}
    img = C.decode_image(data)
    if img is None:
        row.update(readable=False)
        return rel, row
    h, w = img.shape[:2]
    mean_luma, rms_contrast, dark, bright = C.luma_stats(img)
    row.update(
        readable=True, width=w, height=h,
        is_grayscale=C.is_grayscale(img),
        jpeg_q_est=C.jpeg_quality_estimate(data),
        lap_var=round(C.laplacian_variance(img), 2),
        mean_luma=round(mean_luma, 2), rms_contrast=round(rms_contrast, 2),
        frac_dark=round(dark, 4), frac_bright=round(bright, 4),
        phash=f"{C.phash64(img):016x}",
    )
    return rel, row


def extract(cfg: dict) -> None:
    """Byte-identical extraction; every file's SHA-256 is checked against the zip member bytes."""
    raw_zip = C.repo_path(cfg["raw"]["zip"])
    assert C.sha256_file(raw_zip) == cfg["raw"]["zip_sha256"], "raw zip hash mismatch"
    work = C.repo_path(cfg["work_dir"])
    written = verified = 0
    with zipfile.ZipFile(raw_zip) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            data = z.read(info.filename)  # CRC-32 checked by zipfile
            dst = work / info.filename
            if dst.exists():
                assert C.sha256_bytes(dst.read_bytes()) == C.sha256_bytes(data), f"working copy differs: {dst}"
                verified += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            written += 1
    C.write_json(work / "EXTRACTION.json", {"source_zip": cfg["raw"]["zip"], "source_zip_sha256": cfg["raw"]["zip_sha256"],
                                            "written": written, "verified_existing": verified, "run": C.run_info()})
    print(f"extract: written={written} verified_existing={verified}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2/fairface.yaml")
    ap.add_argument("--stage", choices=["extract", "analyze"], required=True)
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    if a.stage == "extract":
        return extract(cfg)
    report: dict = {"config": cfg["_config_path"], "config_sha256": cfg["_config_sha256"], "checks": {}}

    # 1) raw integrity
    raw_zip = C.repo_path(cfg["raw"]["zip"])
    got = C.sha256_file(raw_zip)
    assert got == cfg["raw"]["zip_sha256"], f"raw zip hash mismatch: {got}"
    labels: dict[str, dict] = {}
    label_issues = collections.Counter()
    for split, spec in cfg["raw"]["labels"].items():
        p = C.repo_path(spec["path"])
        assert C.sha256_file(p) == spec["sha256"], f"label hash mismatch: {p}"
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["file"] in labels:
                    label_issues["duplicate_label_row"] += 1
                for k in LABEL_FIELDS:
                    if not r.get(k):
                        label_issues[f"missing_{k}"] += 1
                labels[r["file"]] = {"split": split, **{k: r[k] for k in LABEL_FIELDS}}
    report["checks"]["raw_hashes"] = "OK"
    report["checks"]["label_rows"] = len(labels)
    report["checks"]["label_issues"] = dict(label_issues)

    # 2) zip structure vs labels (no extraction here; see stage "extract")
    work = C.repo_path(cfg["work_dir"])
    with zipfile.ZipFile(raw_zip) as z:
        bad = z.testzip()
        report["checks"]["zip_testzip_first_bad_member"] = bad
        members = [i for i in z.infolist() if not i.is_dir()]
        names = {i.filename for i in members}
        jpgs = sorted(n for n in names if n.lower().endswith(".jpg"))
        report["checks"]["zip_members"] = len(members)
        report["checks"]["zip_non_jpg_members"] = sorted(names - set(jpgs))
        report["checks"]["in_zip_not_in_labels"] = sorted(set(jpgs) - set(labels))[:50]
        report["checks"]["in_labels_not_in_zip"] = sorted(set(labels) - set(jpgs))[:50]
        report["checks"]["split_dir_mismatch"] = sorted(n for n in jpgs if n in labels and not n.startswith(labels[n]["split"] + "/"))[:50]

    # 3) per-file analysis (parallel)
    rels = [(Path(cfg["work_dir"]) / n).as_posix() for n in jpgs]
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init_worker, initargs=(str(raw_zip),)) as ex:
        for member, row in ex.map(_analyze, jpgs, chunksize=256):
            results[(Path(cfg["work_dir"]) / member).as_posix()] = row

    # 4) duplicates: exact (sha256) and near (pHash)
    by_sha = collections.defaultdict(list)
    for rel, r in results.items():
        by_sha[r["sha256"]].append(rel)
    exact_group = {rel: f"ex-{sha[:12]}" for sha, rels_ in by_sha.items() if len(rels_) > 1 for rel in rels_}
    ph = {rel: int(r["phash"], 16) for rel, r in results.items() if r.get("readable")}
    near_group = C.near_duplicate_groups(ph, cfg["near_duplicate_max_hamming"])

    # 5) manifest rows + stage-1 suitability
    s = cfg["suitability"]
    rows = []
    for n, rel in zip(jpgs, rels):
        r = results[rel]
        lab = labels.get(n, {})
        split = n.split("/")[0]
        reasons = []
        if not r.get("readable"):
            reasons.append("unreadable")
        else:
            if min(r["width"], r["height"]) < s["min_side_px"]:
                reasons.append("too_small")
            if s["exclude_grayscale"] and r["is_grayscale"]:
                reasons.append("grayscale")
        if not lab:
            reasons.append("no_label")
        if rel in exact_group:
            reasons.append("exact_duplicate")
        rows.append({
            "sample_id": f"ff-{split}-{Path(n).stem}",
            "split_source": split,
            "zip_member": n,
            "work_path": rel,
            **{k: r.get(k, "") for k in ["sha256", "size_bytes", "readable", "width", "height", "is_grayscale", "jpeg_q_est",
                                          "lap_var", "mean_luma", "rms_contrast", "frac_dark", "frac_bright", "phash"]},
            "exact_dup_group": exact_group.get(rel, ""),
            "near_dup_group": near_group.get(rel, ""),
            **{k: lab.get(k, "") for k in LABEL_FIELDS},
            "license": "CC BY 4.0",
            "source": "https://github.com/joojs/fairface",
            "stage1_suitable": not reasons,
            "stage1_reasons": ";".join(reasons),
        })
    C.write_csv_gz(C.repo_path(cfg["manifest"]), rows)

    # 6) report
    cnt = collections.Counter
    report["summary"] = {
        "images": len(rows),
        "by_split": dict(cnt(r["split_source"] for r in rows)),
        "unreadable": sum(1 for r in rows if r["readable"] is False),
        "sizes": dict(cnt(f'{r["width"]}x{r["height"]}' for r in rows if r["readable"])),
        "grayscale": sum(1 for r in rows if r["is_grayscale"] is True),
        "exact_duplicate_files": len(exact_group),
        "exact_duplicate_groups": len(set(exact_group.values())),
        "near_duplicate_files": len(near_group),
        "near_duplicate_groups": len(set(near_group.values())),
        "near_dup_groups_crossing_splits": sum(
            1 for g in set(near_group.values())
            if len({r["split_source"] for r in rows if r["near_dup_group"] == g}) > 1),
        "stage1_suitable": sum(1 for r in rows if r["stage1_suitable"]),
        "stage1_reasons": dict(cnt(x for r in rows for x in r["stage1_reasons"].split(";") if x)),
        "label_values": {k: dict(cnt(r[k] for r in rows)) for k in LABEL_FIELDS},
    }
    report["manifest"] = cfg["manifest"]
    report["manifest_sha256"] = C.sha256_file(C.repo_path(cfg["manifest"]))
    report["run"] = C.run_info()
    C.write_json(C.repo_path(cfg["report"]), report)
    print({k: report["summary"][k] for k in ["images", "by_split", "unreadable", "grayscale", "exact_duplicate_files",
                                             "near_duplicate_files", "near_dup_groups_crossing_splits", "stage1_suitable"]})
    print("checks:", {k: v for k, v in report["checks"].items()})


if __name__ == "__main__":
    main()
