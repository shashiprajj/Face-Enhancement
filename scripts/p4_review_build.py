"""P4-T5 — build the blinded human identity/fabrication review kit (reproducible, seeded). No review is performed.

Items are drawn from the Phase 3 `main` (test, SYNTHETIC) outputs, which already exist; nothing is re-run.
  * per configuration (B0, A1–A5): RANDOM items stratified by severity (seeded hash order) + TARGETED items from
    the P3-T7 sample index (identity flags). Random and targeted are analysed separately (targeted rates are biased).
  * controls: KNOWN_SAME (the reference itself, JPEG q85 + 2 % rescale — same photo), KNOWN_DIFFERENT (clean reference
    of another identity from the same race × gender stratum).
  * REPEAT items (exact duplicates under a new id) for intra-rater consistency.
Blinding: the kit shows only random item ids; config / pair / role live in the separate key (never in the kit).

Kit (git-ignored, local only — faces never leave the machine):
    experiments/phase4/outputs/review_kit/<version>/{index.html, img/<item>.png, items.json}
Tracked: data/manifests/phase4/review/{key_<version>.csv.gz, kit_<version>.json}

    scripts/dock.sh python scripts/p4_review_build.py --version rv1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict

import cv2
import numpy as np

from fes_data.common import read_csv_gz, repo_path, run_info, sha256_bytes, sha256_file, write_csv_gz, write_json

SEED = 20260928
CONFIGS = ["B0", "A1", "A2", "A3", "A4", "A5"]
PER_SEVERITY = {"mild": 6, "moderate": 6, "severe": 8, "extreme": 6}     # 26 random per configuration
TARGETED_PER_CONFIG = 4
TARGET_FLAGS = ("id_flip", "id_closer_to_impostor", "id_drift")
N_SAME, N_DIFF, N_REPEAT = 20, 20, 12
KEY_FIELDS = ["item_id", "role", "selection", "config_id", "pair_id", "ref_id", "cell", "severity", "race", "gender",
              "right_source", "repeat_of", "image_sha256", "left_sha256"]


def h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="rv1")
    a = ap.parse_args()
    pop = read_csv_gz(repo_path("data/manifests/phase3/population_main.csv.gz"))
    pairs = {p["pair_id"]: p for p in pop}
    refs = {}
    for p in pop:
        refs.setdefault(p["ref_id"], p)
    runs = {c: {r["pair_id"]: r for r in read_csv_gz(repo_path(f"data/manifests/phase3/runs/main/{c}.csv.gz"))}
            for c in CONFIGS}
    sample_index = _read_csv(repo_path("data/manifests/phase3/results/main/sample_index.csv"))

    kit = repo_path(f"experiments/phase4/outputs/review_kit/{a.version}")
    (kit / "img").mkdir(parents=True, exist_ok=True)
    items, key = [], []

    def add(role, selection, right_img, left_path, meta):
        item_id = "it-" + h(a.version, role, selection, meta.get("config_id", ""), meta.get("pair_id", ""),
                             meta.get("ref_id", ""), meta.get("repeat_of", ""))[:10]
        ok, buf = cv2.imencode(".png", right_img)
        (kit / "img" / f"{item_id}.png").write_bytes(buf.tobytes())
        left_name = "ref-" + h(a.version, "left", meta["ref_id"])[:10] + ".png"
        if not (kit / "img" / left_name).exists():
            (kit / "img" / left_name).write_bytes(repo_path(left_path).read_bytes())
        items.append({"item_id": item_id, "left": f"img/{left_name}", "right": f"img/{item_id}.png"})
        p = pairs.get(meta.get("pair_id", ""), refs[meta["ref_id"]])
        key.append({k: "" for k in KEY_FIELDS} | {"item_id": item_id, "role": role, "selection": selection,
                    "race": p["race"], "gender": p["gender"], "image_sha256": sha256_bytes(buf.tobytes()),
                    "left_sha256": sha256_file(repo_path(left_path))} | meta)
        return item_id

    def read_img(path):
        return cv2.imread(str(repo_path(path)), cv2.IMREAD_COLOR)

    # 1. random, stratified by severity, per configuration (same pair set for every configuration → paired design)
    by_sev = defaultdict(list)
    for pid, p in pairs.items():
        by_sev[p["severity"]].append(pid)
    chosen = []
    for sev, n in PER_SEVERITY.items():
        chosen += sorted(by_sev[sev], key=lambda i: h("random", i))[:n]
    for c in CONFIGS:
        for pid in chosen:
            p = pairs[pid]
            add("TEST", "RANDOM", read_img(runs[c][pid]["output_path"]), p["ref_path"],
                {"config_id": c, "pair_id": pid, "ref_id": p["ref_id"], "cell": p["cell"], "severity": p["severity"],
                 "right_source": "output"})
    # 2. targeted (P3-T7 identity flags), excluding pairs already chosen
    for c in CONFIGS[1:]:
        cand = [r for r in sample_index if r["config_id"] == c and r["flag"] in TARGET_FLAGS and r["pair_id"] not in chosen]
        cand = sorted({r["pair_id"]: r for r in cand}.values(), key=lambda r: h("target", c, r["pair_id"]))
        for r in cand[:TARGETED_PER_CONFIG]:
            p = pairs[r["pair_id"]]
            add("TEST", f"TARGETED:{r['flag']}", read_img(runs[c][r["pair_id"]]["output_path"]), p["ref_path"],
                {"config_id": c, "pair_id": r["pair_id"], "ref_id": p["ref_id"], "cell": p["cell"],
                 "severity": p["severity"], "right_source": "output"})
    # 3. controls
    ref_ids = sorted(refs)
    for rid in sorted(ref_ids, key=lambda i: h("same", i))[:N_SAME]:
        img = read_img(refs[rid]["ref_path"])
        sm = cv2.resize(img, (220, 220), interpolation=cv2.INTER_AREA)
        sm = cv2.copyMakeBorder(sm, 2, 2, 2, 2, cv2.BORDER_REFLECT)
        sm = cv2.imdecode(cv2.imencode(".jpg", sm, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], cv2.IMREAD_COLOR)
        add("CONTROL_SAME", "CONTROL", sm, refs[rid]["ref_path"],
            {"ref_id": rid, "right_source": "reference_jpeg85_rescale"})
    strata = defaultdict(list)
    for rid in ref_ids:
        strata[(refs[rid]["race"], refs[rid]["gender"])].append(rid)
    for rid in sorted(ref_ids, key=lambda i: h("diff", i))[:N_DIFF]:
        others = [o for o in strata[(refs[rid]["race"], refs[rid]["gender"])] if o != rid]
        other = sorted(others, key=lambda o: h("diffpick", rid, o))[0]
        add("CONTROL_DIFFERENT", "CONTROL", read_img(refs[other]["ref_path"]), refs[rid]["ref_path"],
            {"ref_id": rid, "right_source": f"reference_of:{other}"})
    # 4. repeats (exact duplicates of TEST items)
    tests = [k for k in key if k["role"] == "TEST"]
    for k in sorted(tests, key=lambda k: h("repeat", k["item_id"]))[:N_REPEAT]:
        img = cv2.imread(str(kit / "img" / f"{k['item_id']}.png"), cv2.IMREAD_COLOR)
        add("REPEAT", "REPEAT", img, pairs[k["pair_id"]]["ref_path"],
            {kk: k[kk] for kk in ("config_id", "pair_id", "ref_id", "cell", "severity", "right_source")}
            | {"repeat_of": k["item_id"]})

    # blinded item list: sorted by id (reviewer order is re-shuffled per reviewer inside the tool)
    items.sort(key=lambda i: i["item_id"])
    assert len({i["item_id"] for i in items}) == len(items)
    (kit / "items.json").write_text(json.dumps(items, indent=0), encoding="utf-8")
    html = repo_path("scripts/p4_review_tool.html").read_text(encoding="utf-8")
    (kit / "index.html").write_text(html.replace("/*__ITEMS__*/[]", json.dumps(items))
                                    .replace("__KIT_VERSION__", a.version), encoding="utf-8")
    out = repo_path("data/manifests/phase4/review")
    write_csv_gz(out / f"key_{a.version}.csv.gz", sorted(key, key=lambda k: k["item_id"]), KEY_FIELDS)
    counts = defaultdict(int)
    for k in key:
        counts[f"{k['role']}|{k['selection'].split(':')[0]}"] += 1
    write_json(out / f"kit_{a.version}.json", {
        "run": run_info(), "version": a.version, "seed": SEED, "source_population": "phase3 main (test, SYNTHETIC)",
        "n_items": len(items), "counts": dict(sorted(counts.items())), "configs": CONFIGS,
        "per_severity_random": PER_SEVERITY, "targeted_per_config": TARGETED_PER_CONFIG,
        "key_sha256": sha256_file(out / f"key_{a.version}.csv.gz"),
        "items_json_sha256": sha256_file(kit / "items.json"),
        "tool_html_sha256": sha256_file(kit / "index.html"),
        "kit_location": f"experiments/phase4/outputs/review_kit/{a.version} (git-ignored; local only)",
        "status": "KIT READY — REVIEW NOT PERFORMED (G-5 open: needs >= 2 human reviewers)"})
    print(len(items), dict(counts))


def _read_csv(path):
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
