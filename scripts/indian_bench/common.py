"""Shared helpers of the Indian-expression FES benchmark (benchmark code only; nothing here changes the service).

Layout (all under `bench_root`, never inside the source dataset):
  manifests/  source_inventory.csv · selected_originals.csv · holdout_originals.csv · degradation_matrix.json ·
              generated_variants.csv · enhancement_results.csv · failures.csv · comparisons.csv · metrics.csv
  clean_originals/<Expression>/<sample_id>.png              the clean reference (see config.reference)
  degraded/<Expression>/<sample_id>/<type>/level_<n>.<ext>  ext = jpg when the last operation is JPEG, else png
  enhanced/<Expression>/<sample_id>/<type>/level_<n>.<ext>  the bytes the service returned (png, or the input bytes)
  comparisons/ by_expression/ · by_degradation/ · contact_sheets/
  reports/ · logs/
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def load_config(path: str | None = None) -> dict:
    cfg = json.loads(Path(path or HERE / "config.json").read_text(encoding="utf-8"))
    cfg["source_root"] = str((ROOT / cfg["source_root"]).resolve())
    cfg["bench_root"] = str((ROOT / cfg["bench_root"]).resolve())
    return cfg


def bench(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["bench_root"]).joinpath(*parts)
    (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def scope(cfg: dict, pilot: bool = False) -> list[dict]:
    """The selected originals this run works on: the pilot samples, or the first `scope_per_expression` of every
    expression in selection order (null = the whole 50 % selection). A smaller scope is a prefix of a larger one, so
    raising the number later continues the same benchmark."""
    if pilot:
        return read_csv(bench(cfg, "manifests", "pilot_samples.csv"))
    rows = read_csv(bench(cfg, "manifests", "selected_originals.csv"))
    n = cfg.get("scope_per_expression")
    return [r for r in rows if int(r["selection_rank"]) < n] if n else rows


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from its parts (independent of PYTHONHASHSEED and of processing order)."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:        # a line cut by an interrupted run: redone on resume
                pass
    return out


def md_table(rows: list[dict], fields: list[str]) -> str:
    out = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    out += ["| " + " | ".join(str(r.get(f, "")) for f in fields) + " |" for r in rows]
    return "\n".join(out)
