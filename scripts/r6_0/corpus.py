"""R6-0 — merged corpus table: case metadata + final label + every signal (+ identity when available). Host, stdlib + numpy.

Final label (statement about the output the R5 build RETURNED or would return, v0):
  GOOD  | BAD (kind: clear / faint / destructive / stress) | UNCERTAIN
Sources, in priority order: R6-0 visual labels (labels_r6.json) > R5-8 DEV labels > named R5 escapes.
Corpus group (prompt section 3): A = GOOD-source pools, B = known-failure pools, C = borderline pools.
"""
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "data/manifests/phase5_rapid/r6_0"

PRIOR_MAP = {"LEGITIMATE": ("GOOD", ""), "LOCALIZED_ARTIFACT": ("BAD", "clear"), "DESTRUCTIVE": ("BAD", "destructive"),
             "UNCERTAIN": ("UNCERTAIN", ""), "FAINT_ARTIFACT": ("BAD", "faint"), "COLOUR_MOTTLING": ("BAD", "clear")}
R6_MAP = {"GOOD": ("GOOD", ""), "BAD_CLEAR": ("BAD", "clear"), "BAD_FAINT": ("BAD", "faint"),
          "BAD_DESTRUCTIVE": ("BAD", "destructive"), "UNCERTAIN": ("UNCERTAIN", "")}


def group(c):
    if c["src"] in ("dev_mod", "p4", "clean") or c["src"].startswith("real"):
        return "C"
    if c["prior"] in ("LEGITIMATE",) or (c["src"] == "test_r56" and c["prior"] == "UNLABELLED"):
        return "A"
    return "B"


def num(v):
    if v in ("", None):
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def load(with_identity=True):
    cases = {c["id"]: c for c in json.loads((M / "cases.json").read_text())}
    r6 = json.loads((M / "labels_r6.json").read_text())
    for cid, c in cases.items():
        if cid in r6:
            c["label"], c["kind"] = R6_MAP[r6[cid]["label"]]
            c["label_src"] = "R6-0 visual (" + r6[cid]["note"] + ")"
        else:
            c["label"], c["kind"] = PRIOR_MAP[c["prior"]]
            c["label_src"] = "R5 prior: " + c["note"]
        if c["src"].startswith("real") and c["name"] in ("img165.jpg", "img72.jpg"):
            c["kind"] = "stress"            # halftone prints the service routes to A4; NAFNet forced here
        c["group"] = group(c)
        c["sig"] = {}
    with gzip.open(M / "signals.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r.get("error"):
                continue
            cases[r["id"]]["sig"][r["variant"]] = {k: num(v) for k, v in r.items() if k not in ("id", "variant")}
    p = M / "identity.csv.gz"
    if with_identity and p.exists():
        with gzip.open(p, "rt") as fh:
            for r in csv.DictReader(fh):
                d = cases[r["id"]]["sig"].get(r["variant"])
                if d is not None:
                    d.update({k: num(v) for k, v in r.items() if k not in ("id", "variant")})
    return [c for c in cases.values() if c["sig"]]
