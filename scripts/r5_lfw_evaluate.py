"""R5-2 — identity evaluation of ORIGINAL vs FROZEN-GATE vs FORCED-A4 on the LFW research-proxy population.

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python scripts/r5_lfw_evaluate.py [--verify-service 5]

THIS IS A RESEARCH PROXY: identity = SFace (E01) and dlib ResNet (E07), the Phase 3/4 test-only evaluators.
It is NOT Feature-Match (not available in this environment) and the data is LFW (licence REVIEW_REQUIRED).

Per query (degraded photo of identity X; reference = a DIFFERENT clean photo of X):
  A ORIGINAL   the service's original: the face chip in the 224 E-frame (bicubic), i.e. B0 of the harness
  B GATED      frozen p5-rwgate-1.0.0: G1 → A4 output if the service output guard passes (face found on it), else A
  C FORCED A4  A4 output for every G0–G2 query regardless of the gate (research only; never for G3)
  G3 (unsafe) queries: only A; scored on the full query photo with its own detection (no face → no match).
Alignment: one landmark set per query, used for every condition (YuNet on A; if none, on C) — identity content is
compared, not detector luck. References: YuNet landmarks on the clean reference photo.
Measures per evaluator (never pooled): genuine cosine, verification match at the FAR-1e-2 threshold of clean
reference impostor pairs (480 identities), rank-1 identification among the 480 references, best impostor score;
per pair Δ vs A with δ verdicts (fes_bench.p4eval), flips (match→no match), rescues, new false accepts.
--verify-service N re-runs N G1 queries through the real R5-1 service (rw gate, real isolated A4) and checks the output
pixels equal the forced-A4 run (condition B is what the service returns).
Writes data/manifests/phase5_rapid/lfw/{eval_rows.csv.gz, summary.json, tables.md} and review sheets in
experiments/phase5_rapid/outputs/lfw_review/ (+ human_review_template.csv for a human reviewer).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict

import cv2
import numpy as np

from fes_bench import evaluators as EV
from fes_bench import p4eval as E
from fes_bench.metrics import SFace
from fes_data.common import read_csv_gz, repo_path, run_info, write_csv_gz, write_json
from fes_data.preprocess import YuNetDetector

ROOT = "data/manifests/phase5_rapid/lfw"
FAR = 1e-2
CONDS = ("A", "B", "C")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-service", type=int, default=5)
    a = ap.parse_args()
    cv2.setNumThreads(4)
    det = YuNetDetector(repo_path("models/D01_yunet/face_detection_yunet_2026may.onnx"), 0.7, 0.3, max_side=640)
    sf, dl = SFace(), EV.DlibResNet()
    delta = E.deltas_from_validation()
    imread = lambda p: cv2.imread(str(repo_path(p)), cv2.IMREAD_COLOR)  # noqa: E731

    def lm_of(img):
        d = det.detect(img)
        return np.asarray(d[0].landmarks, np.float64) if d else None

    def emb(img, lm):
        if lm is None:
            return None
        return {"E01": sf.embed(img, lm), "E07": dl.embed(img, lm)}

    # ---- references + impostor pool
    sel = read_csv_gz(repo_path(f"{ROOT}/selection.csv.gz"))
    names = [r["identity"] for r in sel]
    R = {}
    for r in sel:
        img = imread(r["ref_path"])
        R[r["identity"]] = emb(img, lm_of(img))
    ids = [n for n in names if R[n] is not None]
    M = {e: np.stack([R[n][e] for n in ids]) for e in E.EVALUATORS}
    thr = {}
    for e in E.EVALUATORS:
        s = (M[e] @ M[e].T)[np.triu_indices(len(ids), 1)]
        thr[e] = float(np.quantile(s, 1 - FAR))
    idx = {n: i for i, n in enumerate(ids)}

    def score(q, ident):
        """genuine cos, best impostor cos, rank-1 correct — per evaluator."""
        out = {}
        for e in E.EVALUATORS:
            if q is None:
                out[e] = (float("nan"), float("nan"), False)
                continue
            s = M[e] @ q[e]
            g = float(s[idx[ident]])
            imp = float(np.max(np.delete(s, idx[ident])))
            out[e] = (g, imp, bool(np.argmax(s) == idx[ident]))
        return out

    gate = {r["query_id"]: r for r in read_csv_gz(repo_path(f"{ROOT}/gate.csv.gz"))}
    queries = {r["query_id"]: r for r in read_csv_gz(repo_path(f"{ROOT}/queries.csv.gz"))}
    runs = {c: {r["pair_id"]: r for r in read_csv_gz(repo_path(f"{ROOT}/runs/lfwq/{c}.csv.gz"))} for c in ("B0", "A4")}
    rows = []
    for qid in sorted(gate):
        gr, qr = gate[qid], queries[qid]
        ident = gr["identity"]
        rec = {"query_id": qid, "identity": ident, "cell": gr["cell"], "group": gr["group"], "reason": gr["reason"],
               "route": gr["route"], "chip_px": gr["chip_px"], "iod_px": gr.get("q_iod_px", "")}
        if gr["group"] == "G3":
            img = imread(qr["query_path"])
            sA = score(emb(img, lm_of(img)), ident)
            res = {"A": sA, "B": sA, "C": None}
            rec["guard_ok"] = ""
        else:
            b0 = imread(runs["B0"][qid]["output_path"])
            a4r = runs["A4"].get(qid)
            a4 = imread(a4r["output_path"]) if a4r and a4r["status"] == "OK" else None
            lm = lm_of(b0)
            if lm is None and a4 is not None:
                lm = lm_of(a4)
            sA = score(emb(b0, lm), ident)
            sC = score(emb(a4, lm), ident) if a4 is not None else sA
            guard = a4 is not None and len(det.detect(a4)) > 0          # service output guard (INVALID_OUTPUT)
            sB = sC if (gr["group"] == "G1" and guard) else sA
            res = {"A": sA, "B": sB, "C": sC}
            rec["guard_ok"] = guard
        for c in CONDS:
            for e in E.EVALUATORS:
                v = res[c][e] if res[c] is not None else (float("nan"), float("nan"), False)
                rec[f"{c}_cos_{e}"] = round(v[0], 5) if v[0] == v[0] else ""
                rec[f"{c}_imp_{e}"] = round(v[1], 5) if v[1] == v[1] else ""
                rec[f"{c}_match_{e}"] = bool(v[0] == v[0] and v[0] >= thr[e]) if res[c] is not None else ""
                rec[f"{c}_rank1_{e}"] = v[2] if res[c] is not None else ""
                rec[f"{c}_fa_{e}"] = bool(v[1] == v[1] and v[1] >= thr[e]) if res[c] is not None else ""
        rows.append(rec)

    # ---- aggregation
    def f(x):
        return float(x) if x not in ("", None) else float("nan")

    def agg(sel_rows, c):
        """condition c vs A on the selected rows, per evaluator + multi-evaluator verdicts."""
        out = {"n": len(sel_rows)}
        verd = Counter()
        for r in sel_rows:
            v1 = E.verdict(f(r[f"{c}_cos_E01"]) - f(r["A_cos_E01"]), delta["E01"])
            v7 = E.verdict(f(r[f"{c}_cos_E07"]) - f(r["A_cos_E07"]), delta["E07"])
            verd[E.multi_verdict(v1, v7)] += 1
        out["verdicts"] = dict(verd)
        out["confirmed_benefit"] = verd.get("CONFIRMED_BENEFIT", 0)
        out["harm_any"] = sum(verd.get(k, 0) for k in ("CONFIRMED_HARM", "PARTIAL_HARM", "DISAGREE"))
        for e in E.EVALUATORS:
            by = defaultdict(list)
            for r in sel_rows:
                d = f(r[f"{c}_cos_{e}"]) - f(r[f"A_cos_{e}"])
                by[r["identity"]].append(0.0 if d != d else d)
            est, lo, hi = E.cluster_boot(by)
            out[f"d_{e}"] = [round(est, 5), round(lo, 5), round(hi, 5), E.tie_mark(lo, hi)]
            m0 = [r["A_match_" + e] is True for r in sel_rows]
            m1 = [r[f"{c}_match_{e}"] is True for r in sel_rows]
            out[f"match_A_{e}"], out[f"match_{c}_{e}"] = sum(m0), sum(m1)
            out[f"flips_{e}"] = sum(x and not y for x, y in zip(m0, m1))
            out[f"rescues_{e}"] = sum(y and not x for x, y in zip(m0, m1))
            k0 = [r["A_rank1_" + e] is True for r in sel_rows]
            k1 = [r[f"{c}_rank1_{e}"] is True for r in sel_rows]
            out[f"rank1_A_{e}"], out[f"rank1_{c}_{e}"] = sum(k0), sum(k1)
            out[f"rank1_lost_{e}"] = sum(x and not y for x, y in zip(k0, k1))
            out[f"rank1_gained_{e}"] = sum(y and not x for x, y in zip(k0, k1))
            out[f"new_fa_{e}"] = sum(r[f"{c}_fa_{e}"] is True and r["A_fa_" + e] is not True for r in sel_rows)
        return out

    summary = {"run": run_info(), "proxy": "SFace E01 + dlib E07 (NOT Feature-Match)", "data": "LFW research proxy",
               "gate_seal": "78837a8d (p5-rwgate-1.0.0, frozen)", "thr_far_1e-2": thr, "delta": delta,
               "n_refs": len(ids), "groups": dict(Counter(r["group"] for r in rows)), "tables": {}}
    groups = {"G1": lambda r: r["group"] == "G1", "G2": lambda r: r["group"] == "G2",
              "G1+G2": lambda r: r["group"] in ("G1", "G2")}
    for gname, gf in groups.items():
        sel_g = [r for r in rows if gf(r)]
        summary["tables"][gname] = {c: agg(sel_g, c) for c in ("B", "C")}
        summary["tables"][gname]["by_cell"] = {
            cell: {c: agg([r for r in sel_g if r["cell"] == cell], c) for c in ("B", "C")}
            for cell in sorted({r["cell"] for r in sel_g})}
    g3 = [r for r in rows if r["group"] == "G3"]
    summary["tables"]["G3"] = {"n": len(g3), "a4_run": 0,
                               "match_A_E01": sum(r["A_match_E01"] is True for r in g3),
                               "match_A_E07": sum(r["A_match_E07"] is True for r in g3),
                               "reasons": dict(Counter(r["reason"].split(":")[0] for r in g3)),
                               "by_cell": dict(Counter(r["cell"] for r in g3))}
    # G2 opportunity analysis: where would forcing A4 have helped?
    g2 = [r for r in rows if r["group"] == "G2"]
    summary["tables"]["G2_reasons"] = {
        reason: agg([r for r in g2 if r["reason"] == reason], "C") for reason in sorted({r["reason"] for r in g2})}
    summary["tables"]["guard_failures_G1"] = sum(r["group"] == "G1" and r["guard_ok"] is False for r in rows)

    # ---- service verification (condition B == service output)
    if a.verify_service:
        from fes_service.pipeline import EnhancementPipeline
        from fes_service.worker import IsolatedModel
        m = IsolatedModel("fes_service.models:a4_factory", timeout_s=300, threads=4)
        pipe = EnhancementPipeline(m, timeout_s=300, gate="rw")
        picks = sorted((r for r in rows if r["group"] == "G1"),
                       key=lambda r: hashlib.sha256(f"r5-2|{r['query_id']}".encode()).hexdigest())[:a.verify_service]
        checks = []
        for r in picks:
            res = pipe.process(repo_path(queries[r["query_id"]]["query_path"]).read_bytes(), "full_image")
            out = cv2.imdecode(np.frombuffer(res.image, np.uint8), cv2.IMREAD_COLOR)
            same = res.status == "ENHANCED" and hashlib.sha256(out.tobytes()).hexdigest() == \
                runs["A4"][r["query_id"]]["output_pixel_sha256"]
            checks.append({"query_id": r["query_id"], "status": res.status, "pixels_equal_forced_a4": same})
        m.close()
        summary["service_verification"] = checks

    write_csv_gz(repo_path(f"{ROOT}/eval_rows.csv.gz"), rows, list(rows[0]))
    write_json(repo_path(f"{ROOT}/summary.json"), summary)
    md = render(summary)
    repo_path(f"{ROOT}/tables.md").write_text(md, encoding="utf-8")
    review_sheets(rows, queries, runs)
    print(md)


def render(s: dict) -> str:
    fm = lambda d: f"{d[0]:+.4f} {d[3]}"  # noqa: E731
    L = ["# R5-2 LFW research proxy — SFace (E01) / dlib (E07); NOT Feature-Match", "",
         f"Thresholds FAR 1e-2: {s['thr_far_1e-2']} · groups {s['groups']}", ""]
    hdr = ("| Group / cell | cond | n | ΔID SFace | ΔID dlib | benefit | harm | match SFace A→X | match dlib A→X | "
           "flips S/d | rescues S/d | rank-1 SFace A→X | new FA S/d |")
    for g in ("G1", "G2", "G1+G2"):
        L += [f"## {g}", "", hdr, "|" + "---|" * 13]
        for key, t in [("all", s["tables"][g])] + list(s["tables"][g]["by_cell"].items()):
            for c in ("B", "C"):
                x = t[c]
                if g == "G2" and c == "B":
                    continue                                   # B == A on G2 by construction
                L.append(f"| {key} | {c} | {x['n']} | {fm(x['d_E01'])} | {fm(x['d_E07'])} | {x['confirmed_benefit']} | "
                         f"{x['harm_any']} | {x['match_A_E01']}→{x[f'match_{c}_E01']} | {x['match_A_E07']}→{x[f'match_{c}_E07']} | "
                         f"{x['flips_E01']}/{x['flips_E07']} | {x['rescues_E01']}/{x['rescues_E07']} | "
                         f"{x['rank1_A_E01']}→{x[f'rank1_{c}_E01']} | {x['new_fa_E01']}/{x['new_fa_E07']} |")
        L.append("")
    L += ["## G2 by gate reason — forced A4 (C) vs original", "", hdr, "|" + "---|" * 13]
    for reason, x in s["tables"]["G2_reasons"].items():
        L.append(f"| {reason} | C | {x['n']} | {fm(x['d_E01'])} | {fm(x['d_E07'])} | {x['confirmed_benefit']} | "
                 f"{x['harm_any']} | {x['match_A_E01']}→{x['match_C_E01']} | {x['match_A_E07']}→{x['match_C_E07']} | "
                 f"{x['flips_E01']}/{x['flips_E07']} | {x['rescues_E01']}/{x['rescues_E07']} | "
                 f"{x['rank1_A_E01']}→{x['rank1_C_E01']} | {x['new_fa_E01']}/{x['new_fa_E07']} |")
    L += ["", f"G3: {s['tables']['G3']}", f"G1 output-guard failures: {s['tables']['guard_failures_G1']}",
          f"Service verification: {s.get('service_verification')}"]
    return "\n".join(L) + "\n"


def review_sheets(rows, queries, runs) -> None:
    """Query photo | chip→A4 output | reference, for a HUMAN reviewer; template CSV left blank."""
    out = repo_path("experiments/phase5_rapid/outputs/lfw_review")
    out.mkdir(parents=True, exist_ok=True)
    pick = []
    for g in ("G1", "G2", "G3"):
        cand = sorted((r for r in rows if r["group"] == g), key=lambda r: hashlib.sha256(f"rv|{r['query_id']}".encode()).hexdigest())
        pick += cand[:24 if g != "G3" else 12]
    tmpl = []
    for g in ("G1", "G2", "G3"):
        tiles = []
        for r in [p for p in pick if p["group"] == g]:
            q = cv2.imread(str(repo_path(queries[r["query_id"]]["query_path"])))
            ref = cv2.imread(str(repo_path(queries[r["query_id"]]["ref_path"])))
            a4r = runs["A4"].get(r["query_id"])
            a4 = cv2.imread(str(repo_path(a4r["output_path"]))) if a4r and a4r["status"] == "OK" else np.full((224, 224, 3), 200, np.uint8)
            cells = [cv2.resize(x, (180, 180), interpolation=cv2.INTER_AREA) for x in (q, a4, ref)]
            t = np.hstack(cells)
            t = np.vstack([t, np.full((18, t.shape[1], 3), 255, np.uint8)])
            cv2.putText(t, f"{r['query_id'][:40]}  {r['reason'][:30]}", (2, 193), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 200), 1)
            tiles.append(t)
            tmpl.append({"query_id": r["query_id"], "group": g, "cell": r["cell"],
                         "human_recognizable_query": "", "human_same_person_as_reference_query": "",
                         "human_same_person_as_reference_A4": "", "human_fabrication_visible_A4": "", "reviewer": ""})
        rowsimg = [np.hstack(tiles[i:i + 3] + [np.full_like(tiles[0], 255)] * (3 - len(tiles[i:i + 3])))
                   for i in range(0, len(tiles), 3)]
        cv2.imwrite(str(out / f"review_{g}.jpg"), np.vstack(rowsimg), [cv2.IMWRITE_JPEG_QUALITY, 88])
    write_csv_gz(out / "human_review_template.csv.gz", tmpl, list(tmpl[0]))


if __name__ == "__main__":
    main()
