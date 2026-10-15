"""R6-ONNX: tables from experiments/r6_onnx/{export.json, bench/rows.json} → experiments/r6_onnx/MODEL_ONLY_REPORT.md and
bench/per_image.csv (standard library only, runs on the host).

    python scripts/r6_onnx/report.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "r6_onnx"
MODELS = ("r6", "r6_ft")
NAME = {"r6": "NAFNet-GoPro (R6)", "r6_ft": "Face fine-tune (R6-FT)"}
KIND = {"gauss": "gaussian blur", "disk": "defocus blur", "motion": "motion blur", "none": "already sharp",
        "shake": "heavy shake blur"}
SET = {"val300": "FairFace val, 300 faces (128 px)", "synth160": "FairFace whole faces, 160 (128–224 px)",
       "archive100": "Archive test, 100 faces (heavy shake)"}


def p95(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def s(ms):
    return f"{ms / 1000:.2f} s"


def main():
    ex = json.loads((EXP / "export.json").read_text())
    bench = json.loads((EXP / "bench" / "rows.json").read_text())
    rows, probes = bench["rows"], {p["backend"]: p for p in bench["probes"]}
    sets = [k for k in SET if any(r["set"] == k for r in rows)]
    md = ["# R6-ONNX: PyTorch vs ONNX Runtime for the two NAFNet models", "",
          f"{len(rows)} blurred faces, CPU only, {bench.get('threads', 2)} threads for both backends. "
          f"torch {ex['torch']} · onnxruntime {ex['onnxruntime']} · opset {ex['models'][0]['opset']} · fp32.", ""]

    md += ["## 1. Exported files", "", table(
        ["Model", "ONNX file", "ONNX size", "PyTorch size", "Parameters", "ONNX SHA-256 (first 16)"],
        [[NAME[m["profile"]], m["onnx"], f"{m['onnx_bytes'] / 1e6:.1f} MB", f"{m['pth_bytes'] / 1e6:.1f} MB",
          f"{m['n_params'] / 1e6:.1f} M", m["onnx_sha256"][:16]] for m in ex["models"]]), "",
        f"Input: float32 RGB, any H × W that is a multiple of 16 and at most {ex['models'][0]['max_side']} px "
        "(face chips are at most 224 px).", ""]

    md += ["## 2. Time per face", "", "One full single-output request = both models × 4 mirror runs + selection "
           "(8 forward passes). Median over the faces of each set.", ""]
    trows = []
    for k in sets + ["all"]:
        sel = [r for r in rows if k == "all" or r["set"] == k]
        t, o = [r["torch"]["best_ms"] for r in sel], [r["onnx"]["best_ms"] for r in sel]
        trows.append([SET.get(k, f"All {len(sel)} faces"), s(st.median(t)), s(st.median(o)),
                      f"{(1 - st.median(o) / st.median(t)) * 100:.0f} % faster", s(p95(t)), s(p95(o))])
    md += [table(["Set", "PyTorch", "ONNX", "ONNX vs PyTorch", "PyTorch p95", "ONNX p95"], trows), ""]
    md += ["One forward pass of one model (median):", ""]
    prows = []
    for k in sets:
        sel = [r for r in rows if r["set"] == k]
        for m in MODELS:
            t, o = [r["torch"][m]["single_ms"] for r in sel], [r["onnx"][m]["single_ms"] for r in sel]
            prows.append([SET[k], NAME[m], f"{st.median(t):.0f} ms", f"{st.median(o):.0f} ms",
                          f"{(1 - st.median(o) / st.median(t)) * 100:.0f} % faster"])
    md += [table(["Set", "Model", "PyTorch", "ONNX", "ONNX vs PyTorch"], prows), ""]
    by = {}
    for r in rows:
        b = min(224, ((max(r["h"], r["w"]) + 31) // 32) * 32)
        by.setdefault(b, []).append(r)
    md += ["By face size (full request, median):", "", table(
        ["Face side up to", "Faces", "PyTorch", "ONNX"],
        [[f"{b} px", len(v), s(st.median([r["torch"]["best_ms"] for r in v])),
          s(st.median([r["onnx"]["best_ms"] for r in v]))] for b, v in sorted(by.items())]), ""]

    md += ["## 3. Start-up and memory (both models loaded in one process)", "", table(
        ["Backend", "Load time (both models)", "Memory after load", "Peak memory while running"],
        [[b, f"{sum(probes[b]['load_s'].values()):.1f} s", f"{probes[b]['rss_loaded_mb']} MB",
          f"{probes[b]['rss_peak_mb']} MB"] for b in ("torch", "onnx")]), ""]

    md += ["## 4. Are the outputs the same?", ""]
    drows = []
    for key, lab in (("r6_single", "R6, one pass"), ("r6_ft_single", "R6-FT, one pass"), ("r6_r6", "R6, 4-run consensus"),
                     ("r6_ft_r6", "R6-FT, 4-run consensus"), ("best", "Final single output")):
        mx = [r["diff"][key]["max"] for r in rows]
        drows.append([lab, f"{sum(v == 0 for v in mx)} / {len(mx)}", f"{sum(v <= 1 for v in mx)} / {len(mx)}",
                      f"{sum(v > 2 for v in mx)}", max(mx), f"{st.mean(r['diff'][key]['mean'] for r in rows):.5f}"])
    md += [table(["Output", "Bit-identical faces", "Within 1 grey level", "Faces with a pixel off by > 2",
                  "Largest pixel difference (0–255)", "Mean pixel difference"], drows), "",
           f"- Same model chosen by both backends: **{sum(r['same_chosen'] for r in rows)} / {len(rows)}**",
           f"- Same stability decision for both models: **{sum(r['same_stable'] for r in rows)} / {len(rows)}**", ""]
    bad = [r for r in rows if not r["same_chosen"] or r["diff"]["best"]["max"] > 2]
    if bad:
        md += ["Faces where the final output differs visibly or the choice differs (pictures in `bench/diff/`):", "", table(
            ["Set", "Face", "Blur", "PyTorch chose", "ONNX chose", "Largest pixel difference", "Pixels off by > 2"],
            [[r["set"], r["id"], KIND.get(r["kind"], r["kind"]), r["torch"]["chosen"], r["onnx"]["chosen"],
              r["diff"]["best"]["max"], f"{r['diff']['best']['gt2'] * 100:.2f} %"] for r in bad[:40]]), ""]

    md += ["## 5. Quality against the sharp original", "", "PSNR in dB and SFace similarity to the sharp original "
           "(higher is better; SFace here is a same-image fidelity check, not an identity test).", ""]
    qrows = []
    groups = [(SET[k], [r for r in rows if r["set"] == k]) for k in sets]
    groups += [(f"— {KIND[k]}", [r for r in rows if r["kind"] == k and r["set"] != "archive100"]) for k in ("gauss", "disk", "motion", "none")]
    for lab, sel in groups:
        if not sel:
            continue
        f = [r for r in sel if "sface_input" in r and "sface_best" in r["torch"]]
        qrows.append([lab, len(sel), f"{st.mean(r['psnr_input'] for r in sel):.2f}",
                      f"{st.mean(r['torch']['psnr_best'] for r in sel):.2f}", f"{st.mean(r['onnx']['psnr_best'] for r in sel):.2f}",
                      f"{st.mean(r['sface_input'] for r in f):.3f}" if f else "-",
                      f"{st.mean(r['torch']['sface_best'] for r in f):.3f}" if f else "-",
                      f"{st.mean(r['onnx']['sface_best'] for r in f):.3f}" if f else "-"])
    md += [table(["Faces", "N", "PSNR input", "PSNR PyTorch", "PSNR ONNX", "SFace input", "SFace PyTorch", "SFace ONNX"], qrows), ""]
    crows = []
    for k in sets:
        sel = [r for r in rows if r["set"] == k]
        for b in ("torch", "onnx"):
            c = [r[b]["chosen"] for r in sel]
            crows.append([SET[k], b, c.count("r6"), c.count("r6_ft"), c.count(None)])
    md += ["Which model the single-output rule kept:", "", table(["Set", "Backend", "R6", "R6-FT", "None (original returned)"], crows), ""]

    (EXP / "MODEL_ONLY_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    head = ["set", "face", "blur", "h", "w", "torch_pass_r6_ms", "onnx_pass_r6_ms", "torch_pass_r6_ft_ms", "onnx_pass_r6_ft_ms",
            "torch_full_ms", "onnx_full_ms", "torch_chosen", "onnx_chosen", "final_max_pixel_diff", "psnr_input",
            "psnr_torch", "psnr_onnx", "sface_input", "sface_torch", "sface_onnx"]
    lines = [",".join(head)]
    for r in rows:
        t, o = r["torch"], r["onnx"]
        lines.append(",".join(str(v) for v in [
            r["set"], r["id"], r["kind"], r["h"], r["w"], t["r6"]["single_ms"], o["r6"]["single_ms"], t["r6_ft"]["single_ms"],
            o["r6_ft"]["single_ms"], t["best_ms"], o["best_ms"], t["chosen"], o["chosen"], r["diff"]["best"]["max"],
            r["psnr_input"], t["psnr_best"], o["psnr_best"], r.get("sface_input", ""), t.get("sface_best", ""),
            o.get("sface_best", "")]))
    (EXP / "bench" / "per_image.csv").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
