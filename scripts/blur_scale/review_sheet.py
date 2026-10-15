"""Review sheet for the default "give up above" blur percentage: already-enhanced benchmark faces grouped by their
blur percent (configs/phase5/blur_scale.json), with how often enhancement helped per band and sample image pairs.
Reads existing benchmark records and side-by-side images only; runs nothing on the service.

    python scripts/blur_scale/review_sheet.py
    → Dataset-and-code/fes_indian_expression_benchmark/blur_scale/give_up_review.html (local; personal images)
"""

from __future__ import annotations

import glob
import html
import json
import os
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
BENCH = ROOT / "Dataset-and-code" / "fes_indian_expression_benchmark"
OUT = BENCH / "blur_scale" / "give_up_review.html"
BANDS = [(36, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 99.95), (99.95, 100.01)]
SHOW_FROM, PER_BAND, SEED = 60, 16, 20261014
SIDE_RUNS = ("dev", "dev_comp", "test", "test_v11")          # later runs replace earlier records of a variant


def records() -> list[dict]:
    from fes_service import blur_scale as BS
    sc = BS.load_scale()
    out = []
    q = {}
    for line in (BENCH / "manifests" / "enhancement_results.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["outcome"] == "enhanced" and r.get("quality"):
            q[r["variant_id"]] = json.loads(r["quality"])["lapd1_c"]
    for line in (BENCH / "manifests" / "comparisons.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        if c["variant_id"] in q and c.get("psnr_output") is not None and c.get("comparison_path"):
            sf = (c["sface_clean_vs_output"] - c["sface_clean_vs_degraded"]
                  if c.get("sface_clean_vs_output") is not None and c.get("sface_clean_vs_degraded") is not None
                  else None)
            out.append({"framing": "front", "variant_id": c["variant_id"], "type": c["type"], "level": c["level"],
                        "pct": sc.percent("front", q[c["variant_id"]]), "dpsnr": c["psnr_output"] - c["psnr_degraded"],
                        "dsface": sf, "img": BENCH / c["comparison_path"]})
    side = {}
    for run in SIDE_RUNS:
        for p in sorted(glob.glob(str(BENCH / "side_profile_run" / run / "results*.jsonl"))):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                img = BENCH / "side_profile_run" / run / "side_by_side" / f"{r['variant_id']}.jpg"
                if (r.get("framing") == "detection_box" and r["outcome"] == "enhanced" and "psnr_output" in r
                        and img.exists()):
                    side[r["variant_id"]] = {"framing": "side", "variant_id": r["variant_id"], "type": r["type"],
                                             "level": r["level"],
                                             "pct": sc.percent("side", json.loads(r["quality"])["lapd1_c"]),
                                             "dpsnr": r["psnr_output"] - r["psnr_degraded"], "dsface": None, "img": img}
    return out + list(side.values())


def band_of(pct: float):
    return next((b for b in BANDS if b[0] <= pct < b[1]), None)


def label(b) -> str:
    return "100 % (σ 4 or blurrier)" if b[0] >= 99.95 else f"{b[0]:g}–{min(b[1], 100):g} %"


def main() -> None:
    recs = records()
    rows = []
    for b in BANDS:
        for fr in ("front", "side", "all"):
            g = [r for r in recs if band_of(r["pct"]) == b and fr in ("all", r["framing"])]
            if not g:
                continue
            sf = [r["dsface"] for r in g if r["dsface"] is not None]
            rows.append(f"<tr class='{fr}'><td>{label(b)}</td><td>{fr}</td><td>{len(g)}</td>"
                        f"<td>{100 * sum(r['dpsnr'] > 0 for r in g) / len(g):.0f} %</td>"
                        f"<td>{st.fmean(r['dpsnr'] for r in g):+.2f}</td>"
                        f"<td>{f'{st.fmean(sf):+.3f}' if sf else '–'}</td></tr>")
    rnd = random.Random(SEED)
    sections = []
    for b in BANDS:
        if b[0] < SHOW_FROM:
            continue
        g = sorted((r for r in recs if band_of(r["pct"]) == b), key=lambda r: r["variant_id"])
        pick = rnd.sample(g, min(PER_BAND, len(g)))
        pick.sort(key=lambda r: r["pct"])
        cards = "".join(
            f"<figure><img loading='lazy' src='{html.escape(os.path.relpath(r['img'], OUT.parent).replace(os.sep, '/'))}'>"
            f"<figcaption><b>{r['pct']:.0f} %</b> · {r['framing']} · {html.escape(r['type'])} L{r['level']} · "
            f"ΔPSNR {r['dpsnr']:+.2f} dB<br><small>{html.escape(r['variant_id'])}</small></figcaption></figure>"
            for r in pick)
        sections.append(f"<h2>{label(b)} <small>({len(g)} enhanced faces, {len(pick)} shown)</small></h2>"
                        f"<div class='grid'>{cards}</div>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Give-up review</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--mut:#aaa;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}}
table{{border-collapse:collapse;margin:8px 0 24px}}td,th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right}}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){{text-align:left}}tr.all td{{font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}}
figure{{margin:0;border:1px solid var(--line);padding:6px}}img{{width:100%;height:auto;display:block}}
figcaption{{font-size:13px;margin-top:4px}}small{{color:var(--mut)}}
</style></head><body>
<h1>“Give up above” — which blur level is too much?</h1>
<p>Every face below was <b>enhanced</b> by the service in the benchmark (synthetic blur on the Indian-expression photos).
Blur % is the new admin scale: 0 % sharp, 100 % = Gaussian σ 4 on an 800-px photo or blurrier. Each image is the
benchmark's side-by-side (clean reference · blurred input · service output). Question for the review: from which band on
are the outputs no longer worth returning? Faces above the chosen value would be returned unchanged.</p>
<h2>How often enhancement helped, by blur band</h2>
<table><tr><th>Blur band</th><th>Faces</th><th>n</th><th>Better (ΔPSNR&gt;0)</th><th>Mean ΔPSNR (dB)</th>
<th>Mean ΔSFace (front only)</th></tr>{''.join(rows)}</table>
<p><small>ΔPSNR = output vs blurred input, both against the clean reference. ΔSFace = change in identity similarity to
the clean face (front faces only; not reliable on profiles). Side records combine the best_side runs (dev, dev_comp,
test, test_v11; latest record per variant).</small></p>
{''.join(sections)}
</body></html>""", encoding="utf-8")
    print("written", OUT, f"({len(recs)} enhanced faces)")
    print("\n".join(r.replace("<tr class='", "").replace("'><td>", " | ").replace("</td><td>", " | ")
                    .replace("</td></tr>", "") for r in rows))


if __name__ == "__main__":
    main()
