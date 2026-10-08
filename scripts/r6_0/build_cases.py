"""R6-0 — evaluation corpus from EXISTING R5 evidence (host, stdlib only; deterministic).

    python scripts/r6_0/build_cases.py   →  data/manifests/phase5_rapid/r6_0/cases.json

Every case = one NAFNet output the R5 work produced, identified by (source, mode, name, profile):
  profile "live" = R5-6 build (NAFNet zero-pads to /16, black out-of-image chip area);
  profile "cand" = R5-7/8/10 build (reflect-pad16, reflect-filled chip; out-of-image area set back to black).
Prior labels are carried over verbatim (`prior`); cases without a prior label get prior "UNLABELLED" and are labelled
visually in R6-0 (labels_r6.json) — never silently.
"""
import csv, gzip, json, random, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "data/manifests/phase5_rapid"
SV = ROOT / "experiments/phase5_rapid/outputs/r5_6"
OUT = M / "r6_0/cases.json"


def enhanced(run):
    rows = []
    for line in (SV / run / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("status") == "ENHANCED" and r.get("model") != "A4" and r.get("used") != "A4":
            rows.append(r["input_id"].split("/")[-1])
    return sorted(set(rows))


def main():
    cases = {}

    def add(src, mode, name, profile, prior, note=""):
        cid = f"{src}:{mode}:{name}"
        cases.setdefault(cid, {"id": cid, "src": src, "mode": mode, "name": name, "profile": profile,
                               "prior": prior, "note": note})

    # A. R5-8 DEV labels (archive val, candidate build, every label kept)
    for k, v in json.loads((M / "r5_8/labels.json").read_text()).items():
        mode, name = k.split("/")
        add("dev", mode, name, "cand", v["label"], v["source"])
    # B. R5-7 escapes on TEST (candidate build without local guard) — R5-8 §13
    for mode, names in json.loads((M / "r5_8/r57_escapes_fate.json").read_text()).items():
        for n in names:
            add("test_r57", "fc" if mode == "face_chip" else "fi", f"image_{n}.jpg", "cand", "LOCALIZED_ARTIFACT",
                "R5-7 TEST escape (R5-8 §13)")
    # C. R5-6 live outputs above the R5-8 local limit (R5-8 §13 'R5-6 control') — prior = R5-8 visual statement
    for mode, f in (("fc", "test_scan_test_face_chip"), ("fi", "test_scan_test_full_image")):
        rows = list(csv.DictReader(gzip.open(M / f"r5_8/{f}.csv.gz", "rt")))
        for r in rows:
            if float(r["l_dot_c_max"] or 0) > 24.45:
                add("test_r56", mode, r["file"].replace("test__", "").replace(".png", ""), "live",
                    "LOCALIZED_ARTIFACT_R58_SCAN", "R5-6 live, l_dot_c_max > 24.45 (R5-8 §13)")
    # D. named escapes of R5-8 / R5-10 on TEST
    add("test_r58", "fi", "image_89280.jpg", "cand", "LOCALIZED_ARTIFACT", "R5-8 clear escape (dotted cheek)")
    add("test_r58", "fi", "image_91965.jpg", "cand", "FAINT_ARTIFACT", "R5-8 faint escape (edge specks)")
    add("test_r510", "fc", "image_84257.jpg", "cand", "LOCALIZED_ARTIFACT", "R5-10 clear escape (cheek speckle)")
    # E. real photos: every NAFNet output ever returned/attempted on the 200 CC0 photos
    for run, prof in (("real_full_image", "live"), ("r510_real_full_image", "cand")):
        for n in sorted(os.listdir(SV / run / "out")):
            name = n.replace("real__", "").replace(".png", "")
            prior = "COLOUR_MOTTLING" if name == "img118.jpg" and prof == "cand" else "UNLABELLED"
            add("real_r56" if prof == "live" else "real_r510", "fi", name, prof, prior, f"real photo, {run}")
    # F. unlabelled random TEST outputs actually returned by the live R5-6 build (seed 0) — false-reject denominator
    rnd = random.Random(0)
    for mode, run in (("fc", "test_face_chip"), ("fi", "test_full_image")):
        for n in rnd.sample(enhanced(run), 60):
            add("test_r56", mode, n, "live", "UNLABELLED", "random returned R5-6 output (seed 0)")
    # G. BORDERLINE pools (R5-10 DEV outputs, candidate build): moderate-blur archive val (newly admitted by r5_10),
    #    p4/r5gap synthetic degradation families (motion/gauss blur, noise, JPEG, low light, combined; all saved outputs),
    #    and NAFNet on SHARP GT faces ("clean": suspicious change on an input that needs no enhancement)
    R10 = ROOT / "experiments/phase5_rapid/outputs/r5_10"
    for mode in ("fc", "fi"):
        for n in rnd.sample(sorted(os.listdir(R10 / "archive" / mode)), 40):
            add("dev_mod", mode, n[:-4], "cand", "UNLABELLED", "R5-10 moderate-blur archive val output (seed 0)")
    for n in sorted(os.listdir(R10 / "p4" / "fc")):
        add("p4", "fc", n[:-4], "cand", "UNLABELLED", "R5-10 p4/r5gap synthetic family output")
    for n in rnd.sample(sorted(os.listdir(R10 / "clean" / "fc")), 30):
        add("clean", "fc", n[:-4], "cand", "UNLABELLED", "R5-10 NAFNet on a sharp GT face")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sorted(cases.values(), key=lambda c: c["id"]), indent=1))
    from collections import Counter
    print(len(cases), Counter((c["src"], c["prior"]) for c in cases.values()))


if __name__ == "__main__":
    main()
