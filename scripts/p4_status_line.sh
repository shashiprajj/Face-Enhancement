#!/usr/bin/env bash
# Status table for the 15-minute updates (P4-INFRA-RESUME): inference, scoring, resources.
cd "$(dirname "$0")/.."
PYTHONIOENCODING=utf-8 python - <<'PY'
import json, os, time, subprocess
exp = 896
R = "data/manifests/phase4"
print(f"Status {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC")
NAMES = {"B0": "Original input (no enhancement)", "B1": "Lanczos-4 upscaling", "B2": "Classical (Lanczos+NLM+unsharp)",
         "B2c": "Classical + CLAHE", "A1": "RestoreFormer++", "A2": "VQFR v1", "A3": "Real-ESRGAN x4plus",
         "A4": "SwinIR-L (real-SR GAN)", "A5": "FBCNN → NAFNet-SIDD → Real-ESRGAN"}
print("| ID | Model | Inference | Rate | ETA | Failed | SFace score | dlib score |")
print("|---|---|---|---|---|---|---|---|")
for c in ["B0", "B1", "B2", "B2c", "A3", "A5", "A1", "A4", "A2"]:
    f = f"{R}/runs/p4dev/{c}.partial.jsonl"
    ok = fail = 0; rate = eta = "—"
    if os.path.exists(f):
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        ok = len({r["pair_id"] for r in rows if r["status"] == "OK"}); fail = sum(r["status"] != "OK" for r in rows)
        st = f"{R}/runs/p4dev/{c}.resume_start"
        if os.path.exists(st) and ok < exp:
            t0, n0 = map(float, open(st).read().split()); el = time.time() - t0
            if ok > n0 and el > 60:
                r = (ok - n0) / el * 3600; rate = f"{r:.0f}/h"; eta = f"{(exp - ok) / r:.1f} h"
    inf = "✅ 896/896" if ok == exp else (f"⏳ {ok}/{exp} ({100*ok/exp:.0f}%)" if ok else "queued")
    s1 = "✅" if os.path.exists(f"{R}/metrics/p4dev/{c}.csv.gz") else "—"
    s7 = "✅" if os.path.exists(f"{R}/metrics_e07/p4dev/{c}.csv.gz") else "—"
    print(f"| {c} | {NAMES[c]} | {inf} | {rate} | {eta} | {fail} | {s1} | {s7} |")
mem = subprocess.run(["docker", "run", "--rm", "fes-dev:cpu-p2", "sh", "-c", "grep MemAvailable /proc/meminfo; cat /proc/loadavg"],
                     capture_output=True, text=True).stdout.split()
wo = subprocess.run(["docker", "ps", "-q", "--filter", "name=welorbit"], capture_output=True, text=True).stdout.split()
st = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"], capture_output=True, text=True).stdout
fes = [l.split(" /")[0] for l in st.splitlines() if l.startswith("fes-p4dev")]
print()
print("| Resource | Value |")
print("|---|---|")
print(f"| VM free memory | {int(mem[1]) / 1e6:.1f} GB of 15.3 GB |")
print(f"| Load avg (1 min) | {mem[3]} on 16 CPU |")
print(f"| Inference container | {', '.join(fes) or 'none'} |")
print(f"| WelOrbit containers up | {len(wo)} / 12 |")
PY
