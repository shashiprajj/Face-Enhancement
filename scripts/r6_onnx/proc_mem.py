"""R6-ONNX-INTEGRATION: memory of every process in THIS container (run with `docker exec` inside the service container).
Prints JSON: per process pid, parent, resident memory now (VmRSS) and its peak (VmHWM), plus the container totals."""

import json
import os


def status(pid):
    d = {}
    for line in open(f"/proc/{pid}/status", encoding="utf-8"):
        k, _, v = line.partition(":")
        d[k] = v.strip()
    return d


procs = []
for pid in sorted(int(p) for p in os.listdir("/proc") if p.isdigit()):
    try:
        s = status(pid)
        if "VmRSS" not in s or pid == os.getpid():
            continue
        procs.append({"pid": pid, "ppid": int(s["PPid"]), "name": s["Name"], "threads": int(s["Threads"]),
                      "rss_mb": round(int(s["VmRSS"].split()[0]) / 1024, 1),
                      "peak_rss_mb": round(int(s["VmHWM"].split()[0]) / 1024, 1)})
    except (FileNotFoundError, ProcessLookupError):
        pass
cg = {}
for name, path in (("current_mb", "/sys/fs/cgroup/memory.current"), ("peak_mb", "/sys/fs/cgroup/memory.peak")):
    try:
        cg[name] = round(int(open(path).read()) / 2 ** 20, 1)
    except (OSError, ValueError):
        cg[name] = None
cpu = {}
try:
    for line in open("/sys/fs/cgroup/cpu.stat"):
        k, v = line.split()
        cpu[k] = int(v)
except (OSError, ValueError):
    pass
cg["cpu_usage_s"] = round(cpu["usage_usec"] / 1e6, 2) if "usage_usec" in cpu else None
print(json.dumps({"processes": procs, "sum_rss_mb": round(sum(p["rss_mb"] for p in procs), 1),
                  "sum_peak_rss_mb": round(sum(p["peak_rss_mb"] for p in procs), 1), "cgroup": cg}))
