"""R6-ONNX: export the two NAFNet width-64 models of the single-output mode (r6 = NAFNet-GoPro, r6_ft = face-blur
fine-tune) to ONNX (fp32, dynamic H / W) and check each export against PyTorch.

  scripts/dock.sh python scripts/r6_onnx/export_nafnet.py            (FES_IMAGE=fes-dev:cpu-p5-onnx)

NAFNetLocal (TLC) replaces global average pooling with a LOCAL average whose window is 1.5 × the 256 px training
patch (384 px at full resolution, halved per encoder level). local_arch.AvgPool2d branches on the tensor size in
Python, which an ONNX trace would freeze. For inputs with H, W ≤ 384 the window covers the whole feature map at every
level, so the layer IS a global mean: the export swaps it for AdaptiveAvgPool2d(1), which is exact in that range (the
service's face chips are ≤ 224 px) and is checked below against the untouched network. MAX_SIDE is written into the
model metadata; the runtime wrapper refuses larger inputs.

Output: models/M28_nafnet/onnx/<name>.onnx (git-ignored, like every weight file) + experiments/r6_onnx/export.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fes_service import models as M  # noqa: E402

MAX_SIDE = 384                                        # TLC window at full resolution (1.5 × 256)
OUT_DIR = ROOT / "models" / "M28_nafnet" / "onnx"
REPORT = ROOT / "experiments" / "r6_onnx" / "export.json"
MODELS = {
    "nafnet_gopro": {"profile": "r6", "weights": None, "sha256": None},
    "nafnet_face_ft": {"profile": "r6_ft", "weights": M.FACE_FT_WEIGHTS, "sha256": M.FACE_FT_SHA256},
}
CHECK_SIZES = [(224, 224), (192, 192), (96, 96), (160, 208), (384, 384)]
OPSET = 17


class Clamped(nn.Module):
    """net(x).clamp(0, 1): what nafnet_factory applies before converting back to uint8."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).clamp(0, 1)


def globalize(net: nn.Module) -> int:
    """Swap every TLC AvgPool2d for AdaptiveAvgPool2d(1) in place → number of layers swapped."""
    n = 0
    for mod in list(net.modules()):
        for name, child in list(mod.named_children()):
            if type(child).__name__ == "AvgPool2d" and hasattr(child, "base_size"):
                setattr(mod, name, nn.AdaptiveAvgPool2d(1))
                n += 1
    return n


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def export(name: str, spec: dict) -> dict:
    import onnx
    import onnxruntime as ort

    torch.manual_seed(0)
    ref, rep = M.load_nafnet(weights=spec["weights"], sha256=spec["sha256"])
    ref.eval()
    xs = {s: torch.rand(1, 3, *s) for s in CHECK_SIZES}
    with torch.inference_mode():
        want = {s: ref(x).clamp(0, 1).numpy() for s, x in xs.items()}     # untouched network (TLC layers)

    net, _ = M.load_nafnet(weights=spec["weights"], sha256=spec["sha256"])
    net.eval()
    swapped = globalize(net)
    wrapped = Clamped(net).eval()
    with torch.inference_mode():
        swap_diff = max(float(np.abs(wrapped(x).numpy() - want[s]).max()) for s, x in xs.items())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.onnx"
    t0 = time.perf_counter()
    torch.onnx.export(wrapped, (xs[(224, 224)],), str(path), input_names=["image"], output_names=["output"],
                      dynamic_axes={"image": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}}, opset_version=OPSET,
                      dynamo=False, do_constant_folding=True)
    export_s = time.perf_counter() - t0

    m = onnx.load(str(path))
    onnx.checker.check_model(str(path))
    meta = {"fes_model": name, "fes_profile": spec["profile"], "fes_source_weights": rep["weights"],
            "fes_source_sha256": rep["sha256"], "fes_max_side": str(MAX_SIDE), "fes_multiple": "16",
            "fes_io": "float32 RGB NCHW in [0,1] -> float32 RGB NCHW clamped to [0,1]"}
    del m.metadata_props[:]
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(path))
    ops = sorted({n.op_type for n in m.graph.node})

    so = ort.SessionOptions()
    so.intra_op_num_threads = torch.get_num_threads()
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    diffs = {}
    for s, x in xs.items():
        got = sess.run(None, {"image": x.numpy()})[0]
        d = np.abs(got - want[s])
        u8 = np.abs(np.rint(got * 255) - np.rint(want[s] * 255))
        diffs[f"{s[0]}x{s[1]}"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                   "u8_max": int(u8.max()), "u8_changed_frac": float((u8 > 0).mean())}
    return {"name": name, "profile": spec["profile"], "onnx": str(path.relative_to(ROOT)).replace("\\", "/"),
            "onnx_bytes": path.stat().st_size, "onnx_sha256": sha256(path),
            "pth": rep["weights"], "pth_bytes": (ROOT / rep["weights"]).stat().st_size, "pth_sha256": rep["sha256"],
            "n_params": rep["n_model_params"], "opset": OPSET, "ops": ops, "n_nodes": len(m.graph.node),
            "tlc_layers_swapped": swapped, "swap_max_abs_diff_torch": swap_diff, "export_s": round(export_s, 1),
            "max_side": MAX_SIDE, "onnx_vs_torch": diffs}


def main():
    import onnx
    import onnxruntime as ort
    out = {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "models": []}
    for name, spec in MODELS.items():
        r = export(name, spec)
        out["models"].append(r)
        worst = max(v["max_abs"] for v in r["onnx_vs_torch"].values())
        print(f"{name}: {r['onnx_bytes'] / 1e6:.1f} MB, {r['n_nodes']} nodes, swap diff {r['swap_max_abs_diff_torch']:.2e}, "
              f"onnx-vs-torch max {worst:.2e}, export {r['export_s']} s", flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
