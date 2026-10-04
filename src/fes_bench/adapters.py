"""Candidate adapters: one uniform call for every configuration (P3-T2).

    adapter.enhance(degraded_bgr_uint8 (s×s), ctx) -> (output_bgr_uint8 224×224 in the E-frame, meta dict)

`ctx` carries only what every adapter may use: the input size `s` and the reference's recorded
E→F transform `M_e2f` (P2-T3). No adapter sees the clean reference, the degradation label or its parameters
(no oracle information). Model-specific range / channel order / padding live inside the adapter only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fes_bench import archs
from fes_bench import geometry as G
from fes_data.common import repo_path, sha256_file


# ---------------------------------------------------------------- tensor helpers (upstream conventions)

def bgr_u8_to_rgb_t(img: np.ndarray) -> torch.Tensor:
    """uint8 BGR HxWx3 → float32 RGB 1x3xHxW in [0, 1] (basicsr img2tensor / KAIR uint2tensor4)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]


def rgb_t_to_bgr_u8(t: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Clamp to [lo, hi], rescale to [0, 255], round (basicsr tensor2img / KAIR single2uint)."""
    x = t.detach().float().clamp(lo, hi)[0]
    x = (x - lo) / (hi - lo)
    rgb = (x.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- weights registry (P1-T9 manifest)

WEIGHTS = {
    "M03": ("A01", "models/M03_restoreformerpp/RestoreFormer++.ckpt",
            "613fe52805f86bf8c2bffff08ae9f7a0b99f408be1bf221767af6183038be3a2"),
    "M05": ("A05", "models/M05_vqfr/VQFR_v1-33a1fac5.pth",
            "33a1fac51e2b286705fc3e561d87330d1562b574e3cf09fb4201d55d73fe57bf"),
    "M23": ("A02", "models/M23_realesrgan/RealESRGAN_x4plus.pth",
            "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"),
    "M24": ("A03", "models/M24_swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
            "99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a"),
    "M30": ("A04", "models/M30_fbcnn/fbcnn_color.pth",
            "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8"),
    "M28-GoPro": ("A06", "models/M28_nafnet/NAFNet-GoPro-width64.pth",
                  "329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f"),
    "M28-SIDD": ("A07", "models/M28_nafnet/NAFNet-SIDD-width64.pth",
                 "cd685efaae01f7c4e9951f2deab05780079c8eb1e49ed664b72f6db04dabb445"),
    # Phase 5 Rapid, research-only (official GitHub releases, downloaded 2026-10-04)
    "M01": ("P5-GFPGAN", "models/M01_gfpgan/GFPGANv1.4.pth",
            "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad"),
    "M02": ("P5-CODEFORMER", "models/M02_codeformer/codeformer.pth",
            "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7"),
}

# builder, checkpoint key, prefix to strip, strict
NETS = {
    "M03": (archs.build_restoreformerpp, "state_dict", "vqvae.", False),  # upstream loads strict=False; keys reported
    "M05": (archs.build_vqfr_v1, "params", None, True),
    "M23": (archs.build_rrdbnet_x4, "params_ema", None, True),
    "M24": (archs.build_swinir_realsr_l_x4, "params_ema", None, True),
    "M30": (archs.build_fbcnn_color, None, None, True),
    "M28-GoPro": (archs.build_nafnet_gopro, "params", None, True),
    "M28-SIDD": (archs.build_nafnet_sidd, "params", None, True),
    "M01": (archs.build_gfpgan_v14, "params_ema", None, True),
    "M02": (archs.build_codeformer, "params_ema", None, True),
}


def load_net(key: str, verify_sha: bool = True) -> tuple[torch.nn.Module, dict]:
    asset, rel, sha = WEIGHTS[key]
    builder, ck_key, prefix, strict = NETS[key]
    t0 = time.perf_counter()
    net = builder()
    # SHA-256 is verified while the file is streamed to local storage (archs.staged_copy)
    rep = archs.load_weights(net, repo_path(rel), ck_key, prefix, strict, expect_sha256=sha if verify_sha else None)
    rep.update(asset_id=asset, weights=rel, sha256=sha, strict=strict, load_s=round(time.perf_counter() - t0, 3))
    if rep["missing_keys"]:
        raise RuntimeError(f"{key}: missing keys after load: {rep['missing_keys'][:5]}")
    return net, rep


# ---------------------------------------------------------------- adapter base

@dataclass
class Adapter:
    config_id: str
    label: str
    kind: str                           # baseline | face | general_sr | chain
    components: list[str] = field(default_factory=list)
    load_reports: dict = field(default_factory=dict)

    def load(self) -> None:
        pass

    def enhance(self, deg: np.ndarray, ctx: dict) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"config_id": self.config_id, "label": self.label, "kind": self.kind,
                "components": self.components, "settings": self.settings()}

    def settings(self) -> dict:
        return {}


# ---------------------------------------------------------------- baselines (P0-T4 §6 B0–B2; P1-T5 §3)

class B0Identity(Adapter):
    """Degraded input, brought to the E-frame only when it is LR (bicubic, as for P2-T4 input statistics)."""

    def __init__(self):
        super().__init__("B0", "Degraded input (no enhancement)", "baseline")

    def enhance(self, deg, ctx):
        return G.upsample_to_e(deg, cv2.INTER_CUBIC), {}

    def settings(self):
        return {"lr_to_e": "INTER_CUBIC", "same_size": "identity"}


class B1Lanczos(Adapter):
    def __init__(self):
        super().__init__("B1", "Classical upscaling (Lanczos-4)", "baseline")

    def enhance(self, deg, ctx):
        return G.upsample_to_e(deg, cv2.INTER_LANCZOS4), {}

    def settings(self):
        return {"lr_to_e": "INTER_LANCZOS4", "same_size": "identity"}


class B2Classical(Adapter):
    """Lanczos → [CLAHE] → non-local means → unsharp mask. Fixed a-priori parameters (not tuned; P3-T1 §6)."""

    NLM = dict(h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)
    UNSHARP = dict(sigma=1.0, amount=0.5)
    CLAHE = dict(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(self, clahe: bool = False):
        super().__init__("B2c" if clahe else "B2",
                         "Classical restoration + CLAHE" if clahe else "Classical restoration",
                         "baseline")
        self.clahe = clahe
        self._clahe = cv2.createCLAHE(**self.CLAHE) if clahe else None

    def enhance(self, deg, ctx):
        x = G.upsample_to_e(deg, cv2.INTER_LANCZOS4)
        if self.clahe:
            lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
            lab[..., 0] = self._clahe.apply(lab[..., 0])
            x = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        x = cv2.fastNlMeansDenoisingColored(x, None, self.NLM["h"], self.NLM["hColor"],
                                            self.NLM["templateWindowSize"], self.NLM["searchWindowSize"])
        blur = cv2.GaussianBlur(x.astype(np.float32), (0, 0), self.UNSHARP["sigma"])
        x = np.clip(np.rint(x.astype(np.float32) * (1 + self.UNSHARP["amount"]) - blur * self.UNSHARP["amount"]),
                    0, 255).astype(np.uint8)
        return x, {}

    def settings(self):
        s = {"upscale": "INTER_LANCZOS4", "nlm": self.NLM, "unsharp": self.UNSHARP}
        if self.clahe:
            s["clahe_lab_L"] = {"clipLimit": 2.0, "tileGridSize": [8, 8], "order": "after upscale, before NLM"}
        return s


# ---------------------------------------------------------------- face-specific (F-frame, 512)

class FaceModel(Adapter):
    """D-frame → F-frame (one bicubic warp, facexlib fill) → model in [-1, 1] RGB → F→E (area + linear)."""

    def __init__(self, config_id, label, key):
        super().__init__(config_id, label, "face", [key])
        self.key = key
        self.net = None

    def load(self):
        self.net, rep = load_net(self.key)
        self.load_reports = {self.key: rep}

    def _forward(self, x):
        raise NotImplementedError

    def enhance(self, deg, ctx):
        s, M = ctx["s"], ctx["M_e2f"]
        f_in = G.warp(deg, G.d_to_f(s, M), (G.F_SIZE, G.F_SIZE), cv2.INTER_CUBIC)
        x = bgr_u8_to_rgb_t(f_in) * 2.0 - 1.0          # normalize(mean=0.5, std=0.5)
        with torch.inference_mode():
            y = self._forward(x)
        f_out = rgb_t_to_bgr_u8(y, -1.0, 1.0)
        out = G.warp(f_out, G.f_to_e(M), (G.E_SIZE, G.E_SIZE))
        cov = G.coverage_mask(M)
        if not cov.all():   # P1-T5 §5.1 paste-back rule: background = bicubic for pixels the model never saw
            out = np.where(cov[..., None], out, G.upsample_to_e(deg, cv2.INTER_CUBIC))
        return out, {"f_coverage": round(float(cov.mean()), 4)}

    def settings(self):
        return {"frame": "FFHQ-512 (facexlib template)", "d_to_f": "single warp INTER_CUBIC",
                "border": "BORDER_CONSTANT (135,133,132) BGR", "f_to_e": "INTER_AREA pre-reduction + INTER_LINEAR",
                "uncovered_e_pixels": "bicubic degraded input (P1-T5 paste-back rule; excluded from metrics)",
                "input_norm": "RGB, (x-0.5)/0.5", "output": "clamp[-1,1] → round", "precision": "fp32"}


class RestoreFormerPP(FaceModel):
    def __init__(self):
        super().__init__("A1", "RestoreFormer++", "M03")

    def _forward(self, x):
        return self.net(x)[0]


class VQFRv1(FaceModel):
    def __init__(self):
        super().__init__("A2", "VQFR v1", "M05")

    def _forward(self, x):
        return self.net(x)["main_dec"]   # demo_util.py: output of the main (fidelity) decoder

    def settings(self):
        s = super().settings()
        s["output_branch"] = "main_dec (v1 has no fidelity_ratio control; argument unused upstream)"
        return s


class GFPGANv14(FaceModel):
    """GFPGAN v1.4 (gfpgan/utils.py: return_rgb=False, weight=0.5). Upstream injects random StyleGAN noise on every
    call; the seed is fixed per call so the same input always gives the same output."""

    def __init__(self):
        super().__init__("A6", "GFPGAN v1.4 (research-only)", "M01")

    def _forward(self, x):
        torch.manual_seed(0)
        return self.net(x, return_rgb=False, weight=0.5)[0]


class CodeFormerW(FaceModel):
    """CodeFormer (inference_codeformer.py: net(x, w=fidelity, adain=True)[0]). w=0.5 upstream default; w=1.0 =
    maximum fidelity to the input (identity-oriented)."""

    def __init__(self, config_id, w):
        super().__init__(config_id, f"CodeFormer w={w} (research-only, NC licence)", "M02")
        self.w = w

    def _forward(self, x):
        return self.net(x, w=self.w, adain=True)[0]

    def settings(self):
        return {**super().settings(), "fidelity_w": self.w, "adain": True}


# ---------------------------------------------------------------- general SR (D-frame native, ×4)

def _realesrgan_x4(net, x: torch.Tensor) -> torch.Tensor:
    # RealESRGANer @a4abfb2 with inference_realesrgan.py defaults: tile=0, pre_pad=0, scale 4 → no mod pad
    return net(x)


def _swinir_x4(net, x: torch.Tensor, window: int = 8) -> torch.Tensor:
    # main_test_swinir.py @6545850: flip-pad to a multiple of window_size (always adds ≥1 px), crop h*4
    _, _, h, w = x.shape
    hp = (h // window + 1) * window - h
    wp = (w // window + 1) * window - w
    x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h + hp, :]
    x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w + wp]
    return net(x)[..., :h * 4, :w * 4]


class GeneralSR(Adapter):
    """Native-resolution input → ×4 → resize to the E-frame (area when >224, Lanczos when <224)."""

    def __init__(self, config_id, label, key, fn):
        super().__init__(config_id, label, "general_sr", [key])
        self.key, self.fn, self.net = key, fn, None

    def load(self):
        self.net, rep = load_net(self.key)
        self.load_reports = {self.key: rep}

    def enhance(self, deg, ctx):
        with torch.inference_mode():
            y = self.fn(self.net, bgr_u8_to_rgb_t(deg))
        return G.resize_to_e(rgb_t_to_bgr_u8(y)), {"sr_out": int(y.shape[-1])}

    def settings(self):
        return {"input": "D-frame native, RGB [0,1]", "scale": 4, "tile": 0,
                "to_e": "INTER_AREA if 4s>224, INTER_LANCZOS4 if 4s<224", "precision": "fp32"}


def RealESRGAN():
    return GeneralSR("A3", "Real-ESRGAN x4plus", "M23", _realesrgan_x4)


def SwinIR():
    return GeneralSR("A4", "SwinIR-L real-SR x4 (GAN)", "M24", _swinir_x4)


# ---------------------------------------------------------------- chain A5

class Chain(Adapter):
    """FBCNN (blind QF) → NAFNet → Real-ESRGAN ×4 → E-frame. Every stage runs on every input (no oracle routing)."""

    def __init__(self, config_id="A5", nafnet="M28-SIDD"):
        super().__init__(config_id, f"Chain FBCNN → NAFNet-{nafnet.split('-')[1]} → Real-ESRGAN", "chain",
                         ["M30", nafnet, "M23"])
        self.nafnet_key = nafnet
        self.nets = {}

    def load(self):
        for k in self.components:
            self.nets[k], self.load_reports[k] = load_net(k)

    def enhance(self, deg, ctx):
        x = bgr_u8_to_rgb_t(deg)
        with torch.inference_mode():
            x, qf = self.nets["M30"](x)                    # FBCNN.forward(x, qf_input=None) → blind QF prediction
            x = x.clamp(0, 1)
            x = self.nets[self.nafnet_key](x).clamp(0, 1)
            y = _realesrgan_x4(self.nets["M23"], x)
        return G.resize_to_e(rgb_t_to_bgr_u8(y)), {"fbcnn_pred_qf": round(float(1 - qf.item()) * 100, 1)}

    def settings(self):
        return {"order": ["FBCNN color (blind QF)", self.nafnet_key, "Real-ESRGAN x4plus"],
                "between_stages": "float32, clamp [0,1] (no 8-bit re-quantization)",
                "routing": "none — all stages always run", "to_e": "as general SR", "precision": "fp32"}


def build(config_id: str) -> Adapter:
    table = {
        "B0": B0Identity, "B1": B1Lanczos, "B2": lambda: B2Classical(False), "B2c": lambda: B2Classical(True),
        "A1": RestoreFormerPP, "A2": VQFRv1, "A3": RealESRGAN, "A4": SwinIR, "A5": Chain,
        "A6": GFPGANv14, "A7": lambda: CodeFormerW("A7", 0.5), "A8": lambda: CodeFormerW("A8", 1.0),
    }
    return table[config_id]()
