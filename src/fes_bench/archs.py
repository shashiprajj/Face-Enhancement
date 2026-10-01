"""Load the vendored upstream network definitions (third_party/, byte-identical to the pinned commits).

The upstream files import training-time packages that are not in the pinned image (timm, einops, sklearn,
the repos' own `utils`/registry/`ops` packages). Instead of editing the vendored files, this module installs
minimal import shims in `sys.modules` and loads each file under its upstream module name. The shims only
provide what the *inference* code paths use; anything training-only raises if it is ever reached.

Weights are loaded exclusively with `torch.load(weights_only=True)` (P1-T9 §1, P0-T5 §8).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

from fes_data.common import REPO_ROOT, sha256_file

THIRD_PARTY = REPO_ROOT / "third_party"

# Upstream file → SHA-256 recorded when vendored (third_party/PROVENANCE.md). Checked before loading.
VENDORED_SHA256 = {
    "restoreformerpp/RestoreFormer/modules/vqvae/vqvae_arch.py": "ba4a1275d938a416ea5ce848d7c5418d34cc82ca8d6f1dfb1998a8ed225aa138",
    "vqfr/vqfr/archs/quantizer_arch.py": "f85f34dedb8a0275699d2d2bcc6c574bd26a249ab6bb64b312bb5a83468b0239",
    "vqfr/vqfr/archs/vqganv1_arch.py": "f705db4367fb1c943178c12d7ae25c1176896f170cbf7b472d87f07cc03cf5fb",
    "vqfr/vqfr/archs/vqfrv1_arch.py": "63a1c50094abe51dcd354f8706dee4884500c39123620bb776e905f528e8787a",
    "vqfr/vqfr/ops/dcn/deform_conv.py": "ff5dc4db2e8d1053a493c2ae0a7a3cb829b573403ced63d46bd6ea14f6a45382",
    "basicsr/basicsr/archs/arch_util.py": "be2cec6482fba5455cc1f03ff40da84a89e27feadf886018a4fa7b97b8a1bb55",
    "basicsr/basicsr/archs/rrdbnet_arch.py": "6aa8eb3f77e918b90eeff4ebfaf8745abe0c40eb8fcc03abbc1928693d70ab58",
    "swinir/models/network_swinir.py": "9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913",
    "fbcnn/models/network_fbcnn.py": "e1afc1f6483d57b774365467393216508f29a708b576dc973bad41bae67e8d03",
    "nafnet/basicsr/models/archs/arch_util.py": "5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005",
    "nafnet/basicsr/models/archs/local_arch.py": "c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26",
    "nafnet/basicsr/models/archs/NAFNet_arch.py": "01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967",
}


# ---------------------------------------------------------------- shims

def _training_only(*_a, **_k):
    raise RuntimeError("training-only code path reached in an inference shim")


def _package(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as package
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        sys.modules[name] = mod
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(_package(parent), child, mod)
    return mod


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)  # torch._dynamo calls find_spec on sys.modules
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(_package(parent), child, mod)
    return mod


class _DropPath(nn.Module):
    """timm DropPath; identity at inference (drop_path_rate is 0 for every model here)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        _training_only()


def _to_2tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)


def _rearrange(t: torch.Tensor, pattern: str, **_kw) -> torch.Tensor:
    """The only einops patterns on the VQFR v1 inference path."""
    p = pattern.replace(" ", "")
    if p == "bchw->bhwc":
        return t.permute(0, 2, 3, 1)
    if p == "bhwc->bchw":
        return t.permute(0, 3, 1, 2)
    raise NotImplementedError(f"einops shim: unsupported pattern {pattern!r}")


class _NoDCN(nn.Module):
    """Base for BasicSR's DCNv2Pack, which RRDBNet never instantiates."""

    def __init__(self, *_a, **_k):
        _training_only()


class _Registry:
    def __init__(self, name: str):
        self._name, self._obj = name, {}

    def register(self, obj=None):
        def deco(o):
            self._obj[o.__name__] = o
            return o
        return deco(obj) if obj is not None else deco

    def get(self, name):
        return self._obj[name]


def _install_shims() -> None:
    if "timm.models.layers" not in sys.modules:
        _module("timm.models.layers", DropPath=_DropPath, to_2tuple=_to_2tuple,
                trunc_normal_=nn.init.trunc_normal_)
    if "einops" not in sys.modules:
        _module("einops", rearrange=_rearrange)
    if "sklearn.cluster" not in sys.modules:
        _module("sklearn.cluster", k_means=_training_only)
    log = lambda *a, **k: logging.getLogger("fes_bench.vendored")  # noqa: E731
    # VQFR package skeleton (upstream vqfr/__init__.py imports data/models/training code; not vendored)
    _package("vqfr.archs").__path__ = [str(THIRD_PARTY / "vqfr/vqfr/archs")]
    _package("vqfr.ops.dcn").__path__ = [str(THIRD_PARTY / "vqfr/vqfr/ops/dcn")]
    _module("vqfr.utils", get_root_logger=log, all_gather=_training_only)
    _module("vqfr.utils.registry", ARCH_REGISTRY=_Registry("arch"), QUANTIZER_REGISTRY=_Registry("quantizer"))
    # BasicSR skeleton (RRDBNet) and NAFNet's own basicsr fork share the top-level name; their submodules differ.
    _module("basicsr.utils", get_root_logger=log)
    _module("basicsr.utils.registry", ARCH_REGISTRY=_Registry("arch"))
    _module("basicsr.ops.dcn", ModulatedDeformConvPack=_NoDCN, modulated_deform_conv=_training_only)
    _package("basicsr.archs")
    _package("basicsr.models.archs")


def _load(mod_name: str, rel: str) -> types.ModuleType:
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = THIRD_PARTY / rel
    digest = sha256_file(path)
    if digest != VENDORED_SHA256[rel]:
        raise RuntimeError(f"vendored file modified: {rel} ({digest})")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    parent, child = mod_name.rsplit(".", 1)
    setattr(_package(parent), child, mod)
    spec.loader.exec_module(mod)
    return mod


def _vqfr_modules():
    _install_shims()
    dcn = _load("vqfr.ops.dcn.deform_conv", "vqfr/vqfr/ops/dcn/deform_conv.py")
    pkg = sys.modules["vqfr.ops.dcn"]
    pkg.ModulatedDeformConvPack = dcn.ModulatedDeformConvPack
    pkg.modulated_deform_conv = dcn.modulated_deform_conv
    _load("vqfr.archs.quantizer_arch", "vqfr/vqfr/archs/quantizer_arch.py")
    _load("vqfr.archs.vqganv1_arch", "vqfr/vqfr/archs/vqganv1_arch.py")
    return _load("vqfr.archs.vqfrv1_arch", "vqfr/vqfr/archs/vqfrv1_arch.py")


# ---------------------------------------------------------------- builders (upstream inference configurations)

def build_restoreformerpp() -> nn.Module:
    _install_shims()
    m = _load("RestoreFormer.modules.vqvae.vqvae_arch", "restoreformerpp/RestoreFormer/modules/vqvae/vqvae_arch.py")
    # RestoreFormer/RestoreFormer.py @59d250f: arch 'RestoreFormer++'
    return m.VQVAEGANMultiHeadTransformer(head_size=4, ex_multi_scale_num=1)


def build_vqfr_v1() -> nn.Module:
    m = _vqfr_modules()
    # options/test/VQFR/test_vqfr_v1.yml @239c77a (demo_util.py names 'L2VectorQuantizer', which does not accept
    # these arguments; the official test config's 'L2VectorQuantizerKmeans' is used)
    levels = {f"Level_{2 ** i}": {"cond_channels": 32, "cond_downscale_rate": 2 ** i, "deformable_groups": 4}
              for i in range(5, -1, -1)}
    return m.VQFRv1(
        base_channels=128, proj_patch_size=32, resolution_scale_rates=[1, 2, 2, 2, 2, 2],
        channel_multipliers=[1, 1, 2, 2, 2, 4], encoder_num_blocks=2, decoder_num_blocks=3,
        quant_level=["Level_32"], fix_keys=["embedding"], inpfeat_extraction_opt={"in_dim": 3, "out_dim": 32},
        align_from_patch=32, align_opt=levels,
        quantizer_opt={"Level_32": {"type": "L2VectorQuantizerKmeans", "in_dim": 512, "num_code": 1024,
                                    "code_dim": 256, "reservoir_size": 16384, "reestimate_iters": 2000,
                                    "reestimate_maxiters": -1, "warmup_iters": -1}})


def build_rrdbnet_x4() -> nn.Module:
    _install_shims()
    _load("basicsr.archs.arch_util", "basicsr/basicsr/archs/arch_util.py")
    m = _load("basicsr.archs.rrdbnet_arch", "basicsr/basicsr/archs/rrdbnet_arch.py")
    # inference_realesrgan.py @a4abfb2: RealESRGAN_x4plus
    return m.RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)


def build_swinir_realsr_l_x4() -> nn.Module:
    _install_shims()
    m = _load("swinir_models.network_swinir", "swinir/models/network_swinir.py")
    # main_test_swinir.py @6545850: task real_sr, --large_model, scale 4
    return m.SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
                    depths=[6] * 9, embed_dim=240, num_heads=[8] * 9, mlp_ratio=2,
                    upsampler="nearest+conv", resi_connection="3conv")


def build_fbcnn_color() -> nn.Module:
    m = _load("fbcnn_models.network_fbcnn", "fbcnn/models/network_fbcnn.py")
    # main_test_fbcnn_color_real.py @54d1831: n_channels=3, nc=[64,128,256,512], nb=4
    return m.FBCNN(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="R")


def _nafnet_modules():
    _install_shims()
    _load("basicsr.models.archs.arch_util", "nafnet/basicsr/models/archs/arch_util.py")
    _load("basicsr.models.archs.local_arch", "nafnet/basicsr/models/archs/local_arch.py")
    return _load("basicsr.models.archs.NAFNet_arch", "nafnet/basicsr/models/archs/NAFNet_arch.py")


def build_nafnet_sidd() -> nn.Module:
    # options/test/SIDD/NAFNet-width64.yml @2b4af71
    return _nafnet_modules().NAFNet(img_channel=3, width=64, middle_blk_num=12,
                                    enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])


def build_nafnet_gopro() -> nn.Module:
    # options/test/GoPro/NAFNet-width64.yml @2b4af71: NAFNetLocal (TLC, train_size 1x3x256x256)
    return _nafnet_modules().NAFNetLocal(img_channel=3, width=64, middle_blk_num=1,
                                         enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])


# ---------------------------------------------------------------- weights

def staged_copy(path: str | Path, expect_sha256: str | None = None, chunk: int = 4 << 20) -> Path:
    """Stream `path` to container-local temp storage in small chunks (hashing on the way) and return the copy.

    torch.load issues one read() per tensor record (up to ~150 MB). Through the Windows→Docker bind mount those
    large reads fail with ENOMEM when the Docker VM is under memory pressure (P3-T2 F5); small reads do not.
    Loading from a local copy is byte-identical: the SHA-256 is checked during the copy.
    """
    import hashlib
    import os
    import tempfile
    h = hashlib.sha256()
    fd, tmp = tempfile.mkstemp(suffix=Path(path).suffix, prefix="fesw_")
    with os.fdopen(fd, "wb") as dst:
        for attempt in range(6):
            try:
                with open(path, "rb") as src:
                    dst.seek(0)
                    dst.truncate()
                    h = hashlib.sha256()
                    while block := src.read(chunk):
                        dst.write(block)
                        h.update(block)
                break
            except OSError:
                if attempt == 5:
                    raise
                import time
                time.sleep(2 * (attempt + 1))
    if expect_sha256 and h.hexdigest() != expect_sha256:
        os.unlink(tmp)
        raise RuntimeError(f"checksum mismatch while staging {path}: {h.hexdigest()}")
    return Path(tmp)


def load_weights(net: nn.Module, path: str | Path, key: str | None, strip_prefix: str | None = None,
                 strict: bool = True, expect_sha256: str | None = None) -> dict:
    """Safe-load a checkpoint and report exactly which keys matched (no silent partial loads)."""
    local = staged_copy(path, expect_sha256)
    try:
        ckpt = torch.load(local, map_location="cpu", weights_only=True)
    finally:
        local.unlink(missing_ok=True)
    sd = ckpt[key] if key else ckpt
    if strip_prefix:
        sd = {(k[len(strip_prefix):] if k.startswith(strip_prefix) else k): v for k, v in sd.items()}
    res = net.load_state_dict(sd, strict=strict)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in net.parameters())
    return {"missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys),
            "n_checkpoint_tensors": len(sd), "n_model_params": int(n_params)}
