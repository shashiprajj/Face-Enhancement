"""R6-FT — fine-tune NAFNet-GoPro width-64 (MIT) on FACE blur, synthesised on FairFace (CC BY 4.0).

Why: NAFNet-GoPro was trained only on camera-shake (motion) blur from street video; on out-of-focus / soft face blur
it changes little (R6-0 §19). Same architecture, same weights format: the result drops into the service unchanged.

Runs on Kaggle (GPU, kernel "fes-nafnet-face-ft") or locally (CPU smoke test):
    python train_nafnet_face.py --data DIR --out DIR [--hours 8.5] [--iters 40000] [--smoke]
DIR must contain fairface-img-margin025-trainval.zip and NAFNet-GoPro-width64.pth (SHA-256 checked).

Training pairs (on the fly, seeded per worker): FairFace train face (224 px, margin 0.25) → random scale (the service
sees 96–224 px chips) → random 128 crop = SHARP target; input = the same crop degraded by ONE blur family
  gaussian σ 0.8–4.0 (isotropic / anisotropic) · defocus disk r 1.5–6 · motion line 5–25 px any angle · none (10 %)
then optionally down-up resampling (×1–2.5), mild Gaussian noise σ ≤ 4/255, JPEG q 60–95; 5 % grayscale; random
horizontal flip. Validation: 300 FairFace VAL faces with fixed seeded degradations (never trained on).
Loss: Charbonnier + 0.05 · L1 of the FFT amplitude (keeps edges sharp). AdamW 1e-4 → cosine; EMA 0.999.
Output: nafnet_face_ft.pth ({"params": EMA weights}, same key layout as the original checkpoint), log.json,
val_sheet.png. Fine-tuned weights are a derivative of MIT weights; FairFace attribution: Karkkainen & Joo, WACV 2021.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ZIP = "fairface-img-margin025-trainval.zip"
WEIGHTS = "NAFNet-GoPro-width64.pth"
WEIGHTS_SHA = "329d3ab4077b8d6b"           # prefix of the pinned registry SHA-256 (asset A06)
CROP = 128


# ---------------------------------------------------------------- NAFNet (megvii-research/NAFNet @2b4af71, MIT)
# Plain NAFNet; LayerNorm2d written with standard ops (same maths as the custom autograd function, AMP-friendly).
# The service wraps the same weights in NAFNetLocal (pool replacement only; no extra parameters).

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        x32 = x.float()
        mu = x32.mean(1, keepdim=True)
        var = (x32 - mu).pow(2).mean(1, keepdim=True)
        y = (x32 - mu) / (var + self.eps).sqrt()
        return (self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)).to(x.dtype)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c * 2
        self.conv1 = nn.Conv2d(c, d, 1)
        self.conv2 = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.conv3 = nn.Conv2d(d // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(c, 2 * c, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width=64, middle_blk_num=1, enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.decoders, self.ups, self.downs = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        _, _, h, w = inp.shape
        p = self.padder_size
        x0 = F.pad(inp, (0, (p - w % p) % p, 0, (p - h % p) % p))
        x = self.intro(x0)
        encs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = dec(up(x) + skip)
        return (self.ending(x) + x0)[:, :, :h, :w]


# ---------------------------------------------------------------- degradations

def _gauss_kernel(rng):
    s1 = rng.uniform(0.8, 4.0)
    s2 = s1 if rng.random() < 0.6 else rng.uniform(0.8, 4.0)
    th = rng.uniform(0, math.pi)
    r = int(math.ceil(3 * max(s1, s2)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    xr, yr = xx * math.cos(th) + yy * math.sin(th), -xx * math.sin(th) + yy * math.cos(th)
    k = np.exp(-0.5 * (xr ** 2 / s1 ** 2 + yr ** 2 / s2 ** 2))
    return k / k.sum()


def _disk_kernel(rng):
    rad = rng.uniform(1.5, 6.0)
    r = int(math.ceil(rad)) + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    k = np.clip(rad + 0.5 - np.hypot(xx, yy), 0, 1)              # anti-aliased disk
    k = cv2.GaussianBlur(k, (0, 0), 0.5)
    return k / k.sum()


def _motion_kernel(rng):
    ln = rng.uniform(5, 25)
    th = rng.uniform(0, math.pi)
    r = int(math.ceil(ln / 2)) + 1
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    for t in np.linspace(-ln / 2, ln / 2, int(ln * 3)):
        x, y = r + t * math.cos(th), r + t * math.sin(th)
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= y0 + dy < k.shape[0] and 0 <= x0 + dx < k.shape[1]:
                k[y0 + dy, x0 + dx] += wgt
    return k / k.sum()


def degrade(sharp_full, rng):
    """sharp_full: float32 RGB [0,1] (already scaled). Returns the degraded image (same size)."""
    u = rng.random()
    kind = "none" if u < 0.10 else "gauss" if u < 0.45 else "disk" if u < 0.70 else "motion"
    x = sharp_full
    if kind != "none":
        k = {"gauss": _gauss_kernel, "disk": _disk_kernel, "motion": _motion_kernel}[kind](rng)
        x = cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.3:                                         # resolution loss
        f = rng.uniform(1.0, 2.5)
        h, w = x.shape[:2]
        x = cv2.resize(cv2.resize(x, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA),
                       (w, h), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.5:
        x = x + rng.normal(0, rng.uniform(0, 4.0) / 255.0, x.shape).astype(np.float32)
    x = np.clip(x, 0, 1)
    if rng.random() < 0.4:
        ok, enc = cv2.imencode(".jpg", (x[..., ::-1] * 255 + 0.5).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    return x, kind


def make_pair(img_rgb_u8, rng):
    side = int(rng.integers(CROP, 201))                           # chip scale variety (service: 96–224 px)
    im = cv2.resize(img_rgb_u8, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if rng.random() < 0.05:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = np.repeat(g[..., None], 3, -1)
    if rng.random() < 0.5:
        im = im[:, ::-1].copy()
    deg, kind = degrade(im, rng)
    y0, x0 = (int(rng.integers(0, side - CROP + 1)) for _ in range(2))
    sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    return deg[sl], im[sl], kind


class FaceBlur(torch.utils.data.Dataset):
    def __init__(self, zip_path, names, seed, fixed=False):
        self.zip_path, self.names, self.seed, self.fixed, self.z = zip_path, names, seed, fixed, None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if os.path.isdir(self.zip_path):                 # Kaggle unpacks uploaded zips into a folder
            buf = np.fromfile(os.path.join(self.zip_path, self.names[i]), np.uint8)
        else:
            if self.z is None:
                self.z = zipfile.ZipFile(self.zip_path)
            buf = np.frombuffer(self.z.read(self.names[i]), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
        rng = np.random.default_rng(self.seed * 1_000_003 + i if self.fixed else None)
        d, s, kind = make_pair(img, rng)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))  # noqa: E731
        return t(d), t(s), kind


def _worker_init(wid):
    seed = (torch.initial_seed() + wid) % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------- training

def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps ** 2).mean()


def fft_loss(a, b):
    return (torch.fft.rfft2(a.float()).abs() - torch.fft.rfft2(b.float()).abs()).abs().mean()


def psnr(a, b):
    mse = ((a.clamp(0, 1) - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1 / mse)).mean().item()


def load_pretrained(net, path):
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert sha.startswith(WEIGHTS_SHA), f"unexpected weights {sha}"
    sd = torch.load(path, map_location="cpu", weights_only=True)["params"]
    net.load_state_dict(sd, strict=True)
    return sha


def main(a):
    t_start = time.time()
    torch.manual_seed(0)
    random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    zp = os.path.join(a.data, ZIP)
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
    else:                                                  # unpacked: <data>/<zip stem>/{train,val}/*.jpg
        zp = os.path.join(a.data, ZIP[:-4])
        names = sorted(f"{sp}/{f}" for sp in ("train", "val") for f in os.listdir(os.path.join(zp, sp))
                       if f.lower().endswith(".jpg"))
    train = [n for n in names if n.startswith("train/")]
    val = [n for n in names if n.startswith("val/")][::max(1, len([n for n in names if n.startswith("val/")]) // 300)][:300]
    if a.smoke:
        train, val = train[:64], val[:8]
    print(f"device {dev} · train {len(train)} · val {len(val)}", flush=True)

    net = NAFNet().to(dev)
    sha = load_pretrained(net, os.path.join(a.data, WEIGHTS))
    ema = NAFNet().to(dev)
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)

    dl = torch.utils.data.DataLoader(FaceBlur(zp, train, 1), batch_size=a.bs, shuffle=True, drop_last=True,
                                     num_workers=a.workers, worker_init_fn=_worker_init, persistent_workers=a.workers > 0)
    vdl = torch.utils.data.DataLoader(FaceBlur(zp, val, 7, fixed=True), batch_size=16, num_workers=a.workers)
    vset = [(d, s, k) for d, s, k in vdl]                          # fixed validation pairs, built once

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.9), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters, eta_min=1e-7)
    # fp32 only: pretrained NAFNet-GoPro overflows in fp16 (Kaggle run 1: eval PSNR NaN under autocast)
    use_amp = False
    ngpu = torch.cuda.device_count() if dev.type == "cuda" else 0
    fwd = nn.DataParallel(net) if ngpu > 1 else net              # both T4s for the training forward/backward
    print(f"gpus {ngpu}", flush=True)

    def evaluate(model):
        model.eval()
        tot, n, by = 0.0, 0, {}
        base = 0.0
        with torch.no_grad():
            for d, s, kinds in vset:
                d, s = d.to(dev), s.to(dev)
                with torch.autocast(dev.type, enabled=use_amp):
                    o = model(d)
                o = torch.nan_to_num(o.float(), nan=0.0, posinf=1.0, neginf=0.0)
                for j, k in enumerate(kinds):
                    pj, bj = psnr(o[j:j + 1], s[j:j + 1]), psnr(d[j:j + 1], s[j:j + 1])
                    by.setdefault(k, []).append(pj - bj)
                    tot, base, n = tot + pj, base + bj, n + 1
        model.train()
        return {"psnr": tot / n, "psnr_input": base / n, "gain_by_kind": {k: float(np.mean(v)) for k, v in by.items()}}

    log = {"weights_sha256": sha, "args": vars(a), "evals": []}
    e0 = evaluate(ema)
    e0["iter"] = 0
    log["evals"].append(e0)
    print("eval@0 (original NAFNet-GoPro)", json.dumps(e0), flush=True)
    best = -1e9
    it, t0 = 0, time.time()
    net.train()
    while it < a.iters:
        for d, s, _ in dl:
            d, s = d.to(dev, non_blocking=True), s.to(dev, non_blocking=True)
            o = fwd(d)
            loss = charbonnier(o, s) + 0.05 * fft_loss(o, s)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at iter {it}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(a.ema).add_(pn.detach(), alpha=1 - a.ema)
            it += 1
            if it % 100 == 0:
                print(f"it {it} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / it:.3f}s/it", flush=True)
            done = it >= a.iters or (time.time() - t_start) > a.hours * 3600
            if it % a.eval_every == 0 or done:
                e = evaluate(ema)
                e["iter"] = it
                log["evals"].append(e)
                print(f"eval@{it}", json.dumps(e), flush=True)
                if math.isfinite(e["psnr"]) and e["psnr"] > best:
                    best = e["psnr"]
                    torch.save({"params": {k: v.detach().cpu() for k, v in ema.state_dict().items()}},
                               os.path.join(a.out, "nafnet_face_ft.pth"))
                    log["best"] = e
                with open(os.path.join(a.out, "log.json"), "w") as fh:
                    json.dump(log, fh, indent=1)
            if done:
                break
        if (time.time() - t_start) > a.hours * 3600:
            break

    # visual sheet: input | original NAFNet | fine-tuned | sharp
    orig = NAFNet().to(dev)
    load_pretrained(orig, os.path.join(a.data, WEIGHTS))
    orig.eval()
    best_net = NAFNet().to(dev)
    best_net.load_state_dict(torch.load(os.path.join(a.out, "nafnet_face_ft.pth"), weights_only=True)["params"])
    best_net.eval()
    rows = []
    with torch.no_grad():
        d, s, _ = vset[0]
        d = d[:8].to(dev)
        o1, o2 = orig(d), best_net(d)
        for j in range(d.shape[0]):
            row = [x[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy() for x in (d, o1, o2, s[:8].to(dev))]
            rows.append(np.hstack(row))
    sheet = (np.vstack(rows)[..., ::-1] * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(a.out, "val_sheet.png"), cv2.resize(sheet, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    log["total_s"] = time.time() - t_start
    with open(os.path.join(a.out, "log.json"), "w") as fh:
        json.dump(log, fh, indent=1)
    print("DONE best", json.dumps(log.get("best")), flush=True)


def find_data():
    if os.path.isdir("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if WEIGHTS in files and (ZIP in files or os.path.isdir(os.path.join(root, ZIP[:-4]))):
                return root
    return "."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "out_ft")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--hours", type=float, default=8.5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.data = a.data or find_data()
    if a.smoke:
        a.iters, a.bs, a.eval_every, a.workers = 3, 2, 3, 0
    main(a)
