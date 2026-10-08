"""R6-0 — post-enhancement verifier signals (deterministic, numpy/OpenCV only; research code, not used by the service).

Six families, each computed from (input chip, enhanced candidate). Family F also uses the same NAFNet run on the
flipped input:
  G geometry           YuNet on input vs output: detector score, bbox IoU, landmark displacement / IOD
  P photometric        channel-mean shift, luminance shift, global chroma delta (p99 / p99.9), 8x8 local chroma
                       delta max, colour invented on a near-grey input, fraction of extreme pixel changes
  T texture            HF luma energy ratio, HF colour ratio (+ growth), dottedness (colour / luma),
                       residual spectral peak
  L local diff         |out - in|: changed-pixel %, largest connected changed region, top-2 % window
                       concentration, share of change energy in the 8-px chip border, 8x8 max of the change
  I identity           SFace / dlib cos(input, output), computed in Docker by identity.py (not here)
  F flip consistency   disagreement between NAFNet(x) and flip^-1(NAFNet(flip(x))) for the h / v / hv flips.
                       A real deblur is (nearly) equivariant to mirroring; padding-, position- and
                       activation-driven artifacts are not. Reports the 8x8 max and p99.9 of the per-pixel max
                       disagreement (Lab), for luma and chroma separately.
"""
from __future__ import annotations

import cv2
import numpy as np

WIN = 8


def lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def box(x, w=WIN):
    return cv2.blur(x.astype(np.float32), (w, w), borderType=cv2.BORDER_REFLECT_101)


def iso2(x):
    k = np.array([[1.0, -2.0, 1.0]], np.float32)
    x = x.astype(np.float32)
    return np.minimum(cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT_101) ** 2,
                      cv2.filter2D(x, -1, k.T, borderType=cv2.BORDER_REFLECT_101) ** 2)


def hp(x, s=1.0):
    return x - cv2.GaussianBlur(x, (0, 0), s)


def geometry(inp, out, det):
    a, b = det.detect(inp), det.detect(out)
    f = {"g_score_in": a[0].score if a else 0.0, "g_score_out": b[0].score if b else 0.0, "g_face_out": bool(b)}
    f["g_score_drop"] = f["g_score_in"] - f["g_score_out"]
    if a and b:
        la, lb = np.asarray(a[0].landmarks), np.asarray(b[0].landmarks)
        iod = max(np.linalg.norm(la[1] - la[0]), 1.0)
        f["g_lm_disp"] = float(np.linalg.norm(la - lb, axis=1).max() / iod)
        x1, y1, w1, h1 = a[0].bbox_xywh
        x2, y2, w2, h2 = b[0].bbox_xywh
        iw = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
        ih = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
        f["g_bbox_iou"] = float(iw * ih / (w1 * h1 + w2 * h2 - iw * ih))
    else:
        f["g_lm_disp"], f["g_bbox_iou"] = 9.0, 0.0
    return f


def photometric(inp, out, valid):
    A, B = lab(out), lab(inp)
    o = out.reshape(-1, 3)[valid.ravel()].astype(np.float32)
    i = inp.reshape(-1, 3)[valid.ravel()].astype(np.float32)
    dc = np.hypot(A[..., 1] - B[..., 1], A[..., 2] - B[..., 2])
    dcv = dc[valid]
    gray_in = float(np.percentile(np.hypot(B[..., 1] - 128, B[..., 2] - 128)[valid], 99))
    gray_out = float(np.percentile(np.hypot(A[..., 1] - 128, A[..., 2] - 128)[valid], 99))
    return {"p_chan_shift": float(np.abs(o.mean(0) - i.mean(0)).max()),
            "p_luma_shift": float(abs(A[..., 0][valid].mean() - B[..., 0][valid].mean())),
            "p_chroma_p99": float(np.percentile(dcv, 99)), "p_chroma_p999": float(np.percentile(dcv, 99.9)),
            "p_chroma_local_max": float(box(dc * valid).max()),
            "p_gray_in_c99": gray_in, "p_colour_invented": max(0.0, gray_out - gray_in),
            "p_extreme_frac": float((np.abs(o - i) > 64).any(1).mean())}


def texture(inp, out, valid):
    A, B = lab(out), lab(inp)
    ha, hb = hp(A, 1.5), hp(B, 1.5)
    el_o, el_i = float((ha[..., 0] ** 2)[valid].mean()), float((hb[..., 0] ** 2)[valid].mean())
    ec_o = float((ha[..., 1] ** 2 + ha[..., 2] ** 2)[valid].mean())
    ec_i = float((hb[..., 1] ** 2 + hb[..., 2] ** 2)[valid].mean())
    dot_c = np.maximum(iso2(A[..., 1]) + iso2(A[..., 2]) - iso2(B[..., 1]) - iso2(B[..., 2]), 0) * valid
    dot_l = np.maximum(iso2(A[..., 0]) - iso2(B[..., 0]), 0) * valid
    # spectral peak of the luma residual (out - in): a periodic grid / dot pattern is a sharp off-centre peak
    r = (A[..., 0] - B[..., 0]) * valid
    r = r - r.mean()
    S = np.abs(np.fft.fftshift(np.fft.fft2(r * np.outer(np.hanning(r.shape[0]), np.hanning(r.shape[1])))))
    cy, cx = S.shape[0] // 2, S.shape[1] // 2
    yy, xx = np.mgrid[:S.shape[0], :S.shape[1]]
    hf = np.hypot((yy - cy) / S.shape[0], (xx - cx) / S.shape[1]) > 0.12
    return {"t_hf_luma_ratio": el_o / (el_i + 1.0), "t_hf_colour_ratio": ec_o / (el_o + 1.0),
            "t_hf_colour_growth": (ec_o + 1.0) / (ec_i + 1.0),
            "t_dot_c_max": float(box(dot_c).max()), "t_dot_l_max": float(box(dot_l).max()),
            "t_spec_peak": float(S[hf].max() / (np.median(S[hf]) + 1e-6))}


def local_diff(inp, out, valid):
    D = np.abs(lab(out) - lab(inp))
    dl = D[..., 0] * valid
    dall = np.sqrt((D ** 2).sum(-1)) * valid
    n = valid.sum()
    f = {"l_changed_pct": float((dall > 8).sum() / n)}
    m = (dall > 16).astype(np.uint8)
    k, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    f["l_largest_cc"] = float(st[1:, cv2.CC_STAT_AREA].max() / n) if k > 1 else 0.0
    e = box(dall ** 2)[valid]
    srt = np.sort(e)[::-1]
    f["l_top2_conc"] = float(srt[: max(1, len(srt) // 50)].sum() / (srt.sum() + 1e-6))
    bd = np.zeros_like(valid)
    bd[:WIN], bd[-WIN:], bd[:, :WIN], bd[:, -WIN:] = True, True, True, True
    f["l_border_share"] = float((dall ** 2)[bd & valid].sum() / ((dall ** 2).sum() + 1e-6))
    f["l_win_max"] = float(box(dall).max())
    f["l_win_max_excess"] = float(box(dall).max() / (np.median(dall[valid]) + 1.0))
    f["l_luma_win_max"] = float(box(dl).max())
    return f


def flip_maps(v0, flips, valid):
    """Per-pixel max Lab disagreement between v0 and each flip run: (luma map, chroma map)."""
    A = lab(v0)
    dl = np.zeros(v0.shape[:2], np.float32)
    dc = np.zeros(v0.shape[:2], np.float32)
    for v in flips:
        B = lab(v)
        dl = np.maximum(dl, np.abs(A[..., 0] - B[..., 0]))
        dc = np.maximum(dc, np.hypot(A[..., 1] - B[..., 1], A[..., 2] - B[..., 2]))
    return dl * valid, dc * valid


def flip_consistency(v0, flips, valid, inp):
    dl, dc = flip_maps(v0, flips, valid)
    chg = np.sqrt(((lab(v0) - lab(inp)) ** 2).sum(-1)) * valid
    return {"f_luma_win_max": float(box(dl).max()), "f_chroma_win_max": float(box(dc).max()),
            "f_luma_p999": float(np.percentile(dl[valid], 99.9)),
            "f_chroma_p999": float(np.percentile(dc[valid], 99.9)),
            "f_dis_over_change": float(box(dl + dc).max() / (np.percentile(box(chg)[valid], 95) + 1.0)),
            "f_mean": float((dl + dc)[valid].mean())}


def all_signals(inp, out, flips, mask, det=None):
    valid = np.ones(inp.shape[:2], bool) if mask is None else ~mask
    f = {}
    if det is not None:
        f.update(geometry(inp, out, det))
    f.update(photometric(inp, out, valid))
    f.update(texture(inp, out, valid))
    f.update(local_diff(inp, out, valid))
    if flips:
        f.update(flip_consistency(out, flips, valid, inp))
    return f


# ---------------------------------------------------------------- mitigations (post-processing of the SAME model output)

def luma_only(inp, out):
    """Output luminance (Y) + input chrominance (Cr, Cb): the enhancement cannot change colour at all."""
    a, b = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb), cv2.cvtColor(inp, cv2.COLOR_BGR2YCrCb)
    a[..., 1:] = b[..., 1:]
    return cv2.cvtColor(a, cv2.COLOR_YCrCb2BGR)


def flip_mean(v0, flips):
    s = np.mean([v0.astype(np.float32)] + [f.astype(np.float32) for f in flips], 0)
    return np.clip(s + 0.5, 0, 255).astype(np.uint8)


def flip_median(v0, flips):
    return np.median(np.stack([v0] + list(flips)).astype(np.float32), 0).round().clip(0, 255).astype(np.uint8)
