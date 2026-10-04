"""P4-T7 — selective model routing: quality analysis → frozen T6 gate → ORIGINAL or a degradation-specific route.

Contract (ORIGINAL = B0, the input returned untouched, is always valid and is the fallback for everything):
  1. features  = fes_bench.quality.features(input)                      (reference-free)
  2. decision  = fes_bench.gate.decide(features, frozen T6 rules)
       UNSAFE_OR_UNCERTAIN → ORIGINAL + escalate flag          DO_NOT_ENHANCE → ORIGINAL
       ENHANCE_CANDIDATE   → 3.
  3. bucket    = degradation bucket from features (thresholds = clean DEV reference quantiles; no outcome data)
     route     = frozen table bucket → candidate | ORIGINAL (chosen on DEV, see `fit_table`)
  4. run the routed candidate with guards; any failure → ORIGINAL with a recorded fallback reason:
       MODEL_FAILURE (exception) · TIMEOUT (> timeout_s) · RESOURCE_FAILURE (MemoryError / allocator OOM) ·
       INVALID_OUTPUT (wrong shape/dtype, non-finite, near-constant image std < 1, or no face found on the output by
       the FES detection contract — catches colour-noise collapse such as Phase 3's A5 dark-input failure)
Every call returns a decision record (the decision log). Candidates are only the five already benchmarked.
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import time
from dataclasses import dataclass, field

import numpy as np

from fes_bench import gate as GT

ROUTING_VERSION = "p4-routing-1.0.0"
ORIGINAL = "B0"
CANDIDATES = ("A1", "A2", "A3", "A4", "A5")
# tie-break / cost order: CPU warm latency at 224 px, P3-T8 (A1 5.5 s < A3 12.6 < A5 12.7 < A2 20.1 < A4 49.4)
COST_ORDER = ("A1", "A3", "A5", "A2", "A4")
BUCKETS = ("PHOTOMETRIC", "MIXED", "LOWRES", "NOISE", "BLUR", "COMPRESS", "NONE")
MIN_BUCKET_N = 10          # a route needs ≥ 10 gated DEV pairs in its bucket, else ORIGINAL
CLEAN_Q_LO, CLEAN_Q_HI = 0.02, 0.98


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def bucket_thresholds(clean_feats: list[dict]) -> dict:
    """Thresholds = quantiles of CLEAN dev references (what 'not degraded' looks like). No outcome data used."""
    q = lambda k, p: float(np.nanquantile(np.array([f.get(k, np.nan) for f in clean_feats], float), p))  # noqa: E731
    return {"iod_px_lo": q("iod_px", CLEAN_Q_LO), "noise_sigma_hi": q("noise_sigma", CLEAN_Q_HI),
            "lap_norm_c_lo": q("lap_norm_c", CLEAN_Q_LO), "blockiness_hi": q("blockiness", CLEAN_Q_HI),
            "luma_mean_lo": q("luma_mean", CLEAN_Q_LO), "rms_contrast_lo": q("rms_contrast", CLEAN_Q_LO)}


def flags(f: dict, t: dict) -> dict[str, bool]:
    g = lambda k: f.get(k)  # noqa: E731
    lowres = not _nan(g("iod_px")) and g("iod_px") < t["iod_px_lo"]
    return {"LOWRES": lowres,
            # small native inputs inflate the noise estimate (P4-T2 §3) → noise only counts at full resolution
            "NOISE": not lowres and not _nan(g("noise_sigma")) and g("noise_sigma") > t["noise_sigma_hi"],
            "BLUR": not lowres and not _nan(g("lap_norm_c")) and g("lap_norm_c") < t["lap_norm_c_lo"],
            "COMPRESS": not _nan(g("blockiness")) and g("blockiness") > t["blockiness_hi"],
            "PHOTOMETRIC": (not _nan(g("luma_mean")) and g("luma_mean") < t["luma_mean_lo"]) or
                           (not _nan(g("rms_contrast")) and g("rms_contrast") < t["rms_contrast_lo"])}


def bucket_of(f: dict, t: dict) -> str:
    fl = flags(f, t)
    if fl["PHOTOMETRIC"]:
        return "PHOTOMETRIC"
    structural = [k for k in ("LOWRES", "NOISE", "BLUR", "COMPRESS") if fl[k]]
    if len(structural) >= 2:
        return "MIXED"
    return structural[0] if structural else "NONE"


def fit_table(buckets: list[str], verdicts: dict[str, list[str]]) -> tuple[dict[str, str], dict]:
    """DEV pairs already gated ENHANCE_CANDIDATE. Per bucket: candidate with the largest
    U = #CONFIRMED_BENEFIT − LAMBDA·#HARM, only if U > 0 and n ≥ MIN_BUCKET_N; ties → cheaper; else ORIGINAL."""
    table, detail = {}, {}
    for b in BUCKETS:
        idx = [i for i, x in enumerate(buckets) if x == b]
        us = {c: float(sum(verdicts[c][i] in GT.BENEFIT for i in idx) - GT.LAMBDA * sum(verdicts[c][i] in GT.HARM for i in idx))
              for c in verdicts}
        best = sorted(us, key=lambda c: (-us[c], COST_ORDER.index(c)))[0] if us else None
        table[b] = best if (best and len(idx) >= MIN_BUCKET_N and us[best] > 0) else ORIGINAL
        detail[b] = {"n": len(idx), "utility": us}
    return table, detail


# ------------------------------------------------------------------ runtime router (fallbacks)

@dataclass
class RouteRecord:
    decision: str
    reason: str
    bucket: str = ""
    route: str = ORIGINAL
    used: str = ORIGINAL
    fallback: str = ""
    error: str = ""
    latency_s: float = 0.0
    escalate: bool = False
    extra: dict = field(default_factory=dict)


def _is_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return isinstance(e, MemoryError) or any(k in s for k in ("out of memory", "defaultcpuallocator", "not enough memory",
                                                              "cannot allocate memory"))


class Router:
    """`adapters`: config id → callable(img, ctx) -> (out uint8 H×W×3, meta). `features_fn(img)` → feature dict.
    `face_check(img)` → True if the FES detection contract finds a face on the output."""

    def __init__(self, gate_rules, thresholds: dict, table: dict, adapters: dict, features_fn, face_check,
                 timeout_s: float = 120.0, out_shape=(224, 224, 3)):
        self.rules, self.t, self.table = gate_rules, thresholds, table
        self.adapters, self.features_fn, self.face_check = adapters, features_fn, face_check
        self.timeout_s, self.out_shape = timeout_s, out_shape
        self._pool = cf.ThreadPoolExecutor(1)

    def plan(self, f: dict) -> RouteRecord:
        d, why = GT.decide(f, self.rules)
        rec = RouteRecord(decision=d, reason=why, escalate=d == GT.UNSAFE)
        if d == GT.ENH:
            rec.bucket = bucket_of(f, self.t)
            rec.route = self.table.get(rec.bucket, ORIGINAL)
        return rec

    def valid(self, out) -> str:
        if not isinstance(out, np.ndarray) or out.dtype != np.uint8 or out.shape != self.out_shape:
            return f"shape/dtype {getattr(out, 'shape', None)} {getattr(out, 'dtype', None)}"
        if float(out.std()) < 1.0:
            return "near-constant image"
        if not self.face_check(out):
            return "no face detected on output"
        return ""

    def route(self, img: np.ndarray, ctx: dict, original_e: np.ndarray) -> tuple[np.ndarray, RouteRecord]:
        """`original_e` = the ORIGINAL in the output frame (B0 rule). Returns (output, record)."""
        t0 = time.perf_counter()
        rec = self.plan(self.features_fn(img))
        if rec.route == ORIGINAL:
            rec.latency_s = time.perf_counter() - t0
            return original_e, rec
        fut = self._pool.submit(self.adapters[rec.route], img, ctx)
        try:
            out, _ = fut.result(timeout=self.timeout_s)
            bad = self.valid(out)
            if bad:
                rec.fallback, rec.error = "INVALID_OUTPUT", bad
            else:
                rec.used = rec.route
        except cf.TimeoutError:
            rec.fallback = "TIMEOUT"          # thread cannot be killed in-process: production must isolate (Phase 5)
            self._pool = cf.ThreadPoolExecutor(1)
        except BaseException as e:  # noqa: BLE001 — every failure must end in a defined status
            rec.fallback = "RESOURCE_FAILURE" if _is_oom(e) else "MODEL_FAILURE"
            rec.error = f"{type(e).__name__}: {e}"[:300]
        rec.latency_s = time.perf_counter() - t0
        return (out if rec.used != ORIGINAL else original_e), rec


def load_routing(cfg: dict, t6_gate: dict) -> tuple[dict, dict]:
    """Verify the T7 seal and that it was built on THIS frozen T6 gate. → (thresholds, table)."""
    assert cfg["sha256"] == GT.gate_hash(cfg), "routing file modified after freeze"
    assert cfg["t6_gate_seal"] == t6_gate["sha256"], "routing was calibrated on a different T6 gate"
    assert set(cfg["table"].values()) <= set(CANDIDATES) | {ORIGINAL}
    return cfg["bucket_thresholds"], cfg["table"]
