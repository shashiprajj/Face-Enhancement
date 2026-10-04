"""R5-1 — standalone enhancement pipeline around the FROZEN Phase 4 components (no redesign, no new models).

    bytes → validation → face detection (FES contract, YuNet) → face chip → quality features (p4-quality-1.0.0)
          → T6 gate (frozen, seal b8ee2c43) → T7 routing (frozen, seal 2f026f22) → ORIGINAL | A4 (isolated process)
          → output validation (router guard) → result + metadata

Input modes
  face_chip   the Phase 4 contract: a square, face-centred chip (FairFace/dlib framing, side ≤ 224). Used as is.
  full_image  any photo. Detection uses the existing contract (largest face; `multiple_faces` warning). The face is
              cut out as a square chip in the same FairFace/dlib framing the gate was calibrated on
              (DLIB_CHIP_TEMPLATE, landmark similarity fit) at NATIVE resolution (no upsampling; downscaled to
              224 only if larger), so A4 and the gate see what they saw in Phase 4.
Research option (R5-UI)
  force_a4   run A4 on the face chip even when the gate does not route it (what the R5 "forced A4" evaluations did).
             The gate still runs and its decision/reason are reported unchanged; `forced` marks the override and
             `gate_route` keeps the gate's own route. Detection failures still return ORIGINAL (no chip to enhance);
             the router's timeout/failure fallbacks and output guard still apply. Default off = the frozen behaviour.
Outputs
  ENHANCED   the A4 face chip, 224×224 (E-frame), PNG
  ORIGINAL   the input bytes exactly as received (never re-encoded) — for every non-A4 outcome
  REJECTED   undecodable / out-of-limit input: nothing to return
Validity: the gate/router thresholds are SYNTHETIC-calibrated (Phase 4). This pipeline is a research candidate,
not production-approved; A4 weights and YuNet are licence REVIEW_REQUIRED.
"""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from fes_bench import gate as GT
from fes_bench import geometry as G
from fes_bench import quality as Q
from fes_bench import routing as RT
from fes_bench.evaluators import DLIB_CHIP_TEMPLATE_150
from fes_data import preprocess as P
from fes_data.common import repo_path

SERVICE_VERSION = "fes-service-r5-0.1.0"
YUNET = "models/D01_yunet/face_detection_yunet_2026may.onnx"
CHIP_MAX = G.E_SIZE                                   # 224: the largest input size the gate was calibrated on
TEMPLATE_IOD_150 = float(np.linalg.norm(DLIB_CHIP_TEMPLATE_150[1] - DLIB_CHIP_TEMPLATE_150[0]))
QUALITY_KEYS = ("noise_sigma", "lap_norm_c", "lapd1_c", "blockiness", "luma_mean", "rms_contrast", "iod_px", "det_score",
                "is_gray")

ENHANCED, ORIGINAL_S, REJECTED = "ENHANCED", "ORIGINAL", "REJECTED"


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 40_000_000
    min_side: int = 8
    max_aspect: float = 10.0


@dataclass
class Result:
    status: str                                  # ENHANCED | ORIGINAL | REJECTED
    request_id: str
    input_mode: str
    image: bytes | None = None                   # output bytes (ORIGINAL: the input bytes unchanged)
    image_format: str = ""                       # png | as_received
    decision: str = ""                           # T6 decision
    reason: str = ""
    bucket: str = ""
    route: str = RT.ORIGINAL
    used: str = RT.ORIGINAL
    fallback: str = ""
    escalate: bool = False
    error: str = ""
    warnings: list = field(default_factory=list)
    n_faces: int = 0
    chip_px: int = 0
    quality: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    forced: bool = False                         # force_a4 overrode the gate's route
    gate_route: str = ""                         # the route the gate chose (before any force_a4 override)
    chip: bytes | None = None                    # PNG of the face chip the gate analysed / A4 received

    def metadata(self) -> dict:
        """Structured, non-sensitive metadata: no landmarks, boxes, embeddings or pixels."""
        return {"service_version": SERVICE_VERSION, "request_id": self.request_id, "status": self.status,
                "input_mode": self.input_mode, "enhanced": self.status == ENHANCED,
                "output": "a4_face_chip_224" if self.status == ENHANCED else
                          ("original_input" if self.status == ORIGINAL_S else "none"),
                "decision": self.decision, "reason": self.reason, "bucket": self.bucket, "route": self.route,
                "used": self.used, "fallback": self.fallback, "escalate": self.escalate, "error": self.error,
                "warnings": self.warnings, "n_faces": self.n_faces, "chip_px": self.chip_px,
                "quality": self.quality, "timings_ms": self.timings_ms, "forced": self.forced,
                "gate_route": self.gate_route}


def _json_num(v):
    """NaN/inf (e.g. no detection score on an UNSAFE input) → None: JSON has no NaN."""
    return None if isinstance(v, float) and not np.isfinite(v) else v


class Busy(Exception):
    pass


def face_chip(img: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, int]:
    """Square FairFace/dlib-framed chip cut at INTEGER pixel positions (no resampling at native scale).

    Framing: the similarity fit of the 5 landmarks to DLIB_CHIP_TEMPLATE gives the chip centre and side in source
    pixels (side ≈ 224 · IOD / 81.4); rotation is NOT applied. Any sub-pixel warp is a low-pass filter and erases the
    noise/sharpness statistics the frozen gate reads (measured: bilinear re-warp kept only 6 of 73 Phase-4 A4 routes).
    Out-of-image area is filled black; sides above 224 are reduced with INTER_AREA (the FairFace chip rule)."""
    M = P.umeyama_similarity(landmarks, DLIB_CHIP_TEMPLATE_150)          # source → 150-px chip
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    A_inv = np.linalg.inv(M[:, :2])
    cx, cy = A_inv @ (np.array([75.0, 75.0]) - M[:, 2])                 # chip centre in source pixels
    side = int(max(8, round(150.0 / scale)))
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)) if pad else img
    chip = src[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    if side > CHIP_MAX:
        chip = cv2.resize(chip, (CHIP_MAX, CHIP_MAX), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(chip), side


class EnhancementPipeline:
    """Thread-safe; one request is processed at a time (YuNet and the router are stateful), at most `max_inflight`
    requests may wait — beyond that `Busy` is raised (HTTP 503) instead of queueing without bound."""

    def __init__(self, a4, timeout_s: float = 120.0, limits: Limits = Limits(), max_inflight: int = 4,
                 gate: str = "t6"):
        """gate="t6": the frozen Phase 4 T6/T7 policy, exactly (synthetic-calibrated).
        gate="rw": the R5-3 real-world gate (fes_service.rwgate, configs/phase5/r5_rwgate.json)."""
        assert gate in ("t6", "rw"), gate
        self.gate = gate
        g6 = json.loads(repo_path("configs/phase4/t6_gate.json").read_text(encoding="utf-8"))
        r7 = json.loads(repo_path("configs/phase4/t7_routing.json").read_text(encoding="utf-8"))
        self.rules = GT.load_rules(g6)                          # verifies the T6 seal
        self.t, self.table = RT.load_routing(r7, g6)            # verifies the T7 seal + link to T6
        assert set(self.table.values()) <= {RT.ORIGINAL, "A4"}, "only A4 may be routed (Phase 4 gate)"
        self.seals = {"t6_gate": g6["sha256"], "t7_routing": r7["sha256"]}
        if gate == "rw":
            from fes_service import rwgate as RW
            self.rw_cfg = RW.load_config()
            self.seals["rw_gate"] = self.rw_cfg["sha256"]
        self.det = P.YuNetDetector(repo_path(YUNET), 0.7, 0.3, max_side=640)
        self.det_cfg = P.PreprocessConfig(out_size=112)
        self.a4 = a4
        self.limits = limits
        self._feat = {}
        # the router's own thread timeout is a backstop; the isolated worker enforces (and kills at) timeout_s
        face_check = lambda out: len(self.det.detect(out)) > 0  # noqa: E731
        if gate == "rw":
            self.router = RW.RWRouter(self.rw_cfg, {"A4": a4}, features_fn=self._features, face_check=face_check,
                                      timeout_s=timeout_s + 30.0)
        else:
            self.router = RT.Router(self.rules, self.t, self.table, {"A4": a4}, features_fn=self._features,
                                    face_check=face_check, timeout_s=timeout_s + 30.0)
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_inflight)

    def _features(self, chip: np.ndarray) -> dict:
        if self.gate == "rw":
            from fes_service.rwgate import rw_features
            self._feat = rw_features(chip, self.det)
        else:
            self._feat = Q.features(chip, self.det)
        return self._feat

    # ---------------------------------------------------------------- validation
    def _decode(self, data: bytes) -> tuple[np.ndarray | None, str]:
        L = self.limits
        if not data:
            return None, "empty input"
        if len(data) > L.max_bytes:
            return None, f"input exceeds {L.max_bytes} bytes"
        buf = np.frombuffer(data, np.uint8)
        # header-only size check before the full decode (decompression-bomb guard)
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
            if w * h > L.max_pixels:
                return None, f"image exceeds {L.max_pixels} pixels"
        except Image.DecompressionBombError:
            return None, f"image exceeds {L.max_pixels} pixels"
        except Exception:  # noqa: BLE001 — not a PIL-readable header: cv2 decides below
            pass
        try:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except cv2.error as e:
            return None, f"decode failed: {str(e)[:80]}"
        if img is None:
            return None, "not a decodable image"
        h, w = img.shape[:2]
        if min(h, w) < L.min_side:
            return None, f"image smaller than {L.min_side} px"
        if h * w > L.max_pixels:
            return None, f"image exceeds {L.max_pixels} pixels"
        if max(h, w) / min(h, w) > L.max_aspect:
            return None, f"aspect ratio above {L.max_aspect}"
        return img, ""

    # ---------------------------------------------------------------- main entry
    def process(self, data: bytes, input_mode: str = "full_image", force_a4: bool = False) -> Result:
        if not self._slots.acquire(blocking=False):
            raise Busy("too many requests in flight")
        try:
            with self._lock:
                return self._process(data, input_mode, force_a4)
        finally:
            self._slots.release()

    def _route(self, chip: np.ndarray, force_a4: bool):
        """The frozen router; with force_a4 its plan is kept but the route is overridden to A4 (under the lock)."""
        if not force_a4:
            return self.router.route(chip, {"s": chip.shape[0]}, G.upsample_to_e(chip))
        gate_plan = self.router.plan

        def forced_plan(f):
            rec = gate_plan(f)
            rec.extra["gate_route"] = rec.route
            rec.route = "A4"
            return rec
        self.router.plan = forced_plan
        try:
            return self.router.route(chip, {"s": chip.shape[0]}, G.upsample_to_e(chip))
        finally:
            del self.router.plan                                     # back to the class method

    def _process(self, data: bytes, input_mode: str, force_a4: bool = False) -> Result:
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
        res = Result(status=REJECTED, request_id=uuid.uuid4().hex, input_mode=input_mode)

        def original(**kw) -> Result:
            res.status, res.image, res.image_format = ORIGINAL_S, data, "as_received"
            for k, v in kw.items():
                setattr(res, k, v)
            res.timings_ms["total"] = ms()
            return res

        if input_mode not in ("full_image", "face_chip"):
            res.error = f"unknown input_mode {input_mode!r}"
            return res
        img, err = self._decode(data)
        res.timings_ms["decode"] = ms()
        if img is None:
            res.error = err
            res.timings_ms["total"] = ms()
            return res

        # ---- detection + chip
        if input_mode == "face_chip":
            h, w = img.shape[:2]
            if h != w or h > CHIP_MAX:
                res.error = f"face_chip input must be square with side ≤ {CHIP_MAX} (got {w}×{h})"
                res.timings_ms["total"] = ms()
                return res
            chip = img
        else:
            fr = P.preprocess(img, self.det, self.det_cfg, input_mode="face_crop")[0]
            res.n_faces, res.warnings = fr.n_faces, list(fr.warnings)
            res.timings_ms["detect"] = ms()
            if fr.status != P.OK or fr.detection is None:
                return original(decision=GT.UNSAFE, reason=f"no_valid_face:{fr.status}", escalate=True)
            lm = np.asarray(fr.detection.landmarks, np.float64)
            if not fr.aligned or not P.landmarks_plausible(lm):
                return original(decision=GT.UNSAFE, reason="alignment_unreliable:landmarks", escalate=True)
            chip, _ = face_chip(img, lm)
        res.chip_px = int(chip.shape[0])
        ok, chip_png = cv2.imencode(".png", chip)
        res.chip = chip_png.tobytes() if ok else None

        # ---- frozen T6 → T7 → ORIGINAL | A4 (with the router's fallbacks and output guard)
        try:
            out, rec = self._route(chip, force_a4)
        except Exception as e:  # noqa: BLE001 — feature extraction itself failed: never enhance
            return original(decision=GT.UNSAFE, reason="quality_analysis_failed", escalate=True,
                            error=f"{type(e).__name__}: {e}"[:200])
        res.timings_ms["route"] = ms()
        res.quality = {k: _json_num(self._feat.get(k)) for k in QUALITY_KEYS}
        res.decision, res.reason, res.bucket, res.route = rec.decision, rec.reason, rec.bucket, rec.route
        res.fallback, res.escalate, res.error, res.used = rec.fallback, rec.escalate, rec.error, rec.used
        res.gate_route = rec.extra.get("gate_route", rec.route)
        res.forced = force_a4 and res.gate_route != "A4"
        if rec.used == RT.ORIGINAL:
            return original()
        if self.gate == "rw" and self._feat.get("is_gray"):
            # greyscale in → greyscale out: A4 runs in RGB and may tint a grey face; never add colour that was not there
            out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            res.warnings.append("grayscale_preserved")
        ok, png = cv2.imencode(".png", out)
        if not ok:
            return original(used=RT.ORIGINAL, fallback="INVALID_OUTPUT", error="png encode failed")
        res.status, res.image, res.image_format = ENHANCED, png.tobytes(), "png"
        res.timings_ms["total"] = ms()
        return res
