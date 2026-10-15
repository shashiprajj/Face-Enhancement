"""Enhancement pipeline: one photo or face chip in, one face chip (or the untouched input) out.

    bytes → validation → face detection (YuNet) → face chip → input gate (sealed NAFNet route config)
          → NAFNet r6 and NAFNet r6_ft (isolated ONNX workers, 4-run stability consensus each)
          → stability check + artifact guards → the sharper accepted output, or the ORIGINAL

Input modes
  face_chip   a square, face-centred chip (FairFace/dlib framing, side ≤ 224). Used as is.
  full_image  any photo. Detection uses the FES contract (largest face; `multiple_faces` warning). The face is cut
              out as a square chip in the FairFace/dlib framing the gate was calibrated on (DLIB_CHIP_TEMPLATE_150,
              landmark similarity fit) at NATIVE resolution (no upsampling; downscaled to 224 only if larger). Chip area
              outside the image is reflect-filled for the gate and the models and black in the returned chip.
Profiles (per request; `profile`)
  best        (default) r6 and r6_ft both run their guarded route on the same face; the accepted output with the
              higher lapd1_c (nafnet_route.select_best) is returned and `selection` reports the candidates.
  best_side   `best`; where `best` stops because the face is turned, the face is framed by its detection box instead
              (face_chip_box) and the same route runs on that chip with the sealed side-face thresholds.
  r6, r6_ft   one candidate only (NAFNet-GoPro / the face-blur fine-tune).
Outputs
  ENHANCED   the deblurred face chip at its native size, PNG
  ORIGINAL   the input bytes exactly as received (never re-encoded) — for every non-enhanced outcome
  REJECTED   undecodable / out-of-limit input or an invalid request: nothing to return
Validity: the gate thresholds are calibrated on synthetic DEV blur. Research candidate, not production-approved;
YuNet is licence REVIEW_REQUIRED.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from fes_service import blur_scale as BS
from fes_service import gate as GT
from fes_service import nafnet_route as NR
from fes_service import preprocess as P
from fes_service import quality as QL
from fes_service import repo_path

SERVICE_VERSION = "fes-service-r6-1.0.0"
YUNET = "models/D01_yunet/face_detection_yunet_2026may.onnx"
CHIP_MAX = 224                                        # the largest chip side the gate was calibrated on
# Mean YuNet 5-point landmarks of the 112 dev_pilot references (dlib chips, padding 0.25) scaled 224 → 150.
# Derived on DEV data only (P4-T3 §3); frozen.
DLIB_CHIP_TEMPLATE_150 = np.array([[46.396, 45.712], [100.894, 45.828], [72.459, 77.261],
                                   [50.880, 102.266], [95.750, 102.370]], np.float64)
TEMPLATE_IOD_150 = float(np.linalg.norm(DLIB_CHIP_TEMPLATE_150[1] - DLIB_CHIP_TEMPLATE_150[0]))
QUALITY_KEYS = ("noise_sigma", "lap_norm_c", "lapd1_c", "blockiness", "luma_mean", "rms_contrast", "iod_px", "det_score",
                "is_gray", "pixelation", "hf_peak_frac", "hf_energy")
ORIGINAL_ROUTE = "B0"                                 # route / used code of "the input returned unchanged"

ENHANCED, ORIGINAL_S, REJECTED = "ENHANCED", "ORIGINAL", "REJECTED"
# fallback codes: why an eligible face was returned unchanged
ARTIFACT_GUARD = "ARTIFACT_GUARD"                 # the output failed the artifact guard
LOCAL_ARTIFACT_GUARD = "LOCAL_ARTIFACT_GUARD"     # the output failed the local-patch guard
STABILITY_GUARD = "STABILITY_GUARD"               # a mirror run of NAFNet was unstable

R6, R6_FT = "r6", "r6_ft"
BEST_CANDIDATES = (R6, R6_FT)
PROFILE_MODEL = {R6: NR.MODEL, R6_FT: "NAFNET_FACE_FT"}          # model INVOKED per candidate (models.FACE_FT_MODEL)
PROFILE_BEST = "best"
# "best_side": exactly "best" for a face the landmark framing accepts. Only where "best" stops because the face is
# turned — the detector's landmarks are not usable (FACE_TOO_SMALL / alignment_unreliable) or the gate answers
# NAFNET_REJECT_UNSAFE_ALIGNMENT — the face is framed by its DETECTION BOX instead (face_chip_box) and the same "best"
# route runs on that chip. Two things replace what needs two visible eyes: the safety rule (NR.box_safety: confident
# detection in the full image) and the face-size feature (the chip's template-equivalent eye distance). Evidence:
# Indian-expression benchmark (2026-10-13/14). Identity on profile faces is not measured (SFace is not reliable there).
PROFILE_BEST_SIDE = "best_side"
BEST_PROFILES = (PROFILE_BEST, PROFILE_BEST_SIDE)
PROFILES = BEST_CANDIDATES + BEST_PROFILES
DEFAULT_PROFILE = PROFILE_BEST
# chip side / max(box width, box height): the median over 80 frontal references of that benchmark, so a box-framed
# chip shows a face at the scale the models and the gate see in the landmark framing
SIDE_BOX_SCALE = 0.982


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
    profile: str = ""
    image: bytes | None = None                   # output bytes (ORIGINAL: the input bytes unchanged)
    image_format: str = ""                       # png | as_received
    decision: str = ""                           # gate decision
    reason: str = ""
    bucket: str = ""
    route: str = ORIGINAL_ROUTE
    used: str = ORIGINAL_ROUTE
    fallback: str = ""
    escalate: bool = False
    error: str = ""
    warnings: list = field(default_factory=list)
    n_faces: int = 0
    chip_px: int = 0
    quality: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    chip: bytes | None = None                    # PNG of the face chip the gate analysed
    model: str | None = None                     # model INVOKED: NAFNET_GOPRO | NAFNET_FACE_FT | None
    artifact_check: str = "NOT_RUN"              # PASS | FAIL | NOT_RUN
    nafnet: dict | None = None                   # route record (eligibility, reason, guard results)
    selection: dict | None = None                # profiles best / best_side: candidates, scores and the one chosen

    def metadata(self) -> dict:
        """Structured, non-sensitive metadata: no landmarks, boxes, embeddings or pixels."""
        extra = {"selection": self.selection} if self.selection is not None else {}
        return {"profile": self.profile, **extra, "service_version": SERVICE_VERSION, "request_id": self.request_id,
                "status": self.status, "input_mode": self.input_mode, "enhanced": self.status == ENHANCED,
                "output": ("nafnet_face_chip_native" if self.status == ENHANCED
                           else "original_input" if self.status == ORIGINAL_S else "none"),
                "decision": self.decision, "reason": self.reason, "bucket": self.bucket, "route": self.route,
                "used": self.used, "fallback": self.fallback, "escalate": self.escalate, "error": self.error,
                "warnings": self.warnings, "n_faces": self.n_faces, "chip_px": self.chip_px,
                "quality": self.quality, "timings_ms": self.timings_ms, "model": self.model,
                "artifact_check": self.artifact_check, "nafnet": self.nafnet}


def _json_num(v):
    """NaN/inf (e.g. no detection score on an UNSAFE input) → None: JSON has no NaN."""
    return None if isinstance(v, float) and not np.isfinite(v) else v


def _is_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return isinstance(e, MemoryError) or any(k in s for k in ("out of memory", "defaultcpuallocator", "not enough memory",
                                                              "cannot allocate memory"))


def fallback_code(e: BaseException) -> str:
    """A model call that raised (fes_service.worker): TIMEOUT | RESOURCE_FAILURE | MODEL_FAILURE."""
    return "TIMEOUT" if isinstance(e, TimeoutError) else "RESOURCE_FAILURE" if _is_oom(e) else "MODEL_FAILURE"


class Busy(Exception):
    pass


def face_chip(img: np.ndarray, landmarks: np.ndarray, border: int = cv2.BORDER_CONSTANT) -> tuple[np.ndarray, int]:
    """Square FairFace/dlib-framed chip cut at INTEGER pixel positions (no resampling at native scale).

    Framing: the similarity fit of the 5 landmarks to DLIB_CHIP_TEMPLATE gives the chip centre and side in source
    pixels (side ≈ 224 · IOD / 81.4); rotation is NOT applied. Any sub-pixel warp is a low-pass filter and erases the
    noise/sharpness statistics the gate reads. Out-of-image area is filled black (or `border`); sides above 224 are
    reduced with INTER_AREA (the FairFace chip rule)."""
    M = P.umeyama_similarity(landmarks, DLIB_CHIP_TEMPLATE_150)          # source → 150-px chip
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    A_inv = np.linalg.inv(M[:, :2])
    cx, cy = A_inv @ (np.array([75.0, 75.0]) - M[:, 2])                 # chip centre in source pixels
    side = int(max(8, round(150.0 / scale)))
    x0, y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=(0, 0, 0)) if pad else img
    chip = src[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    if side > CHIP_MAX:
        chip = cv2.resize(chip, (CHIP_MAX, CHIP_MAX), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(chip), side


def face_chip_filled(img: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The SAME chip as face_chip (same framing, same pixels where the image exists) with its out-of-image area filled
    by reflecting the image instead of black, plus that area's mask (chip coordinates). R5-7 DEV: the black fill
    multiplied lapd1 ×4 and lost 95 % of the affected eligible faces."""
    filled, _ = face_chip(img, landmarks, cv2.BORDER_REFLECT_101)
    mask = face_chip(np.full(img.shape[:2] + (3,), 255, np.uint8), landmarks)[0][..., 0] < 128
    return filled, mask


def face_chip_box(img: np.ndarray, bbox_xywh, border: int = cv2.BORDER_CONSTANT) -> tuple[np.ndarray, int]:
    """SIDE-FACE framing (profile "best_side"): square chip centred on the detection box, side = SIDE_BOX_SCALE ×
    the box's longer side, cut at integer pixel positions and reduced to 224 with INTER_AREA when larger — the
    face_chip rules, with the box in place of the landmark fit."""
    x, y, bw, bh = bbox_xywh
    side = int(max(8, round(SIDE_BOX_SCALE * max(bw, bh))))
    x0, y0 = int(round(x + bw / 2.0 - side / 2.0)), int(round(y + bh / 2.0 - side / 2.0))
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=(0, 0, 0)) if pad else img
    chip = src[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    if side > CHIP_MAX:
        chip = cv2.resize(chip, (CHIP_MAX, CHIP_MAX), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(chip), side


class EnhancementPipeline:
    """Thread-safe; one request is processed at a time (YuNet is stateful), at most `max_inflight` requests may
    wait — beyond that `Busy` is raised (HTTP 503) instead of queueing without bound."""

    def __init__(self, models: dict, limits: Limits = Limits(), max_inflight: int = 4,
                 blur_settings: BS.BlurSettings | None = None, blur_settings_source: str = "default"):
        """models: {"r6": callable, "r6_ft": callable}, each callable(chip, ctx) -> (same-size chip, meta) with
        meta["runs_dot_c"] (the 4-run consensus), normally an IsolatedModel of an ONNX factory (fes_service.models).
        blur_settings: the deployment's admin blur range (fes_service.blur_scale; None = both off). Validated here: an
        invalid value raises ValueError (the service does not become ready)."""
        assert set(models) == set(BEST_CANDIDATES), f"models for {', '.join(BEST_CANDIDATES)} are needed"
        self.models = dict(models)
        self.cfg = NR.load_config()
        self.side_cfg = NR.load_side_config()
        self.blur_scale = BS.load_scale()
        self.seals = {"nafnet_route": self.cfg["sha256"], "side_face_route": self.side_cfg["sha256"],
                      "blur_scale": self.blur_scale.cfg["sha256"]}
        self.oob_fill = bool(self.cfg.get("framing", {}).get("oob_reflect_fill", False))
        self.blur_settings = BS.validate(blur_settings or BS.BlurSettings(), self.blur_scale)
        self.blur_settings_source = blur_settings_source
        self._req_blur: tuple[BS.BlurSettings, dict] | None = None    # the current request's range (under the lock)
        self.det = P.YuNetDetector(repo_path(YUNET), 0.7, 0.3, max_side=640)
        self.det_cfg = P.PreprocessConfig(out_size=112)
        self.limits = limits
        self._face_check = lambda out: len(self.det.detect(out)) > 0  # noqa: E731
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_inflight)

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
    def process(self, data: bytes, input_mode: str = "full_image", profile: str | None = None,
                blur: BS.BlurSettings | None = None) -> Result:
        """profile None/"" = DEFAULT_PROFILE; an unknown profile is REJECTED, never silently replaced.
        blur: per-request admin blur range; it may only narrow the deployment's (otherwise REJECTED)."""
        if not self._slots.acquire(blocking=False):
            raise Busy("too many requests in flight")
        try:
            with self._lock:
                profile = profile or DEFAULT_PROFILE
                try:
                    self._req_blur = self._request_blur(blur)
                except ValueError as e:
                    return Result(status=REJECTED, request_id=uuid.uuid4().hex, input_mode=input_mode,
                                  profile=profile, error=str(e))
                return self._process(data, input_mode, profile)
        finally:
            self._slots.release()

    def _request_blur(self, blur: BS.BlurSettings | None) -> tuple[BS.BlurSettings, dict]:
        """→ (effective blur range, where each value came from)."""
        req = blur or BS.BlurSettings()
        eff = BS.validate(req, self.blur_scale, base=self.blur_settings)

        def src(r, d, unset):
            return "request" if r is not None else "deployment" if d is not None else unset
        return eff, {"enhance_from": src(req.enhance_from_pct, self.blur_settings.enhance_from_pct, "validated_limit"),
                     "give_up_above": src(req.give_up_above_pct, self.blur_settings.give_up_above_pct, "off")}

    def _process(self, data: bytes, input_mode: str, profile: str) -> Result:
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
        res = Result(status=REJECTED, request_id=uuid.uuid4().hex, input_mode=input_mode, profile=profile)

        def original(**kw) -> Result:
            res.status, res.image, res.image_format = ORIGINAL_S, data, "as_received"
            for k, v in kw.items():
                setattr(res, k, v)
            res.timings_ms["total"] = ms()
            return res

        fill = None                         # (reflect-filled chip, out-of-image mask) when the chip leaves the image
        if input_mode not in ("full_image", "face_chip"):
            res.error = f"unknown input_mode {input_mode!r}"
            return res
        if profile not in PROFILES:
            res.error = f"unknown profile {profile!r} (expected one of {', '.join(PROFILES)})"
            return res
        side = profile == PROFILE_BEST_SIDE and input_mode == "full_image"
        det = None                          # the full-image detection (profile "best_side": its box frames a side face)
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
            det = fr.detection
            if fr.status != P.OK or fr.detection is None:
                if side and det is not None:
                    return self._side_path(res, data, img, det, ms, f"no_valid_face:{fr.status}")
                return original(decision=GT.UNSAFE, reason=f"no_valid_face:{fr.status}", escalate=True)
            lm = np.asarray(fr.detection.landmarks, np.float64)
            if not fr.aligned or not P.landmarks_plausible(lm):
                if side:
                    return self._side_path(res, data, img, det, ms, "alignment_unreliable:landmarks")
                return original(decision=GT.UNSAFE, reason="alignment_unreliable:landmarks", escalate=True)
            chip, _ = face_chip(img, lm)
            if self.oob_fill:
                filled, oob = face_chip_filled(img, lm)
                if oob.any():
                    fill = (filled, oob)
        res.chip_px = int(chip.shape[0])
        ok, chip_png = cv2.imencode(".png", chip)
        res.chip = chip_png.tobytes() if ok else None
        if profile in BEST_PROFILES:
            out = self._best_path(res, data, chip, ms, fill)
            if (side and det is not None and out.status == ORIGINAL_S
                    and (out.nafnet or {}).get("reason") == NR.R_ALIGN):     # turned face: the gate's pose rule said no
                return self._side_path(res, data, img, det, ms, NR.R_ALIGN)
            return out
        return self._nafnet_path(res, chip, original, ms, fill, profile)

    def _side_path(self, res: Result, data: bytes, img: np.ndarray, det, ms, why: str) -> Result:
        """Profile "best_side": the landmark framing stopped at `why`; frame the face by its detection box and run
        the unchanged "best" route on that chip (see PROFILE_BEST_SIDE). Out-of-image area: reflect-filled for the
        gate and the models, black in the returned chip."""
        res = replace(res, warnings=[*res.warnings, "side_face_box_framing"], timings_ms=dict(res.timings_ms))
        chip, _ = face_chip_box(img, det.bbox_xywh)
        oob = face_chip_box(np.full(img.shape[:2] + (3,), 255, np.uint8), det.bbox_xywh)[0][..., 0] < 128
        fill = (face_chip_box(img, det.bbox_xywh, cv2.BORDER_REFLECT_101)[0], oob) if oob.any() else None
        res.chip_px = int(chip.shape[0])
        ok, chip_png = cv2.imencode(".png", chip)
        res.chip = chip_png.tobytes() if ok else None
        box = {"landmark_framing": why, "det_score": float(det.score),
               "iod_equiv_px": round(chip.shape[0] * TEMPLATE_IOD_150 / 150.0, 3)}
        return self._best_path(res, data, chip, ms, fill, box)

    def _best_path(self, res: Result, data: bytes, chip: np.ndarray, ms, fill=None, box: dict | None = None) -> Result:
        """Profiles best / best_side: run every BEST_CANDIDATES profile's guarded route on the same chip and return ONE
        result — the accepted output NR.select_best prefers, or the original when no candidate was accepted. The
        candidates share one sealed input gate, so a face the first one finds not eligible is final and the others are
        not run."""
        subs: dict[str, Result] = {}
        infer_for = self._parallel_infer()
        for name in BEST_CANDIDATES:
            sub = replace(res, warnings=list(res.warnings), timings_ms=dict(res.timings_ms))

            def sub_original(_s=sub, **kw) -> Result:
                _s.status, _s.image, _s.image_format = ORIGINAL_S, data, "as_received"
                for k, v in kw.items():
                    setattr(_s, k, v)
                _s.timings_ms["total"] = ms()
                return _s
            self._nafnet_path(sub, chip, sub_original, ms, fill, name, infer_for(name), box)
            subs[name] = sub
            if not (sub.nafnet or {}).get("eligible"):
                break
        outs = {n: cv2.imdecode(np.frombuffer(s.image, np.uint8), cv2.IMREAD_COLOR)
                for n, s in subs.items() if s.status == ENHANCED and s.used == NR.MODEL}
        chosen, scores = NR.select_best(chip, outs) if outs else (None, {})
        final = subs[chosen] if chosen else subs[BEST_CANDIDATES[0]]
        final.selection = {"rule": NR.SELECT_RULE, "chosen": chosen,
                           "candidates": {n: {"status": s.status, "fallback": s.fallback or "", "model": s.model,
                                              "score": scores.get(n)} for n, s in subs.items()}}
        final.timings_ms["total"] = ms()
        return final

    def _parallel_infer(self):
        """→ infer_for(name) → infer(work, ctx), a drop-in for `self.models[name](work, ctx)` in _nafnet_path.
        Nothing runs until the FIRST candidate reaches its model call (so a face the gate rejects never starts a
        model). At that moment the model of EVERY candidate is started on that same input, each in its own
        IsolatedModel (own process, own lock, own timeout: a hang, crash or error stays with that worker and surfaces
        as the same exception, to the same candidate, as a direct call). Each candidate then collects ITS result. A
        later candidate whose input is not the very same one is run directly."""
        from concurrent.futures import ThreadPoolExecutor
        state: dict = {}

        def infer_for(name):
            def infer(work, ctx):
                if not state:
                    pool = ThreadPoolExecutor(len(self.models), thread_name_prefix="fes-best")
                    state.update(work=work, ctx=dict(ctx),
                                 fut={n: pool.submit(self.models[n], work, dict(ctx)) for n in BEST_CANDIDATES})
                    pool.shutdown(wait=False)
                first = state["ctx"]
                same = work is state["work"] and ctx.keys() == first.keys() and all(
                    ctx[k] is first[k] or (not isinstance(ctx[k], np.ndarray) and ctx[k] == first[k]) for k in ctx)
                if not same or name not in state["fut"]:
                    return self.models[name](work, ctx)
                return state["fut"].pop(name).result()
            return infer
        return infer_for

    def _nafnet_path(self, res: Result, chip: np.ndarray, original, ms, fill=None, name: str = R6, infer=None,
                     box: dict | None = None) -> Result:
        """The guarded route of one candidate: input gate → NAFNet (4 mirror runs, isolated worker) → stability check
        → artifact guard → local-patch guard; ORIGINAL otherwise. `fill` = (reflect-filled chip, out-of-image mask)
        when the full_image chip extends past the image: gate + NAFNet see the filled chip, the output's out-of-image
        area is set back to black. `box`: side-face framing (profile best_side)."""
        cfg = self.cfg
        work = fill[0] if fill is not None else chip
        try:
            f = QL.rw_features(work, self.det)                    # rw features (p5-rwfeat-1.0.0)
            f.update(NR.texture_features(work))                   # p5-nafnet-feat-1.0.0
        except Exception as e:  # noqa: BLE001 — feature extraction failed: never enhance
            return original(decision=GT.UNSAFE, reason="quality_analysis_failed", escalate=True,
                            error=f"{type(e).__name__}: {e}"[:200])
        if box is not None:     # side-face framing: the box's detection score and the chip's template-equivalent IOD
            f = {**f, "det_score": box["det_score"], "iod_px": box["iod_equiv_px"]}
        res.quality = {k: _json_num(f.get(k)) for k in QUALITY_KEYS}
        gate_cfg = cfg if box is None else {         # box framing: its own sealed blur boundary
            **cfg, "thresholds": {**cfg["thresholds"], **self.side_cfg["thresholds"]}}
        br = None                                    # admin blur range: the face's blur % and the range applied
        if self._req_blur is not None and not NR._nan(f.get("lapd1_c")):
            br = BS.report(self.blur_scale, "front" if box is None else "side", f["lapd1_c"],
                           gate_cfg["thresholds"]["blur_max"], *self._req_blur)
        nd = NR.decide(f, gate_cfg, GT.safety if box is None else NR.box_safety,
                       br if br is not None and self._req_blur[0].active else None)
        res.nafnet = {**nd.as_dict(), "route_version": cfg["route_version"], "infer_s": None, "artifact": None}
        if br is not None:
            res.nafnet["blur_range"] = br
        if box is not None:
            res.nafnet.update(framing="detection_box", landmark_framing=box["landmark_framing"],
                              side_route_version=self.side_cfg["route_version"])
        res.nafnet.update(profile=name, local_artifact=None,
                          oob_fill_frac=round(float(fill[1].mean()), 4) if fill is not None else 0.0)
        res.timings_ms["gate"] = ms()
        if not nd.eligible:
            return original(decision=nd.decision, reason=nd.reason, bucket=nd.route, route=ORIGINAL_ROUTE,
                            escalate=nd.decision == GT.UNSAFE)
        res.decision, res.reason, res.bucket = nd.decision, nd.reason, nd.route
        res.route = NR.MODEL
        res.model = PROFILE_MODEL[name]
        try:
            ctx = {"s": work.shape[0]}
            if fill is not None:                                  # out-of-image area excluded from the tracer
                ctx["oob_mask"] = fill[1]
            out, meta = (infer or self.models[name])(work, ctx)
            res.nafnet["infer_s"] = (meta or {}).get("infer_s")
            if (meta or {}).get("backend"):
                res.nafnet["backend"] = meta["backend"]
            if fill is not None and isinstance(out, np.ndarray) and out.shape[:2] == fill[1].shape:
                out = out.copy()
                out[fill[1]] = 0                                  # out-of-image area back to black: geometry unchanged
        except Exception as e:  # noqa: BLE001 — timeout / OOM / crash / load failure: always the original
            res.timings_ms["nafnet"] = ms()
            return original(used=ORIGINAL_ROUTE, fallback=fallback_code(e), error=f"{type(e).__name__}: {e}"[:300])
        res.timings_ms["nafnet"] = ms()
        try:                                                      # 4-run stability check before any other guard
            sg = NR.stability_check((meta or {}).get("runs_dot_c"), cfg)
        except Exception as e:  # noqa: BLE001 — an uncertain stability result counts as a failed one
            sg = NR.ArtifactGuardResult(False, [NR.A_UNSTABLE_ERROR], {"error": f"{type(e).__name__}: {e}"[:200]})
        res.nafnet["stability"] = sg.as_dict()
        if not sg.passed:
            res.artifact_check = "FAIL"
            return original(used=ORIGINAL_ROUTE, fallback=STABILITY_GUARD, error="+".join(sg.reasons))
        try:
            g = NR.artifact_guard(chip, out, cfg, self._face_check, bool(f.get("is_gray")))
        except Exception as e:  # noqa: BLE001 — an uncertain safety result counts as a failed one
            g = NR.ArtifactGuardResult(False, [NR.A_NUMERIC], {"guard_error": f"{type(e).__name__}: {e}"[:200]})
        res.timings_ms["guard"] = ms()
        res.nafnet["artifact"] = g.as_dict()
        res.artifact_check = "PASS" if g.passed else "FAIL"
        if not g.passed:
            return original(used=ORIGINAL_ROUTE, fallback=ARTIFACT_GUARD, error="+".join(g.reasons))
        try:
            lg = NR.local_artifact_guard(chip, out, cfg, bool(f.get("is_gray")))
        except Exception as e:  # noqa: BLE001 — detector failure counts as a rejection
            lg = NR.ArtifactGuardResult(False, [NR.A_LOCAL_ERROR], {"error": f"{type(e).__name__}: {e}"[:200]})
        res.timings_ms["local_guard"] = ms()
        res.nafnet["local_artifact"] = lg.as_dict()
        if not lg.passed:
            res.artifact_check = "FAIL"
            return original(used=ORIGINAL_ROUTE, fallback=LOCAL_ARTIFACT_GUARD, error="+".join(lg.reasons))
        if f.get("is_gray"):
            # greyscale in → greyscale out (the guard has already rejected any STRONG invented colour)
            out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            res.warnings.append("grayscale_preserved")
        ok, png = cv2.imencode(".png", out)
        if not ok:
            return original(used=ORIGINAL_ROUTE, fallback="INVALID_OUTPUT", error="png encode failed")
        res.used = NR.MODEL
        res.status, res.image, res.image_format = ENHANCED, png.tobytes(), "png"
        res.timings_ms["total"] = ms()
        return res
