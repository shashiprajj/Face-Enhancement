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
Route modes (R5-5; `route_mode`, per pipeline and per request)
  a4         the R5-1/R5-3 behaviour, unchanged: gate (t6 | rw) → A4 SwinIR-L or ORIGINAL. Legacy / comparison route.
  nafnet     the guarded NAFNet-GoPro route only (fes_service.nafnet_route): dedicated input gate → NAFNet (isolated
             worker) → artifact guard → NAFNet output; ORIGINAL on any rejection / failure / guard failure.
  auto       NAFNet route first; if NAFNet is ineligible, the one evidence-backed A4 route of p5-rwgate is retained
             (strong_noise → A4, synthetic Phase 4 evidence); everything else → ORIGINAL. The rwgate strong_blur → A4
             route is NOT used in auto: R5-ARCHIVE measured no blur benefit for A4. Never calls both models.
  force_a4 (research) always takes the a4 path.
Outputs
  ENHANCED   A4: the A4 face chip, 224×224 (E-frame), PNG · NAFNet: the deblurred face chip at its native size, PNG
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
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from fes_bench import gate as GT
from fes_bench import geometry as G
from fes_bench import quality as Q
from fes_bench import routing as RT
from fes_bench.evaluators import DLIB_CHIP_TEMPLATE_150
from fes_data import preprocess as P
from fes_data.common import repo_path
from fes_service import blur_scale as BS
from fes_service import nafnet_route as NR

SERVICE_VERSION = "fes-service-r5-0.2.0"
YUNET = "models/D01_yunet/face_detection_yunet_2026may.onnx"
CHIP_MAX = G.E_SIZE                                   # 224: the largest input size the gate was calibrated on
TEMPLATE_IOD_150 = float(np.linalg.norm(DLIB_CHIP_TEMPLATE_150[1] - DLIB_CHIP_TEMPLATE_150[0]))
QUALITY_KEYS = ("noise_sigma", "lap_norm_c", "lapd1_c", "blockiness", "luma_mean", "rms_contrast", "iod_px", "det_score",
                "is_gray")
NAFNET_QUALITY_KEYS = ("pixelation", "hf_peak_frac", "hf_energy")
ROUTE_MODES = ("a4", "nafnet", "auto")
A4_MODEL = "A4_SWINIR_L"
ARTIFACT_GUARD = "ARTIFACT_GUARD"                 # fallback code: the NAFNet output failed the artifact guard

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
    route_mode: str = "a4"                       # a4 | nafnet | auto (R5-5)
    model: str | None = None                     # model INVOKED: A4_SWINIR_L | NAFNET_GOPRO | NAFNET_FACE_FT | None
    artifact_check: str = "NOT_RUN"              # NAFNet artifact guard: PASS | FAIL | NOT_RUN
    nafnet: dict | None = None                   # NAFNet route record (eligibility, reason, guard result)
    profile: str = ""                            # R5-9: the explicitly requested profile ("" = service default)
    selection: dict | None = None                # R6-BEST (profile "best"): candidates, scores and the one chosen

    def metadata(self) -> dict:
        """Structured, non-sensitive metadata: no landmarks, boxes, embeddings or pixels."""
        extra = {"profile": self.profile} if self.profile else {}     # default requests: metadata exactly as before
        if self.selection is not None:
            extra["selection"] = self.selection
        return {**extra, "service_version": SERVICE_VERSION, "request_id": self.request_id, "status": self.status,
                "input_mode": self.input_mode, "enhanced": self.status == ENHANCED,
                "output": ("nafnet_face_chip_native" if self.used == NR.MODEL else "a4_face_chip_224")
                          if self.status == ENHANCED else ("original_input" if self.status == ORIGINAL_S else "none"),
                "decision": self.decision, "reason": self.reason, "bucket": self.bucket, "route": self.route,
                "used": self.used, "fallback": self.fallback, "escalate": self.escalate, "error": self.error,
                "warnings": self.warnings, "n_faces": self.n_faces, "chip_px": self.chip_px,
                "quality": self.quality, "timings_ms": self.timings_ms, "forced": self.forced,
                "gate_route": self.gate_route, "route_mode": self.route_mode, "model": self.model,
                "artifact_check": self.artifact_check, "nafnet": self.nafnet}

    def validation_record(self, input_id: str) -> dict:
        """Flat per-input row for service-level validation (R5-6): no pixels, no landmarks."""
        nf = self.nafnet or {}
        art = nf.get("artifact") or {}
        return {"input_id": input_id, "service_version": SERVICE_VERSION, "route_mode": self.route_mode,
                "status": self.status, "decision": self.decision, "reason": self.reason, "route": self.route,
                "used": self.used, "model": self.model or "", "fallback": self.fallback, "error": self.error,
                "nafnet_eligible": nf.get("eligible"), "nafnet_reason": nf.get("reason", ""),
                "nafnet_infer_s": nf.get("infer_s"), "artifact_check": self.artifact_check,
                "artifact_reasons": "+".join(art.get("reasons", [])),
                **{f"artifact_{k}": v for k, v in (art.get("metrics") or {}).items()},
                "output_valid": self.status == ENHANCED or (self.status == ORIGINAL_S and self.image is not None),
                "n_faces": self.n_faces, "chip_px": self.chip_px, "escalate": self.escalate,
                **{f"q_{k}": v for k, v in self.quality.items()}, "total_ms": self.timings_ms.get("total")}


def _json_num(v):
    """NaN/inf (e.g. no detection score on an UNSAFE input) → None: JSON has no NaN."""
    return None if isinstance(v, float) and not np.isfinite(v) else v


class Busy(Exception):
    pass


def face_chip(img: np.ndarray, landmarks: np.ndarray, border: int = cv2.BORDER_CONSTANT) -> tuple[np.ndarray, int]:
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
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=(0, 0, 0)) if pad else img
    chip = src[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    if side > CHIP_MAX:
        chip = cv2.resize(chip, (CHIP_MAX, CHIP_MAX), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(chip), side


def face_chip_filled(img: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R5-8 candidate profile only: the SAME chip as face_chip (same framing, same pixels where the image exists) with
    its out-of-image area filled by reflecting the image instead of black, plus that area's mask (chip coordinates).
    R5-7 DEV: the black fill multiplied lapd1 ×4 and lost 95 % of the affected NAFNet-eligible faces."""
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


# r5_6 = live/default; r5_8 / r5_10 / r6 / r6_ft = research candidates (opt-in)
NAFNET_PROFILES = ("r5_6", "r5_8", "r5_10", "r6", "r6_ft")
# route config per NAFNet profile (each file is SHA-256 sealed; load_config verifies it). r6_ft = the r6 route
# unchanged (same gate, stability check and guards) with ONE change: the model's weights (models:nafnet_factory_r6_ft)
PROFILE_CONFIG = {"r5_6": NR.CONFIG, "r5_8": NR.CANDIDATE_CONFIG, "r5_10": NR.R5_10_CONFIG, "r6": NR.R6_CONFIG,
                  "r6_ft": NR.R6_CONFIG}
EXPERIMENTAL_PROFILES = ("r5_8", "r5_10", "r6", "r6_ft")
PROFILE_MODEL = {"r6_ft": "NAFNET_FACE_FT"}        # model INVOKED when it is not NR.MODEL (models.FACE_FT_MODEL)
STABILITY_GUARD = "STABILITY_GUARD"                # fallback code: R6 — a mirror run of NAFNet was unstable
LOCAL_ARTIFACT_GUARD = "LOCAL_ARTIFACT_GUARD"     # fallback code: the candidate's local-patch guard rejected the output
# R5-9 per-request profiles: "original" = input returned unchanged, no model invoked (comparison baseline);
# r5_6 / r5_8 / r5_10 = the NAFNet profiles above. r5_8 / r5_10 are servable per request ONLY when the pipeline was
# built with that experimental model (app: FES_EXPERIMENTAL_PROFILES=r5_8,r5_10); a request without `profile` = the
# service default.
PROFILE_ORIGINAL = "original"
# R6-BEST per-request profile "best": 1 input → 1 output. Every BEST_CANDIDATES profile runs its full guarded route on
# the same face; the output chosen by nafnet_route.select_best is returned. Servable only when all of them are enabled.
PROFILE_BEST = "best"
BEST_CANDIDATES = ("r6", "r6_ft")
# SIDE-FACE research profile "best_side" (opt-in per request, full_image only): exactly profile "best" for a face the
# landmark framing accepts. Only where "best" stops because the face is turned — the detector's landmarks are not
# usable (FACE_TOO_SMALL / alignment_unreliable) or the gate answers NAFNET_REJECT_UNSAFE_ALIGNMENT — the face is
# framed by its DETECTION BOX instead (face_chip_box) and the same "best" route runs on that chip: same blur / noise /
# compression rules and thresholds, same models, stability check, guards and selection. Two things replace what
# needs two visible eyes: the safety rule (NR.box_safety: confident detection in the full image) and the face-size
# feature (the chip's template-equivalent eye distance). Evidence: experiments on the Indian-expression benchmark
# (2026-10-13: 641 blurred side faces, synthetic level-3 blur). Not production-approved; identity on profile faces
# is not measured (SFace is not reliable there).
PROFILE_BEST_SIDE = "best_side"
BEST_PROFILES = (PROFILE_BEST, PROFILE_BEST_SIDE)
# chip side / max(box width, box height): the median over 80 frontal references of that benchmark, so a box-framed
# chip shows a face at the scale the models and the gate see in the landmark framing
SIDE_BOX_SCALE = 0.982
REQUEST_PROFILES = (PROFILE_ORIGINAL,) + NAFNET_PROFILES + BEST_PROFILES
# admin blur range (fes_service.blur_scale): the scale is calibrated against the r6 route's blur limits, so it applies
# to the r6-based profiles only (r6, r6_ft and therefore best / best_side, /v2, /v3); r5_6 / r5_8 / r5_10 are unchanged
BLUR_RANGE_PROFILES = ("r6", "r6_ft")


@dataclass(frozen=True)
class _NafnetProfile:
    name: str
    cfg: dict
    model: object
    oob_fill: bool


class EnhancementPipeline:
    """Thread-safe; one request is processed at a time (YuNet and the router are stateful), at most `max_inflight`
    requests may wait — beyond that `Busy` is raised (HTTP 503) instead of queueing without bound."""

    def __init__(self, a4, timeout_s: float = 120.0, limits: Limits = Limits(), max_inflight: int = 4,
                 gate: str = "t6", nafnet=None, route_mode: str = "a4", nafnet_profile: str = "r5_6",
                 experimental_nafnet: dict | None = None, best_parallel: bool = False,
                 blur_settings: BS.BlurSettings | None = None, blur_settings_source: str = "default"):
        """gate="t6": the frozen Phase 4 T6/T7 policy, exactly (synthetic-calibrated).
        gate="rw": the R5-3 real-world gate (fes_service.rwgate, configs/phase5/r5_rwgate.json).
        nafnet: callable(chip, ctx) -> (same-size chip, meta), normally an IsolatedModel(nafnet_factory). Route modes
        nafnet / auto need it and gate="rw" (they use the rw features and, in auto, its noise route).
        nafnet_profile: "r5_6" (default, the validated live route) | "r5_8" (R5-8 research candidate: candidate route
        config with the local-patch guard, out-of-image reflect fill; pair it with models:nafnet_factory_pad16) |
        "r5_10" (R5-10 research candidate: r5_8 with the DEV-recalibrated blur boundary; same model factory) |
        "r6" (R6 research candidate: the r5_8 route + 4-run stability check + median/luma-only consensus output;
        pair it with models:nafnet_factory_r6) | "r6_ft" (R6-FT research candidate: r6 with the face-blur fine-tuned
        weights; pair it with models:nafnet_factory_r6_ft).
        experimental_nafnet: R5-9 — {"r5_8": model, "r5_10": model} makes those candidates selectable PER REQUEST
        (profile="r5_8" / "r5_10") next to the default profile r5_6; it never changes what a request without
        `profile` does.
        best_parallel: PERF Step 1 (opt-in, default off = the sequential behaviour) — profile "best" starts the model
        inference of all its candidates at the same moment, each in its own isolated worker, instead of one after
        the other. Gate, stability check, guards, selection and every fallback are the same code on the same inputs;
        see _parallel_infer.
        blur_settings: the deployment's admin blur range (fes_service.blur_scale; None = both off = unchanged
        behaviour). Validated here: an invalid value raises ValueError (the service does not become ready)."""
        assert gate in ("t6", "rw"), gate
        self.blur_scale, self.blur_settings = None, BS.BlurSettings()
        self.blur_settings_source = blur_settings_source
        self._req_blur: tuple[BS.BlurSettings, dict] | None = None    # the current request's range (under the lock)
        self.best_parallel = bool(best_parallel)
        self.side_cfg = None
        assert nafnet_profile in NAFNET_PROFILES, nafnet_profile
        self.nafnet_profile = nafnet_profile
        assert route_mode in ROUTE_MODES, route_mode
        if route_mode != "a4":
            assert gate == "rw" and nafnet is not None, f"route_mode={route_mode} needs gate='rw' and a NAFNet model"
        self.gate, self.route_mode, self.nafnet = gate, route_mode, nafnet
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
            self.nafnet_cfg = NR.load_config(PROFILE_CONFIG[nafnet_profile])
            self.seals["nafnet_route"] = self.nafnet_cfg["sha256"]
            self.oob_fill = bool(self.nafnet_cfg.get("framing", {}).get("oob_reflect_fill", False))
            if repo_path(NR.SIDE_CONFIG).exists():              # profile "best_side" is servable only with its seal
                self.side_cfg = NR.load_side_config()
                self.seals["side_face_route"] = self.side_cfg["sha256"]
            if repo_path(BS.SCALE_CONFIG).exists() and self.side_cfg is not None:
                self.blur_scale = BS.load_scale()
                self.seals["blur_scale"] = self.blur_scale.cfg["sha256"]
                self.blur_settings = BS.validate(blur_settings or BS.BlurSettings(), self.blur_scale)
        else:
            self.oob_fill = False
        if (blur_settings or BS.BlurSettings()).active and self.blur_scale is None:
            raise ValueError("blur settings need gate 'rw' and the sealed " + BS.SCALE_CONFIG)
        self.profiles: dict[str, _NafnetProfile] = {}
        if gate == "rw" and nafnet is not None:
            self.profiles[nafnet_profile] = _NafnetProfile(nafnet_profile, self.nafnet_cfg, nafnet, self.oob_fill)
        for name, model in (experimental_nafnet or {}).items():
            assert name in EXPERIMENTAL_PROFILES and nafnet_profile == "r5_6", \
                "only r5_8 / r5_10 / r6 / r6_ft can be added next to the default r5_6"
            assert gate == "rw" and nafnet is not None and model is not None, f"experimental {name} needs gate 'rw'"
            cfg = NR.load_config(PROFILE_CONFIG[name])                  # verifies the candidate seal
            self.profiles[name] = _NafnetProfile(name, cfg, model,
                                                 bool(cfg.get("framing", {}).get("oob_reflect_fill", False)))
            self.seals[f"nafnet_route_{name}"] = cfg["sha256"]
        self.det = P.YuNetDetector(repo_path(YUNET), 0.7, 0.3, max_side=640)
        self.det_cfg = P.PreprocessConfig(out_size=112)
        self.a4 = a4
        self.limits = limits
        self._feat = {}
        # the router's own thread timeout is a backstop; the isolated worker enforces (and kills at) timeout_s
        face_check = lambda out: len(self.det.detect(out)) > 0  # noqa: E731
        self._face_check = face_check
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
    def process(self, data: bytes, input_mode: str = "full_image", force_a4: bool = False,
                route_mode: str | None = None, profile: str | None = None,
                blur: BS.BlurSettings | None = None) -> Result:
        """route_mode None = the pipeline's configured mode. profile None/"" = the service default (R5-9: "original" |
        "r5_6" | "r5_8" | "r5_10"; an unknown or not-enabled profile is REJECTED, never silently replaced).
        blur: per-request admin blur range; it may only narrow the deployment's (otherwise REJECTED)."""
        if not self._slots.acquire(blocking=False):
            raise Busy("too many requests in flight")
        try:
            with self._lock:
                self._req_blur = None
                try:
                    self._req_blur = self._request_blur(blur)
                    target = BEST_CANDIDATES[0] if profile in BEST_PROFILES else profile or self.nafnet_profile
                    if blur is not None and blur.active and target not in BLUR_RANGE_PROFILES + (PROFILE_ORIGINAL,):
                        raise ValueError("blur settings apply to profiles "
                                         f"{', '.join(BLUR_RANGE_PROFILES + BEST_PROFILES)} only (this request: "
                                         f"{target!r})")
                except ValueError as e:
                    res = Result(status=REJECTED, request_id=uuid.uuid4().hex, input_mode=input_mode,
                                 route_mode=route_mode or self.route_mode, profile=profile or "", error=str(e))
                    return res
                return self._process(data, input_mode, force_a4, route_mode or self.route_mode, profile or "")
        finally:
            self._slots.release()

    def _request_blur(self, blur: BS.BlurSettings | None) -> tuple[BS.BlurSettings, dict] | None:
        """→ (effective blur range, where each value came from), or None when no blur scale is loaded."""
        if self.blur_scale is None:
            if blur is not None and blur.active:
                raise ValueError("blur settings are not available on this service (gate 'rw' + "
                                 f"{BS.SCALE_CONFIG} needed)")
            return None
        req = blur or BS.BlurSettings()
        eff = BS.validate(req, self.blur_scale, base=self.blur_settings)
        def src(r, d, unset):
            return "request" if r is not None else "deployment" if d is not None else unset
        return eff, {"enhance_from": src(req.enhance_from_pct, self.blur_settings.enhance_from_pct, "validated_limit"),
                     "give_up_above": src(req.give_up_above_pct, self.blur_settings.give_up_above_pct, "off")}

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

    def _process(self, data: bytes, input_mode: str, force_a4: bool = False, route_mode: str = "a4",
                 profile: str = "") -> Result:
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
        res = Result(status=REJECTED, request_id=uuid.uuid4().hex, input_mode=input_mode, route_mode=route_mode,
                     profile=profile)

        def original(**kw) -> Result:
            res.status, res.image, res.image_format = ORIGINAL_S, data, "as_received"
            for k, v in kw.items():
                setattr(res, k, v)
            res.timings_ms["total"] = ms()
            return res

        fill = None                         # (reflect-filled chip, out-of-image mask): r5_8 / r5_10 profiles only
        if input_mode not in ("full_image", "face_chip"):
            res.error = f"unknown input_mode {input_mode!r}"
            return res
        if route_mode not in ROUTE_MODES:
            res.error = f"unknown route_mode {route_mode!r}"
            return res
        if route_mode != "a4" and not force_a4 and (self.gate != "rw" or self.nafnet is None):
            res.error = f"route_mode {route_mode!r} is not configured on this service (needs gate 'rw' + NAFNet)"
            return res
        if profile and profile not in REQUEST_PROFILES:
            res.error = f"unknown profile {profile!r} (expected one of {', '.join(REQUEST_PROFILES)})"
            return res
        if profile in NAFNET_PROFILES and profile not in self.profiles:
            res.error = (f"profile {profile!r} is not enabled on this service"
                         + (f" (start it with FES_EXPERIMENTAL_PROFILES={profile})"
                            if profile in EXPERIMENTAL_PROFILES else ""))
            return res
        if profile in BEST_PROFILES and not all(p in self.profiles for p in BEST_CANDIDATES):
            res.error = (f"profile {profile!r} is not enabled on this service"
                         f" (start it with FES_EXPERIMENTAL_PROFILES={','.join(BEST_CANDIDATES)})")
            return res
        if profile == PROFILE_BEST_SIDE and self.side_cfg is None:
            res.error = f"profile {profile!r} is not enabled on this service (no {NR.SIDE_CONFIG})"
            return res
        if profile and force_a4:
            res.error = "force_a4 cannot be combined with an explicit profile"
            return res
        prof = self.profiles.get(BEST_CANDIDATES[0] if profile in BEST_PROFILES else profile or self.nafnet_profile)
        side = profile == PROFILE_BEST_SIDE and route_mode != "a4" and input_mode == "full_image"
        det = None                          # the full-image detection (profile "best_side": its box frames a side face)
        img, err = self._decode(data)
        res.timings_ms["decode"] = ms()
        if img is None:
            res.error = err
            res.timings_ms["total"] = ms()
            return res
        if profile == PROFILE_ORIGINAL:                  # comparison baseline: no detection, no gate, no model
            return original(reason="profile_original")

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
                    return self._side_path(res, data, img, det, route_mode, ms, f"no_valid_face:{fr.status}")
                return original(decision=GT.UNSAFE, reason=f"no_valid_face:{fr.status}", escalate=True)
            lm = np.asarray(fr.detection.landmarks, np.float64)
            if not fr.aligned or not P.landmarks_plausible(lm):
                if side:
                    return self._side_path(res, data, img, det, route_mode, ms, "alignment_unreliable:landmarks")
                return original(decision=GT.UNSAFE, reason="alignment_unreliable:landmarks", escalate=True)
            chip, _ = face_chip(img, lm)
            if prof is not None and prof.oob_fill and route_mode != "a4" and not force_a4:
                filled, oob = face_chip_filled(img, lm)
                if oob.any():
                    fill = (filled, oob)
        res.chip_px = int(chip.shape[0])
        ok, chip_png = cv2.imencode(".png", chip)
        res.chip = chip_png.tobytes() if ok else None
        if route_mode == "a4" or force_a4:
            return self._a4_path(res, chip, force_a4, original, ms)
        if profile in BEST_PROFILES:
            out = self._best_path(res, data, chip, route_mode, ms, fill)
            if (side and det is not None and out.status == ORIGINAL_S
                    and (out.nafnet or {}).get("reason") == NR.R_ALIGN):     # turned face: the gate's pose rule said no
                return self._side_path(res, data, img, det, route_mode, ms, NR.R_ALIGN)
            return out
        return self._nafnet_path(res, chip, route_mode, original, ms, fill, prof)

    def _side_path(self, res: Result, data: bytes, img: np.ndarray, det, route_mode: str, ms, why: str) -> Result:
        """Profile "best_side": the landmark framing stopped at `why`; frame the face by its detection box and run
        the unchanged "best" route on that chip (see PROFILE_BEST_SIDE). Out-of-image area as in the r6 profiles:
        reflect-filled for the gate and the model, black in the returned chip."""
        res = replace(res, warnings=[*res.warnings, "side_face_box_framing"], timings_ms=dict(res.timings_ms))
        chip, _ = face_chip_box(img, det.bbox_xywh)
        oob = face_chip_box(np.full(img.shape[:2] + (3,), 255, np.uint8), det.bbox_xywh)[0][..., 0] < 128
        fill = (face_chip_box(img, det.bbox_xywh, cv2.BORDER_REFLECT_101)[0], oob) if oob.any() else None
        res.chip_px = int(chip.shape[0])
        ok, chip_png = cv2.imencode(".png", chip)
        res.chip = chip_png.tobytes() if ok else None
        box = {"landmark_framing": why, "det_score": float(det.score),
               "iod_equiv_px": round(chip.shape[0] * TEMPLATE_IOD_150 / 150.0, 3)}
        return self._best_path(res, data, chip, route_mode, ms, fill, box)

    def _best_path(self, res: Result, data: bytes, chip: np.ndarray, route_mode: str, ms, fill=None,
                   box: dict | None = None) -> Result:
        """R6-BEST (profile "best"): run every BEST_CANDIDATES profile's unchanged guarded route on the same chip and
        return ONE result — the accepted output NR.select_best prefers, or the original when no candidate was accepted.
        The candidates share one sealed input gate, so a face the first one finds not eligible (ORIGINAL, or the
        retained A4 strong-noise route in auto mode) is final and the others are not run."""
        subs: dict[str, Result] = {}
        infer_for = self._parallel_infer() if self.best_parallel else None
        for name in BEST_CANDIDATES:
            sub = replace(res, warnings=list(res.warnings), timings_ms=dict(res.timings_ms))

            def sub_original(_s=sub, **kw) -> Result:
                _s.status, _s.image, _s.image_format = ORIGINAL_S, data, "as_received"
                for k, v in kw.items():
                    setattr(_s, k, v)
                _s.timings_ms["total"] = ms()
                return _s
            self._nafnet_path(sub, chip, route_mode if not subs else "nafnet", sub_original, ms, fill,
                              self.profiles[name], infer_for(name) if infer_for else None, box)
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
        """PERF Step 1: → infer_for(name) → infer(work, ctx), a drop-in for `profile.model(work, ctx)` in _nafnet_path.
        Nothing runs until the FIRST candidate reaches its model call (so a face the gate rejects never starts a
        model). At that moment the model of EVERY candidate is started on that same input, each in its own
        IsolatedModel (own process, own lock, own timeout: a hang, crash or error stays with that worker and surfaces
        as the same exception, to the same candidate, as in the sequential path). Each candidate then collects ITS
        result. A later candidate whose input is not the very same one is run directly, as before."""
        from concurrent.futures import ThreadPoolExecutor
        state: dict = {}

        def infer_for(name):
            def infer(work, ctx):
                if not state:
                    models = {n: self.profiles[n].model for n in BEST_CANDIDATES}
                    pool = ThreadPoolExecutor(len(models), thread_name_prefix="fes-best")
                    state.update(work=work, ctx=dict(ctx),
                                 fut={n: pool.submit(m, work, dict(ctx)) for n, m in models.items()})
                    pool.shutdown(wait=False)
                first = state["ctx"]
                same = work is state["work"] and ctx.keys() == first.keys() and all(
                    ctx[k] is first[k] or (not isinstance(ctx[k], np.ndarray) and ctx[k] == first[k]) for k in ctx)
                if not same or name not in state["fut"]:
                    return self.profiles[name].model(work, ctx)
                return state["fut"].pop(name).result()
            return infer
        return infer_for

    def _a4_path(self, res: Result, chip: np.ndarray, force_a4: bool, original, ms) -> Result:
        """The R5-1/R5-3 behaviour, unchanged: frozen T6 → T7 (or rwgate) → ORIGINAL | A4, with the router's
        fallbacks and output guard."""
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
        res.model = A4_MODEL if rec.route == "A4" else None
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

    def _nafnet_path(self, res: Result, chip: np.ndarray, route_mode: str, original, ms, fill=None,
                     prof: _NafnetProfile | None = None, infer=None, box: dict | None = None) -> Result:
        """R5-5 guarded NAFNet route: input gate → NAFNet (isolated worker) → artifact guard; ORIGINAL otherwise.
        r5_8 / r5_10 candidate profiles: `fill` = (reflect-filled chip, out-of-image mask) when the full_image chip extends past
        the image (gate + NAFNet see the filled chip, the output's out-of-image area is set back to black), and the
        local-patch guard runs after the artifact guard."""
        from fes_service import rwgate as RW
        prof = prof or self.profiles[self.nafnet_profile]      # R5-9: the request's profile (default r5_6)
        cfg = prof.cfg
        work = fill[0] if fill is not None else chip
        try:
            f = self._features(work)                              # rw features (p5-rwfeat-1.0.0)
            f.update(NR.texture_features(work))                   # p5-nafnet-feat-1.0.0
        except Exception as e:  # noqa: BLE001 — feature extraction failed: never enhance
            return original(decision=GT.UNSAFE, reason="quality_analysis_failed", escalate=True,
                            error=f"{type(e).__name__}: {e}"[:200])
        if box is not None:     # side-face framing: the box's detection score and the chip's template-equivalent IOD
            f = {**f, "det_score": box["det_score"], "iod_px": box["iod_equiv_px"]}
        res.quality = {k: _json_num(f.get(k)) for k in QUALITY_KEYS + NAFNET_QUALITY_KEYS}
        gate_cfg = cfg if box is None else {         # box framing: its own sealed blur boundary
            **cfg, "thresholds": {**cfg["thresholds"], **self.side_cfg["thresholds"]}}
        br = None                                    # admin blur range: the face's blur % and the range applied
        if self._req_blur is not None and prof.name in BLUR_RANGE_PROFILES and not NR._nan(f.get("lapd1_c")):
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
        if prof.name != "r5_6":
            res.nafnet.update(profile=prof.name, local_artifact=None,
                              oob_fill_frac=round(float(fill[1].mean()), 4) if fill is not None else 0.0)
        res.timings_ms["gate"] = ms()
        if not nd.eligible:
            if route_mode == "auto" and box is None:              # the A4 route keeps its own (landmark) safety rules
                fa = f if fill is None else self._features(chip)  # the A4 route always decides on the unchanged chip
                _, why, _, route = RW.decide(fa, self.rw_cfg)
                if route == "A4" and why == "strong_noise":       # the retained, evidence-backed A4 route
                    return self._a4_path(res, chip, False, original, ms)
            res.gate_route = RT.ORIGINAL
            return original(decision=nd.decision, reason=nd.reason, bucket=nd.route, route=RT.ORIGINAL,
                            escalate=nd.decision == GT.UNSAFE)
        res.decision, res.reason, res.bucket = nd.decision, nd.reason, nd.route
        res.route = res.gate_route = res.model = NR.MODEL
        res.model = PROFILE_MODEL.get(prof.name, NR.MODEL)       # r6_ft: same route, fine-tuned weights
        try:
            ctx = {"s": work.shape[0]}
            if "stability" in cfg and fill is not None:          # r6: out-of-image area excluded from the tracer
                ctx["oob_mask"] = fill[1]
            out, meta = (infer or prof.model)(work, ctx)          # infer: PERF Step 1 (profile "best" only)
            res.nafnet["infer_s"] = (meta or {}).get("infer_s")
            if (meta or {}).get("backend"):                       # R6-ONNX: reported only by a non-default backend
                res.nafnet["backend"] = meta["backend"]
            if fill is not None and isinstance(out, np.ndarray) and out.shape[:2] == fill[1].shape:
                out = out.copy()
                out[fill[1]] = 0                                  # out-of-image area back to black: geometry unchanged
        except Exception as e:  # noqa: BLE001 — timeout / OOM / crash / load failure: always the original
            fb = ("TIMEOUT" if isinstance(e, TimeoutError) else
                  "RESOURCE_FAILURE" if RT._is_oom(e) else "MODEL_FAILURE")
            res.timings_ms["nafnet"] = ms()
            return original(used=RT.ORIGINAL, fallback=fb, error=f"{type(e).__name__}: {e}"[:300])
        res.timings_ms["nafnet"] = ms()
        if "stability" in cfg:                            # r6 candidate: 4-run stability check before any other guard
            try:
                sg = NR.stability_check((meta or {}).get("runs_dot_c"), cfg)
            except Exception as e:  # noqa: BLE001 — an uncertain stability result counts as a failed one
                sg = NR.ArtifactGuardResult(False, [NR.A_UNSTABLE_ERROR], {"error": f"{type(e).__name__}: {e}"[:200]})
            res.nafnet["stability"] = sg.as_dict()
            if not sg.passed:
                res.artifact_check = "FAIL"
                return original(used=RT.ORIGINAL, fallback=STABILITY_GUARD, error="+".join(sg.reasons))
        try:
            g = NR.artifact_guard(chip, out, cfg, self._face_check, bool(f.get("is_gray")))
        except Exception as e:  # noqa: BLE001 — an uncertain safety result counts as a failed one
            g = NR.ArtifactGuardResult(False, [NR.A_NUMERIC], {"guard_error": f"{type(e).__name__}: {e}"[:200]})
        res.timings_ms["guard"] = ms()
        res.nafnet["artifact"] = g.as_dict()
        res.artifact_check = "PASS" if g.passed else "FAIL"
        if not g.passed:
            return original(used=RT.ORIGINAL, fallback=ARTIFACT_GUARD, error="+".join(g.reasons))
        if "local_artifact" in cfg:                       # r5_8 / r5_10 candidates: local-patch guard
            try:
                lg = NR.local_artifact_guard(chip, out, cfg, bool(f.get("is_gray")))
            except Exception as e:  # noqa: BLE001 — detector failure counts as a rejection
                lg = NR.ArtifactGuardResult(False, [NR.A_LOCAL_ERROR], {"error": f"{type(e).__name__}: {e}"[:200]})
            res.timings_ms["local_guard"] = ms()
            res.nafnet["local_artifact"] = lg.as_dict()
            if not lg.passed:
                res.artifact_check = "FAIL"
                return original(used=RT.ORIGINAL, fallback=LOCAL_ARTIFACT_GUARD, error="+".join(lg.reasons))
        if f.get("is_gray"):
            # greyscale in → greyscale out (the guard has already rejected any STRONG invented colour)
            out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            res.warnings.append("grayscale_preserved")
        ok, png = cv2.imencode(".png", out)
        if not ok:
            return original(used=RT.ORIGINAL, fallback="INVALID_OUTPUT", error="png encode failed")
        res.used = NR.MODEL
        res.status, res.image, res.image_format = ENHANCED, png.tobytes(), "png"
        res.timings_ms["total"] = ms()
        return res
