"""Face detection, cropping and alignment — deterministic, traceable, detection kept separate from enhancement.

Behaviour contract (P0-T2 §2.6, P0 master C2, P1-T4 §6):
  * one face        → aligned crop, status OK
  * multiple faces  → policy "largest" (default) / "all" / "index"; n_faces recorded
  * no face         → status NO_FACE_DETECTED; nothing is invented
  * very small face → status FACE_TOO_SMALL (inter-ocular distance below the configured minimum)
  * partial face    → aligned crop + warning "partial_face" when the canonical window leaves the image
  * low confidence  → warning "low_confidence" (possible occlusion/profile; occlusion itself is not detectable)
  * face crop input → detection retried on a padded copy (YuNet misses faces touching the frame);
                      still nothing → NO_FACE_DETECTED (never restore a non-face);
                      face found but landmarks implausible → unaligned resize + warning "no_landmarks_found"
Every result carries detector id, score, bbox, landmarks, the similarity transform and the
pipeline version, so any crop can be re-derived from its source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

PREPROC_VERSION = "preproc-1.0.0"

# FFHQ 5-point template at 512x512 (image-left eye, image-right eye, nose tip, image-left mouth corner,
# image-right mouth corner) — the canonical alignment used by FFHQ-trained restoration models.
FFHQ_TEMPLATE_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118]], dtype=np.float64)
TEMPLATE_EYE_DIST_512 = float(np.linalg.norm(FFHQ_TEMPLATE_512[1] - FFHQ_TEMPLATE_512[0]))

# status codes
OK = "OK"
NO_FACE = "NO_FACE_DETECTED"
TOO_SMALL = "FACE_TOO_SMALL"
DECODE_ERROR = "DECODE_ERROR"
INVALID_INDEX = "INVALID_FACE_INDEX"


@dataclass
class Detection:
    bbox_xywh: list[float]          # in source-image pixel coordinates
    score: float
    landmarks: list[list[float]]    # 5x2, image-left eye, image-right eye, nose, mouth-left, mouth-right

    @property
    def area(self) -> float:
        return self.bbox_xywh[2] * self.bbox_xywh[3]

    @property
    def eye_dist(self) -> float:
        lm = np.asarray(self.landmarks)
        return float(np.linalg.norm(lm[1] - lm[0]))


@dataclass
class FaceResult:
    status: str
    warnings: list[str] = field(default_factory=list)
    n_faces: int = 0
    face_index: int | None = None
    detection: Detection | None = None
    aligned: bool = False
    transform_2x3: list[list[float]] | None = None   # source → canonical crop
    scale: float | None = None                        # canonical px per source px
    roll_deg: float | None = None
    yaw_proxy: float | None = None                    # nose offset from eye midline / eye distance (≈0 frontal)
    oob_frac: float | None = None                     # fraction of crop pixels outside the source image
    crop: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def record(self) -> dict:
        """JSON/CSV-friendly metadata (no pixels)."""
        d = {k: v for k, v in asdict(self).items() if k not in ("crop", "detection")}
        if self.detection is not None:
            d.update(det_bbox_xywh=[round(x, 2) for x in self.detection.bbox_xywh],
                     det_score=round(self.detection.score, 4),
                     det_landmarks=[[round(x, 2) for x in p] for p in self.detection.landmarks],
                     eye_dist_src=round(self.detection.eye_dist, 3))
        return d


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Closed-form least-squares similarity (rotation, uniform scale, translation) src→dst. Deterministic."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1, 1] = -1
    r = u @ d @ vt
    var_s = (s_c ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(sig) @ d) / var_s)
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]])


class YuNetDetector:
    """YuNet (OpenCV Zoo, MIT) wrapper. Detection runs on a copy downscaled to `max_side`."""

    def __init__(self, model_path: str | Path, score_threshold: float = 0.7, nms_threshold: float = 0.3,
                 top_k: int = 50, max_side: int = 640):
        self.model_path = str(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.max_side = max_side
        self._net = cv2.FaceDetectorYN.create(self.model_path, "", (320, 320), score_threshold, nms_threshold, top_k)
        self.id = f"yunet:{Path(self.model_path).name}:thr{score_threshold}:nms{nms_threshold}:max{max_side}"

    def detect(self, img_bgr: np.ndarray, score_threshold: float | None = None) -> list[Detection]:
        h, w = img_bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        img = img_bgr if s == 1.0 else cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        self._net.setInputSize((img.shape[1], img.shape[0]))
        self._net.setScoreThreshold(self.score_threshold if score_threshold is None else score_threshold)
        _, faces = self._net.detect(img)
        out: list[Detection] = []
        if faces is None:
            return out
        for f in faces:
            f = f.astype(np.float64) / s
            f[14] *= s  # score is not a coordinate
            lm = f[4:14].reshape(5, 2)
            out.append(Detection(bbox_xywh=f[:4].tolist(), score=float(f[14]), landmarks=lm.tolist()))
        # deterministic order: largest first, then score, then position
        out.sort(key=lambda d: (-d.area, -d.score, d.bbox_xywh[0], d.bbox_xywh[1]))
        return out


def landmarks_plausible(lm: np.ndarray) -> bool:
    """Basic geometric sanity: eyes separated left→right, nose and mouth below the eye line."""
    (le, re, nose, ml, mr) = lm
    eye_d = np.linalg.norm(re - le)
    if eye_d < 1.0 or re[0] <= le[0]:
        return False
    eye_mid_y = (le[1] + re[1]) / 2
    mouth_mid_y = (ml[1] + mr[1]) / 2
    return bool(nose[1] > eye_mid_y - 0.1 * eye_d and mouth_mid_y > nose[1] - 0.1 * eye_d)


def align_face(img_bgr: np.ndarray, landmarks: np.ndarray, out_size: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Warp to the FFHQ template at `out_size`. Returns (crop, M_src_to_crop 2x3, scale, oob_frac).

    Anti-aliased and non-sharpening: when the transform shrinks the image (scale < 1) the source is first
    reduced with INTER_AREA and the residual (≈unit-scale) similarity is applied with INTER_LINEAR.
    No kernels with negative lobes (cubic/Lanczos) are used, so no sharpening/overshoot is introduced.
    """
    template = FFHQ_TEMPLATE_512 * (out_size / 512.0)
    M = umeyama_similarity(landmarks, template)
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    src = img_bgr
    M_eff = M
    if scale < 1.0:
        h, w = img_bgr.shape[:2]
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        src = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        pre = np.array([[nw / w, 0, 0], [0, nh / h, 0], [0, 0, 1]], np.float64)
        M_eff = np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(pre)
        M_eff = M_eff[:2]
    crop = cv2.warpAffine(src, M_eff, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    ones = np.full(src.shape[:2], 255, np.uint8)
    valid = cv2.warpAffine(ones, M_eff, (out_size, out_size), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    oob = float((valid == 0).mean())
    return crop, M, scale, oob


def canonical_transform(landmarks: np.ndarray, out_size: int, src_shape: tuple[int, int]) -> tuple[np.ndarray, float, float]:
    """FFHQ-template similarity for `landmarks` at `out_size` WITHOUT warping pixels.
    Returns (M 2x3, scale, oob_frac) — used to record how a reference maps into FFHQ framing."""
    template = FFHQ_TEMPLATE_512 * (out_size / 512.0)
    M = umeyama_similarity(landmarks, template)
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    ones = np.full(src_shape[:2], 255, np.uint8)
    valid = cv2.warpAffine(ones, M, (out_size, out_size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return M, scale, float((valid == 0).mean())


def pose_proxies(lm: np.ndarray) -> tuple[float, float]:
    """Roll (degrees) from the eye line; yaw proxy = horizontal nose offset from the eye midpoint / eye distance."""
    le, re, nose = lm[0], lm[1], lm[2]
    roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    eye_mid = (le + re) / 2
    eye_vec = (re - le) / np.linalg.norm(re - le)
    yaw = float(np.dot(nose - eye_mid, eye_vec) / np.linalg.norm(re - le))
    return roll, yaw


@dataclass
class PreprocessConfig:
    out_size: int = 256
    selection: str = "largest"          # largest | all | index
    face_index: int = 0
    min_eye_dist_px: float = 12.0       # below → FACE_TOO_SMALL
    low_conf_warn: float = 0.85         # score below → warning low_confidence
    partial_oob_warn: float = 0.02      # oob fraction above → warning partial_face
    yaw_warn: float = 0.25              # |yaw proxy| above → warning unusual_pose
    crop_pad_frac: float = 0.5          # padding added around a face_crop input before re-detection
    crop_relaxed_threshold: float = 0.5


def preprocess(img_bgr: np.ndarray | None, detector: YuNetDetector, cfg: PreprocessConfig,
               input_mode: str = "full_image") -> list[FaceResult]:
    """Run the full detection → selection → alignment contract. Returns one FaceResult per selected face
    (or a single failure result)."""
    base_meta = {"preproc_version": PREPROC_VERSION, "detector": detector.id, "input_mode": input_mode,
                 "out_size": cfg.out_size, "selection": cfg.selection}
    if img_bgr is None:
        return [FaceResult(status=DECODE_ERROR, meta=base_meta)]

    offset = np.zeros(2)
    work = img_bgr
    dets = detector.detect(work)
    retried = False
    if not dets and input_mode == "face_crop":
        # faces filling the whole frame are often missed: pad (constant black), retry with relaxed threshold
        h, w = img_bgr.shape[:2]
        p = int(round(cfg.crop_pad_frac * max(h, w)))
        work = cv2.copyMakeBorder(img_bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        offset = np.array([p, p], np.float64)
        dets = detector.detect(work, score_threshold=cfg.crop_relaxed_threshold)
        retried = True
        # map back to original coordinates
        for d in dets:
            d.bbox_xywh[0] -= p
            d.bbox_xywh[1] -= p
            d.landmarks = (np.asarray(d.landmarks) - offset).tolist()
    base_meta["crop_retry"] = retried

    if not dets:
        return [FaceResult(status=NO_FACE, n_faces=0, meta=base_meta)]

    if cfg.selection == "largest":
        chosen = [0]
    elif cfg.selection == "all":
        chosen = list(range(len(dets)))
    elif cfg.selection == "index":
        if not 0 <= cfg.face_index < len(dets):
            return [FaceResult(status=INVALID_INDEX, n_faces=len(dets), meta=base_meta)]
        chosen = [cfg.face_index]
    else:
        raise ValueError(f"unknown selection {cfg.selection}")

    results = []
    for i in chosen:
        d = dets[i]
        lm = np.asarray(d.landmarks)
        r = FaceResult(status=OK, n_faces=len(dets), face_index=i, detection=d, meta=dict(base_meta))
        if len(dets) > 1:
            r.warnings.append("multiple_faces")
        if retried:
            r.warnings.append("detected_after_crop_retry")
        if d.score < cfg.low_conf_warn:
            r.warnings.append("low_confidence")
        if d.eye_dist < cfg.min_eye_dist_px:
            r.status = TOO_SMALL
            results.append(r)
            continue
        if not landmarks_plausible(lm):
            if input_mode == "face_crop":
                r.warnings.append("no_landmarks_found")
                r.crop = cv2.resize(img_bgr, (cfg.out_size, cfg.out_size), interpolation=cv2.INTER_AREA)
                r.aligned = False
            else:
                r.status = "LANDMARK_FAILURE"
            results.append(r)
            continue
        crop, M, scale, oob = align_face(img_bgr, lm, cfg.out_size)
        roll, yaw = pose_proxies(lm)
        r.crop, r.aligned = crop, True
        r.transform_2x3 = np.round(M, 8).tolist()
        r.scale, r.oob_frac = round(scale, 5), round(oob, 5)
        r.roll_deg, r.yaw_proxy = round(roll, 3), round(yaw, 4)
        if oob > cfg.partial_oob_warn:
            r.warnings.append("partial_face")
        if abs(yaw) > cfg.yaw_warn:
            r.warnings.append("unusual_pose")
        results.append(r)
    return results
