"""Face safety rule and decision vocabulary of the input gate, and the SHA-256 seal of the frozen config files.

UNSAFE_OR_UNCERTAIN — fixed safety rules, NOT calibrated (P2-T6 constants):
  * the detection contract did not return an aligned face (NO_FACE_DETECTED, FACE_TOO_SMALL, ...) → 'no_valid_face'
  * detector confidence < 0.85                                                                  → 'low_detection_confidence'
  * |yaw proxy| > 0.25 or implausible landmarks (NaN pose)                                        → 'alignment_unreliable'
The service returns the ORIGINAL and flags/escalates; it never enhances such a face.
"""

from __future__ import annotations

import hashlib
import json
import math

LOW_CONF = 0.85          # preprocess.PreprocessConfig.low_conf_warn (P2-T6)
YAW_WARN = 0.25          # preprocess.PreprocessConfig.yaw_warn (P2-T6)

DNE, ENH, UNSAFE = "DO_NOT_ENHANCE", "ENHANCE_CANDIDATE", "UNSAFE_OR_UNCERTAIN"


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def safety(f: dict) -> str | None:
    """Fixed UNSAFE_OR_UNCERTAIN reasons (None = safe to consider)."""
    if f.get("det_status") != "OK":
        return f"no_valid_face:{f.get('det_status')}"
    if _nan(f.get("det_score")) or f["det_score"] < LOW_CONF:
        return "low_detection_confidence"
    if _nan(f.get("yaw_proxy")) or abs(f["yaw_proxy"]) > YAW_WARN:
        return "alignment_unreliable"
    return None


def gate_hash(cfg: dict) -> str:
    """The seal of a frozen config file: SHA-256 of its canonical JSON without the `sha256` key."""
    body = {k: v for k, v in cfg.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
