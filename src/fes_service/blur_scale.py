"""Admin blur range — the 0-100 % blur scale and the two settings that use it.

    blur scale   the gate's sharpness feature lapd1_c → blur percent, per chip framing (front = landmark framing,
                 side = detection-box framing, profile best_side). 0 % = sharp, 100 % = Gaussian σ 4.0 on an 800-px
                 photo; percent = 25 × σ. Sealed in configs/phase5/blur_scale.json
                 (scripts/blur_scale/calibrate_blur_scale.py). One percentage means the same amount of lost detail
                 on both framings, although their raw lapd1_c values differ (side chips read less sharp).
    enhance_from_pct     faces LESS blurred than this are returned unchanged (NAFNET_REJECT_MILD_BLUR).
                         null = the validated limits (front 0.008 / side 0.003 lapd1_c). It can only be STRICTER:
                         below MIN_ENHANCE_FROM sharp faces would start being "enhanced", so it is refused.
    give_up_above_pct    faces MORE blurred than this are returned unchanged (NAFNET_REJECT_TOO_BLURRED).
                         null = off (every heavily blurred face is attempted, the behaviour before this setting).

Where they come from: the deployment settings file (FES_SETTINGS, default configs/deployment_settings.json, read at
startup; a missing file = both null) and, optionally, per request — a request may only narrow the deployment range
(an enhance_from_pct below it or a give_up_above_pct above it is refused). With both null nothing changes: the gate
decides exactly as before. Every NAFNet response reports the face's blur percent and the range applied
(nafnet.blur_range).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from fes_data.common import repo_path

SCALE_VERSION = "p8-blur-scale-1.0.0"
SCALE_CONFIG = "configs/phase5/blur_scale.json"
SETTINGS_FILE = "configs/deployment_settings.json"
FRAMINGS = ("front", "side")
NOT_BLURRY_ENOUGH, IN_RANGE, TOO_BLURRED = "NOT_BLURRY_ENOUGH", "IN_RANGE", "TOO_BLURRED"


class BlurScale:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._pts = {}
        for framing in FRAMINGS:
            pts = [(math.log(p["lapd1_c"]), float(p["percent"])) for p in cfg["curves"][framing]]
            assert all(a[0] > b[0] and a[1] < b[1] for a, b in zip(pts, pts[1:])), f"{framing}: scale not monotone"
            self._pts[framing] = pts

    def percent(self, framing: str, lapd1_c: float) -> float:
        """Blur percent of a chip with sharpness lapd1_c (0 = sharp … 100 = σ 4.0 or blurrier)."""
        pts = self._pts[framing]
        if not (isinstance(lapd1_c, (int, float)) and math.isfinite(lapd1_c)) or lapd1_c <= 0:
            return 100.0 if lapd1_c == 0 else float("nan")
        x = math.log(lapd1_c)
        if x >= pts[0][0]:
            return pts[0][1]
        if x <= pts[-1][0]:
            return pts[-1][1]
        for (x0, p0), (x1, p1) in zip(pts, pts[1:]):
            if x1 <= x <= x0:
                return p0 + (p1 - p0) * (x0 - x) / (x0 - x1)
        raise AssertionError("unreachable")

    def floor(self) -> dict:
        """{framing: percent of the validated (sealed) blur limit}."""
        return {k: float(v["percent"]) for k, v in self.cfg["validated_floor"].items()}

    def min_enhance_from(self) -> float:
        """The lowest enhance_from_pct an admin may set: the stricter of the two validated limits, rounded up to a
        whole percent, so one number is valid (never looser than validated) on both framings."""
        return float(math.ceil(max(self.floor().values())))


def load_scale(path: str = SCALE_CONFIG) -> BlurScale:
    from fes_bench import gate as GT
    from fes_service import nafnet_route as NR
    cfg = json.loads(repo_path(path).read_text(encoding="utf-8"))
    assert cfg["sha256"] == GT.gate_hash(cfg), "blur_scale.json modified after freeze"
    assert cfg["scale_version"] == SCALE_VERSION, cfg["scale_version"]
    sealed = {"front": NR.load_config(NR.R6_CONFIG)["thresholds"]["blur_max"],
              "side": NR.load_side_config()["thresholds"]["blur_max"]}
    assert {k: v["blur_max"] for k, v in cfg["validated_floor"].items()} == sealed, \
        "blur_scale.json was calibrated against other validated blur limits"
    return BlurScale(cfg)


@dataclass(frozen=True)
class BlurSettings:
    enhance_from_pct: float | None = None
    give_up_above_pct: float | None = None

    @property
    def active(self) -> bool:
        return self.enhance_from_pct is not None or self.give_up_above_pct is not None


def _num(name: str, v):
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: expected a number from 0 to 100, got {v!r}") from None
    if not math.isfinite(x) or not 0 <= x <= 100:
        raise ValueError(f"{name}: expected a number from 0 to 100, got {v!r}")
    return x


def validate(s: BlurSettings, scale: BlurScale, base: BlurSettings | None = None) -> BlurSettings:
    """Check a settings pair; `base` = the deployment settings a per-request pair may only narrow. Returns the
    EFFECTIVE pair (request values where given, else the deployment's). Raises ValueError with a readable message."""
    lo = scale.min_enhance_from()
    e, g = s.enhance_from_pct, s.give_up_above_pct
    if e is not None and e < lo:
        raise ValueError(f"enhance_from_pct {e:g} is below the validated minimum {lo:g} (lower values would "
                         f"\"enhance\" sharp faces)")
    if base is not None:
        if e is not None and base.enhance_from_pct is not None and e < base.enhance_from_pct:
            raise ValueError(f"enhance_from_pct {e:g} is below the service setting {base.enhance_from_pct:g} "
                             f"(a request may only narrow the range)")
        if g is not None and base.give_up_above_pct is not None and g > base.give_up_above_pct:
            raise ValueError(f"give_up_above_pct {g:g} is above the service setting {base.give_up_above_pct:g} "
                             f"(a request may only narrow the range)")
        e = e if e is not None else base.enhance_from_pct
        g = g if g is not None else base.give_up_above_pct
    if g is not None and g >= 100:
        raise ValueError(f"give_up_above_pct {g:g}: must be below 100 (leave it empty to attempt every blurred face)")
    start = e if e is not None else max(scale.floor().values())
    if g is not None and g <= start:
        raise ValueError(f"give_up_above_pct {g:g} must be above enhance_from_pct {start:g}")
    return BlurSettings(e, g)


def parse(enhance_from_pct, give_up_above_pct) -> BlurSettings:
    return BlurSettings(_num("enhance_from_pct", enhance_from_pct), _num("give_up_above_pct", give_up_above_pct))


def load_settings() -> tuple[BlurSettings, str]:
    """The deployment settings (FES_SETTINGS or configs/deployment_settings.json) → (settings, source file). The
    pipeline validates them against the scale (validate) and refuses to start on an invalid value."""
    rel = os.environ.get("FES_SETTINGS", SETTINGS_FILE)
    p = repo_path(rel)
    if not p.exists():
        if "FES_SETTINGS" in os.environ:
            raise ValueError(f"FES_SETTINGS: {rel} does not exist")
        return BlurSettings(), "default"
    doc = json.loads(p.read_text(encoding="utf-8"))
    blur = doc.get("blur", {})
    unknown = set(blur) - {"enhance_from_pct", "give_up_above_pct"}
    if unknown:
        raise ValueError(f"{rel}: unknown blur setting(s) {', '.join(sorted(unknown))}")
    return parse(blur.get("enhance_from_pct"), blur.get("give_up_above_pct")), rel


def report(scale: BlurScale, framing: str, lapd1_c: float, sealed_blur_max: float, s: BlurSettings,
           sources: dict) -> dict:
    """The blur-range record of one gate decision (nafnet.blur_range)."""
    pct = scale.percent(framing, lapd1_c)
    start = max(scale.percent(framing, sealed_blur_max), s.enhance_from_pct or 0.0)
    verdict = (TOO_BLURRED if s.give_up_above_pct is not None and pct > s.give_up_above_pct
               else NOT_BLURRY_ENOUGH if not (lapd1_c <= sealed_blur_max and pct >= start) else IN_RANGE)
    return {"scale_version": SCALE_VERSION, "framing": framing, "blur_pct": round(pct, 1),
            "enhance_from_pct": round(start, 1), "give_up_above_pct": s.give_up_above_pct,
            "verdict": verdict, "source": sources}
