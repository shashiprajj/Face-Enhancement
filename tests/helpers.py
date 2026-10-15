"""In-process stand-ins for the isolated model workers and small image helpers shared by the tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fes_service import models as MD
from fes_service import pipeline as SP
from fes_service import repo_path

FIXTURES = repo_path("tests/fixtures")
FRONT, SIDE = "front_blur.png", "side_blur.png"     # Indian-expression benchmark, motion blur level 3


def fixture_bytes(name: str) -> bytes:
    p = FIXTURES / name
    if not p.exists():
        pytest.skip(f"test image {name} missing (tests/fixtures is local only)")
    return p.read_bytes()


def fixture_img(name: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(fixture_bytes(name), np.uint8), cv2.IMREAD_COLOR)


def png(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def sharpen(img, ctx=None):
    b = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, b, -0.5, 0), {}


def strong(img, ctx=None):
    b = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 2.5, b, -1.5, 0), {}


def chroma(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 1:]


class Model:
    """In-process stand-in for an isolated worker: r6_consensus around `core`, records every ctx."""

    def __init__(self, core):
        self.fn, self.ctx = MD.r6_consensus(core), []

    def __call__(self, img, ctx):
        self.ctx.append(dict(ctx))
        out, meta = self.fn(img, ctx)
        return out, {**meta, "infer_s": 0.01}

    def alive(self):
        return True


def build(core6=sharpen, core_ft=strong, settings=None, **kw):
    """→ (pipeline, r6 model, r6_ft model) with in-process consensus models around the given cores."""
    m6, ft = Model(core6), Model(core_ft)
    return SP.EnhancementPipeline({"r6": m6, "r6_ft": ft}, blur_settings=settings, **kw), m6, ft
