"""Phase 3 harness: geometry contract, evaluation mask, baseline adapters, metrics (no model weights needed)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from fes_bench import geometry as G  # noqa: E402
from fes_bench import metrics as MT  # noqa: E402

# typical recorded E→F transform (scale ≈ 1.55, centred face) and one whose window misses the chip corner
M_FULL = np.array([[1.55, 0.0, 82.4], [0.0, 1.55, 140.0]])
M_PARTIAL = np.array([[2.2, 0.1, 10.0], [-0.1, 2.2, 80.0]])


def _img(seed=0, s=224):
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 256, (s // 8, s // 8, 3), dtype=np.uint8), (s, s), interpolation=cv2.INTER_CUBIC)
    return base


def test_d_to_e_matches_area_downscale_centres():
    # a pixel centre in the 56-px image maps to the centre of the 4×4 block it averages in the E-frame
    M = G.d_to_e(56)
    x = M @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(x, [1.5, 1.5])


def test_d_to_f_composes_d_to_e_then_e_to_f():
    Mdf = G.d_to_f(112, M_FULL)
    p_d = np.array([10.0, 20.0, 1.0])
    p_e = G.d_to_e(112) @ p_d
    assert np.allclose(Mdf @ p_d, M_FULL @ np.append(p_e, 1.0))


def test_e_f_e_roundtrip_is_near_identity():
    img = _img()
    f = G.warp(img, M_FULL, (512, 512))
    back = G.warp(f, G.f_to_e(M_FULL), (224, 224))
    assert MT.psnr_rgb(back, img) > 35.0


def test_coverage_mask_full_and_partial():
    assert G.coverage_mask(M_FULL).all()
    m = G.coverage_mask(M_PARTIAL)
    assert 0.3 < m.mean() < 1.0
    er = G.coverage_mask(M_PARTIAL, erode_px=2)
    assert er.sum() < m.sum() and not (er & ~m).any()


def test_eval_mask_does_not_erode_image_border():
    assert G.coverage_mask(M_FULL, erode_px=2).all()


def test_resize_to_e_rules():
    assert G.resize_to_e(_img(s=224)).shape == (224, 224, 3)
    assert G.resize_to_e(_img(s=448)).shape == (224, 224, 3)
    assert G.resize_to_e(_img(s=56)).shape == (224, 224, 3)


def test_masked_metrics_ignore_pixels_outside_mask():
    ref = _img(1)
    out = ref.copy()
    out[:20] = 0
    mask = np.ones((224, 224), bool)
    mask[:20] = False
    assert MT.psnr_rgb(out, ref, mask) == float("inf")
    assert MT.ssim_y(out, ref, mask) > MT.ssim_y(out, ref)


def test_face_row_format_for_sface():
    lm = np.array([[70, 90], [150, 90], [110, 130], [80, 170], [140, 170]], float)
    row = MT.face_row(lm)
    assert row.shape == (15,) and row[0] == 70 and row[2] == 80 and row[14] == 1.0


torch = pytest.importorskip("torch")
from fes_bench import adapters as AD  # noqa: E402


@pytest.mark.parametrize("cid", ["B0", "B1", "B2", "B2c"])
@pytest.mark.parametrize("s", [14, 56, 224])
def test_baseline_contract_and_determinism(cid, s):
    a = AD.build(cid)
    a.load()
    deg = _img(2, s)
    o1, _ = a.enhance(deg, {"s": s, "M_e2f": M_FULL})
    o2, _ = a.enhance(deg, {"s": s, "M_e2f": M_FULL})
    assert o1.shape == (224, 224, 3) and o1.dtype == np.uint8 and np.array_equal(o1, o2)


def test_b0_is_identity_for_same_size_input():
    deg = _img(3)
    out, _ = AD.build("B0").enhance(deg, {"s": 224, "M_e2f": M_FULL})
    assert np.array_equal(out, deg)


class _EchoFace(AD.FaceModel):
    """Face adapter whose 'model' returns its input: isolates the harness geometry + paste-back."""

    def __init__(self):
        super().__init__("T", "echo", "none")

    def load(self):
        pass

    def _forward(self, x):
        return x


def test_face_adapter_pastes_back_bicubic_where_uncovered():
    deg = _img(4, 56)
    out, meta = _EchoFace().enhance(deg, {"s": 56, "M_e2f": M_PARTIAL})
    cov = G.coverage_mask(M_PARTIAL)
    b0 = G.upsample_to_e(deg)
    assert meta["f_coverage"] < 1.0
    assert np.array_equal(out[~cov], b0[~cov])
    # covered region ≈ bicubic of the same input (only resampling differences)
    er = G.coverage_mask(M_PARTIAL, erode_px=3)
    assert MT.psnr_rgb(out, b0, er) > 30.0


def test_face_adapter_full_coverage_geometry():
    deg = _img(5, 224)
    out, meta = _EchoFace().enhance(deg, {"s": 224, "M_e2f": M_FULL})
    assert meta["f_coverage"] == 1.0 and MT.psnr_rgb(out, deg) > 35.0
