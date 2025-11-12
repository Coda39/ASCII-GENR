import numpy as np

from eval_pipeline import (
    structural_similarity_score,
    edge_preservation_ratio,
    temporal_consistency_score,
    detail_retention_index,
)


def make_checkerboard(size=64, n=8):
    # create a simple checkerboard
    x = np.indices((size, size)).sum(axis=0) // (size // n)
    return (x % 2).astype(float)


def test_ssim_identical():
    im = make_checkerboard(64, 8)
    s = structural_similarity_score(im, im)
    assert pytest_approx(s, 1.0)


def test_edge_preservation_perfect():
    im = make_checkerboard(64, 8)
    e = edge_preservation_ratio(im, im, sigma=1.0, dilate_radius=1)
    assert pytest_approx(e, 1.0)


def test_detail_retention_same():
    im = make_checkerboard(64, 8)
    d = detail_retention_index(im, im, patch_size=8)
    assert pytest_approx(d, 1.0)


def test_temporal_constant_frames():
    im = make_checkerboard(32, 4)
    frames = [im, im, im]
    t = temporal_consistency_score(frames)
    assert pytest_approx(t, 1.0)


def pytest_approx(a, b, tol=1e-6):
    return abs(a - b) <= tol
