"""Phase 3 cooking: mesh → collision proxy (no GL)."""
import numpy as np
import pytest

from omi_physics import model, hull
from omi_physics.cookery import cook_shape, clear_cache


def box_surface(half=(1.0, 1.0, 1.0), n=4):
    h = np.asarray(half)
    grid = np.linspace(-1, 1, n)
    pts = []
    for a in grid:
        for b in grid:
            pts += [(a, b, 1), (a, b, -1), (a, 1, b), (a, -1, b), (1, a, b), (-1, a, b)]
    return np.unique(np.array(pts) * h, axis=0)


def sphere_surface(radius=1.3, n=200):
    rng = np.random.RandomState(0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius


def l_shape():
    pts = []
    for x in np.linspace(0, 2, 6):
        for y in np.linspace(0, 2, 6):
            for z in np.linspace(0, 1, 3):
                if x <= 0.7 or y <= 0.7:            # L cross-section
                    pts.append((x, y, z))
    return np.array(pts, dtype='d')


def test_boxish_mesh_cooks_to_box():
    shape = cook_shape(box_surface((1.0, 0.5, 0.7)), strategy='primitive')
    assert shape.type == 'box'
    assert np.allclose(shape.size, (2.0, 1.0, 1.4), atol=0.2)


def test_sphereish_mesh_cooks_to_sphere():
    shape = cook_shape(sphere_surface(1.3), strategy='primitive')
    assert shape.type == 'sphere'
    assert shape.radius == pytest.approx(1.3, abs=0.1)


def test_convex_strategy_spans_the_cube_corners():
    pts = box_surface((1, 1, 1))
    shape = cook_shape(pts, strategy='convex')
    assert shape.type == 'convex'
    # the hull must reach all 8 cube corners under diagonal support directions
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                d = np.array([sx, sy, sz], dtype='d')
                support = shape.points[np.argmax(shape.points @ d)]
                assert np.allclose(support, (sx, sy, sz), atol=1e-9)


def test_static_mesh_auto_cooks_to_trimesh():
    pts = box_surface((1, 1, 1))
    _, faces = hull.convex_hull(pts)
    shape = cook_shape(pts, indices=faces, strategy='auto', dynamic=False)
    assert shape.type == 'trimesh'
    assert len(shape.points) == len(pts)


def test_concave_mesh_decomposes_to_compound_covering_mesh():
    pts = l_shape()
    pieces = cook_shape(pts, strategy='decompose')
    assert isinstance(pieces, list)
    assert len(pieces) >= 2
    # every original vertex lies inside some convex piece's AABB
    for p in pts:
        covered = any(np.all(p >= s.points.min(axis=0) - 1e-6) and
                      np.all(p <= s.points.max(axis=0) + 1e-6) for s in pieces)
        assert covered


def test_cache_returns_same_object():
    clear_cache()
    pts = box_surface((1, 1, 1))
    a = cook_shape(pts, strategy='convex')
    b = cook_shape(pts, strategy='convex')
    assert a is b


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
