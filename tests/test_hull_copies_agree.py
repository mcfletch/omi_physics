"""omi_physics' hull and OpenGLContext's ``physics.cooking`` give the same answers.

OpenGLContext carries its own copy of :mod:`omi_physics.hull`, so that the
engine cooks collision shapes without omi_physics installed. A fix to one copy
belongs in the other; these tests compare the two over the clouds cooking finds
hard, and skip where OpenGLContext is not installed.
"""
import numpy as np
import pytest

from omi_physics import hull

cooking = pytest.importorskip('OpenGLContext.physics.cooking',
                              reason="OpenGLContext's copy is the one compared")


def hard_clouds(count=200, seed=0):
    """Seeded clouds: plain, snapped to a half-metre grid, nearly flat and a millimetre across."""
    rng = np.random.default_rng(seed)
    clouds = []
    for index in range(count):
        points = rng.normal(size=(int(rng.integers(4, 60)), 3))
        kind = index % 4
        if kind == 1:
            points = np.round(points * 2.0) / 2.0
        elif kind == 2:
            points[:, 2] = points[:, 0] * 1e-9
        elif kind == 3:
            points = points * 1e-3
        clouds.append(points)
    # The cloud omi_physics' hull once built with a seed corner twice.
    clouds.append(np.array([[0, 0, 0], [0, 0, 1], [1, -1, 0], [-1, 1e-9, 1], [-2, 0, 0]],
                           dtype='d'))
    return clouds


def test_the_hulls_are_the_same():
    for points in hard_clouds():
        ours, theirs = hull.convex_hull(points), cooking.convex_hull(points)
        assert np.array_equal(ours[0], theirs[0])
        assert np.array_equal(ours[1], theirs[1])


def test_the_volumes_and_concavities_are_the_same():
    for points in hard_clouds():
        vertices, faces = hull.convex_hull(points)
        assert hull.hull_volume(vertices, faces) == cooking.hull_volume(vertices, faces)
        assert hull.concavity(points) == cooking.concavity(points)


def test_the_convex_pieces_are_the_same():
    for points in hard_clouds(count=40, seed=1):
        ell = np.concatenate([points * (10.0, 1.0, 1.0), points * (1.0, 10.0, 1.0)])
        ours = hull.approximate_convex_decomposition(ell)
        theirs = cooking.approximate_convex_decomposition(ell)
        assert len(ours) == len(theirs)
        for mine, other in zip(ours, theirs, strict=True):
            assert np.array_equal(mine, other)
