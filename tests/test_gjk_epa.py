"""Phase 3 GJK/EPA on convex proxies (no GL)."""
import numpy as np
import pytest

from omi_physics import mathutil
from omi_physics.body import ConvexProxy, BoxProxy, SphereProxy
from omi_physics import gjk


def box_cloud(center, half=(0.5, 0.5, 0.5), R=np.eye(3)):
    corners = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1)
                        for sz in (-1, 1)], dtype='d') * half
    return ConvexProxy(corners, center, R)


def aabb_overlap(ca, ha, cb, hb):
    return np.all(np.abs(np.subtract(ca, cb)) <= np.add(ha, hb))


def test_gjk_matches_axis_aligned_overlap():
    rng = np.random.RandomState(7)
    ha = hb = (0.5, 0.5, 0.5)
    for _ in range(200):
        ca = rng.uniform(-2, 2, 3)
        cb = rng.uniform(-2, 2, 3)
        hit, _ = gjk.gjk_intersect(box_cloud(ca), box_cloud(cb))
        expected = aabb_overlap(ca, ha, cb, hb)
        assert hit == expected, (ca, cb)


def test_epa_depth_and_normal_on_axis_overlap():
    A = box_cloud((0, 0, 0), (0.5, 0.5, 0.5))
    B = box_cloud((0.7, 0, 0), (0.5, 0.5, 0.5))     # overlap 0.3 on X
    contacts = gjk.collide_convex(0, 1, A, B)
    assert len(contacts) == 1
    c = contacts[0]
    assert c.depth == pytest.approx(0.3, abs=1e-3)
    assert np.allclose(np.abs(c.normal), (1, 0, 0), atol=1e-3)


def test_epa_separated_returns_no_contact():
    A = box_cloud((0, 0, 0))
    B = box_cloud((3, 0, 0))
    assert gjk.collide_convex(0, 1, A, B) == []


def test_convex_matches_analytic_sphere_box_depth():
    """A convex box hull vs a sphere: EPA depth ~ analytic penetration."""
    S = SphereProxy((0, 1.2, 0), 0.5)
    B = box_cloud((0, 0, 0), (1, 1, 1))
    contacts = gjk.collide_convex(0, 1, S, B)
    assert contacts
    assert contacts[0].depth == pytest.approx(0.3, abs=0.05)


def test_gjk_oriented_boxes():
    q = mathutil.quat_from_axis_angle((0, 0, 1), np.pi / 4)
    R = mathutil.quat_to_matrix(q)
    A = box_cloud((0, 0, 0), (0.5, 0.5, 0.5))
    B = box_cloud((1.0, 0, 0), (0.5, 0.5, 0.5), R)   # rotated, corner pokes in
    hit, _ = gjk.gjk_intersect(A, B)
    assert hit


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
