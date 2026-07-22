"""Phase 1 narrow phase: analytic primitive contacts (no GL)."""
import numpy as np
import pytest

from omi_physics import mathutil
from omi_physics.body import SphereProxy, BoxProxy
from omi_physics import collide

I3 = np.eye(3)


def box(center, half=(0.5, 0.5, 0.5), R=I3):
    return BoxProxy(center, half, R)


def test_sphere_sphere_hit():
    A = SphereProxy((0, 0, 0), 1.0)
    B = SphereProxy((1.5, 0, 0), 1.0)
    (c,) = collide.collide(0, 1, A, B)
    assert c.depth == pytest.approx(0.5)
    assert np.allclose(c.normal, (1, 0, 0))
    assert c.point[0] == pytest.approx(0.75)


def test_sphere_sphere_miss():
    A = SphereProxy((0, 0, 0), 1.0)
    B = SphereProxy((3, 0, 0), 1.0)
    assert collide.collide(0, 1, A, B) == []


def test_sphere_box_face():
    S = SphereProxy((0, 1.3, 0), 0.5)
    B = box((0, 0, 0), (1, 1, 1))
    (c,) = collide.collide(0, 1, S, B)
    assert c.depth == pytest.approx(0.2, abs=1e-9)
    # normal points from sphere (A) toward box (B): downward
    assert np.allclose(c.normal, (0, -1, 0), atol=1e-9)


def test_sphere_box_miss():
    S = SphereProxy((0, 2.0, 0), 0.5)
    B = box((0, 0, 0), (1, 1, 1))
    assert collide.collide(0, 1, S, B) == []


def test_sphere_inside_box_gets_min_axis():
    S = SphereProxy((0.1, 0.0, 0.0), 0.2)
    B = box((0, 0, 0), (1, 0.3, 1))
    (c,) = collide.collide(0, 1, S, B)
    assert abs(c.normal[1]) == pytest.approx(1.0)     # nearest face is +/-Y


def test_box_box_axis_aligned_overlap():
    A = box((0, 0, 0), (1, 1, 1))
    B = box((1.5, 0, 0), (1, 1, 1))
    contacts = collide.collide(0, 1, A, B)
    assert contacts
    assert all(np.allclose(np.abs(c.normal), (1, 0, 0), atol=1e-6) for c in contacts)
    assert max(c.depth for c in contacts) == pytest.approx(0.5, abs=1e-6)


def test_box_box_resting_gives_multipoint_manifold():
    ground = box((0, 0, 0), (5, 0.5, 5))
    crate = box((0, 0.99, 0), (0.5, 0.5, 0.5))
    contacts = collide.collide(0, 1, ground, crate)
    assert len(contacts) >= 3                    # stable face manifold
    assert all(np.allclose(np.abs(c.normal), (0, 1, 0), atol=1e-6) for c in contacts)


def test_box_box_miss():
    A = box((0, 0, 0), (1, 1, 1))
    B = box((3, 0, 0), (1, 1, 1))
    assert collide.collide(0, 1, A, B) == []


def test_oriented_box_box_hit():
    A = box((0, 0, 0), (1, 1, 1))
    q = mathutil.quat_from_axis_angle((0, 1, 0), np.pi / 4)
    R = mathutil.quat_to_matrix(q)
    B = box((2.2, 0, 0), (1, 1, 1), R)
    contacts = collide.collide(0, 1, A, B)
    assert contacts                              # corner of rotated box pokes in


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
