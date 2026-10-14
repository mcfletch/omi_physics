"""Phase 1 narrow phase: analytic primitive contacts (no GL)."""
import numpy as np
import pytest

from omi_physics import collide, mathutil
from omi_physics.body import BoxProxy, SphereProxy

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


#: A separation whose square is below the smallest double, so a distance taken
#: as the root of a sum of squares reads zero across it.
TINY_GAP = 1e-200
#: A radius far smaller than TINY_GAP: shapes this size, that far apart, are apart.
TINY_RADIUS = 6e-294


def along(axis, distance):
    """The point ``distance`` from the origin along coordinate ``axis``."""
    point = np.zeros(3)
    point[axis] = distance
    return point


@pytest.mark.parametrize('axis', range(3))
def test_spheres_a_tiny_gap_apart_do_not_touch(axis):
    A = SphereProxy((0, 0, 0), TINY_RADIUS)
    B = SphereProxy(along(axis, TINY_GAP), TINY_RADIUS)
    assert collide.collide(0, 1, A, B) == []


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('radius', [0.0, TINY_RADIUS])
def test_sphere_a_tiny_gap_outside_a_box_does_not_touch(axis, radius):
    S = SphereProxy(along(axis, TINY_GAP), radius)
    B = box((0, 0, 0), (0, 0, 0))               # a box flattened to a point
    assert collide.collide(0, 1, S, B) == []
    assert collide.collide(0, 1, B, S) == []


@pytest.mark.parametrize('axis', range(3))
def test_compiled_spheres_a_tiny_gap_apart_do_not_touch(axis):
    native = pytest.importorskip('omi_physics._collide_native')
    count, _normal, _points, _depths = native.sphere_sphere(
        np.zeros(3), TINY_RADIUS, along(axis, TINY_GAP), TINY_RADIUS)
    assert count == 0


@pytest.mark.parametrize('axis', range(3))
def test_compiled_sphere_a_tiny_gap_outside_a_box_does_not_touch(axis):
    native = pytest.importorskip('omi_physics._collide_native')
    count, _normal, _points, _depths = native.sphere_box(
        along(axis, TINY_GAP), 0.0, np.zeros(3), I3.copy(), np.zeros(3))
    assert count == 0

if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
