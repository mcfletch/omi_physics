"""Phase 3 GJK/EPA on convex proxies (no GL)."""
import numpy as np
import pytest

from omi_physics import gjk, mathutil, model
from omi_physics.body import ConvexProxy, SphereProxy, make_proxy

I3 = np.eye(3)


def box_cloud(center, half=(0.5, 0.5, 0.5), R=I3):
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


@pytest.mark.parametrize('x', [0.0, 1e-9, -2.34e-13, 0.1, 1.0, 7.3])
def test_a_capsule_just_into_a_long_flat_box_is_found_wherever_it_stands(x):
    """GJK hands EPA a tetrahedron in whatever winding it ended in.

    On a long thin box the first vertices it finds lie almost in a line, and a
    tetrahedron taken with its faces wound inward had EPA expand the wrong way
    and answer a sideways normal at no depth: a capsule 2 cm into the floor
    was reported as not touching it.
    """
    from omi_physics import collide
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    floor = make_proxy(model.Shape.box((60, 1, 60)), np.array([0.0, -0.5, 0.0]), ident)
    capsule = make_proxy(model.Shape.capsule(height=1.2, radius=0.32),
                         np.array([x, 0.9001, -x]), ident)
    contacts = collide.collide(0, 1, capsule, floor)
    assert len(contacts) == 1
    assert contacts[0].depth == pytest.approx(0.0199, abs=1e-6)
    assert np.allclose(contacts[0].normal, (0, -1, 0), atol=1e-6)


def test_an_origin_on_the_edge_of_a_thin_tetrahedron_is_inside_it():
    """A small sphere almost wholly inside a large one, nearly along an axis.

    The first two support points put the origin on the edge between them to
    within rounding, so each tetrahedron built around that edge has the origin
    on two of its faces. Read as outside, the search turns round the edge
    until it runs out of iterations.
    """
    small = SphereProxy(np.array([0.0, 5.960464477539063e-08, 1.6210007107555384]), 0.01)
    large = SphereProxy(np.array([0.01, 0.0, -5.960464477539063e-08]),
                        1.6210007107555384)
    hit, _simplex = gjk.gjk_intersect(small, large)
    assert hit
    (contact,) = gjk.collide_convex(0, 1, small, large)
    assert contact.depth == pytest.approx(0.00997, abs=1e-4)
