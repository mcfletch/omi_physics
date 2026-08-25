"""Properties the narrow phase holds for any pair of shapes.

A contact is a promise to the solver: a unit normal pointing from A to B, a
point in world space, and a depth that is how far apart the two have to move to
stop overlapping. The solver divides by quantities derived from all three, so a
normal that is not a direction, or a depth that does not separate, is an impulse
of the wrong size applied in the wrong direction -- and the body it is applied
to leaves the level.

The shapes generated here include the ones an asset pipeline produces and nobody
writes a test for: a collider with a zero radius, a box flattened to a plane or
to a point, two colliders at the same position, a "triangle" whose three
vertices are one point.
"""
import numpy as np
import pytest

# Hypothesis is a CPython-only test dependency; see test_properties_math.py.
pytest.importorskip('hypothesis', reason='hypothesis has no wheel for this interpreter')

from hypothesis import given
from hypothesis import strategies as st
from property_strategies import (
    SCALES,
    SETTINGS,
    analytic_proxies,
    box_proxies,
    capsule_proxies,
    proxy_scale,
    rotations,
    shrunk,
    sphere_proxies,
    support_proxies,
    translated,
    triangles,
    vectors,
)

from omi_physics import collide, gjk
from omi_physics.body import BoxProxy, CapsuleProxy, SphereProxy


def boxes_are_apart(A, B) -> bool:
    """True when the two proxies' bounding boxes do not overlap at all."""
    alo, ahi = A.aabb()
    blo, bhi = B.aabb()
    return bool(np.any(ahi < blo) or np.any(bhi < alo))


class TestEveryContactIsWellFormed:
    """Whatever comes in, what goes to the solver is usable."""

    @given(A=support_proxies(), B=support_proxies())
    @SETTINGS
    def test_the_normal_is_a_direction(self, A, B):
        for c in collide.collide(0, 1, A, B):
            assert np.isfinite(c.normal).all()
            assert np.linalg.norm(c.normal) == pytest.approx(1.0, rel=1e-6)

    @given(A=support_proxies(), B=support_proxies())
    @SETTINGS
    def test_the_point_and_depth_are_numbers(self, A, B):
        for c in collide.collide(0, 1, A, B):
            assert np.isfinite(c.point).all()
            assert np.isfinite(c.depth)
            assert c.depth >= 0.0

    @given(A=support_proxies(), B=support_proxies())
    @SETTINGS
    def test_the_manifold_stays_small(self, A, B):
        """The solver iterates over every contact of every pair; a manifold that
        grew with the shape's complexity would put the step's cost there."""
        assert len(collide.collide(0, 1, A, B)) <= 4

    @given(A=support_proxies(), B=support_proxies())
    @SETTINGS
    def test_the_contact_names_the_bodies_it_was_asked_about(self, A, B):
        for c in collide.collide(3, 7, A, B):
            assert (c.a, c.b) == (3, 7)

    @given(scale=st.sampled_from(SCALES),
           data=st.data())
    @SETTINGS
    def test_the_depth_is_no_larger_than_the_shapes(self, scale, data):
        """A depth greater than the shapes involved is a sign flip somewhere, and
        the solver would answer it with an impulse to match."""
        A = data.draw(analytic_proxies(scale))
        B = data.draw(analytic_proxies(scale))
        reach = proxy_scale(A) + proxy_scale(B)
        for c in collide.collide(0, 1, A, B):
            assert c.depth <= reach * (1.0 + 1e-6)


class TestShapesThatAreApartDoNotTouch:
    @given(A=support_proxies(), B=support_proxies())
    @SETTINGS
    def test_disjoint_bounding_boxes_mean_no_contact(self, A, B):
        if not boxes_are_apart(A, B):
            return
        assert collide.collide(0, 1, A, B) == []

    @given(A=analytic_proxies(), B=analytic_proxies(), away=vectors(1.0))
    @SETTINGS
    def test_moving_far_enough_apart_ends_the_contact(self, A, B, away):
        direction = away / max(np.linalg.norm(away), 1e-12)
        if np.linalg.norm(away) < 1e-6:
            return
        gap = (proxy_scale(A) + proxy_scale(B)) * 4.0
        assert collide.collide(0, 1, A, translated(B, direction * gap)) == []


def overlap_along(A, B, normal) -> float:
    """How far A and B overlap measured along ``normal``, which points A to B.

    The far side of A minus the near side of B, both from the proxies' own
    support functions -- so it is the same question for a sphere, a box or a
    point cloud.
    """
    return float(np.dot(A.support(normal), normal) - np.dot(B.support(-normal), normal))


class TestTheDepthIsNeverMoreThanTheWayOut:
    """`depth` is what the position pass moves a pair apart by, so a depth
    larger than the overlap it describes pushes them past each other, and the
    pair meets again next step with the sign reversed -- which is a stack that
    shivers instead of resting.

    Stated one-sided because a clipped manifold reports each point's own
    penetration against the reference face rather than the distance that would
    separate the pair. The deepest corner of the incident face can be clipped
    away by the reference face's side planes, leaving the manifold shallower
    than the pair is deep; the position pass then recovers what is left over the
    next few steps, which is also how it treats a body that arrives deep inside
    another.
    """

    @given(scale=st.sampled_from(SCALES), data=st.data())
    @SETTINGS
    def test_no_contact_asks_for_more_than_the_pair_overlaps(self, scale, data):
        A = data.draw(analytic_proxies(scale))
        B = data.draw(analytic_proxies(scale))
        for c in collide.collide(0, 1, A, B):
            room = overlap_along(A, B, c.normal)
            assert c.depth <= room + 1e-9 * (proxy_scale(A) + proxy_scale(B))

    @given(A=support_proxies(), B=support_proxies())
    @SETTINGS
    def test_that_holds_for_the_pairs_gjk_answers_too(self, A, B):
        for c in gjk.collide_convex(0, 1, A, B):
            room = overlap_along(A, B, c.normal)
            assert c.depth <= room + 1e-6 * (proxy_scale(A) + proxy_scale(B))

    @given(A=sphere_proxies(degenerate=False), B=sphere_proxies(degenerate=False))
    @SETTINGS
    def test_two_spheres_report_exactly_the_way_out(self, A, B):
        """No clipping is involved, so here the depth is the whole distance --
        as long as the two centres are far enough apart to name a direction.

        Nearer than a nanometre they are one point as far as the routine is
        concerned, and it picks a way out instead of computing one; what is left
        over is then that nanometre, which is the floor rather than an error.
        """
        if np.linalg.norm(B.center - A.center) <= 1e-9:
            return
        for c in collide.collide(0, 1, A, B):
            moved = translated(B, c.normal * (c.depth + 1e-9 * (A.radius + B.radius)))
            assert collide.collide(0, 1, A, moved) == []


def solidly_overlapping(A, B) -> bool:
    """Whether the pair still meets with both shrunk about their centres.

    A pair that only just touches is on a knife edge -- which axis offers the
    shorter way out, and which corners of a clipped face count as inside, come
    down to the last bit of the coordinates -- so the properties that ask for
    one answer rather than another ask them of pairs that are meeting solidly.
    """
    return bool(collide.collide(0, 1, shrunk(A), shrunk(B)))


class TestOrderDoesNotMatter:
    """Which body the caller happens to have called A cannot decide whether the
    two are touching, nor -- where the answer is computed rather than picked --
    which way they part.

    Two things are deliberately not asserted of a box pair. Where its contacts
    *sit* is one: a box manifold is the incident face clipped against the
    reference face and seated on it, and which of the two boxes is the reference
    is decided by which is better aligned with the normal -- so a pair equally
    aligned with it is seated on one box's face one way round and on the other's
    the other. How many points survive that clipping is the other. Both describe
    the same meeting from the two sides, and the solver is handed one of them.
    """

    @given(A=analytic_proxies(), B=analytic_proxies())
    @SETTINGS
    def test_swapping_the_pair_does_not_change_whether_they_meet(self, A, B):
        if not solidly_overlapping(A, B):
            return
        assert collide.collide(0, 1, A, B)
        assert collide.collide(1, 0, B, A)

    @given(A=sphere_proxies(), B=sphere_proxies())
    @SETTINGS
    def test_two_spheres_answer_the_same_either_way_round(self, A, B):
        """No clipping and no choice of reference face, so here the whole
        contact is required to be the same one seen from the other side."""
        if np.allclose(A.center, B.center):
            return
        forward = collide.collide(0, 1, A, B)
        backward = collide.collide(1, 0, B, A)
        assert len(forward) == len(backward)
        for f, r in zip(forward, backward, strict=True):
            assert np.allclose(f.normal, -r.normal, atol=1e-9)
            assert f.depth == pytest.approx(r.depth, rel=1e-9, abs=1e-12)
            assert np.allclose(f.point, r.point, atol=1e-9)

    @given(A=analytic_proxies(), B=analytic_proxies())
    @SETTINGS
    def test_coincident_shapes_are_still_given_a_way_apart(self, A, B):
        """Which way is arbitrary; that there is one is not, or the pair stays
        stuck inside one another for as long as the scene runs."""
        moved = translated(B, A.center - B.center)
        for c in collide.collide(0, 1, A, moved):
            assert np.linalg.norm(c.normal) == pytest.approx(1.0, rel=1e-6)


class TestWhereTheShapesAreDoesNotMatter:
    """Every threshold in the narrow phase should be relative to the shapes, so
    the same pair a kilometre from the origin meets in the same way. An absolute
    one shows up here as a pair that touches at the origin and does not touch
    where the level actually is."""

    @given(A=analytic_proxies(), B=analytic_proxies(), offset=vectors(1000.0))
    @SETTINGS
    def test_moving_both_does_not_change_whether_they_meet(self, A, B, offset):
        if not solidly_overlapping(A, B):
            return
        assert collide.collide(0, 1, translated(A, offset), translated(B, offset))

    @given(A=sphere_proxies(degenerate=False), B=sphere_proxies(degenerate=False),
           offset=vectors(1000.0))
    @SETTINGS
    def test_two_spheres_meet_in_the_same_way_wherever_they_are(self, A, B, offset):
        """Exactly, and not only in whether they touch: sphere against sphere is
        a subtraction of two lengths, with nothing in it that could be relative
        to where the pair happens to be."""
        here = collide.collide(0, 1, A, B)
        there = collide.collide(0, 1, translated(A, offset), translated(B, offset))
        assert len(here) == len(there)
        tol = 1e-9 * max(np.max(np.abs(offset)), 1.0)
        for a, b in zip(here, there, strict=True):
            assert np.allclose(a.normal, b.normal, atol=1e-9)
            assert a.depth == pytest.approx(b.depth, rel=1e-9, abs=tol)
            assert np.allclose(a.point + offset, b.point, atol=tol)


class TestGJKAndTheAnalyticTestsAgree:
    """GJK answers the pairs no closed form covers, so nothing else says whether
    it is right. Two spheres have both, which makes them the case where its
    answer can be checked against one."""

    @given(A=sphere_proxies(degenerate=False), B=sphere_proxies(degenerate=False))
    @SETTINGS
    def test_clearly_overlapping_spheres_are_found(self, A, B):
        gap = np.linalg.norm(B.center - A.center) - (A.radius + B.radius)
        if gap > -1e-3 * (A.radius + B.radius):
            return
        hit, _ = gjk.gjk_intersect(A, B)
        assert hit

    @given(A=sphere_proxies(degenerate=False), B=sphere_proxies(degenerate=False))
    @SETTINGS
    def test_clearly_separated_spheres_are_not(self, A, B):
        gap = np.linalg.norm(B.center - A.center) - (A.radius + B.radius)
        if gap < 1e-3 * (A.radius + B.radius):
            return
        hit, _ = gjk.gjk_intersect(A, B)
        assert not hit

    @given(A=support_proxies(), B=support_proxies())
    @SETTINGS
    def test_the_contact_it_builds_is_well_formed(self, A, B):
        for c in gjk.collide_convex(0, 1, A, B):
            assert np.isfinite(c.point).all()
            assert np.linalg.norm(c.normal) == pytest.approx(1.0, rel=1e-6)
            assert c.depth > 0.0


class TestBatchedTriangleTests:
    """What a body standing on a landscape goes through, several thousand
    triangles at a time. A degenerate triangle among them has no plane to
    separate along, and must not become a NaN normal pushing the character."""

    @given(box=box_proxies(), verts=triangles())
    @SETTINGS
    def test_a_box_against_any_triangles_gives_usable_pushes(self, box, verts):
        hit, points, normals, depths = collide.box_triangle_batch(box, verts)
        assert len(hit) == len(verts)
        for i in np.flatnonzero(hit):
            assert np.isfinite(points[i]).all()
            assert np.linalg.norm(normals[i]) == pytest.approx(1.0, rel=1e-6)
            assert np.isfinite(depths[i]) and depths[i] >= 0.0

    @given(cap=capsule_proxies(), verts=triangles())
    @SETTINGS
    def test_a_capsule_against_any_triangles_gives_usable_pushes(self, cap, verts):
        hit, points, normals, depths = collide.capsule_triangle_batch(cap, verts)
        assert len(hit) == len(verts)
        for i in np.flatnonzero(hit):
            assert np.isfinite(points[i]).all()
            assert np.linalg.norm(normals[i]) == pytest.approx(1.0, rel=1e-6)
            assert np.isfinite(depths[i]) and depths[i] >= 0.0

    @given(cap=capsule_proxies(degenerate=False), verts=triangles())
    @SETTINGS
    def test_a_capsule_far_from_every_triangle_hits_nothing(self, cap, verts):
        lo = verts.reshape(-1, 3).min(axis=0)
        hi = verts.reshape(-1, 3).max(axis=0)
        span = float(np.max(hi - lo)) + cap.radius + cap.half_height
        moved = CapsuleProxy(hi + (span + 1.0), cap.half_height, cap.radius, cap.R)
        hit, _, _, _ = collide.capsule_triangle_batch(moved, verts)
        assert not hit.any()


class TestDegenerateCollidersAreQuiet:
    """Named cases, because an asset pipeline produces every one of them and the
    property files above would only reach them by luck."""

    @given(centre=vectors(10.0), R=rotations())
    @SETTINGS
    def test_two_shapes_with_no_size_at_the_same_point(self, centre, R):
        pairs = [
            (SphereProxy(centre, 0.0), SphereProxy(centre, 0.0)),
            (BoxProxy(centre, np.zeros(3), R), BoxProxy(centre, np.zeros(3), R)),
            (SphereProxy(centre, 0.0), BoxProxy(centre, np.zeros(3), R)),
            (CapsuleProxy(centre, 0.0, 0.0, R), SphereProxy(centre, 0.0)),
        ]
        for A, B in pairs:
            for c in collide.collide(0, 1, A, B):
                assert np.isfinite(c.point).all()
                assert np.isfinite(c.depth)
                assert np.linalg.norm(c.normal) == pytest.approx(1.0, rel=1e-6)

    @given(centre=vectors(10.0), R=rotations(), radius=st.floats(0.1, 2.0))
    @SETTINGS
    def test_a_sphere_at_the_centre_of_a_flattened_box(self, centre, R, radius):
        """A box with one axis collapsed is a plane, which is what a
        badly-authored floor collider is."""
        flat = BoxProxy(centre, np.array([2.0, 0.0, 2.0]), R)
        (c,) = collide.collide(0, 1, SphereProxy(centre, radius), flat)
        assert np.linalg.norm(c.normal) == pytest.approx(1.0, rel=1e-6)
        assert c.depth == pytest.approx(radius, rel=1e-6)
