"""Properties the load-time cooking helpers hold for any point cloud.

Cooking runs once, on whatever geometry an artist saved, and its answer becomes
the collider a whole level is played against. The clouds it is given are not the
tidy ones: a flat piece of ground is coplanar, a railing is very nearly a line,
a mesh exported twice over has every vertex duplicated. A hull that comes back
inside-out, or that leaves points outside itself, is a collider with a hole in
it, and the hole is found by a player falling through it.
"""
import numpy as np
import pytest

# Hypothesis is a CPython-only test dependency; see test_properties_math.py.
pytest.importorskip('hypothesis', reason='hypothesis has no wheel for this interpreter')

from hypothesis import given
from hypothesis import strategies as st
from property_strategies import SETTINGS, coordinates, vectors

from omi_physics import hull


def clouds(min_size: int = 1, max_size: int = 30, spread: float = 10.0):
    """An arbitrary point cloud: duplicates, collinear runs and all."""
    return st.lists(vectors(spread), min_size=min_size, max_size=max_size).map(
        lambda pts: np.array(pts, dtype='d'))


@st.composite
def flat_clouds(draw, max_size: int = 20):
    """A cloud with no thickness -- every point on one plane.

    What a floor, a wall or a decal comes in as, and the case the hull has no
    volume to build from.
    """
    count = draw(st.integers(min_value=1, max_value=max_size))
    axis = draw(st.integers(min_value=0, max_value=2))
    points = np.array([draw(vectors(10.0)) for _ in range(count)])
    points[:, axis] = draw(coordinates(10.0))
    return points


def cloud_scale(points: np.ndarray) -> float:
    """A length the cloud's own size, for tolerances that must not be absolute."""
    if not len(points):
        return 1.0
    return max(float(np.max(points.max(axis=0) - points.min(axis=0))), 1.0)


class TestTheHullIsMadeOfTheCloud:
    @given(points=clouds())
    @SETTINGS
    def test_every_hull_vertex_came_from_the_input(self, points):
        """A hull that invents a vertex is not the hull of anything."""
        verts, _faces = hull.convex_hull(points)
        for v in verts:
            assert np.any(np.all(np.isclose(points, v), axis=1))

    @given(points=clouds())
    @SETTINGS
    def test_the_faces_index_the_vertices_it_returned(self, points):
        verts, faces = hull.convex_hull(points)
        if len(faces):
            assert faces.min() >= 0 and faces.max() < len(verts)

    @given(points=clouds())
    @SETTINGS
    def test_no_face_is_a_triangle_of_one_or_two_points(self, points):
        _verts, faces = hull.convex_hull(points)
        for face in faces:
            assert len(set(face.tolist())) == 3

    @given(points=clouds())
    @SETTINGS
    def test_the_volume_is_a_volume(self, points):
        verts, faces = hull.convex_hull(points)
        volume = hull.hull_volume(verts, faces)
        assert np.isfinite(volume) and volume >= 0.0


class TestTheHullContainsWhatItWasBuiltFrom:
    @given(points=clouds(min_size=4))
    @SETTINGS
    def test_no_point_lies_outside_a_face(self, points):
        """The property that makes it a collider: a point outside the hull is
        geometry the hull does not cover, and a place to fall through."""
        verts, faces = hull.convex_hull(points)
        if not len(faces):
            return
        inside = verts.mean(axis=0)
        tolerance = 1e-6 * cloud_scale(points)
        for face in faces:
            a, b, c = verts[face]
            normal = np.cross(b - a, c - a)
            length = float(np.linalg.norm(normal))
            if length <= 1e-12:
                continue
            normal = normal / length
            # Outward: away from a point known to be within the hull.
            if float(np.dot(normal, inside - a)) > 0:
                normal = -normal
            assert np.max((points - a) @ normal) <= tolerance


class TestDegenerateCloudsAreAnswered:
    """None of these has a hull; what matters is that saying so is cheap and
    that the answer is still an array pair the caller can use."""

    @given(points=clouds(max_size=3))
    @SETTINGS
    def test_too_few_points_gives_no_faces(self, points):
        verts, faces = hull.convex_hull(points)
        assert len(faces) == 0
        assert len(verts) <= len(points)

    @given(point=vectors(10.0), count=st.integers(min_value=1, max_value=20))
    @SETTINGS
    def test_the_same_point_over_and_over_gives_no_faces(self, point, count):
        _verts, faces = hull.convex_hull(np.tile(point, (count, 1)))
        assert len(faces) == 0

    @given(points=flat_clouds())
    @SETTINGS
    def test_a_cloud_with_no_thickness_gives_no_faces(self, points):
        _verts, faces = hull.convex_hull(points)
        assert len(faces) == 0

    @given(start=vectors(10.0), along=vectors(1.0),
           ts=st.lists(st.floats(min_value=-5.0, max_value=5.0),
                       min_size=1, max_size=20))
    @SETTINGS
    def test_a_cloud_on_one_line_gives_no_faces(self, start, along, ts):
        points = np.array([start + along * t for t in ts])
        _verts, faces = hull.convex_hull(points)
        assert len(faces) == 0


class TestTheConcavityMeasure:
    @given(points=clouds())
    @SETTINGS
    def test_it_stays_within_its_documented_range(self, points):
        """`auto` cooking compares it against a threshold, so a value outside
        [0, 1] decides the wrong way about what a shape is."""
        value = hull.concavity(points)
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0


class TestTheDecomposition:
    @given(points=clouds(min_size=1, max_size=40))
    @SETTINGS
    def test_the_pieces_account_for_every_point(self, points):
        """Splitting is by a median plane, so the pieces partition the cloud:
        a point in none of them is geometry the compound collider has lost."""
        pieces = hull.approximate_convex_decomposition(points)
        assert pieces
        assert sum(len(piece) for piece in pieces) == len(points)

    @given(points=clouds(min_size=1, max_size=40),
           budget=st.integers(min_value=1, max_value=8))
    @SETTINGS
    def test_no_piece_is_empty(self, points, budget):
        for piece in hull.approximate_convex_decomposition(points, max_pieces=budget):
            assert len(piece)
