"""Narrowing a triangle soup down to the part a query actually touches.

A level is one mesh of tens of thousands of triangles, and both things that ask
questions of it — the character controller resolving contacts, and a ray asking
what it hits — ask many times a frame.  Answering either by looking at every
triangle is the difference between a cast nobody notices and a cast that shows
up in the frame time.

So the grid is tested for the two properties that matter and not for its
contents: it must never *lose* a triangle the query touches, and it must not
hand back the whole mesh for a query that only reaches across one room.
"""

import numpy as np
import pytest

from omi_physics.trigrid import TriangleGrid


def soup(side=40, spacing=1.0):
    """A plane of ``side * side * 2`` triangles, one metre apart.

    Big enough that "all of them" and "the ones near the query" are obviously
    different numbers.
    """
    lows, highs = [], []
    for i in range(side):
        for j in range(side):
            x, z = i * spacing, j * spacing
            for _ in range(2):
                lows.append([x, 0.0, z])
                highs.append([x + spacing, 0.0, z + spacing])
    return np.array(lows, dtype='d'), np.array(highs, dtype='d')


def brute(lows, highs, low, high):
    """Every triangle whose bound overlaps a box, the slow honest way."""
    return set(np.flatnonzero(np.all(lows <= high, axis=1)
                              & np.all(highs >= low, axis=1)).tolist())


def slab(lows, highs, origin, heading, limit):
    """Every triangle whose bound the ray passes through, the slow honest way."""
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse = 1.0 / heading
        first = (lows - origin) * inverse
        second = (highs - origin) * inverse
    near = np.nanmax(np.minimum(first, second), axis=1)
    far = np.nanmin(np.maximum(first, second), axis=1)
    hit = (far >= np.maximum(near, 0.0)) & (near <= limit)
    return set(np.flatnonzero(hit).tolist())


@pytest.fixture
def grid():
    lows, highs = soup()
    return TriangleGrid(lows, highs), lows, highs


class TestAskingWithABox:

    def test_it_finds_everything_the_box_touches(self, grid):
        """The one property that cannot be traded away for speed."""
        built, lows, highs = grid
        low, high = np.array([3.0, -1.0, 3.0]), np.array([6.0, 1.0, 6.0])
        assert brute(lows, highs, low, high) <= set(built.box(low, high).tolist())

    def test_it_does_not_hand_back_the_whole_mesh(self, grid):
        built, lows, _highs = grid
        low, high = np.array([3.0, -1.0, 3.0]), np.array([6.0, 1.0, 6.0])
        assert len(built.box(low, high)) < len(lows) // 10

    def test_a_box_outside_the_mesh_finds_nothing(self, grid):
        built, _lows, _highs = grid
        found = built.box(np.array([-50.0, -50.0, -50.0]),
                          np.array([-40.0, -40.0, -40.0]))
        assert len(found) == 0

    def test_a_small_mesh_is_answered_without_a_grid(self):
        """Below a handful of triangles the grid costs more than it saves."""
        lows = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        highs = np.array([[1.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
        built = TriangleGrid(lows, highs)
        found = built.box(np.array([-1.0, -1.0, -1.0]), np.array([2.0, 2.0, 2.0]))
        assert found.tolist() == [0]


class TestAskingAlongARay:
    """The part that did not exist, and that the bots pay for.

    A ray's *bounding box* is a poor stand-in for the ray: a cast from one end
    of a level to the other has a box containing the whole level, so narrowing
    by the box narrows to nothing.  Walking the cells the ray actually enters
    is what makes a long cast cheap.
    """

    def test_it_finds_everything_the_ray_passes_through(self, grid):
        built, lows, highs = grid
        origin = np.array([-5.0, 0.0, 10.5])
        heading = np.array([1.0, 0.0, 0.0])
        wanted = slab(lows, highs, origin, heading, 100.0)
        assert wanted <= set(built.ray(origin, heading, 100.0).tolist())

    def test_a_diagonal_ray_loses_nothing_either(self, grid):
        built, lows, highs = grid
        origin = np.array([-5.0, 0.0, -5.0])
        heading = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        wanted = slab(lows, highs, origin, heading, 100.0)
        assert wanted <= set(built.ray(origin, heading, 100.0).tolist())

    def test_a_ray_across_the_whole_mesh_still_narrows(self, grid):
        """The case a bounding box cannot help with at all."""
        built, lows, _highs = grid
        origin = np.array([-5.0, 0.0, 10.5])
        found = built.ray(origin, np.array([1.0, 0.0, 0.0]), 100.0)
        assert len(found) < len(lows) // 10

    def test_a_short_ray_looks_at_very_little(self, grid):
        """What a bot's step probe is: half a metre, on a mesh of thousands."""
        built, lows, _highs = grid
        found = built.ray(np.array([10.5, 0.5, 10.5]),
                          np.array([1.0, 0.0, 0.0]), 0.5)
        assert len(found) < len(lows) // 100

    def test_a_ray_that_never_reaches_the_mesh_finds_nothing(self, grid):
        built, _lows, _highs = grid
        found = built.ray(np.array([10.5, 50.0, 10.5]),
                          np.array([0.0, 1.0, 0.0]), 10.0)
        assert len(found) == 0

    def test_a_ray_stopping_short_does_not_reach_past_its_limit(self, grid):
        """A cast is a segment, not a line; the limit has to bound the walk."""
        built, lows, highs = grid
        origin = np.array([-5.0, 0.0, 10.5])
        heading = np.array([1.0, 0.0, 0.0])
        near = set(built.ray(origin, heading, 8.0).tolist())
        assert near <= set(built.ray(origin, heading, 100.0).tolist())
        assert not near - slab(lows, highs, origin, heading, 8.0 + 1e-6) - {
            index for index in near
            if lows[index][0] <= 3.0 + 1e-9}

    def test_an_axis_aligned_ray_does_not_divide_by_zero(self, grid):
        """Two of the three components are exactly zero, which is the common case."""
        built, _lows, _highs = grid
        assert len(built.ray(np.array([10.5, -1.0, 10.5]),
                             np.array([0.0, 1.0, 0.0]), 5.0)) > 0

    def test_a_ray_going_nowhere_is_harmless(self, grid):
        built, _lows, _highs = grid
        assert len(built.ray(np.array([10.5, 0.0, 10.5]),
                             np.array([0.0, 0.0, 0.0]), 5.0)) == 0

    def test_a_ray_beginning_inside_the_mesh_finds_what_it_starts_in(self, grid):
        built, lows, highs = grid
        origin = np.array([10.5, 0.0, 10.5])
        heading = np.array([1.0, 0.0, 0.0])
        wanted = slab(lows, highs, origin, heading, 3.0)
        assert wanted <= set(built.ray(origin, heading, 3.0).tolist())


class TestTrianglesTooBigToBin:
    """A single triangle can span a level; binning it into every cell it
    touches would put thousands of entries in the grid for one triangle, so it
    goes on a list every query looks at."""

    def test_an_enormous_triangle_is_always_a_candidate(self):
        lows, highs = soup()
        lows = np.vstack([lows, [[-100.0, -100.0, -100.0]]])
        highs = np.vstack([highs, [[100.0, 100.0, 100.0]]])
        built = TriangleGrid(lows, highs)
        huge = len(lows) - 1
        assert huge in built.box(np.array([1.0, -1.0, 1.0]),
                                np.array([2.0, 1.0, 2.0])).tolist()
        assert huge in built.ray(np.array([1.5, -5.0, 1.5]),
                                 np.array([0.0, 1.0, 0.0]), 10.0).tolist()
