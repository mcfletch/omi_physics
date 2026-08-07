"""The grid answering a box query cheaply enough to ask it hundreds of times.

The grid's *algorithm* was never the problem: it narrows tens of thousands of
triangles to a handful. Its per-query *constant* was -- around a hundred
microseconds whether it answered with two triangles or two hundred, because
every query built intermediate arrays, ran ``np.unique`` over them and then
filtered by fancy indexing. A character controller asks nine to twelve times a
frame, so at a hundred characters that constant alone is tens of milliseconds
before a single triangle is tested.

These pin the cheap path: the same answer, without the per-query array
machinery. The contract is unchanged and is what the answer is checked against
-- a superset of the truly overlapping triangles, never missing one.
"""

import numpy as np
import pytest

from omi_physics.trigrid import TriangleGrid


def _soup(count=4000, extent=40.0, size=0.4, seed=7):
    """Triangle bounds scattered through a level-sized volume."""
    rng = np.random.default_rng(seed)
    lows = rng.uniform(-extent, extent, size=(count, 3))
    return lows, lows + rng.uniform(0.05, size, size=(count, 3))


@pytest.fixture
def grid():
    lows, highs = _soup()
    return TriangleGrid(lows, highs), lows, highs


def _truth(lows, highs, low, high):
    """Every triangle whose bounds really do overlap the query."""
    return set(np.flatnonzero(np.all(lows <= high, axis=1)
                              & np.all(highs >= low, axis=1)).tolist())


class TestTheAnswerIsStillRight:
    @pytest.mark.parametrize('seed', range(25))
    def test_a_query_never_loses_a_triangle_it_touches(self, grid, seed):
        built, lows, highs = grid
        rng = np.random.default_rng(1000 + seed)
        low = rng.uniform(-40.0, 38.0, size=3)
        high = low + rng.uniform(0.2, 6.0, size=3)
        assert _truth(lows, highs, low, high) <= set(built.box(low, high).tolist())

    @pytest.mark.parametrize('seed', range(25))
    def test_and_the_superset_it_hands_back_stays_small(self, grid, seed):
        """A box answers a *superset*; what matters is that it is a tight one.

        Exact equality is not the contract and is not worth buying: filtering
        the last few costs more numpy dispatch than the caller's own exact test
        now costs on them.
        """
        built, lows, highs = grid
        rng = np.random.default_rng(2000 + seed)
        low = rng.uniform(-40.0, 38.0, size=3)
        high = low + rng.uniform(0.2, 6.0, size=3)
        found = built.box(low, high)
        truth = _truth(lows, highs, low, high)
        assert truth <= set(found.tolist())
        assert len(found) <= max(24, 4 * len(truth))

    def test_a_query_off_the_end_of_the_grid_is_empty_not_an_error(self, grid):
        built, _lows, _highs = grid
        assert len(built.box(np.array([500.0] * 3), np.array([501.0] * 3))) == 0

    def test_a_query_covering_everything_answers_everything(self, grid):
        built, lows, _highs = grid
        found = built.box(np.array([-1e4] * 3), np.array([1e4] * 3))
        assert len(found) == len(lows)

    def test_indices_come_back_sorted_and_unique(self, grid):
        built, _lows, _highs = grid
        found = built.box(np.array([-5.0] * 3), np.array([5.0] * 3))
        assert list(found) == sorted(set(found.tolist()))

    def test_a_mesh_too_small_to_bin_is_still_answered_exactly(self):
        lows, highs = _soup(count=32, extent=4.0)
        built = TriangleGrid(lows, highs)
        low, high = np.array([-1.0] * 3), np.array([1.0] * 3)
        assert set(built.box(low, high).tolist()) == _truth(lows, highs, low, high)

    def test_an_oversized_triangle_is_never_missed(self):
        """One triangle spanning the level is not binned, and must still show up."""
        lows, highs = _soup(count=600, extent=20.0)
        lows[0] = (-20.0, -20.0, -20.0)
        highs[0] = (20.0, 20.0, 20.0)
        built = TriangleGrid(lows, highs)
        found = built.box(np.array([0.0] * 3), np.array([0.1] * 3))
        assert 0 in found.tolist()


class TestItIsCheapEnoughToAskOften:
    """The point of the change: the constant, not the complexity."""

    def test_a_small_query_does_not_cost_a_large_one(self, grid):
        """A capsule-sized query is what a character asks, nine times a frame."""
        import time
        built, _lows, _highs = grid
        low = np.array([0.0, 0.0, 0.0])
        high = np.array([0.8, 1.8, 0.8])
        built.box(low, high)                       # warm
        started = time.perf_counter()
        for _ in range(2000):
            built.box(low, high)
        each = (time.perf_counter() - started) / 2000
        # Generous against the ~100us it used to take; the point is the order
        # of magnitude, not the number. Twelve of these per character per frame
        # at a hundred characters has to fit in a frame with room to spare.
        assert each < 20e-6, '%.1f us per query' % (each * 1e6,)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
