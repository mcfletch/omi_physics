"""A uniform grid over a triangle soup, asked by box or along a ray.

Both things that interrogate a level's collision mesh do it many times a frame
— the character controller resolving contacts, and :mod:`omi_physics.raycast`
asking what a shot or a line of sight meets — and a level is one mesh of tens
of thousands of triangles.  Answering either by testing every triangle is O(T)
per query, which is felt: at 66k triangles a single cast costs milliseconds,
and a handful of bots each casting twice a tick will not fit in a frame.

The grid bins every triangle into the cells its bound overlaps, so a query
looks only at the triangles near it.  **A query is never allowed to lose a
triangle it touches** — the grid narrows, and the caller still does the exact
test on what comes back — so its contract is a *superset* of the true answer
and the only other thing that matters is that the superset is small.

Two ways to ask, because the callers want different things:

- :meth:`TriangleGrid.box` for a region, which is what a moving body sweeps.
- :meth:`TriangleGrid.ray` for a segment, which is what a cast is.  A ray's
  bounding box is a poor stand-in for the ray — a cast from one end of a level
  to the other has a box containing the whole level — so this walks the cells
  the ray actually enters, in order, and stops at its limit.
"""

from __future__ import annotations

import numpy as np

__all__ = ['TriangleGrid']

#: Roughly how many cells the longest side of the mesh is cut into.  The cost
#: of a query is the triangles in the cells it touches plus the cells
#: themselves, so a finer grid is not automatically better: past this the
#: bookkeeping of walking cells outgrows the triangles it saves.
TARGET_CELLS = 64

#: A triangle spanning more cells than this is not binned at all.  One triangle
#: can span a whole level -- a skybox face, a floor drawn as two triangles --
#: and binning it would put thousands of entries in the grid for one triangle,
#: which costs more memory and more query time than simply always checking it.
BIG_CELL_LIMIT = 64

#: Below this many triangles the grid costs more to build and walk than the
#: full scan it replaces, so there is no grid and every query is exact.
SMALLEST_WORTH_BINNING = 256

#: How far past a cell boundary a walk may drift before it is treated as having
#: crossed.  The walk is in units of distance along the ray, so this is metres.
EDGE = 1e-9


class TriangleGrid:
    """Triangle indices binned by where their bounds are.

    Built from the triangles' bounds rather than their vertices because that is
    what both queries compare against, and because the caller has usually
    computed them already.
    """

    __slots__ = ('_big', '_cell', '_cells', '_high', '_highs',
                 '_low', '_lows', '_shape')

    def __init__(self, lows: np.ndarray, highs: np.ndarray,
                 target: int = TARGET_CELLS,
                 big_cell_limit: int = BIG_CELL_LIMIT) -> None:
        self._lows = np.asarray(lows, dtype='d')
        self._highs = np.asarray(highs, dtype='d')
        self._cells: dict[tuple[int, int, int], np.ndarray] | None = None
        self._big = np.zeros(0, dtype='i')
        if len(self._lows) < SMALLEST_WORTH_BINNING:
            return
        self._low = self._lows.min(axis=0)
        self._high = self._highs.max(axis=0)
        extent = np.maximum(self._high - self._low, 1e-9)
        self._cell = float(extent.max()) / max(target, 1)
        self._build(big_cell_limit)

    def _build(self, big_cell_limit: int) -> None:
        """Bin every triangle, setting the oversized ones aside."""
        first = self._cell_of(self._lows)
        last = self._cell_of(self._highs)
        self._shape = tuple(int(value) for value in last.max(axis=0) + 1)
        spans = np.prod(last - first + 1, axis=1)
        binned: dict[tuple[int, int, int], list[int]] = {}
        big: list[int] = []
        for index in range(len(self._lows)):
            if spans[index] > big_cell_limit:
                big.append(index)
                continue
            for i in range(first[index, 0], last[index, 0] + 1):
                for j in range(first[index, 1], last[index, 1] + 1):
                    for k in range(first[index, 2], last[index, 2] + 1):
                        binned.setdefault((i, j, k), []).append(index)
        self._cells = {key: np.array(value, dtype='i')
                       for key, value in binned.items()}
        self._big = np.array(big, dtype='i')

    def _cell_of(self, point: np.ndarray) -> np.ndarray:
        """Which cell a point falls in, clamped to the grid."""
        raw = np.floor((point - self._low) / self._cell).astype(int)
        return np.clip(raw, 0, np.array(getattr(self, '_shape',
                                                (1 << 30, 1 << 30, 1 << 30))) - 1)

    # -- asking -------------------------------------------------------------

    def box(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Triangles whose bounds may overlap ``[low, high]``."""
        if self._cells is None:
            return self._exact(low, high)
        if np.any(low > self._high) or np.any(high < self._low):
            return self._big.copy()
        first, last = self._corner(low), self._corner(high)
        found = [self._cells[key]
                 for i in range(first[0], last[0] + 1)
                 for j in range(first[1], last[1] + 1)
                 for k in range(first[2], last[2] + 1)
                 if (key := (i, j, k)) in self._cells]
        return self._gathered(found, low, high, tighten=False)

    def _corner(self, point: np.ndarray) -> tuple[int, int, int]:
        """:meth:`_cell_of` for a single point, in plain Python arithmetic.

        The same answer as the array form, which stays for the build and for
        the ray walk. Asked twice per box query and doing three divisions, it
        spent most of its time entering and leaving numpy rather than dividing
        -- and a character controller asks it more often than anything else
        here does.
        """
        cell, low, shape = self._cell, self._low, self._shape
        return (
            min(max(int((point[0] - low[0]) // cell), 0), shape[0] - 1),
            min(max(int((point[1] - low[1]) // cell), 0), shape[1] - 1),
            min(max(int((point[2] - low[2]) // cell), 0), shape[2] - 1),
        )

    def ray(self, origin: np.ndarray, heading: np.ndarray,
            limit: float) -> np.ndarray:
        """Triangles whose bounds the segment ``origin -> limit`` may meet.

        The cells are walked in the order the ray enters them, which is what
        makes a long cast cheap: a full-length cast across a level touches a
        line of cells rather than the volume its bounding box describes.
        """
        origin = np.asarray(origin, dtype='d')
        heading = np.asarray(heading, dtype='d')
        if self._cells is None:
            return self._exact(*_segment_bounds(origin, heading, limit))
        entry = _enters(origin, heading, self._low, self._high, limit)
        if entry is None:
            return self._big.copy()
        found = [self._cells[key] for key in self._walk(origin, heading, entry)
                 if key in self._cells]
        return self._gathered(found, *_segment_bounds(origin, heading, limit))

    def _walk(self, origin: np.ndarray, heading: np.ndarray,
              entry: tuple[float, float]):
        """The cells the ray crosses, from where it enters to where it stops.

        The standard voxel traversal: hold, per axis, the distance at which the
        ray next crosses a boundary on that axis, and repeatedly advance
        whichever is nearest.  An axis the ray does not move along never comes
        up, which is the common case -- most casts are along one axis or in one
        plane.
        """
        near, far = entry
        cell = self._cell_of(origin + heading * max(near, 0.0))
        step = np.where(heading > 0, 1, -1)
        with np.errstate(divide='ignore', invalid='ignore'):
            crossing = np.abs(self._cell / heading)
            edge = self._low + (cell + (heading > 0)) * self._cell
            following = (edge - origin) / heading
        crossing = np.where(np.isfinite(crossing), crossing, np.inf)
        following = np.where(np.isfinite(following), following, np.inf)
        while True:
            yield (int(cell[0]), int(cell[1]), int(cell[2]))
            axis = int(np.argmin(following))
            if following[axis] > far + EDGE:
                return
            cell[axis] += step[axis]
            if not 0 <= cell[axis] < self._shape[axis]:
                return
            following[axis] += crossing[axis]

    # -- the answer ---------------------------------------------------------

    def _gathered(self, found: list[np.ndarray], low: np.ndarray,
                  high: np.ndarray, tighten: bool = True) -> np.ndarray:
        """One sorted array of candidates, with the oversized triangles added.

        ``tighten`` filters by the query's own bounds on the way out. A ray
        wants it and asks for it: it is what keeps a cast from answering with
        triangles past its own limit, which is semantics rather than speed.

        A box does not. Its contract is a superset -- the caller does the exact
        test either way -- and that test is now a fraction of a microsecond a
        triangle, while the filter is two gathers and four reductions however
        few candidates there are. Paying it to save a cheaper thing is the
        wrong way round, and a character-sized box is asked more often than
        anything else here.

        A single cell's array is already sorted and unique, so the case such a
        box hits most often skips the merge entirely.
        """
        if len(self._big):
            found = found + [self._big]
        if not found:
            return np.zeros(0, dtype='i')
        candidates = (found[0] if len(found) == 1
                      else np.unique(np.concatenate(found)))
        if not tighten:
            return candidates
        keep = (np.all(self._lows[candidates] <= high, axis=1)
                & np.all(self._highs[candidates] >= low, axis=1))
        return candidates[keep]

    def _exact(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Every overlapping triangle, for a mesh too small to be worth binning."""
        return np.flatnonzero(np.all(self._lows <= high, axis=1)
                              & np.all(self._highs >= low, axis=1))


def _segment_bounds(origin: np.ndarray, heading: np.ndarray,
                    limit: float) -> tuple[np.ndarray, np.ndarray]:
    """The box holding a segment, used to reject candidates a cell handed back."""
    finish = origin + heading * limit
    return (np.minimum(origin, finish), np.maximum(origin, finish))


def _enters(origin: np.ndarray, heading: np.ndarray, low: np.ndarray,
            high: np.ndarray, limit: float) -> tuple[float, float] | None:
    """Where along the ray it is inside ``[low, high]``, or None for never.

    Both ends are wanted, not just whether it hits: the near end is where the
    walk starts and the far end is where it stops, and starting a walk at the
    ray's own origin would step through empty space for a cast that begins
    outside the mesh.
    """
    if not np.any(heading):
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse = 1.0 / heading
        first = (low - origin) * inverse
        second = (high - origin) * inverse
    near = float(np.nanmax(np.minimum(first, second)))
    far = float(np.nanmin(np.maximum(first, second)))
    if far < max(near, 0.0) or near > limit:
        return None
    return (near, min(far, limit))
