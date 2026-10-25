"""A grid over the footprints of a world's fixed bodies, for the bodies near a box.

A character depenetrates its capsule from the solid bodies that do not move
under the solver. In a world of a few levels that is every one of them; in a
world of tens of thousands of trunks, posts and rocks it is the handful within
a metre or two of the capsule. :class:`StaticGrid` answers that: the bodies
whose bounding boxes overlap a box, from a uniform grid over their ``(x, z)``
footprints.

The grid holds the bodies of motion type static. Three kinds are kept apart and
tested against every box instead, because a cell cannot hold them:

* bodies whose footprint spans many cells (a ground, a level's floor),
* bodies of kinematic motion, which move without the world's layout changing, and
* the bodies added, removed or placed since the grid was built, up to
  :attr:`StaticGrid.LOOSE_LIMIT` of them, read from the world as they are now.
  A streaming world adds and removes a rock at a time, and a rebuild for each
  would cost a pass over every trunk in it.
"""
import weakref
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .world import PhysicsWorld

__all__ = ['StaticGrid']

#: A body wider than this many cells, on either axis, is tested against every box.
LARGE_CELLS = 4

#: Cells a box may cover before the grid is skipped for a scan of its own bodies.
MAX_CELLS = 4096

#: Keys are ``x * _STRIDE + z`` over cell numbers offset to be positive.
_STRIDE = 1 << 24

#: A box edge past this many metres is clamped to it, so an infinite edge has cells.
_HUGE = 1.0e12

_STATIC, _KINEMATIC, _DYNAMIC = 0, 1, 2


class StaticGrid:
    """The fixed, solid bodies of ``world`` as they were at ``world.layout_epoch``."""

    #: How many changes since the build are read from the world before the grid is rebuilt.
    LOOSE_LIMIT = 4096

    def __init__(self, world: 'PhysicsWorld') -> None:
        # Weak: a grid is kept in a table keyed weakly by its world, and a
        # strong reference back would keep the world alive for as long as the table does.
        self._world = weakref.ref(world)
        #: The world's layout epoch when the grid was built.
        self.epoch = world.layout_epoch
        count = world.body_count
        motion = world.motion_type
        solid = (world.collider_shape >= 0) & ~world.sensor
        fixed = np.flatnonzero(solid & (motion == _STATIC))
        #: Kinematic bodies, whose boxes are read from the world at each query.
        self._movers = np.flatnonzero(solid & (motion == _KINEMATIC))
        lo, hi = world.aabb_min[:count][fixed], world.aabb_max[:count][fixed]
        width = np.maximum(hi[:, 0] - lo[:, 0], hi[:, 2] - lo[:, 2])
        finite = np.isfinite(width) & np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
        typical = float(np.median(width[finite])) if finite.any() else 1.0
        self.cell = max(4.0 * typical, 1.0)
        large = ~finite | (width > LARGE_CELLS * self.cell)
        #: Bodies tested against every box.
        self._large = fixed[large]
        self._large_lo, self._large_hi = lo[large], hi[large]
        gridded = ~large
        self._ids = fixed[gridded]
        self._lo, self._hi = lo[gridded], hi[gridded]
        self._fill_cells()
        self._loose = np.zeros(0, dtype=np.intp)
        self._loose_epoch = self.epoch

    @property
    def has_movers(self) -> bool:
        """Whether any of the bodies is kinematic, and so moves without the layout changing."""
        return bool(len(self._movers))

    def current(self) -> bool:
        """Whether the grid, with the bodies it reads from the world, answers for the world.

        False once a bulk addition, or more than :attr:`LOOSE_LIMIT` changes, has
        been made since the grid was built.
        """
        world = self._world()
        if world is None:
            return False
        if world.layout_epoch == self._loose_epoch:
            return True
        changed = world.layout_changes_since(self.epoch)
        if changed is None or len(changed) > self.LOOSE_LIMIT:
            return False
        self._loose = np.unique(np.asarray(changed, dtype=np.intp))
        self._loose_epoch = world.layout_epoch
        return True

    def _cells(self, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The first and last cell of each ``(x, z)`` range, as ``(n, 2)`` integers."""
        return (np.floor(low / self.cell).astype(np.int64),
                np.floor(high / self.cell).astype(np.int64))

    def _fill_cells(self) -> None:
        """Sort the body-in-cell pairs by cell, so a cell's bodies are a slice."""
        if not len(self._ids):
            self._origin = np.zeros(2, dtype=np.int64)
            self._reach = np.zeros(2, dtype=np.int64)
            self._keys = np.zeros(0, dtype=np.int64)
            self._starts = np.zeros(1, dtype=np.intp)
            self._members = np.zeros(0, dtype=np.intp)
            return
        first, last = self._cells(self._lo[:, [0, 2]], self._hi[:, [0, 2]])
        self._origin = first.min(axis=0) - 1
        first, last = first - self._origin, last - self._origin
        #: The last cell of the grid on each axis.
        self._reach = last.max(axis=0)
        across = last - first + 1
        counts = across[:, 0] * across[:, 1]
        body = np.repeat(np.arange(len(self._ids)), counts)
        within = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        wide = np.repeat(across[:, 0], counts)
        x = np.repeat(first[:, 0], counts) + within % wide
        z = np.repeat(first[:, 1], counts) + within // wide
        keys = x * _STRIDE + z
        order = np.argsort(keys, kind='stable')
        sorted_keys = keys[order]
        self._keys, starts = np.unique(sorted_keys, return_index=True)
        self._starts = np.append(starts, len(sorted_keys)).astype(np.intp)
        self._members = body[order]

    def near(self, lo: np.ndarray, hi: np.ndarray) -> list[int]:
        """The ids of the bodies whose boxes overlap the box ``lo`` to ``hi``, ascending.

        ``lo`` and ``hi`` may be infinite on an axis.
        """
        lo = np.asarray(lo, dtype='d')
        hi = np.asarray(hi, dtype='d')
        gridded = self._gridded(lo, hi)
        if len(self._loose):
            gridded = gridded[~np.isin(gridded, self._loose)]
        found = [gridded, self._others(lo, hi), self._changed(lo, hi)]
        return sorted(set(np.concatenate(found).tolist()))

    def _gridded(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """The bodies in the grid that overlap the box."""
        if not len(self._ids):
            return np.zeros(0, dtype=np.intp)
        first, last = (np.clip(cells - self._origin, 0, self._reach) for cells in self._cells(
            np.clip(lo[[0, 2]], -_HUGE, _HUGE), np.clip(hi[[0, 2]], -_HUGE, _HUGE)))
        if (last - first + 1).prod() > MAX_CELLS:
            candidates = np.arange(len(self._ids))
        else:
            xs = np.arange(first[0], last[0] + 1, dtype=np.int64)
            zs = np.arange(first[1], last[1] + 1, dtype=np.int64)
            wanted = (xs[:, None] * _STRIDE + zs[None, :]).ravel()
            at = np.searchsorted(self._keys, wanted)
            at = at[self._keys[np.minimum(at, len(self._keys) - 1)] == wanted]
            if not len(at):
                return np.zeros(0, dtype=np.intp)
            candidates = np.concatenate([self._members[self._starts[k]:self._starts[k + 1]]
                                         for k in at.tolist()])
        keep = (np.all(self._hi[candidates] >= lo, axis=1)
                & np.all(self._lo[candidates] <= hi, axis=1))
        return self._ids[candidates[keep]]

    def _changed(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """The bodies changed since the build that are fixed and solid now and overlap the box."""
        world = self._world()
        if world is None or not len(self._loose):
            return np.zeros(0, dtype=np.intp)
        rows = self._loose
        solid = ((world.collider_shape[rows] >= 0) & ~world.sensor[rows]
                 & (world.motion_type[rows] != _DYNAMIC))
        rows = rows[solid]
        return rows[np.all(world.aabb_max[rows] >= lo, axis=1)
                    & np.all(world.aabb_min[rows] <= hi, axis=1)]

    def _others(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """The large bodies and the kinematic ones that overlap the box."""
        found = [self._large[np.all(self._large_hi >= lo, axis=1)
                             & np.all(self._large_lo <= hi, axis=1)]]
        world = self._world()
        if len(self._movers) and world is not None:
            low = world.aabb_min[self._movers]
            high = world.aabb_max[self._movers]
            found.append(self._movers[np.all(high >= lo, axis=1) & np.all(low <= hi, axis=1)])
        return np.concatenate(found)
