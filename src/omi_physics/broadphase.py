"""Broad phase — a dynamic AABB tree with fattened boxes (Box2D ``b2DynamicTree``).

Leaves store fattened AABBs so small motions re-fit rather than re-insert.  Pair
queries walk the tree for fat-box overlaps, then keep only pairs whose *real*
boxes overlap and whose ``collisionFilters`` interact — so the reported set
equals the brute-force O(N²) set with no false negatives.
"""
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .world import PhysicsWorld

# The tree stores each box as a plain ``(x, y, z)`` float tuple, not a numpy
# array: every union/area/overlap test is on three scalars, where numpy's
# per-call ufunc/broadcast overhead dwarfs the arithmetic (a union is ~10x
# cheaper as tuple min/max). World AABBs arrive as numpy rows and are converted
# once at the insert/update/query boundary.
_Box = tuple[float, float, float]


class _Node:
    """One tree node: a fat AABB plus parent/child links; a leaf carries ``obj``."""
    __slots__ = ('lo', 'hi', 'parent', 'child1', 'child2', 'obj', 'height')

    lo: _Box | None
    hi: _Box | None
    parent: Optional['_Node']
    child1: Optional['_Node']
    child2: Optional['_Node']
    obj: int
    height: int

    def __init__(self) -> None:
        self.lo = None
        self.hi = None
        self.parent = None
        self.child1 = None
        self.child2 = None
        self.obj = -1          # >=0 for leaves
        self.height = 0

    @property
    def is_leaf(self) -> bool:
        """True when this node holds an object rather than two child nodes."""
        return self.child1 is None


def _area(lo: _Box, hi: _Box) -> float:
    """Surface area of the box ``[lo, hi]`` (the SAH cost metric)."""
    dx = hi[0] - lo[0]
    dy = hi[1] - lo[1]
    dz = hi[2] - lo[2]
    return 2.0 * (dx * dy + dy * dz + dz * dx)


def _union(a_lo: _Box, a_hi: _Box, b_lo: _Box, b_hi: _Box) -> tuple[_Box, _Box]:
    """Smallest box enclosing both input boxes, as ``(lo, hi)``."""
    return ((a_lo[0] if a_lo[0] < b_lo[0] else b_lo[0],
             a_lo[1] if a_lo[1] < b_lo[1] else b_lo[1],
             a_lo[2] if a_lo[2] < b_lo[2] else b_lo[2]),
            (a_hi[0] if a_hi[0] > b_hi[0] else b_hi[0],
             a_hi[1] if a_hi[1] > b_hi[1] else b_hi[1],
             a_hi[2] if a_hi[2] > b_hi[2] else b_hi[2]))


class DynamicAABBTree:
    """Balanced-ish AABB tree of fattened leaves for logarithmic overlap queries."""

    def __init__(self, fatten: float = 0.1) -> None:
        self.fatten = fatten
        self.root: _Node | None = None
        self.leaves: dict[int, _Node] = {}       # obj index -> _Node

    # -- insertion / removal --------------------------------------------
    def insert(self, obj: int, lo: np.ndarray, hi: np.ndarray) -> None:
        """Add object ``obj`` with real box ``[lo, hi]`` (stored fattened by ``fatten``)."""
        leaf = _Node()
        leaf.obj = obj
        f = self.fatten
        leaf.lo = (float(lo[0]) - f, float(lo[1]) - f, float(lo[2]) - f)
        leaf.hi = (float(hi[0]) + f, float(hi[1]) + f, float(hi[2]) + f)
        self.leaves[obj] = leaf
        self._insert_leaf(leaf)

    def remove(self, obj: int) -> None:
        """Remove object ``obj`` from the tree (a no-op if it is not present)."""
        leaf = self.leaves.pop(obj, None)
        if leaf is not None:
            self._remove_leaf(leaf)

    def update(self, obj: int, lo: np.ndarray, hi: np.ndarray) -> None:
        """Refit: only re-insert when the body left its fat box."""
        leaf = self.leaves.get(obj)
        if leaf is None:
            self.insert(obj, lo, hi)
            return
        assert leaf.lo is not None and leaf.hi is not None
        llo, lhi = leaf.lo, leaf.hi
        if (lo[0] >= llo[0] and lo[1] >= llo[1] and lo[2] >= llo[2] and
                hi[0] <= lhi[0] and hi[1] <= lhi[1] and hi[2] <= lhi[2]):
            return
        self._remove_leaf(leaf)
        f = self.fatten
        leaf.lo = (float(lo[0]) - f, float(lo[1]) - f, float(lo[2]) - f)
        leaf.hi = (float(hi[0]) + f, float(hi[1]) + f, float(hi[2]) + f)
        self._insert_leaf(leaf)

    def _insert_leaf(self, leaf: _Node) -> None:
        """Place ``leaf`` at the surface-area-cheapest sibling, then refit ancestors."""
        assert leaf.lo is not None and leaf.hi is not None
        if self.root is None:
            self.root = leaf
            leaf.parent = None
            return
        node = self.root
        while not node.is_leaf:
            assert node.lo is not None and node.hi is not None
            assert node.child1 is not None and node.child2 is not None
            combined = _area(*_union(node.lo, node.hi, leaf.lo, leaf.hi))
            cost = 2 * combined
            inherit = 2 * (combined - _area(node.lo, node.hi))
            cost1 = self._descent_cost(node.child1, leaf, inherit)
            cost2 = self._descent_cost(node.child2, leaf, inherit)
            if cost < cost1 and cost < cost2:
                break
            node = node.child1 if cost1 < cost2 else node.child2

        assert node.lo is not None and node.hi is not None
        old_parent = node.parent
        new_parent = _Node()
        new_parent.parent = old_parent
        new_parent.lo, new_parent.hi = _union(node.lo, node.hi, leaf.lo, leaf.hi)
        new_parent.height = node.height + 1
        new_parent.child1 = node
        new_parent.child2 = leaf
        node.parent = new_parent
        leaf.parent = new_parent
        if old_parent is None:
            self.root = new_parent
        elif old_parent.child1 is node:
            old_parent.child1 = new_parent
        else:
            old_parent.child2 = new_parent
        self._refit_ancestors(new_parent)

    def _descent_cost(self, child: _Node, leaf: _Node, inherit: float) -> float:
        """Surface-area cost of pushing ``leaf`` down into subtree ``child``."""
        assert child.lo is not None and child.hi is not None
        assert leaf.lo is not None and leaf.hi is not None
        lo, hi = _union(child.lo, child.hi, leaf.lo, leaf.hi)
        if child.is_leaf:
            return _area(lo, hi) + inherit
        return (_area(lo, hi) - _area(child.lo, child.hi)) + inherit

    def _remove_leaf(self, leaf: _Node) -> None:
        """Detach ``leaf``, promote its sibling into the parent slot, refit upward."""
        if leaf is self.root:
            self.root = None
            return
        parent = leaf.parent
        assert parent is not None
        grand = parent.parent
        sibling = parent.child2 if parent.child1 is leaf else parent.child1
        assert sibling is not None
        if grand is None:
            self.root = sibling
            sibling.parent = None
        else:
            if grand.child1 is parent:
                grand.child1 = sibling
            else:
                grand.child2 = sibling
            sibling.parent = grand
            self._refit_ancestors(grand)

    def _refit_ancestors(self, node: _Node | None) -> None:
        """Recompute each ancestor's box and height from ``node`` up to the root."""
        while node is not None:
            c1, c2 = node.child1, node.child2
            assert c1 is not None and c2 is not None
            assert c1.lo is not None and c1.hi is not None
            assert c2.lo is not None and c2.hi is not None
            node.lo, node.hi = _union(c1.lo, c1.hi, c2.lo, c2.hi)
            node.height = 1 + max(c1.height, c2.height)
            node = node.parent

    # -- queries ---------------------------------------------------------
    def query(self, lo: np.ndarray, hi: np.ndarray,
              skip: int | None = None) -> list[int]:
        """Object indices whose fat box overlaps ``[lo, hi]`` (excluding ``skip``)."""
        out: list[int] = []
        if self.root is None:
            return out
        lo0, lo1, lo2 = float(lo[0]), float(lo[1]), float(lo[2])
        hi0, hi1, hi2 = float(hi[0]), float(hi[1]), float(hi[2])
        stack = [self.root]
        while stack:
            n = stack.pop()
            nlo, nhi = n.lo, n.hi
            assert nlo is not None and nhi is not None
            if (hi0 < nlo[0] or hi1 < nlo[1] or hi2 < nlo[2] or
                    lo0 > nhi[0] or lo1 > nhi[1] or lo2 > nhi[2]):
                continue
            if n.is_leaf:
                if n.obj != skip:
                    out.append(n.obj)
            else:
                assert n.child1 is not None and n.child2 is not None
                stack.append(n.child1)
                stack.append(n.child2)
        return out

    def all_leaves(self) -> list[_Node]:
        """Every leaf node currently in the tree."""
        return list(self.leaves.values())


def _real_overlap(world: "PhysicsWorld", i: int, j: int) -> bool:
    """True when the bodies' *real* (unfattened) AABBs overlap."""
    imin, imax = world.aabb_min[i], world.aabb_max[i]
    jmin, jmax = world.aabb_min[j], world.aabb_max[j]
    return bool(imin[0] <= jmax[0] and imin[1] <= jmax[1] and imin[2] <= jmax[2] and
                jmin[0] <= imax[0] and jmin[1] <= imax[1] and jmin[2] <= imax[2])


#: The fewest inert bodies (static, no trigger, with a collider) for which
#: :meth:`BroadPhase.pairs` indexes them apart from the bodies that can move.
SPLIT_MIN = 256

#: A body is "large" when its x extent exceeds this many times the median
#: inert extent. Large inert bodies (a floor, a landscape) are tested against
#: every dynamic body; the rest are found by a window of the sorted x order.
_LARGE_EXTENT = 8.0


class _InertIndex:
    """The inert bodies of a world sorted by minimum x, kept while none of them changes.

    An inert body is static, has a collider and no trigger, so it pairs with
    dynamic bodies only. The index lists them in the broad phase's own order
    (minimum x, then body index), the small ones apart from the large, and
    remembers each one's rank in that order. :meth:`BroadPhase.pairs` rebuilds
    it when an inert body is added, removed or moved, which it finds by
    comparing the world's boxes with the ones the index was built from.
    """

    def __init__(self, world: "PhysicsWorld", inert: np.ndarray) -> None:
        count = world.body_count
        self.inert = inert.copy()
        self.lo = world.aabb_min[:count].copy()
        self.hi = world.aabb_max[:count].copy()
        rows = np.flatnonzero(inert)
        order = np.argsort(self.lo[rows, 0], kind='stable')
        self.rows = rows[order]                   # inert bodies by (min x, index)
        self.sorted_lo = self.lo[self.rows, 0]
        self.rank = np.zeros(count, dtype=np.intp)
        self.rank[self.rows] = np.arange(len(self.rows))
        width = self.hi[self.rows, 0] - self.lo[self.rows, 0]
        cutoff = _LARGE_EXTENT * max(float(np.median(width)), 1e-6)
        small = width <= cutoff
        self.large = self.rows[~small]            # tested against every dynamic body
        self.small = self.rows[small]
        self.small_lo = self.lo[self.small, 0]
        self.reach = float(width[small].max()) if small.any() else 0.0

    def current(self, world: "PhysicsWorld", inert: np.ndarray, movers: np.ndarray) -> bool:
        """Whether the world's inert bodies and their boxes are still the ones indexed.

        The bodies that can move have their rows brought up to date first, so
        that one comparison of whole arrays says whether an inert row changed.
        """
        count = world.body_count
        if count != len(self.inert) or not np.array_equal(inert, self.inert):
            return False
        low, high = world.aabb_min[:count], world.aabb_max[:count]
        self.lo[movers] = low[movers]
        self.hi[movers] = high[movers]
        return bool(np.array_equal(low, self.lo) and np.array_equal(high, self.hi))


def _ranks_before(sorted_lo: np.ndarray, sorted_rows: np.ndarray, lo: np.ndarray,
                  rows: np.ndarray) -> np.ndarray:
    """How many of ``sorted_rows`` come before each of ``rows`` in the order (min x, index).

    ``sorted_lo`` holds the minimum x of ``sorted_rows``, which are in that
    order. Bodies of equal minimum x are ordered by index, which takes a
    second search within the run of equals.
    """
    left = np.searchsorted(sorted_lo, lo, side='left')
    right = np.searchsorted(sorted_lo, lo, side='right')
    before = left.copy()
    for k in np.flatnonzero(right > left):
        before[k] += np.searchsorted(sorted_rows[left[k]:right[k]], rows[k], side='left')
    return before


class BroadPhase:
    """Owns the tree, keeps it in sync with the world, yields filtered pairs."""

    def __init__(self, fatten: float = 0.1) -> None:
        self.tree = DynamicAABBTree(fatten=fatten)
        self._known: set[int] = set()
        self._inert: _InertIndex | None = None

    def sync(self, world: "PhysicsWorld") -> None:
        """Insert/remove/refit tree leaves to match the world's current bodies."""
        current = set(range(world.body_count))
        for obj in self._known - current:
            self.tree.remove(obj)
        for i in current:
            self.tree.update(i, world.aabb_min[i], world.aabb_max[i])
        self._known = current

    def pairs(self, world: "PhysicsWorld") -> list[tuple[int, int]]:
        """Candidate colliding body pairs ``(i, j)``, real-box- and filter-checked.

        Skips pairs of two non-dynamic bodies, except a kinematic body against
        a trigger, so a lift or an avatar's proxy enters a trigger volume; and
        skips pairs whose collision filters do not interact. Equals the
        brute-force set.

        Vectorized sweep-and-prune: sort AABBs by min-x, then pair each body with
        the run of later bodies whose min-x lies below its max-x (one
        ``searchsorted``), and complete the y/z overlap and the dynamic/static
        test with numpy masks. The surviving pairs are filtered together
        (:func:`filtered_pairs`), so the Python cost is the output size, not a
        tree walk per body.

        A world of many static bodies (a forest of trunks, a field of rocks) is
        swept without them. They pair with dynamic bodies only, so they are
        sorted once and kept (:class:`_InertIndex`), and each dynamic body asks
        for the few that lie within its x range. The pairs come back in the
        order the sweep of every body would have given them.
        """
        n = world.body_count
        if n < 2:
            return []
        inert = None
        if n >= SPLIT_MIN:
            inert = ((world.motion_type == 0) & (world.trigger_shape < 0)
                     & (world.collider_shape >= 0))
            if np.count_nonzero(inert) < SPLIT_MIN:
                inert = None
        if inert is None:
            self._inert = None
            pair_a, pair_b = _sweep(world, None)
        else:
            pair_a, pair_b = self._pairs_split(world, inert)
        if not len(pair_a):
            return []
        return filtered_pairs(world, np.minimum(pair_a, pair_b), np.maximum(pair_a, pair_b))

    def _pairs_split(self, world: "PhysicsWorld", inert: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
        """The pairs of a world with an indexed inert set, in the order of a full sweep."""
        n = world.body_count
        movers = np.flatnonzero(~inert)
        index = self._inert
        if index is None or not index.current(world, inert, movers):
            index = self._inert = _InertIndex(world, inert)
        lo = world.aabb_min[:n]
        first_a, first_b = _sweep(world, movers)
        dynamic = movers[world.motion_type[movers] == 2]
        second_a, second_b = self._dynamic_with_inert(world, index, dynamic)
        pair_a = np.concatenate((first_a, second_a))
        pair_b = np.concatenate((first_b, second_b))
        if not len(pair_a):
            return pair_a, pair_b
        # Order by where each pair's bodies would have been in a sweep of all of them.
        order_movers = np.argsort(lo[movers, 0], kind='stable')
        sorted_movers = movers[order_movers]
        mover_rank = np.empty(len(movers), dtype=np.intp)
        mover_rank[order_movers] = np.arange(len(movers))
        sorted_mover_lo = lo[sorted_movers, 0]
        ranks = []
        for body in (pair_a, pair_b):
            rank = np.empty(len(body), dtype=np.intp)
            fixed = inert[body]
            if fixed.any():
                rows = body[fixed]
                rank[fixed] = index.rank[rows] + _ranks_before(
                    sorted_mover_lo, sorted_movers, lo[rows, 0], rows)
            loose = ~fixed
            if loose.any():
                rows = body[loose]
                rank[loose] = mover_rank[np.searchsorted(movers, rows)] + _ranks_before(
                    index.sorted_lo, index.rows, lo[rows, 0], rows)
            ranks.append(rank)
        first, last = np.minimum(*ranks), np.maximum(*ranks)
        sweep_order = np.lexsort((last, first))
        return pair_a[sweep_order], pair_b[sweep_order]

    @staticmethod
    def _dynamic_with_inert(world: "PhysicsWorld", index: _InertIndex,
                            dynamic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pairs of a dynamic body and an inert one whose boxes overlap."""
        lo, hi = world.aabb_min, world.aabb_max
        found_a: list[np.ndarray] = []
        found_b: list[np.ndarray] = []
        if len(dynamic) and len(index.small):
            # An inert body overlapping in x starts no further left than the
            # widest of them and no further right than the dynamic body ends.
            start = np.searchsorted(index.small_lo, lo[dynamic, 0] - index.reach, side='left')
            stop = np.searchsorted(index.small_lo, hi[dynamic, 0], side='right')
            counts = np.clip(stop - start, 0, None)
            total = int(counts.sum())
            if total:
                owner = np.repeat(dynamic, counts)
                within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                other = index.small[np.repeat(start, counts) + within]
                found_a.append(owner)
                found_b.append(other)
        for big in index.large:
            if len(dynamic):
                found_a.append(dynamic)
                found_b.append(np.full(len(dynamic), big, dtype=np.intp))
        if not found_a:
            empty = np.zeros(0, dtype=np.intp)
            return empty, empty
        a = np.concatenate(found_a)
        b = np.concatenate(found_b)
        keep = ((hi[a, 0] >= lo[b, 0]) & (hi[b, 0] >= lo[a, 0])
                & (hi[a, 1] >= lo[b, 1]) & (hi[b, 1] >= lo[a, 1])
                & (hi[a, 2] >= lo[b, 2]) & (hi[b, 2] >= lo[a, 2]))
        return a[keep], b[keep]


def _sweep(world: "PhysicsWorld", rows: np.ndarray | None
           ) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping pairs among bodies ``rows`` (all when None) with something to report.

    Sorted by minimum x, each body is paired with the run of later ones whose
    minimum x is within its maximum x, then y, z and the dynamic test are
    applied as masks. The pairs come out in the order of that sweep.
    """
    n = world.body_count
    if rows is None:
        lo = world.aabb_min[:n]
        hi = world.aabb_max[:n]
    else:
        n = len(rows)
        lo = world.aabb_min[rows]
        hi = world.aabb_max[rows]
    empty = np.zeros(0, dtype=np.intp)
    if n < 2:
        return empty, empty
    order = np.argsort(lo[:, 0], kind='stable')
    lo_sx = lo[order, 0]
    hi_sx = hi[order, 0]
    # For sorted body k, [k+1, ends[k]) is the run of later bodies whose
    # min-x is within k's max-x -- exactly the x-overlapping candidates.
    ends = np.searchsorted(lo_sx, hi_sx, side='right')
    k = np.arange(n)
    counts = ends - k - 1
    np.clip(counts, 0, None, out=counts)
    total = int(counts.sum())
    if total == 0:
        return empty, empty
    # Expand each k into its candidate window as flat (ki, kj) index arrays.
    ki = np.repeat(k, counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    kj = ki + 1 + within
    a = order[ki]
    b = order[kj]
    # x already overlaps by construction; finish y/z and require a body
    # that moves on one side of it.
    keep = ((hi[a, 1] >= lo[b, 1]) & (hi[b, 1] >= lo[a, 1]) &
            (hi[a, 2] >= lo[b, 2]) & (hi[b, 2] >= lo[a, 2]))
    if rows is not None:
        a, b = rows[a], rows[b]
    keep &= _moves_against(world, a, b)
    return a[keep], b[keep]


def _moves_against(world: "PhysicsWorld", a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Which pairs have something to report: a dynamic body, or a kinematic one at a trigger."""
    mt = world.motion_type
    trigger = world.trigger_shape >= 0
    return ((mt[a] == 2) | (mt[b] == 2)
            | ((mt[a] == 1) & trigger[b]) | ((mt[b] == 1) & trigger[a]))


def filtered_pairs(world: "PhysicsWorld", first: np.ndarray, second: np.ndarray
                   ) -> list[tuple[int, int]]:
    """The pairs ``(first[k], second[k])`` whose bodies' collision filters admit contact, in order.

    Each body's filter is the one :meth:`~omi_physics.world.PhysicsWorld.filter_for`
    gives its index, so an index of no filter is the default filter. The
    filters the pairs use are compared with each other once, in a table, and
    the pairs are looked up in it together.
    """
    count = len(world.filters)
    index = world.collider_filter[:world.body_count]
    # Every index of no filter means the default one: they share the code ``count``.
    code = np.where((index >= 0) & (index < count), index, count)
    used, inverse = np.unique(code[np.concatenate((first, second))], return_inverse=True)
    filters = [world.filter_for(int(c)) for c in used.tolist()]
    pair_a, pair_b = first.tolist(), second.tolist()
    if len(used) == 1:
        if filters[0].collides_with(filters[0]):
            return list(zip(pair_a, pair_b, strict=True))
        return []
    table = np.array([[one.collides_with(other) for other in filters] for one in filters],
                     dtype=bool)
    keep = table[inverse[:len(first)], inverse[len(first):]]
    return [pair for pair, kept in zip(zip(pair_a, pair_b, strict=True), keep.tolist(), strict=True)
            if kept]


def _filters_interact(world: "PhysicsWorld", a: int, b: int) -> bool:
    """True when bodies ``a`` and ``b``'s collision filters admit contact."""
    fa = world.filter_for(world.collider_filter[a])
    fb = world.filter_for(world.collider_filter[b])
    return fa.collides_with(fb)


def brute_force_pairs(world: "PhysicsWorld") -> list[tuple[int, int]]:
    """Reference O(N²) overlap set for tests."""
    out: list[tuple[int, int]] = []
    n = world.body_count
    for i in range(n):
        for j in range(i + 1, n):
            if not _moves_against(world, np.array([i]), np.array([j]))[0]:
                continue
            if _real_overlap(world, i, j) and _filters_interact(world, i, j):
                out.append((i, j))
    return out
