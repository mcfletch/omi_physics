"""Broad phase — a dynamic AABB tree with fattened boxes (Box2D ``b2DynamicTree``).

Leaves store fattened AABBs so small motions re-fit rather than re-insert.  Pair
queries walk the tree for fat-box overlaps, then keep only pairs whose *real*
boxes overlap and whose ``collisionFilters`` interact — so the reported set
equals the brute-force O(N²) set with no false negatives.
"""
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .world import PhysicsWorld

# The tree stores each box as a plain ``(x, y, z)`` float tuple, not a numpy
# array: every union/area/overlap test is on three scalars, where numpy's
# per-call ufunc/broadcast overhead dwarfs the arithmetic (a union is ~10x
# cheaper as tuple min/max). World AABBs arrive as numpy rows and are converted
# once at the insert/update/query boundary.
_Box = Tuple[float, float, float]


class _Node:
    """One tree node: a fat AABB plus parent/child links; a leaf carries ``obj``."""
    __slots__ = ('lo', 'hi', 'parent', 'child1', 'child2', 'obj', 'height')

    lo: Optional[_Box]
    hi: Optional[_Box]
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


def _union(a_lo: _Box, a_hi: _Box, b_lo: _Box, b_hi: _Box) -> Tuple[_Box, _Box]:
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
        self.root: Optional[_Node] = None
        self.leaves: Dict[int, _Node] = {}       # obj index -> _Node

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

    def _refit_ancestors(self, node: Optional[_Node]) -> None:
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
              skip: Optional[int] = None) -> List[int]:
        """Object indices whose fat box overlaps ``[lo, hi]`` (excluding ``skip``)."""
        out: List[int] = []
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

    def all_leaves(self) -> List[_Node]:
        """Every leaf node currently in the tree."""
        return list(self.leaves.values())


def _real_overlap(world: "PhysicsWorld", i: int, j: int) -> bool:
    """True when the bodies' *real* (unfattened) AABBs overlap."""
    imin, imax = world.aabb_min[i], world.aabb_max[i]
    jmin, jmax = world.aabb_min[j], world.aabb_max[j]
    return bool(imin[0] <= jmax[0] and imin[1] <= jmax[1] and imin[2] <= jmax[2] and
                jmin[0] <= imax[0] and jmin[1] <= imax[1] and jmin[2] <= imax[2])


class BroadPhase:
    """Owns the tree, keeps it in sync with the world, yields filtered pairs."""

    def __init__(self, fatten: float = 0.1) -> None:
        self.tree = DynamicAABBTree(fatten=fatten)
        self._known: Set[int] = set()

    def sync(self, world: "PhysicsWorld") -> None:
        """Insert/remove/refit tree leaves to match the world's current bodies."""
        current = set(range(world.body_count))
        for obj in self._known - current:
            self.tree.remove(obj)
        for i in current:
            self.tree.update(i, world.aabb_min[i], world.aabb_max[i])
        self._known = current

    def pairs(self, world: "PhysicsWorld") -> List[Tuple[int, int]]:
        """Candidate colliding body pairs ``(i, j)``, real-box- and filter-checked.

        Skips pairs of two non-dynamic bodies and pairs whose collision filters
        do not interact; equals the brute-force set.

        Vectorized sweep-and-prune: sort AABBs by min-x, then pair each body with
        the run of later bodies whose min-x lies below its max-x (one
        ``searchsorted``), and complete the y/z overlap and the dynamic/static
        test with numpy masks. Only the surviving real-overlap pairs pay the
        per-pair collision-filter check, so the Python cost is the output size,
        not a tree walk per body.
        """
        n = world.body_count
        if n < 2:
            return []
        lo = world.aabb_min[:n]
        hi = world.aabb_max[:n]
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
            return []
        # Expand each k into its candidate window as flat (ki, kj) index arrays.
        ki = np.repeat(k, counts)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        kj = ki + 1 + within
        a = order[ki]
        b = order[kj]
        # x already overlaps by construction; finish y/z and require a dynamic body.
        keep = ((hi[a, 1] >= lo[b, 1]) & (hi[b, 1] >= lo[a, 1]) &
                (hi[a, 2] >= lo[b, 2]) & (hi[b, 2] >= lo[a, 2]))
        mt = world.motion_type[:n]
        keep &= (mt[a] == 2) | (mt[b] == 2)
        a, b = a[keep], b[keep]
        lo_idx = np.minimum(a, b).tolist()
        hi_idx = np.maximum(a, b).tolist()
        # Fast path: with no custom collision filters (the common case) every
        # surviving pair collides, so skip the per-pair filter probe entirely.
        cf = world.collider_filter[:n]
        if n == 0 or (int(cf.max()) == 0 and int(cf.min()) == 0
                      and world.filter_for(0).collides_with(world.filter_for(0))):
            return list(zip(lo_idx, hi_idx))
        return [(p, q) for p, q in zip(lo_idx, hi_idx)
                if _filters_interact(world, p, q)]


def _filters_interact(world: "PhysicsWorld", a: int, b: int) -> bool:
    """True when bodies ``a`` and ``b``'s collision filters admit contact."""
    fa = world.filter_for(world.collider_filter[a])
    fb = world.filter_for(world.collider_filter[b])
    return fa.collides_with(fb)


def brute_force_pairs(world: "PhysicsWorld") -> List[Tuple[int, int]]:
    """Reference O(N²) overlap set for tests."""
    out: List[Tuple[int, int]] = []
    n = world.body_count
    for i in range(n):
        for j in range(i + 1, n):
            if world.motion_type[i] != 2 and world.motion_type[j] != 2:
                continue
            if _real_overlap(world, i, j) and _filters_interact(world, i, j):
                out.append((i, j))
    return out
