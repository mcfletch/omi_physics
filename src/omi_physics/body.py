"""Engine shape proxies built from :class:`~omi_physics.model.Shape`.

A proxy is the world-space collision form of a shape at a given pose.  The
narrow phase consumes proxies; the debug pass tessellates them.  All proxies
expose enough for analytic tests and a generic support function (for GJK/EPA in
Phase 3).

Every support proxy (everything except :class:`TriangleMeshProxy`) offers the same
three calls the narrow phase relies on: ``aabb()`` for the broad-phase box,
``support(direction)`` for the farthest point along a direction (GJK/EPA), and
``center_hint()`` for a representative interior point.  The type aliases
:data:`SupportProxy` and :data:`Proxy` name these two proxy families for callers.
"""
from typing import Dict, Iterator, Optional, Tuple, Union, TYPE_CHECKING
from numpy.typing import ArrayLike
import numpy as np

from . import mathutil
from . import model
from .mathutil import Vec

_AABB = Tuple[np.ndarray, np.ndarray]           # (lo, hi) corner pair


def _box_corners(half: np.ndarray) -> np.ndarray:
    """The 8 local corner offsets of an axis-aligned box with the given half-extents."""
    x, y, z = half
    return np.array([(sx * x, sy * y, sz * z)
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


class SphereProxy:
    """A world-space sphere: centre plus radius."""
    kind = 'sphere'

    def __init__(self, center: ArrayLike, radius: float):
        self.center = np.asarray(center, dtype='d')
        self.radius = float(radius)

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box as ``(lo, hi)``."""
        r = self.radius
        return self.center - r, self.center + r

    def support(self, direction: Vec) -> np.ndarray:
        """Farthest surface point along ``direction`` (need not be unit length)."""
        return self.center + self.radius * mathutil.normalize(direction)

    def center_hint(self) -> np.ndarray:
        """A representative interior point (the sphere centre)."""
        return self.center


class BoxProxy:
    """A world-space oriented box: centre, half-extents, and a rotation matrix."""
    kind = 'box'

    def __init__(self, center: ArrayLike, half: ArrayLike, rotation: ArrayLike):
        self.center = np.asarray(center, dtype='d')
        self.half = np.asarray(half, dtype='d')
        self.R = np.asarray(rotation, dtype='d')     # columns = world axes

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box as ``(lo, hi)``."""
        corners = _box_corners(self.half) @ self.R.T + self.center
        return corners.min(axis=0), corners.max(axis=0)

    def support(self, direction: Vec) -> np.ndarray:
        """Farthest corner along ``direction``."""
        local = self.R.T @ np.asarray(direction, dtype='d')
        sign = np.sign(local)
        return self.center + self.R @ (sign * self.half)

    def center_hint(self) -> np.ndarray:
        """A representative interior point (the box centre)."""
        return self.center


class CapsuleProxy:
    """Line-segment-plus-radius along the local Y axis."""
    kind = 'capsule'

    def __init__(self, center: ArrayLike, half_height: float, radius: float,
                 rotation: ArrayLike):
        self.center = np.asarray(center, dtype='d')
        self.half_height = float(half_height)
        self.radius = float(radius)
        self.R = np.asarray(rotation, dtype='d')
        axis = self.R @ np.array([0.0, 1.0, 0.0])
        self.p0 = self.center - axis * self.half_height
        self.p1 = self.center + axis * self.half_height

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box as ``(lo, hi)``."""
        lo = np.minimum(self.p0, self.p1) - self.radius
        hi = np.maximum(self.p0, self.p1) + self.radius
        return lo, hi

    def support(self, direction: Vec) -> np.ndarray:
        """Farthest surface point along ``direction`` (segment endpoint + radius)."""
        d = np.asarray(direction, dtype='d')
        end = self.p0 if np.dot(self.p0, d) > np.dot(self.p1, d) else self.p1
        return end + self.radius * mathutil.normalize(d)

    def center_hint(self) -> np.ndarray:
        """A representative interior point (the capsule centre)."""
        return self.center


class ConvexProxy:
    """A world-space convex point cloud (its own convex hull)."""
    kind = 'convex'

    def __init__(self, points: ArrayLike, center: ArrayLike, rotation: ArrayLike):
        self.local = np.asarray(points, dtype='d')
        self.center = np.asarray(center, dtype='d')
        self.R = np.asarray(rotation, dtype='d')
        self.world = self.local @ self.R.T + self.center

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box as ``(lo, hi)``."""
        return self.world.min(axis=0), self.world.max(axis=0)

    def support(self, direction: Vec) -> np.ndarray:
        """Farthest vertex along ``direction``."""
        d = np.asarray(direction, dtype='d')
        return self.world[np.argmax(self.world @ d)]

    def center_hint(self) -> np.ndarray:
        """A representative interior point (the vertex centroid)."""
        return self.world.mean(axis=0)


class TriangleProxy:
    """A single world-space triangle, as a degenerate convex for GJK/EPA."""
    kind = 'triangle'

    def __init__(self, v0: ArrayLike, v1: ArrayLike, v2: ArrayLike):
        self.verts = np.array([v0, v1, v2], dtype='d')

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box as ``(lo, hi)``."""
        return self.verts.min(axis=0), self.verts.max(axis=0)

    def support(self, direction: Vec) -> np.ndarray:
        """Farthest of the three vertices along ``direction``."""
        return self.verts[np.argmax(self.verts @ np.asarray(direction, dtype='d'))]

    def center_hint(self) -> np.ndarray:
        """A representative interior point (the triangle centroid)."""
        return self.verts.mean(axis=0)


class TriangleMeshProxy:
    """Static triangle soup: per-triangle collision against a moving convex.

    Triangle AABBs are precomputed for a cheap vectorized overlap query; the
    proxy is meant to be cached for a (static) body since it never moves.
    """
    kind = 'trimesh'

    def __init__(self, points: ArrayLike, indices: ArrayLike, center: ArrayLike,
                 rotation: ArrayLike):
        self.world_pts = np.asarray(points, dtype='d') @ np.asarray(rotation).T + center
        self.indices = np.asarray(indices, dtype='i')
        tris = self.world_pts[self.indices]
        self.tri_lo = tris.min(axis=1)
        self.tri_hi = tris.max(axis=1)
        # A uniform grid so per-query cost is proportional to the triangles near
        # the query, not the whole mesh — a full-mesh AABB scan is O(T) and the
        # character controller queries many times per frame (85k-tri models
        # otherwise stutter). Only built for meshes large enough to matter.
        self._grid: Optional[Dict[tuple, np.ndarray]] = None
        if len(self.indices) > 256:
            self._build_grid()

    def _build_grid(self, target: int = 64, big_cell_limit: int = 64) -> None:
        """Bin triangles into a uniform spatial grid; oversized triangles go to a
        separate always-check list so no cell holds thousands of them."""
        lo = self.world_pts.min(axis=0)
        self._grid_lo = lo
        extent = np.maximum(self.world_pts.max(axis=0) - lo, 1e-9)
        self._cell = float(extent.max()) / target
        cmin = np.floor((self.tri_lo - lo) / self._cell).astype(int)
        cmax = np.floor((self.tri_hi - lo) / self._cell).astype(int)
        spans = np.prod(cmax - cmin + 1, axis=1)
        grid: Dict[tuple, list] = {}
        big = []
        for t in range(len(self.indices)):
            if spans[t] > big_cell_limit:       # huge triangle: always-check list
                big.append(t)
                continue
            for i in range(cmin[t, 0], cmax[t, 0] + 1):
                for j in range(cmin[t, 1], cmax[t, 1] + 1):
                    for k in range(cmin[t, 2], cmax[t, 2] + 1):
                        grid.setdefault((i, j, k), []).append(t)
        self._grid = {key: np.array(v, dtype='i') for key, v in grid.items()}
        self._big = np.array(big, dtype='i')

    def _candidate_indices(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Triangle indices whose AABBs may overlap the query box ``[lo, hi]``."""
        if self._grid is None:
            return np.nonzero(np.all(self.tri_lo <= hi, axis=1)
                              & np.all(self.tri_hi >= lo, axis=1))[0]
        cmin = np.floor((lo - self._grid_lo) / self._cell).astype(int)
        cmax = np.floor((hi - self._grid_lo) / self._cell).astype(int)
        parts = []
        for i in range(cmin[0], cmax[0] + 1):
            for j in range(cmin[1], cmax[1] + 1):
                for k in range(cmin[2], cmax[2] + 1):
                    cell = self._grid.get((i, j, k))
                    if cell is not None:
                        parts.append(cell)
        if len(self._big):
            parts.append(self._big)
        if not parts:
            return np.zeros(0, dtype='i')
        cand = np.unique(np.concatenate(parts))
        keep = np.all(self.tri_lo[cand] <= hi, axis=1) & np.all(self.tri_hi[cand] >= lo, axis=1)
        return cand[keep]

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box of the whole mesh as ``(lo, hi)``."""
        return self.world_pts.min(axis=0), self.world_pts.max(axis=0)

    def triangles_overlapping(self, lo: np.ndarray, hi: np.ndarray) -> 'Iterator[TriangleProxy]':
        """Yield a :class:`TriangleProxy` for each triangle near the box ``[lo, hi]``."""
        for idx in self._candidate_indices(lo, hi):
            i0, i1, i2 = self.indices[idx]
            yield TriangleProxy(self.world_pts[i0], self.world_pts[i1], self.world_pts[i2])

    def center_hint(self) -> np.ndarray:
        """A representative interior point (the vertex centroid)."""
        return self.world_pts.mean(axis=0)


# A support proxy answers ``support(direction)``; the mesh proxy does not (it is
# collided triangle-by-triangle instead).  ``Proxy`` is any collision proxy.
SupportProxy = Union[SphereProxy, BoxProxy, CapsuleProxy, ConvexProxy, TriangleProxy]
Proxy = Union[SupportProxy, TriangleMeshProxy]


def make_proxy(shape: 'model.Shape', position: ArrayLike,
               orientation: ArrayLike) -> Proxy:
    """Build the world-space collision proxy for ``shape`` at the given pose."""
    R = mathutil.quat_to_matrix(np.asarray(orientation, dtype='d'))
    pos = np.asarray(position, dtype='d')
    t = shape.type
    if t == 'sphere':
        return SphereProxy(pos, shape.radius)
    if t == 'box':
        return BoxProxy(pos, 0.5 * np.asarray(shape.size), R)
    if t == 'capsule':
        return CapsuleProxy(pos, 0.5 * shape.height, shape.radiusBottom, R)
    if t == 'cylinder':
        # treated as a capsule-ish convex for AABB / support; narrow phase uses support
        return CapsuleProxy(pos, 0.5 * shape.height, shape.radiusBottom, R)
    if t == 'convex':
        if shape.points is None:
            raise ValueError('convex shape has no points')
        return ConvexProxy(shape.points, pos, R)
    if t == 'trimesh':
        if shape.points is None or shape.indices is None:
            raise ValueError('trimesh shape needs both points and indices')
        return TriangleMeshProxy(shape.points, shape.indices, pos, R)
    raise ValueError('unknown shape type %r' % t)


def world_aabb(shape: 'model.Shape', position: ArrayLike,
               orientation: ArrayLike) -> _AABB:
    """Axis-aligned bounding box of ``shape`` at the given pose, as ``(lo, hi)``."""
    return make_proxy(shape, position, orientation).aabb()
