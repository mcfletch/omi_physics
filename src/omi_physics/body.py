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
import functools
import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike

from . import mathutil, model
from .heightfield import HeightSurface
from .mathutil import Vec
from .trigrid import TriangleGrid

_AABB = tuple[np.ndarray, np.ndarray]           # (lo, hi) corner pair

#: Two triangles sharing an edge whose faces are within 5 degrees of parallel make
#: one surface for :attr:`TriangleMeshProxy.internal_edges`.
INTERNAL_EDGE_COS = math.cos(math.radians(5.0))


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

    def planes(self) -> tuple[np.ndarray, np.ndarray]:
        """The six faces as world ``(normals, offsets)``, normals out: ``normal . x = offset``."""
        normals = np.vstack((self.R.T, -self.R.T))
        return normals, normals @ self.center + np.concatenate((self.half, self.half))

    def corners(self) -> np.ndarray:
        """The eight corners in the world, ``(8, 3)``."""
        return _box_corners(self.half) @ self.R.T + self.center

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box as ``(lo, hi)``."""
        corners = self.corners()
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
        # The local Y axis turned into the world is R's second column, which is
        # what R @ (0, 1, 0) computes -- as a view, without the matrix product
        # or the vector to multiply it by. A capsule proxy is rebuilt several
        # times a frame for every character in the world.
        axis = self.R[:, 1]
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


class CylinderProxy:
    """A flat-ended cylinder along the local Y axis, narrowing to a cone.

    ``radius_bottom`` is the radius of the end at ``-half_height`` and
    ``radius_top`` of the end at ``+half_height``; either may be zero.
    """
    kind = 'cylinder'

    def __init__(self, center: ArrayLike, half_height: float, radius_bottom: float,
                 radius_top: float, rotation: ArrayLike):
        self.center = np.asarray(center, dtype='d')
        self.half_height = float(half_height)
        self.radius_bottom = float(radius_bottom)
        self.radius_top = float(radius_top)
        self.R = np.asarray(rotation, dtype='d')
        self.axis = self.R[:, 1]
        self.bottom = self.center - self.axis * self.half_height
        self.top = self.center + self.axis * self.half_height

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box as ``(lo, hi)``: the boxes of the two end discs."""
        # A disc of radius r across axis a reaches r * sqrt(1 - a_k^2) along
        # world axis k.
        spread = np.sqrt(np.maximum(1.0 - self.axis * self.axis, 0.0))
        lo = np.minimum(self.bottom - self.radius_bottom * spread,
                        self.top - self.radius_top * spread)
        hi = np.maximum(self.bottom + self.radius_bottom * spread,
                        self.top + self.radius_top * spread)
        return lo, hi

    def support(self, direction: Vec) -> np.ndarray:
        """Farthest point along ``direction``: a point on the rim of one end."""
        d = np.asarray(direction, dtype='d')
        radial = d - np.dot(d, self.axis) * self.axis
        length = float(np.linalg.norm(radial))
        radial = radial / length if length > 1e-12 else np.zeros(3)
        bottom = self.bottom + self.radius_bottom * radial
        top = self.top + self.radius_top * radial
        return top if np.dot(top, d) >= np.dot(bottom, d) else bottom

    def planes(self, sides: int = 24) -> tuple[np.ndarray, np.ndarray]:
        """The two ends and ``sides`` side facets as world ``(normals, offsets)``, normals out."""
        angle = (np.arange(sides) + 0.5) * (2.0 * np.pi / sides)
        radial = np.cos(angle)[:, None] * self.R[:, 0] + np.sin(angle)[:, None] * self.R[:, 2]
        length = 2.0 * self.half_height
        inscribed = np.cos(np.pi / sides)
        # A facet runs from the middle of an edge of the polygon inscribed in one
        # end to the same edge's middle at the other, so it narrows by
        # (bottom - top) * inscribed over its length and tilts toward the axis by that.
        side = length * radial + (self.radius_bottom - self.radius_top) * inscribed * self.axis
        side /= np.linalg.norm(side, axis=1, keepdims=True)
        reach = self.radius_bottom * inscribed
        on_side = self.bottom + reach * radial
        normals = np.vstack((self.axis, -self.axis, side))
        offsets = np.concatenate((
            [self.axis @ self.top, -(self.axis @ self.bottom)],
            np.einsum('ij,ij->i', side, on_side)))
        return normals, offsets

    def center_hint(self) -> np.ndarray:
        """A representative interior point (the middle of the axis)."""
        return self.center


class CompoundProxy:
    """Several proxies that move as one body, collided part by part."""
    kind = 'compound'

    def __init__(self, parts: 'list[Proxy]'):
        self.parts = parts

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box of every part, as ``(lo, hi)``."""
        boxes = [part.aabb() for part in self.parts]
        return (np.min([lo for lo, _ in boxes], axis=0),
                np.max([hi for _, hi in boxes], axis=0))

    def center_hint(self) -> np.ndarray:
        """A representative point (the mean of the parts' own)."""
        return np.mean([part.center_hint() for part in self.parts], axis=0)


class ConvexProxy:
    """A world-space convex point cloud (its own convex hull)."""
    kind = 'convex'

    def __init__(self, points: ArrayLike, center: ArrayLike, rotation: ArrayLike,
                 planes: tuple[np.ndarray, np.ndarray] | None = None):
        self.local = np.asarray(points, dtype='d')
        self.center = np.asarray(center, dtype='d')
        self.R = np.asarray(rotation, dtype='d')
        self.world = self.local @ self.R.T + self.center
        self._local_planes = planes

    def planes(self) -> tuple[np.ndarray, np.ndarray]:
        """The hull's faces as world ``(normals, offsets)``, normals out: ``normal . x = offset``.

        Taken from the shape's cached faces when the proxy was made with them.
        """
        if self._local_planes is None:
            self._local_planes = model.Shape.convex(self.local).hull_planes()
        normals, offsets = self._local_planes
        placed = normals @ self.R.T
        return placed, offsets + placed @ self.center

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

    Triangle AABBs are precomputed for a vectorized overlap query. The proxy
    holds the triangles at one pose and is cached for a body while it keeps that pose.
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
        # the query, not the whole mesh -- a full-mesh AABB scan is O(T) and the
        # character controller queries many times per frame (85k-tri models
        # otherwise stutter).
        self._grid = TriangleGrid(self.tri_lo, self.tri_hi)

    def _candidate_indices(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Triangle indices whose AABBs may overlap the query box ``[lo, hi]``."""
        return self._grid.box(lo, hi)

    def aabb(self) -> _AABB:
        """Axis-aligned bounding box of the whole mesh as ``(lo, hi)``."""
        return self.world_pts.min(axis=0), self.world_pts.max(axis=0)

    def triangles_overlapping(self, lo: np.ndarray, hi: np.ndarray) -> 'Iterator[TriangleProxy]':
        """Yield a :class:`TriangleProxy` for each triangle near the box ``[lo, hi]``.

        One Python object per candidate, which is why the capsule path uses
        :meth:`candidate_vertices` instead: a character asks this many times a
        frame and the proxies are built only to be read once and dropped.
        """
        for idx in self._candidate_indices(lo, hi):
            i0, i1, i2 = self.indices[idx]
            yield TriangleProxy(self.world_pts[i0], self.world_pts[i1], self.world_pts[i2])

    def candidate_vertices(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """World vertices of the triangles near ``[lo, hi]``, as ``(N, 3, 3)``.

        The batch narrow phase wants the numbers, not objects wrapping them:
        one gather instead of a proxy built and discarded per triangle.
        """
        return self.world_pts[self.indices[self._candidate_indices(lo, hi)]]

    def candidate_triangles(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Indices of the triangles near ``[lo, hi]``, into :attr:`indices`."""
        return self._candidate_indices(lo, hi)

    @functools.cached_property
    def internal_edges(self) -> np.ndarray:
        """``(T, 3)`` booleans: edge ``k`` of each triangle lies inside a flat surface.

        Edge ``k`` runs from corner ``k`` to corner ``k + 1``. It is internal where
        exactly one other triangle shares it and the two faces are within
        ``INTERNAL_EDGE_COS`` of parallel, so that a body crossing it is still on
        the same surface. Vertices are matched by position, so a mesh that repeats
        a vertex per triangle has the same edges as one that shares them. A free
        edge, an edge of three or more triangles, and a crease are not internal.
        """
        count = len(self.indices)
        internal = np.zeros((count, 3), dtype=bool)
        if not count:
            return internal
        _unique, welded = np.unique(self.world_pts, axis=0, return_inverse=True)
        corners = welded.reshape(-1)[self.indices].astype(np.int64)
        start, end = corners, np.roll(corners, -1, axis=1)
        low, high = np.minimum(start, end), np.maximum(start, end)
        keys = (low * (int(corners.max()) + 1) + high).reshape(-1)
        order = np.argsort(keys, kind='stable')
        ordered = keys[order]
        same = ordered[1:] == ordered[:-1]
        # A pair is two equal keys with no third beside them.
        before = np.concatenate(([False], same[:-1]))
        after = np.concatenate((same[1:], [False]))
        pairs = np.flatnonzero(same & ~before & ~after)
        first, second = order[pairs], order[pairs + 1]
        tris = self.world_pts[self.indices]
        faces = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        lengths = np.linalg.norm(faces, axis=1)
        faces = np.divide(faces, lengths[:, None], out=np.zeros_like(faces),
                          where=lengths[:, None] > 0.0)
        # Either winding: a triangle soup does not promise a consistent one.
        flat = np.abs(np.einsum('ij,ij->i', faces[first // 3], faces[second // 3]))
        flat = flat >= INTERNAL_EDGE_COS
        flags = internal.reshape(-1)
        flags[first[flat]] = True
        flags[second[flat]] = True
        return internal

    def center_hint(self) -> np.ndarray:
        """A representative interior point (the vertex centroid)."""
        return self.world_pts.mean(axis=0)


class HeightFieldProxy:
    """A height field placed in the world: its surface, its origin and its rotation.

    The surface is shared with the shape and never copied, so building one for a
    body costs nothing however large the field is. It collides in its own frame
    (:meth:`to_local`), which is why it holds the rotation and not a world-space
    copy of anything.
    """
    kind = 'heightfield'

    def __init__(self, surface: HeightSurface, center: ArrayLike, rotation: ArrayLike):
        self.surface = surface
        self.center = np.asarray(center, dtype='d')
        self.R = np.asarray(rotation, dtype='d')
        #: Whether the field is not turned, so its frame differs from the world's by
        #: :attr:`center` alone.
        self.upright = bool(np.array_equal(self.R, np.eye(3)))

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """World ``points`` ``(..., 3)`` in the field's frame."""
        return (points - self.center) @ self.R

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """Points ``(..., 3)`` of the field's frame in the world."""
        return points @ self.R.T + self.center

    def aabb(self) -> _AABB:
        """The world box of the surface's bounds; infinite on an axis the surface leaves open."""
        low, high = self.surface.aabb()
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            box = np.array([(x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1])
                            for z in (low[2], high[2])])
            placed = self.to_world(box)
            return placed.min(axis=0), placed.max(axis=0)
        if np.allclose(self.R, np.eye(3)):
            return low + self.center, high + self.center
        return np.full(3, -np.inf), np.full(3, np.inf)

    def center_hint(self) -> np.ndarray:
        """A representative point (the field's origin)."""
        return self.center


# A support proxy answers ``support(direction)``; the mesh proxy does not (it is
# collided triangle-by-triangle instead), nor does a compound (part by part).
# ``Proxy`` is any collision proxy.
SupportProxy = (SphereProxy | BoxProxy | CapsuleProxy | CylinderProxy | ConvexProxy
                | TriangleProxy)
Proxy = SupportProxy | TriangleMeshProxy | CompoundProxy | HeightFieldProxy


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
        return CylinderProxy(pos, 0.5 * shape.height, shape.radiusBottom,
                             shape.radiusTop, R)
    if t == 'convex':
        if shape.points is None:
            raise ValueError('convex shape has no points')
        return ConvexProxy(shape.points, pos, R, shape.hull_planes())
    if t == 'trimesh':
        if shape.points is None or shape.indices is None:
            raise ValueError('trimesh shape needs both points and indices')
        return TriangleMeshProxy(shape.points, shape.indices, pos, R)
    if t == 'heightfield':
        if shape.surface is None:
            raise ValueError('heightfield shape has no surface')
        return HeightFieldProxy(shape.surface, pos, R)
    if t == 'compound':
        return CompoundProxy([make_proxy(part.shape, *part_pose(part, pos, orientation))
                              for part in shape.parts or ()])
    raise ValueError('unknown shape type %r' % t)


def part_pose(part: 'model.ShapePart', position: ArrayLike,
              orientation: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Where a compound's ``part`` is for a body at the pose given: ``(position, orientation)``."""
    q = np.asarray(orientation, dtype='d')
    placed = (np.asarray(position, dtype='d')
              + mathutil.quat_to_matrix(q) @ np.asarray(part.position, dtype='d'))
    turned = mathutil.quat_mul(q, np.asarray(part.orientation, dtype='d'))
    return placed, turned


def world_aabb(shape: 'model.Shape', position: ArrayLike,
               orientation: ArrayLike) -> _AABB:
    """Axis-aligned bounding box of ``shape`` at the given pose, as ``(lo, hi)``.

    A mesh is measured from its points rather than through a proxy: building
    one indexes the whole mesh into a spatial grid, which is worth doing for a
    body that is about to be *collided* and pure waste for one that is only
    being measured. A streaming world measures a tile the moment it arrives.
    """
    kind = str(shape.type)
    if kind == 'compound':
        boxes = [world_aabb(part.shape, *part_pose(part, position, orientation))
                 for part in shape.parts or ()]
        if not boxes:
            centre = np.asarray(position, dtype='d')
            return centre.copy(), centre.copy()
        return (np.min([lo for lo, _ in boxes], axis=0),
                np.max([hi for _, hi in boxes], axis=0))
    if kind in ('trimesh', 'convex') and shape.points is not None:
        points = np.asarray(shape.points, dtype='d')
        if len(points):
            rotation = mathutil.quat_to_matrix(np.asarray(orientation, dtype='d'))
            placed = points @ rotation.T + np.asarray(position, dtype='d')
            return placed.min(axis=0), placed.max(axis=0)
    return make_proxy(shape, position, orientation).aabb()


def pose_key(position: ArrayLike, orientation: ArrayLike) -> tuple:
    """A hashable stand-in for a pose, for caching what depends on one.

    Seven floats compared is nothing beside building a proxy for a mesh of
    thousands of triangles, and it is the honest test: a body that has not
    moved keeps its proxy, whatever kind of body it says it is. Keying on the
    motion type instead would go wrong the moment a caller repositions a static
    body, which is a normal thing to do to a level.
    """
    p = np.asarray(position, dtype='d')
    q = np.asarray(orientation, dtype='d')
    return (float(p[0]), float(p[1]), float(p[2]),
            float(q[0]), float(q[1]), float(q[2]), float(q[3]))
