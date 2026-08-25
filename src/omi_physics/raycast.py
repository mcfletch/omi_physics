"""Casting a ray at the world: what it hits, where, and which way that faces.

One query, two customers. A hitscan weapon asks *what did I shoot*; a bot asks
*can I see you*. They are the same question with different follow-ups, so it
lives here rather than being written twice in a game.

**What matters is the nearest hit, not any hit.** A cast that returns the first
body it happens to test shoots straight through walls, and the whole of the
structure below — reject by bounding box, test the survivors exactly nearest
first, keep the closest — exists for that.

**The rejection is done for the whole world at once**, as a slab test over the
world's own AABB arrays. A streamed landscape holds hundreds of static
colliders and a vehicle casts a ray per wheel per step; a Python loop over the
bodies costs more than the exact tests it is there to avoid.

Shapes are tested exactly: a sphere analytically, a box in its own frame, a
capsule as a segment, a trimesh triangle by triangle through the proxy's own
spatial grid. **A convex hull is not cast against**, because doing it properly
needs the hull's faces and this holds only its points; such a body is left out
of the answer and named by :func:`unsupported_shapes`, which is a prop you
cannot shoot rather than a hit in the wrong place. The second would be blamed
on the weapon.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mathutil import Vec
from .trigrid import TriangleGrid

log = logging.getLogger(__name__)

__all__ = ['NO_TRIANGLE', 'RayHit', 'line_of_sight', 'raycast',
           'raycast_many', 'unsupported_shapes']

#: :attr:`RayHit.triangle` for a hit on a shape that has no triangles.  A
#: sphere, a box and a capsule are surfaces without parts, so there is nothing
#: to name; a caller looking a material up by triangle gets this and knows to
#: stop rather than to index with -1 and read the last one.
NO_TRIANGLE = -1

#: How far a ray reaches when the caller names no limit.  Long enough to cross
#: any level, short enough that a miss does not walk the whole world.
DEFAULT_RANGE = 10_000.0

#: Below this a direction is not a direction.
_TINY = 1e-12

#: Shape types this module tests exactly.  Anything else is skipped; see the
#: module docstring for why that is the honest answer.
SUPPORTED = frozenset(('box', 'sphere', 'capsule', 'cylinder', 'trimesh'))


@dataclass(frozen=True)
class RayHit:
    """Where a ray met the world."""

    #: Index of the body it hit.
    body: int
    #: How far along the ray, in world units.
    distance: float
    #: Where, in world coordinates.
    point: np.ndarray
    #: The surface normal there, **always facing back along the ray**, so an
    #: impact effect is oriented the same way whichever side of a surface was
    #: struck and whichever way its triangles happened to be wound.
    normal: np.ndarray
    #: Which triangle of a trimesh was struck, or :data:`NO_TRIANGLE` for a
    #: shape that has none.  It is an index into the shape's own ``indices``,
    #: so a caller that built the mesh can look up whatever it knows about
    #: that part of it — which material it is, which surface it came from —
    #: without a second query against the geometry.
    triangle: int = NO_TRIANGLE


def raycast(world: Any, origin: Vec, direction: Vec,
            max_distance: float = DEFAULT_RANGE,
            skip: Iterable[int] = ()) -> RayHit | None:
    """The nearest thing ``direction`` meets from ``origin``, or None.

    ``direction`` need not be normalised — a caller with a velocity should not
    have to care — and a zero direction is a miss rather than an error.

    ``skip`` names bodies the ray passes through: a shooter's own body, and the
    two ends of a line-of-sight query.
    """
    start = np.asarray(origin, dtype='d')
    heading = np.asarray(direction, dtype='d')
    length = float(np.linalg.norm(heading))
    if length < _TINY or max_distance <= 0.0:
        return None
    heading = heading / length
    ignored = set(skip)
    nearest: RayHit | None = None
    limit = float(max_distance)
    for body in _castable(world, ignored, start, heading, limit):
        found = _hit_body(world, body, start, heading, limit)
        if found is not None and (nearest is None or found.distance < nearest.distance):
            nearest = found
            # Every later body only matters if it is nearer still, so the reach
            # shrinks as the answer improves.  On a level's worth of geometry
            # that is most of what keeps a cast cheap.
            limit = found.distance
    return nearest


def raycast_many(world: Any, origins: Any, directions: Any,
                 max_distance: float = DEFAULT_RANGE,
                 skip: Iterable[int] = ()) -> list[RayHit | None]:
    """Several rays at once, sharing the work they have in common.

    A vehicle casts one ray per wheel every step and they all start within a
    car's length of each other: the same bodies are worth testing, and the same
    part of the same mesh comes back from each. Cast one at a time, that work
    is done once per wheel, and for a raycast vehicle it is most of the step.

    So the bundle is answered together. Which bodies matter is decided once,
    for the box the whole bundle sweeps; a triangle mesh among them is asked
    once for the triangles near that box, and every ray is then tested against
    the same candidates. What comes back is what each ray would have answered
    on its own, in the order the rays were given, with ``None`` for a ray that
    met nothing.

    Nothing here assumes the rays are parallel or that they start together --
    a bundle spread across a level simply shares less.
    """
    starts = np.asarray(origins, dtype='d').reshape(-1, 3)
    headings = np.asarray(directions, dtype='d').reshape(-1, 3)
    count = len(starts)
    if not count or max_distance <= 0.0:
        return [None] * count
    lengths = np.linalg.norm(headings, axis=1)
    live = lengths >= _TINY
    headings = headings / np.where(lengths[:, None] > 0, lengths[:, None], 1.0)
    answers: list[RayHit | None] = [None] * count
    if not live.any():
        return answers
    limits = np.where(live, float(max_distance), 0.0)
    for body in _bundle_castable(world, set(skip), starts[live], headings[live],
                                 float(max_distance)):
        _hit_bundle(world, body, starts, headings, live, limits, answers)
    return answers


def _bundle_castable(world: Any, ignored: set, starts: np.ndarray,
                     headings: np.ndarray, limit: float) -> list[int]:
    """The bodies worth testing against any ray in the bundle.

    One box test per body against the box the whole bundle sweeps: a superset
    of what each ray's own slab test would keep, and the exact tests below
    still decide. Not sorted by distance -- there is no one distance for a
    bundle, and each ray shrinks its own reach as it finds something.
    """
    count = int(world.body_count)
    if not count:                                # pragma: no cover - empty world
        return []
    shapes = np.asarray(world.collider_shape[:count])
    live = shapes >= 0
    if ignored:
        live[np.fromiter(ignored, dtype=int, count=len(ignored))] = False
    if not live.any():
        return []
    ends = starts + headings * limit
    low = np.minimum(starts.min(axis=0), ends.min(axis=0))
    high = np.maximum(starts.max(axis=0), ends.max(axis=0))
    boxes_low = np.asarray(world.aabb_min[:count], dtype='d')
    boxes_high = np.asarray(world.aabb_max[:count], dtype='d')
    fitted = (np.all(np.isfinite(boxes_low), axis=1)
              & np.all(np.isfinite(boxes_high), axis=1)
              & np.all(boxes_high >= boxes_low, axis=1)
              & np.any(boxes_high > boxes_low, axis=1))
    overlaps = (np.all(boxes_low <= high, axis=1)
                & np.all(boxes_high >= low, axis=1))
    # A box that is not a box tells the cast nothing, so the body behind it is
    # tested exactly rather than skipped.
    live &= overlaps | ~fitted
    return np.nonzero(live)[0].tolist()


def _hit_bundle(world: Any, body: int, starts: np.ndarray, headings: np.ndarray,
                live: np.ndarray, limits: np.ndarray,
                answers: list) -> None:
    """Test one body against every live ray, keeping each ray's nearest."""
    shape = world.shapes[int(world.collider_shape[body])]
    if str(shape.type) == 'trimesh':
        _hit_bundle_trimesh(world, body, shape, starts, headings, live, limits,
                            answers)
        return
    for index in np.nonzero(live)[0]:
        found = _hit_body(world, body, starts[index], headings[index],
                          float(limits[index]))
        if found is not None:
            answers[index] = found
            limits[index] = found.distance


def _hit_bundle_trimesh(world: Any, body: int, shape: Any, starts: np.ndarray,
                        headings: np.ndarray, live: np.ndarray,
                        limits: np.ndarray, answers: list) -> None:
    """The shared half: one set of candidate triangles for the whole bundle.

    Walking the mesh's grid is what a cast against a landscape costs, and four
    wheels of one car walk very nearly the same cells. Asking for the triangles
    near the bundle's box instead trades a slightly larger candidate set for
    one query, and the exact test each ray then runs is a single numpy pass.
    """
    centre = np.asarray(world.position[body], dtype='d')
    rotation = _rotation(np.asarray(world.orientation[body], dtype='d'))
    placed = _placed(world, body, shape, centre, rotation)
    if placed is None:                           # pragma: no cover - no mesh
        return
    rays = np.nonzero(live)[0]
    ends = starts[rays] + headings[rays] * limits[rays, None]
    low = np.minimum(starts[rays].min(axis=0), ends.min(axis=0))
    high = np.maximum(starts[rays].max(axis=0), ends.max(axis=0))
    if np.any(low > placed.high) or np.any(high < placed.low):
        return
    candidates = placed.grid.box(low, high)
    if not len(candidates):
        return
    triangles = placed.triangles[candidates]
    for index in rays:
        found = _hit_triangles(starts[index], headings[index], triangles,
                               float(limits[index]))
        if found is None:
            continue
        distance, normal, among = found
        if float(np.dot(normal, headings[index])) > 0.0:
            normal = -normal
        answers[index] = RayHit(
            body=int(body), distance=distance,
            point=starts[index] + headings[index] * distance, normal=normal,
            triangle=int(candidates[among]))
        limits[index] = distance


def line_of_sight(world: Any, start: Vec, end: Vec,
                  skip: Iterable[int] = ()) -> bool:
    """Whether nothing lies between two points.

    The cheaper question, and the one a bot asks many times a frame: only what
    is *between* matters, so a wall past the target does not block the view.
    """
    first = np.asarray(start, dtype='d')
    second = np.asarray(end, dtype='d')
    to = second - first
    distance = float(np.linalg.norm(to))
    if distance < _TINY:
        return True
    return raycast(world, first, to, max_distance=distance, skip=skip) is None


def unsupported_shapes(world: Any) -> set[str]:
    """Shape types in ``world`` that a ray cannot be cast against.

    Reported rather than skipped in silence: a body a ray passes through is
    something a game needs to know about, and the answer is either a different
    collider or a hull cast this module does not have.
    """
    found: set[str] = set()
    for body in range(world.body_count):
        index = int(world.collider_shape[body])
        if index < 0:
            continue
        kind = str(world.shapes[index].type)
        if kind not in SUPPORTED:
            found.add(kind)
    return found


def _castable(world: Any, ignored: set[int], origin: np.ndarray,
              heading: np.ndarray, limit: float) -> Iterable[int]:
    """The bodies worth testing exactly, nearest first.

    A body is worth testing when it has a collider, is not being skipped, and
    the ray's segment crosses its world bounding box. **The box test is done
    for every body at once**, as a slab test over the world's own AABB arrays:
    a streamed landscape holds hundreds of static colliders and a vehicle casts
    a ray per wheel per step, so a Python loop over the bodies is the frame.

    Nearest first, because the caller shrinks its reach with every hit and the
    sooner it finds the near one the fewer of the far ones it tests.

    The boxes are the world's own: :meth:`~omi_physics.world.PhysicsWorld.step`
    maintains them and :meth:`~omi_physics.world.PhysicsWorld.add_body` fits a
    new body as it arrives, so a world that has been built but never stepped is
    still castable. A body moved by writing its position directly keeps the box
    it had until the next refit, exactly as it does for collision.
    """
    count = int(world.body_count)
    if not count:                                # pragma: no cover - empty world
        return ()
    shapes = np.asarray(world.collider_shape[:count])
    live = shapes >= 0
    if ignored:
        live[np.fromiter(ignored, dtype=int, count=len(ignored))] = False
    if not live.any():
        return ()
    entry = _slab_entry(world, origin, heading, limit, live)
    order = np.nonzero(live)[0]
    return order[np.argsort(entry[order], kind='stable')].tolist()


def _slab_entry(world: Any, origin: np.ndarray, heading: np.ndarray,
                limit: float, live: np.ndarray) -> np.ndarray:
    """Where the ray enters each body's box, with a miss marked as infinite.

    The standard slab test, vectorised over every body. A body whose box the
    ray misses is dropped from ``live`` in place, so the caller is left with
    the ones worth an exact test.
    """
    count = len(live)
    low = np.asarray(world.aabb_min[:count], dtype='d')
    high = np.asarray(world.aabb_max[:count], dtype='d')
    # A box has to be finite, the right way round, and have some size in it.
    # A world that has never been stepped has all-zero boxes, and believing
    # those would make a cast miss everything in it.
    fitted = np.all(np.isfinite(low), axis=1) & np.all(np.isfinite(high), axis=1)
    fitted &= np.all(high >= low, axis=1) & np.any(high > low, axis=1)
    # A box that is not a box tells the cast nothing, so the body behind it is
    # tested exactly rather than skipped.
    unknown = live & ~fitted
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse = 1.0 / heading
        near = (low - origin) * inverse
        far = (high - origin) * inverse
    lower = np.minimum(near, far).max(axis=1)
    upper = np.maximum(near, far).min(axis=1)
    # NaN appears where a component of the heading is zero and the origin is
    # exactly on a face; treating it as a hit keeps a grazing ray honest.
    lower = np.nan_to_num(lower, nan=0.0)
    upper = np.nan_to_num(upper, nan=limit)
    crosses = (upper >= np.maximum(lower, 0.0)) & (lower <= limit)
    live &= (crosses & fitted) | unknown
    entry = np.where(lower > 0.0, lower, 0.0)
    entry[unknown] = 0.0
    return entry


def _hit_body(world: Any, body: int, origin: np.ndarray, heading: np.ndarray,
              limit: float) -> RayHit | None:
    """The ray's meeting with one body, or None."""
    shape = world.shapes[int(world.collider_shape[body])]
    kind = str(shape.type)
    if kind not in SUPPORTED:
        return None
    centre = np.asarray(world.position[body], dtype='d')
    rotation = _rotation(np.asarray(world.orientation[body], dtype='d'))
    triangle = NO_TRIANGLE
    if kind == 'sphere':
        found = _hit_sphere(origin, heading, centre, float(shape.radius), limit)
    elif kind in ('capsule', 'cylinder'):
        found = _hit_capsule(origin, heading, centre, rotation, shape, limit)
    elif kind == 'trimesh':
        placed = _placed(world, body, shape, centre, rotation)
        met = (None if placed is None
               else _hit_trimesh(origin, heading, limit, placed))
        found = None if met is None else met[:2]
        triangle = NO_TRIANGLE if met is None else met[2]
    else:
        half = np.asarray(shape.size, dtype='d') * 0.5
        found = _hit_box(origin, heading, centre, rotation, half, limit)
    if found is None:
        return None
    distance, normal = found
    # Always back along the ray: an impact effect is oriented the same way
    # whichever side of a surface was struck.
    if float(np.dot(normal, heading)) > 0.0:
        normal = -normal
    return RayHit(body=body, distance=distance,
                  point=origin + heading * distance, normal=normal,
                  triangle=triangle)


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    """A quaternion ``(x, y, z, w)`` as a rotation matrix."""
    x, y, z, w = (float(value) for value in quaternion[:4])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype='d')


def _hit_sphere(origin: np.ndarray, heading: np.ndarray, centre: np.ndarray,
                radius: float, limit: float) -> tuple[float, np.ndarray] | None:
    """Ray against a sphere, analytically."""
    to_centre = origin - centre
    along = float(np.dot(to_centre, heading))
    gap = float(np.dot(to_centre, to_centre)) - radius * radius
    if gap > 0.0 and along > 0.0:               # outside, pointing away
        return None
    discriminant = along * along - gap
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    distance = -along - root
    if distance < 0.0:                          # started inside
        distance = -along + root
    if distance < 0.0 or distance > limit:
        return None
    normal = (origin + heading * distance) - centre
    return (distance, _unit(normal))


def _hit_box(origin: np.ndarray, heading: np.ndarray, centre: np.ndarray,
             rotation: np.ndarray, half: np.ndarray,
             limit: float) -> tuple[float, np.ndarray] | None:
    """Ray against an oriented box, by slabs in the box's own frame."""
    local_origin = rotation.T @ (origin - centre)
    local_heading = rotation.T @ heading
    near, far = -math.inf, math.inf
    axis = 0
    sign = 1.0
    for index in range(3):
        if abs(local_heading[index]) < _TINY:
            if abs(local_origin[index]) > half[index]:
                return None
            continue
        inverse = 1.0 / local_heading[index]
        first = (-half[index] - local_origin[index]) * inverse
        second = (half[index] - local_origin[index]) * inverse
        entering = -1.0 if inverse > 0.0 else 1.0
        if first > second:
            first, second = second, first
            entering = -entering
        if first > near:
            near, axis, sign = first, index, entering
        far = min(far, second)
        if near > far:
            return None
    distance = near if near >= 0.0 else far
    if distance < 0.0 or distance > limit:
        return None
    local_normal = np.zeros(3)
    local_normal[axis] = sign
    return (distance, rotation @ local_normal)


def _hit_capsule(origin: np.ndarray, heading: np.ndarray, centre: np.ndarray,
                 rotation: np.ndarray, shape: Any,
                 limit: float) -> tuple[float, np.ndarray] | None:
    """Ray against a capsule: its segment swollen by a radius.

    The body of the capsule and its two caps are solved separately and the
    nearest kept, which is simpler to be sure of than one closed form and
    costs nothing at these counts.
    """
    radius = float(shape.radiusBottom)
    half = float(shape.height) * 0.5
    axis = rotation @ np.array([0.0, 1.0, 0.0])
    bottom = centre - axis * half
    top = centre + axis * half

    best: tuple[float, np.ndarray] | None = None
    for cap in (bottom, top):
        found = _hit_sphere(origin, heading, cap, radius, limit)
        if found is not None and (best is None or found[0] < best[0]):
            best = found

    # The infinite cylinder about the segment, clipped to the segment's extent.
    to_axis = origin - bottom
    segment = top - bottom
    segment_sq = float(np.dot(segment, segment))
    if segment_sq >= _TINY:
        heading_perp = heading - segment * (float(np.dot(heading, segment)) / segment_sq)
        offset_perp = to_axis - segment * (float(np.dot(to_axis, segment)) / segment_sq)
        a = float(np.dot(heading_perp, heading_perp))
        b = 2.0 * float(np.dot(heading_perp, offset_perp))
        c = float(np.dot(offset_perp, offset_perp)) - radius * radius
        if a >= _TINY:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = math.sqrt(discriminant)
                for distance in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                    if distance < 0.0 or distance > limit:
                        continue
                    point = origin + heading * distance
                    along = float(np.dot(point - bottom, segment)) / segment_sq
                    if not 0.0 <= along <= 1.0:
                        continue
                    if best is None or distance < best[0]:
                        closest = bottom + segment * along
                        best = (distance, _unit(point - closest))
                    break
    return best


class _Placed:
    """One trimesh in world space, kept so a cast does not rebuild it.

    A level is a single trimesh of tens of thousands of triangles that never
    moves, and transforming all of it per cast costs milliseconds — enough that
    a bot checking what it can see would be felt.  What is cached is the pose
    *and the shape* it was built for, so a body that moves rebuilds, one that
    does not pays once, and one whose collider was swapped for another —
    a recycled slot in a streaming world — does not answer with the mesh that
    used to be there.
    """

    __slots__ = (
        'centre',
        'grid',
        'high',
        'highs',
        'low',
        'lows',
        'rotation',
        'shape',
        'triangles',
    )

    def __init__(self, shape: Any, centre: np.ndarray,
                 rotation: np.ndarray) -> None:
        self.centre = centre.copy()
        self.rotation = rotation.copy()
        self.shape = shape
        points = np.asarray(shape.points, dtype='d') @ rotation.T + centre
        indices = np.asarray(shape.indices, dtype='i')
        self.triangles = points[indices]
        self.lows = self.triangles.min(axis=1)
        self.highs = self.triangles.max(axis=1)
        self.low = points.min(axis=0)
        self.high = points.max(axis=0)
        self.grid = TriangleGrid(self.lows, self.highs)

    def matches(self, shape: Any, centre: np.ndarray,
                rotation: np.ndarray) -> bool:
        return (self.shape is shape
                and np.array_equal(self.centre, centre)
                and np.array_equal(self.rotation, rotation))


def _placed(world: Any, body: int, shape: Any, centre: np.ndarray,
            rotation: np.ndarray) -> _Placed | None:
    """This body's trimesh in world space, built once per pose.

    Kept on the world rather than in a module-level table, so it goes when the
    world does and two worlds never share an entry.
    """
    if not len(np.asarray(shape.indices, dtype='i')):
        return None
    cache = getattr(world, '_raycast_meshes', None)
    if cache is None:
        cache = {}
        world._raycast_meshes = cache
    found = cache.get(body)
    if found is None or not found.matches(shape, centre, rotation):
        found = _Placed(shape, centre, rotation)
        cache[body] = found
    return found


def _hit_trimesh(origin: np.ndarray, heading: np.ndarray, limit: float,
                 placed: _Placed) -> tuple[float, np.ndarray, int] | None:
    """Ray against a triangle soup, narrowed to the cells the ray crosses.

    Through the mesh's spatial grid rather than by testing every triangle's
    bound: a level is one mesh of tens of thousands of triangles, and even the
    cheap slab test costs milliseconds when it is done that many times per
    cast.  What the grid hands back is a superset of what the ray can meet, so
    the exact test below still decides.

    ``placed`` is the mesh in world space, cached against the pose it was built
    for; where the mesh is and which shape it came from are already in it.

    The third value is the winner's index in the *mesh*, not among the
    candidates the grid offered: the grid is an optimisation and its numbering
    is nobody else's business.
    """
    if not _hits_box(origin, heading, placed.low, placed.high, limit):
        return None
    candidates = placed.grid.ray(origin, heading, limit)
    if not len(candidates):
        return None
    found = _hit_triangles(origin, heading, placed.triangles[candidates], limit)
    if found is None:
        return None
    distance, normal, among = found
    return (distance, normal, int(candidates[among]))


def _hits_box(origin: np.ndarray, heading: np.ndarray, low: np.ndarray,
              high: np.ndarray, limit: float) -> bool:
    """Whether the ray passes through one axis-aligned box."""
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse = 1.0 / heading
        first = (low - origin) * inverse
        second = (high - origin) * inverse
    near = float(np.nanmax(np.minimum(first, second)))
    far = float(np.nanmin(np.maximum(first, second)))
    return far >= max(near, 0.0) and near <= limit


def _hit_triangles(origin: np.ndarray, heading: np.ndarray,
                   triangles: np.ndarray,
                   limit: float) -> tuple[float, np.ndarray, int] | None:
    """The nearest of many triangles the ray meets, and which one, or None.

    Möller-Trumbore over the whole candidate set at once.  Solving them one at
    a time is the same arithmetic, but it pays Python's call overhead and two
    :func:`numpy.cross` calls per triangle -- and numpy's fixed cost per
    operation dwarfs the work when the operand is three numbers.  A cast's
    candidates run to a few dozen triangles, so that overhead *is* the cast.

    Two-sided deliberately: a map's geometry has no reliable winding from a
    shooter's point of view, and a one-sided test would let a shot through
    every second wall.
    """
    a = triangles[:, 0]
    edge1 = triangles[:, 1] - a
    edge2 = triangles[:, 2] - a
    across = np.cross(heading, edge2)
    determinant = np.einsum('ij,ij->i', edge1, across)
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse = 1.0 / determinant
        to_a = origin - a
        u = np.einsum('ij,ij->i', to_a, across) * inverse
        edge = np.cross(to_a, edge1)
        v = (edge @ heading) * inverse
        distance = np.einsum('ij,ij->i', edge2, edge) * inverse
    met = ((np.abs(determinant) >= _TINY)        # not parallel to the plane
           & np.isfinite(distance)
           & (u >= 0.0) & (u <= 1.0)
           & (v >= 0.0) & (u + v <= 1.0)
           & (distance >= 0.0) & (distance <= limit))
    if not met.any():
        return None
    # ``where`` rather than masking the arrays: the winner's index is wanted
    # for its normal, and a masked argmin gives a position in the mask.
    found = np.flatnonzero(met)
    nearest = found[int(np.argmin(distance[found]))]
    return (float(distance[nearest]),
            _unit(np.cross(edge1[nearest], edge2[nearest])),
            int(nearest))


def _hit_triangle(origin: np.ndarray, heading: np.ndarray,
                  triangle: np.ndarray,
                  limit: float) -> tuple[float, np.ndarray] | None:
    """Ray against one triangle, for a caller holding exactly one.

    The same test :func:`_hit_triangles` does in bulk; kept because a single
    triangle read through it would have to be reshaped into a set of one.
    """
    a, b, c = triangle
    edge1 = b - a
    edge2 = c - a
    across = np.cross(heading, edge2)
    determinant = float(np.dot(edge1, across))
    if abs(determinant) < _TINY:                # parallel to the plane
        return None
    inverse = 1.0 / determinant
    to_a = origin - a
    u = float(np.dot(to_a, across)) * inverse
    if u < 0.0 or u > 1.0:
        return None
    edge = np.cross(to_a, edge1)
    v = float(np.dot(heading, edge)) * inverse
    if v < 0.0 or u + v > 1.0:
        return None
    distance = float(np.dot(edge2, edge)) * inverse
    if distance < 0.0 or distance > limit:
        return None
    return (distance, _unit(np.cross(edge1, edge2)))


def _unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` normalised, or +Y if it has no length to speak of."""
    length = float(np.linalg.norm(vector))
    if length < _TINY:
        return np.array([0.0, 1.0, 0.0])
    return vector / length


def bodies_along(world: Any, origin: Vec, direction: Vec,
                 max_distance: float = DEFAULT_RANGE,
                 skip: Iterable[int] = ()) -> list[RayHit]:
    """Every body the ray meets, nearest first.

    For the cases where the *first* thing hit is not the answer — a shot that
    passes through a trigger volume on its way to a wall, a splash that has to
    know what it can see.
    """
    start = np.asarray(origin, dtype='d')
    heading = np.asarray(direction, dtype='d')
    length = float(np.linalg.norm(heading))
    if length < _TINY or max_distance <= 0.0:
        return []
    heading = heading / length
    ignored = set(skip)
    limit = float(max_distance)
    found = [hit for hit in
             (_hit_body(world, body, start, heading, limit)
              for body in _castable(world, ignored, start, heading, limit))
             if hit is not None]
    return sorted(found, key=lambda hit: hit.distance)
