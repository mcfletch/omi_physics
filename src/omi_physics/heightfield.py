"""Height-field surfaces: a ground that answers a height and a normal anywhere.

A height field is a surface ``y = h(x, z)`` in a body's own frame, with the
body's local +Y as up. Everything below the surface is solid. It has no sides:
a ray that enters the extent from the side, below the surface, passes through.

There are two forms, both :class:`HeightSurface`:

* :class:`GridSurface` interpolates a regular grid of heights bilinearly and
  answers the exact slope of the cell it is in.
* :class:`FunctionSurface` calls a function ``sample(x, z)``. It never leaves the
  process, which is what lets a landscape too large to hold as a grid, or one
  a renderer already samples, serve as the collision surface.

Outside its extent a surface has no height (``-inf``), so nothing collides
there. The routines below work in the field's local frame and return arrays;
:mod:`omi_physics.collide` turns them into contacts and
:mod:`omi_physics.raycast` into hits.
"""
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

#: How many samples a ray march may take before it widens its step.
MAX_RAY_SAMPLES = 1 << 17

#: Bisection halvings of the interval a march found a crossing in.
_BISECTIONS = 40

#: Slides down the tangent plane a sphere's nearest ground point is looked for in,
#: and the move in x and z, in the field's units, below which it has been found.
_SLIDES = 5
_SETTLED = 1e-9

#: The most samples a footprint is searched at along one axis.
_FOOTPRINT_SAMPLES = 7

_TINY = 1e-12

#: A touching sphere of a column (:meth:`HeightSurface.column_contacts`): the contact point, the
#: unit normal pointing out of the ground and the penetration along it.
ColumnContact = tuple[tuple[float, float, float], tuple[float, float, float], float]


def _triple(row: np.ndarray) -> tuple[float, float, float]:
    x, y, z = row.tolist()
    return float(x), float(y), float(z)


class HeightSurface:
    """The interface of a height field: heights, normals, bounds and a ray cast.

    ``low`` and ``high`` are the local bounding box, infinite on an axis the
    surface does not bound. ``feature`` is the smallest horizontal distance over
    which the surface is taken to change, which sets how finely a ray march and a
    footprint search sample it.
    """
    low: np.ndarray
    high: np.ndarray
    feature: float
    #: The steepest the surface is anywhere, as a rise over a run, or None where that is not known.
    #: A sphere further above the surface than its radius times the length of this slope's
    #: normal cannot touch it, which :func:`sphere_contacts` uses to leave most spheres out.
    lipschitz: float | None = None

    def heights(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Surface height at each ``(x, z)``; ``-inf`` outside the extent."""
        raise NotImplementedError

    def normals(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Unit surface normal, pointing up out of the ground, at each ``(x, z)``."""
        raise NotImplementedError

    def sample(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`heights` and :meth:`normals` together, for a surface that can share their work."""
        return self.heights(x, z), self.normals(x, z)

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """The local bounding box as ``(low, high)`` corners."""
        return self.low.copy(), self.high.copy()

    def column_contacts(self, x: float, z: float, heights: list[float], radius: float
                        ) -> list[ColumnContact]:
        """The touching spheres of ``radius`` at ``(x, y, z)``, one for each ``y`` of ``heights``.

        Returns ``(point, normal, depth)`` for each touching sphere, in the order
        of ``heights``: the contacts :func:`sphere_contacts` finds for those
        spheres. An upright capsule is such a column of spheres.
        """
        centres = np.array([(x, y, z) for y in heights], dtype='d').reshape(-1, 3)
        touching, points, normals, depths = sphere_contacts(self, centres, radius)
        return [(_triple(points[k]), _triple(normals[k]), float(depths[k]))
                for k in np.flatnonzero(touching).tolist()]

    def ray(self, origin: np.ndarray, heading: np.ndarray,
            limit: float) -> tuple[float, np.ndarray] | None:
        """Where a ray first meets the surface from above: ``(distance, normal)`` or None.

        ``heading`` is a unit vector in the field's frame. The ray is marched
        in steps of :attr:`feature` over the stretch of it inside the extent, and
        the interval of the first step that goes from above the surface to below
        it is bisected. A ray that starts below the surface, inside
        the extent, meets it at distance 0.
        """
        reach = self._reach(origin, heading, limit)
        if reach is None:
            return None
        first, last = reach
        count = int(np.ceil((last - first) / self.feature)) + 2
        if count > MAX_RAY_SAMPLES:
            count = MAX_RAY_SAMPLES
        along = np.linspace(first, last, count)
        points = origin + along[:, None] * heading
        height = self.heights(points[:, 0], points[:, 2])
        above = points[:, 1] - height          # +inf outside the extent
        if along[0] == 0.0 and np.isfinite(above[0]) and above[0] < 0.0:
            return 0.0, self.normals(points[:1, 0], points[:1, 2])[0]
        crossing = np.flatnonzero((above[:-1] >= 0.0) & np.isfinite(above[:-1])
                                  & (above[1:] < 0.0) & np.isfinite(above[1:]))
        if not len(crossing):
            return None
        index = int(crossing[0])
        near, far = float(along[index]), float(along[index + 1])
        for _ in range(_BISECTIONS):
            middle = 0.5 * (near + far)
            spot = origin + middle * heading
            gap = spot[1] - self.heights(np.array([spot[0]]), np.array([spot[2]]))[0]
            if gap >= 0.0:
                near = middle
            else:
                far = middle
        distance = 0.5 * (near + far)
        spot = origin + distance * heading
        normal = self.normals(np.array([spot[0]]), np.array([spot[2]]))[0]
        return distance, normal

    def _reach(self, origin: np.ndarray, heading: np.ndarray,
               limit: float) -> tuple[float, float] | None:
        """The stretch ``(first, last)`` of the ray inside the extent and the limit, or None."""
        first, last = 0.0, float(limit)
        for axis in (0, 2):
            low, high = float(self.low[axis]), float(self.high[axis])
            if not (np.isfinite(low) or np.isfinite(high)):
                continue
            if abs(heading[axis]) < _TINY:
                if not low <= origin[axis] <= high:
                    return None
                continue
            near = (low - origin[axis]) / heading[axis]
            far = (high - origin[axis]) / heading[axis]
            first = max(first, min(near, far))
            last = min(last, max(near, far))
        if first > last:
            return None
        return first, last


class GridSurface(HeightSurface):
    """A regular grid of heights, interpolated bilinearly.

    ``heights`` is ``(rows, columns)`` with ``heights[j, i]`` the height at
    ``x = origin[0] + i * spacing[0]`` and ``z = origin[2] + j * spacing[1]``,
    added to ``origin[1]``. The slope in a cell is the exact derivative of its
    bilinear patch.
    """

    def __init__(self, heights: ArrayLike, origin: ArrayLike = (0.0, 0.0, 0.0),
                 spacing: float | tuple[float, float] = 1.0) -> None:
        grid = np.asarray(heights, dtype='d')
        if grid.ndim != 2 or grid.shape[0] < 2 or grid.shape[1] < 2:
            raise ValueError('a height grid needs at least 2 rows and 2 columns, got shape %r'
                             % (grid.shape,))
        if not np.all(np.isfinite(grid)):
            raise ValueError('a height grid holds only finite heights')
        step = np.broadcast_to(np.asarray(spacing, dtype='d'), (2,)).copy()
        if np.any(step <= 0.0):
            raise ValueError('a height grid needs a positive spacing, got %r' % (spacing,))
        self.grid = grid
        self.origin = np.asarray(origin, dtype='d').reshape(3)
        self.spacing = step
        rows, columns = grid.shape
        self.low = np.array([self.origin[0], self.origin[1] + grid.min(), self.origin[2]])
        self.high = np.array([self.origin[0] + (columns - 1) * step[0],
                              self.origin[1] + grid.max(),
                              self.origin[2] + (rows - 1) * step[1]])
        self.feature = float(step.min())
        # The slope of a bilinear patch is a blend of the slopes of its edges, so no patch is
        # steeper in x or in z than the steepest edge.
        across = float(np.abs(np.diff(grid, axis=1)).max() / step[0])
        along = float(np.abs(np.diff(grid, axis=0)).max() / step[1])
        self.lipschitz = math.hypot(across, along)
        self._numbers: tuple[list[list[float]], int, int, float, float, float, float, float] | None = None
        #: :meth:`_steepest_table` by its reach in cells across and along.
        self._steepest: dict[tuple[int, int], list[list[float]]] = {}

    def sphere_contact(self, cx: float, cy: float, cz: float, radius: float
                       ) -> tuple[bool, tuple[float, float, float], tuple[float, float, float], float]:
        """The contact of one sphere with the surface, found as :func:`sphere_contacts_by_sliding`
        finds it, with numbers and not arrays: for one sphere that is several times as fast.

        Returns whether it touches, the contact point, the unit normal pointing out of the ground
        and the penetration along it.
        """
        x, z = cx, cz
        ground_x = ground_y = ground_z = 0.0
        nx = nz = 0.0
        ny = 1.0
        distance = 0.0
        inside = False
        for _ in range(_SLIDES):
            height, nx, ny, nz = self._sample_one(x, z)
            inside = height != -math.inf
            ground_x, ground_y, ground_z = x, (height if inside else 0.0), z
            distance = (cx - ground_x) * nx + (cy - ground_y) * ny + (cz - ground_z) * nz
            if inside:
                foot_x, foot_z = cx - distance * nx, cz - distance * nz
            else:
                foot_x, foot_z = x, z
            settled = max(abs(foot_x - x), abs(foot_z - z)) < _SETTLED
            x, z = foot_x, foot_z
            if settled:
                break
        depth = radius - distance
        half = 0.5 * depth
        return (inside and depth > 0.0,
                (ground_x - nx * half, ground_y - ny * half, ground_z - nz * half),
                (nx, ny, nz), depth)

    def _frame(self) -> tuple[list[list[float]], int, int, float, float, float, float, float]:
        """The grid as nested lists, its size, origin and spacing, as numbers, for the one-point lookups."""
        frame = self._numbers
        if frame is None:
            rows = self.grid.tolist()
            frame = self._numbers = (rows, len(rows), len(rows[0]), float(self.origin[0]),
                                     float(self.origin[1]), float(self.origin[2]),
                                     float(self.spacing[0]), float(self.spacing[1]))
        return frame

    def height_at(self, x: float, z: float) -> float:
        """The height at one point, as :meth:`heights` finds it; ``-inf`` off the grid."""
        grid, rows, columns, ox, oy, oz, sx, sz = self._frame()
        gx = (x - ox) / sx
        gz = (z - oz) / sz
        if not (0.0 <= gx <= columns - 1 and 0.0 <= gz <= rows - 1):
            return -math.inf
        i = min(int(gx), columns - 2)
        j = min(int(gz), rows - 2)
        u, v = gx - i, gz - j
        h00, h10 = grid[j][i], grid[j][i + 1]
        h01, h11 = grid[j + 1][i], grid[j + 1][i + 1]
        return (h00 * (1 - u) * (1 - v) + h10 * u * (1 - v) + h01 * (1 - u) * v
                + h11 * u * v) + oy

    def _sample_one(self, x: float, z: float) -> tuple[float, float, float, float]:
        """The height and the unit normal ``(nx, ny, nz)`` at one point: ``-inf`` and up off the grid."""
        grid, rows, columns, ox, oy, oz, sx, sz = self._frame()
        gx = (x - ox) / sx
        gz = (z - oz) / sz
        if not (0.0 <= gx <= columns - 1 and 0.0 <= gz <= rows - 1):
            return -math.inf, 0.0, 1.0, 0.0
        i = min(int(gx), columns - 2)
        j = min(int(gz), rows - 2)
        u, v = gx - i, gz - j
        h00, h10 = grid[j][i], grid[j][i + 1]
        h01, h11 = grid[j + 1][i], grid[j + 1][i + 1]
        height = (h00 * (1 - u) * (1 - v) + h10 * u * (1 - v) + h01 * (1 - u) * v
                  + h11 * u * v) + oy
        slope_x = ((h10 - h00) * (1 - v) + (h11 - h01) * v) / sx
        slope_z = ((h01 - h00) * (1 - u) + (h11 - h10) * u) / sz
        length = math.sqrt(slope_x * slope_x + 1.0 + slope_z * slope_z)
        return height, -slope_x / length, 1.0 / length, -slope_z / length

    def sphere_contacts_together(self, centres: np.ndarray, radius: float
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """:func:`sphere_contacts_by_sliding` for a few spheres, with numbers and not arrays.

        The same arithmetic in the same order, and the spheres slide together
        until the last has settled, so the answer is the same to the bit; for
        a capsule's few spheres it is several times as fast.
        """
        found = self._slide_together(centres.tolist(), radius)
        count = len(found)
        touching = np.zeros(count, dtype=bool)
        contact = np.zeros((count, 3))
        normals = np.zeros((count, 3))
        depths = np.zeros(count)
        for k, (inside, point, normal, depth) in enumerate(found):
            touching[k] = inside
            contact[k] = point
            normals[k] = normal
            depths[k] = depth
        return touching, contact, normals, depths

    def _slide_together(self, points: list[list[float]], radius: float
                        ) -> list[tuple[bool, tuple[float, float, float],
                                        tuple[float, float, float], float]]:
        """Each sphere's ``(touching, point, normal, depth)``, slid until the last settles."""
        xs = [point[0] for point in points]
        zs = [point[2] for point in points]
        found: list[tuple[bool, float, float, float, float, float, float, float]] = []
        for _ in range(_SLIDES):
            found = []
            moved = 0.0
            for (cx, cy, cz), x, z in zip(points, xs, zs, strict=True):
                height, nx, ny, nz = self._sample_one(x, z)
                inside = height != -math.inf
                ground_y = height if inside else 0.0
                distance = (cx - x) * nx + (cy - ground_y) * ny + (cz - z) * nz
                found.append((inside, x, ground_y, z, nx, ny, nz, distance))
            xs, zs = [], []
            for (cx, _cy, cz), (inside, x, _y, z, nx, _ny, nz, distance) in zip(
                    points, found, strict=True):
                foot_x = cx - distance * nx if inside else x
                foot_z = cz - distance * nz if inside else z
                moved = max(moved, abs(foot_x - x), abs(foot_z - z))
                xs.append(foot_x)
                zs.append(foot_z)
            if moved < _SETTLED:
                break
        out = []
        for inside, x, y, z, nx, ny, nz, distance in found:
            depth = radius - distance
            half = 0.5 * depth
            out.append((inside and depth > 0.0, (x - nx * half, y - ny * half, z - nz * half),
                        (nx, ny, nz), depth))
        return out

    def column_contacts(self, x: float, z: float, heights: list[float], radius: float
                        ) -> list[ColumnContact]:
        """:meth:`HeightSurface.column_contacts` as :func:`sphere_contacts` finds them.

        The spheres share their ``(x, z)``, so the height under them is read once.
        """
        bound = self.lipschitz
        if bound is None or len(heights) > _WITH_NUMBERS:
            return super().column_contacts(x, z, heights, radius)
        reach = 2.0 * radius * math.sqrt(1.0 + bound * bound)
        ground = self.height_at(x, z)
        near = [y for y in heights if y - ground < reach]
        nearer = near
        if near:
            limit = self._local_clearance(x, z, radius, reach)
            nearer = [y for y in near if y - ground < limit]
        if len(nearer) > _ONE_BY_ONE or (nearer is near and len(near) > _ONE_BY_ONE):
            found = self._slide_together([[x, y, z] for y in nearer], radius)
        else:
            found = [self.sphere_contact(x, y, z, radius) for y in nearer]
        return [(point, normal, depth) for touching, point, normal, depth in found if touching]

    def _local_clearance(self, x: float, z: float, radius: float, reach: float) -> float:
        """How high above the ground at ``(x, z)`` a sphere of ``radius`` there cannot touch it.

        The bound is :func:`local_clearance` of the steepest slope within ``reach``
        of the point (:meth:`_steepest_near`). ``reach`` is the height under which
        :func:`sphere_contacts` slides a sphere at all, and the slide down the
        tangent planes of a sphere lower than that stays within ``reach`` of the
        point while the slope is gentle enough for a bound to exist; where none
        does, ``inf``.
        """
        steepest = self._steepest_near(x, z, reach)
        return local_clearance(steepest, radius) if clearance_spread(steepest) <= 1.0 else math.inf

    def _steepest_near(self, x: float, z: float, reach: float) -> float:
        """The steepest slope of the cells within ``reach`` of ``(x, z)``, as a rise over a run."""
        _grid, rows, columns, ox, _oy, oz, sx, sz = self._frame()
        wide = (math.ceil(reach / sx), math.ceil(reach / sz))
        table = self._steepest.get(wide)
        if table is None:
            table = self._steepest[wide] = self._steepest_table(*wide)
        gx = (x - ox) / sx
        gz = (z - oz) / sz
        if not (0.0 <= gx <= columns - 1 and 0.0 <= gz <= rows - 1):
            return math.inf
        return table[min(int(gz), rows - 2)][min(int(gx), columns - 2)]

    def _steepest_table(self, across: int, along: int) -> list[list[float]]:
        """Per cell, the steepest slope of the cells ``across`` columns and ``along`` rows away.

        A cell's slope is bounded by the steepest of its edges in x and in z,
        as :attr:`lipschitz` is for the whole grid.
        """
        grid, (sx, sz) = self.grid, self.spacing
        dx = np.abs(np.diff(grid, axis=1)) / sx
        dz = np.abs(np.diff(grid, axis=0)) / sz
        cell_x = np.maximum(dx[:-1, :], dx[1:, :])
        cell_z = np.maximum(dz[:, :-1], dz[:, 1:])
        slope = np.hypot(cell_x, cell_z)
        rows, columns = slope.shape
        padded = np.pad(slope, ((along, along), (across, across)), constant_values=0.0)
        wide = np.zeros((rows + 2 * along, columns))
        for shift in range(2 * across + 1):
            np.maximum(wide, padded[:, shift:shift + columns], out=wide)
        out = np.zeros((rows, columns))
        for shift in range(2 * along + 1):
            np.maximum(out, wide[shift:shift + rows, :], out=out)
        return out.tolist()

    def _cell(self, x: np.ndarray, z: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per point: cell column and row, fractions across the cell, and on-grid or not."""
        rows, columns = self.grid.shape
        gx = (np.asarray(x, dtype='d') - self.origin[0]) / self.spacing[0]
        gz = (np.asarray(z, dtype='d') - self.origin[2]) / self.spacing[1]
        inside = (gx >= 0.0) & (gx <= columns - 1) & (gz >= 0.0) & (gz <= rows - 1)
        gx = np.where(inside, gx, 0.0)
        gz = np.where(inside, gz, 0.0)
        i = np.minimum(gx.astype(np.intp), columns - 2)
        j = np.minimum(gz.astype(np.intp), rows - 2)
        return i, j, gx - i, gz - j, inside

    def _corners(self, i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, ...]:
        g = self.grid
        return g[j, i], g[j, i + 1], g[j + 1, i], g[j + 1, i + 1]

    def heights(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Bilinear height at each ``(x, z)``; ``-inf`` off the grid."""
        i, j, u, v, inside = self._cell(x, z)
        return self._height(i, j, u, v, inside)

    def _height(self, i: np.ndarray, j: np.ndarray, u: np.ndarray, v: np.ndarray,
                inside: np.ndarray) -> np.ndarray:
        h00, h10, h01, h11 = self._corners(i, j)
        height = (h00 * (1 - u) * (1 - v) + h10 * u * (1 - v)
                  + h01 * (1 - u) * v + h11 * u * v) + self.origin[1]
        return np.where(inside, height, -np.inf)

    def normals(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The normal of the cell's bilinear patch at each ``(x, z)``; up off the grid."""
        i, j, u, v, inside = self._cell(x, z)
        return self._normal(i, j, u, v, inside)

    def _normal(self, i: np.ndarray, j: np.ndarray, u: np.ndarray, v: np.ndarray,
                inside: np.ndarray) -> np.ndarray:
        h00, h10, h01, h11 = self._corners(i, j)
        slope_x = ((h10 - h00) * (1 - v) + (h11 - h01) * v) / self.spacing[0]
        slope_z = ((h01 - h00) * (1 - u) + (h11 - h10) * u) / self.spacing[1]
        out = np.stack((-slope_x, np.ones_like(slope_x), -slope_z), axis=-1)
        out /= np.linalg.norm(out, axis=-1, keepdims=True)
        return np.where(inside[..., None], out, (0.0, 1.0, 0.0))

    def sample(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Height and normal at each ``(x, z)`` from one lookup of the cells."""
        cell = self._cell(x, z)
        return self._height(*cell), self._normal(*cell)


class FunctionSurface(HeightSurface):
    """A height given by a function ``sample(x, z)``, in the field's local frame.

    sample -- the height at ``(x, z)``. Called with two floats unless
        ``vectorized``, then with two arrays of one shape that it answers an
        array for
    normal -- optional, the unit normal at ``(x, z)`` as ``(nx, ny, nz)`` with the
        same calling convention, array-valued as ``(..., 3)`` when vectorized.
        Without it the normal is the central difference of ``sample`` over
        ``delta``
    extent -- ``((x_min, z_min), (x_max, z_max))``, or None for a field that
        reaches every ``(x, z)``; either corner may hold infinities
    height_range -- ``(lowest, highest)`` the surface reaches, which bounds the
        body for the broad phase; None leaves it unbounded in y
    feature -- the horizontal distance over which the surface is taken to change,
        the step of a ray march and of a footprint search
    delta -- the offset of the finite differences, in the field's units
    """

    def __init__(self, sample: Callable[..., ArrayLike],
                 normal: Callable[..., ArrayLike] | None = None,
                 extent: tuple[tuple[float, float], tuple[float, float]] | None = None,
                 vectorized: bool = False,
                 height_range: tuple[float, float] | None = None,
                 feature: float = 1.0, delta: float = 1e-3) -> None:
        if feature <= 0.0 or delta <= 0.0:
            raise ValueError('feature and delta are positive distances')
        self._sample = sample
        self._normal = normal
        self._vectorized = vectorized
        self.delta = float(delta)
        self.feature = float(feature)
        (x0, z0), (x1, z1) = extent if extent is not None else ((-np.inf, -np.inf),
                                                                (np.inf, np.inf))
        if x1 < x0 or z1 < z0:
            raise ValueError('the extent runs from its low corner to its high one')
        y0, y1 = height_range if height_range is not None else (-np.inf, np.inf)
        self.low = np.array([x0, y0, z0], dtype='d')
        self.high = np.array([x1, y1, z1], dtype='d')

    def _inside(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return ((x >= self.low[0]) & (x <= self.high[0])
                & (z >= self.low[2]) & (z <= self.high[2]))

    def _call(self, function: Callable[..., ArrayLike], x: np.ndarray, z: np.ndarray,
              width: int) -> np.ndarray:
        """Call ``function`` at each point; ``width`` is one answer's length, 0 for a number."""
        if self._vectorized:
            return np.asarray(function(x, z), dtype='d')
        flat_x, flat_z = x.reshape(-1), z.reshape(-1)
        shape = (len(flat_x), width) if width else (len(flat_x),)
        out = np.empty(shape, dtype='d')
        for index in range(len(flat_x)):
            out[index] = function(float(flat_x[index]), float(flat_z[index]))
        return out.reshape(x.shape + ((width,) if width else ()))

    def heights(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The sampled height at each ``(x, z)``; ``-inf`` outside the extent."""
        x = np.asarray(x, dtype='d')
        z = np.asarray(z, dtype='d')
        inside = self._inside(x, z)
        if inside.all():
            return self._call(self._sample, x, z, 0)
        out = np.full(x.shape, -np.inf)
        if inside.any():
            out[inside] = self._call(self._sample, x[inside], z[inside], 0)
        return out

    def normals(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The supplied normal, or the central difference of the heights, at each ``(x, z)``."""
        x = np.asarray(x, dtype='d')
        z = np.asarray(z, dtype='d')
        if self._normal is not None:
            out = self._call(self._normal, x, z, 3)
            return out / np.linalg.norm(out, axis=-1, keepdims=True)
        d = self.delta
        return self._differenced(self.heights(x + d, z), self.heights(x - d, z),
                                 self.heights(x, z + d), self.heights(x, z - d))

    def sample(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`heights` and :meth:`normals` together.

        Without a normal function, the height and the four heights of its
        central difference are asked for in one call.
        """
        x = np.asarray(x, dtype='d')
        z = np.asarray(z, dtype='d')
        if self._normal is not None:
            return self.heights(x, z), self.normals(x, z)
        d = self.delta
        stencil = self.heights(np.stack((x, x + d, x - d, x, x)), np.stack((z, z, z, z + d, z - d)))
        return stencil[0], self._differenced(*stencil[1:])

    def column_contacts(self, x: float, z: float, heights: list[float], radius: float
                        ) -> list[ColumnContact]:
        """:meth:`HeightSurface.column_contacts`, the spheres slid together as
        :func:`sphere_contacts_by_sliding` slides them, with numbers.

        Each distinct point is sampled once a slide, and the spheres share the
        first, the point under them all, so a column on level ground costs one
        call of the function for the height and its four differences.
        """
        feet_x = [x] * len(heights)
        feet_z = [z] * len(heights)
        found: list[tuple[bool, float, float, float, float, float, float, float]] = []
        for _ in range(_SLIDES):
            samples = self._sample_points(feet_x, feet_z)
            found = []
            moved = 0.0
            for y, foot_x, foot_z, (height, nx, ny, nz) in zip(
                    heights, feet_x, feet_z, samples, strict=True):
                inside = math.isfinite(height)
                ground_y = height if inside else 0.0
                distance = (x - foot_x) * nx + (y - ground_y) * ny + (z - foot_z) * nz
                found.append((inside, foot_x, ground_y, foot_z, nx, ny, nz, distance))
            feet_x, feet_z = [], []
            for inside, foot_x, _y, foot_z, nx, _ny, nz, distance in found:
                next_x = x - distance * nx if inside else foot_x
                next_z = z - distance * nz if inside else foot_z
                moved = max(moved, abs(next_x - foot_x), abs(next_z - foot_z))
                feet_x.append(next_x)
                feet_z.append(next_z)
            if moved < _SETTLED:
                break
        out: list[ColumnContact] = []
        for inside, foot_x, ground_y, foot_z, nx, ny, nz, distance in found:
            depth = radius - distance
            if inside and depth > 0.0:
                half = 0.5 * depth
                out.append(((foot_x - nx * half, ground_y - ny * half, foot_z - nz * half),
                            (nx, ny, nz), depth))
        return out

    def _sample_points(self, xs: list[float], zs: list[float]
                       ) -> list[tuple[float, float, float, float]]:
        """:meth:`sample` at each ``(x, z)`` as numbers ``(height, nx, ny, nz)``.

        Each distinct point is asked for once.
        """
        index: dict[tuple[float, float], int] = {}
        for point in zip(xs, zs, strict=True):
            index.setdefault(point, len(index))
        points = list(index)
        px = np.array([p[0] for p in points], dtype='d')
        pz = np.array([p[1] for p in points], dtype='d')
        answers: list[tuple[float, float, float, float]] = []
        if self._normal is not None:
            heights, normals = self.heights(px, pz).tolist(), self.normals(px, pz).tolist()
            answers = [(height, nx, ny, nz)
                       for height, (nx, ny, nz) in zip(heights, normals, strict=True)]
        else:
            d = self.delta
            stencil = self.heights(np.stack((px, px + d, px - d, px, px)),
                                   np.stack((pz, pz, pz, pz + d, pz - d))).tolist()
            for height, east, west, north, south in zip(*stencil, strict=True):
                slope_x = (east - west) / (2 * d)
                slope_z = (north - south) / (2 * d)
                if not (math.isfinite(slope_x) and math.isfinite(slope_z)):
                    slope_x = slope_z = 0.0
                length = math.sqrt(slope_x * slope_x + 1.0 + slope_z * slope_z)
                answers.append((height, -slope_x / length, 1.0 / length, -slope_z / length))
        return [answers[index[point]] for point in zip(xs, zs, strict=True)]

    def _differenced(self, east: np.ndarray, west: np.ndarray, north: np.ndarray,
                     south: np.ndarray) -> np.ndarray:
        """The unit normal from the heights :attr:`delta` either side along x and along z."""
        d = self.delta
        # A difference taken across the edge of the extent is not a slope.
        with np.errstate(invalid='ignore'):
            slope_x = (east - west) / (2 * d)
            slope_z = (north - south) / (2 * d)
        flat = ~(np.isfinite(slope_x) & np.isfinite(slope_z))
        slope_x = np.where(flat, 0.0, slope_x)
        slope_z = np.where(flat, 0.0, slope_z)
        out = np.stack((-slope_x, np.ones_like(slope_x), -slope_z), axis=-1)
        return out / np.linalg.norm(out, axis=-1, keepdims=True)


#: The steepest slope, as a rise over a run, under which :func:`local_clearance` has a bound.
_GENTLE = 0.45


def clearance_spread(slope: float) -> float:
    """How far, as a multiple of a sphere's height above the ground, the slide down the tangent
    planes can carry the point it samples from the point under the sphere, on ground no steeper than
    ``slope``; ``inf`` past :data:`_GENTLE`.

    Each slide moves the sampled point to the foot of the perpendicular from the centre to the
    tangent plane, at most the distance to the plane times ``slope / sqrt(1 + slope**2)`` from the
    point under the centre, and that distance grows by at most ``1 + slope`` times the move.
    Over the four moves of :data:`_SLIDES` slides that sums to the value returned.
    """
    if slope > _GENTLE:
        return math.inf
    share = slope / math.sqrt(1.0 + slope * slope)
    grow = share * (1.0 + slope)
    return share * (1.0 + grow + grow * grow + grow * grow * grow)


def local_clearance(slope: float, radius: float) -> float:
    """The height above the ground under it from which a sphere of ``radius`` cannot touch ground no
    steeper than ``slope`` within :func:`clearance_spread` times that height; ``inf`` where no
    height is.

    At height ``H`` the sampled point stays within ``e = H * clearance_spread(slope)``, where the
    ground is at most ``slope * e`` higher and its tangent plane tilts toward the centre by at most
    ``slope * e``, so the distance to the plane is at least
    ``(H - 2 * slope * e) / sqrt(1 + slope**2)``.
    The height returned makes that ``radius``, with a margin for rounding.
    """
    spread = clearance_spread(slope)
    room = 1.0 - 2.0 * slope * spread
    if room <= 0.0:
        return math.inf
    return radius * math.sqrt(1.0 + slope * slope) / room * (1.0 + 1e-9) + 1e-9


#: The most spheres a grid finds contacts for one at a time, with numbers; more are found together.
_ONE_BY_ONE = 3
#: The most spheres a grid finds contacts for together with numbers; more are found as arrays.
_WITH_NUMBERS = 16


def sphere_contacts(surface: HeightSurface, centres: np.ndarray, radius: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spheres of ``radius`` at ``centres`` ``(K, 3)`` against the ground.

    The answer is that of :func:`sphere_contacts_by_sliding`. Where the surface knows how steep it
    can be (:attr:`HeightSurface.lipschitz`) one lookup of its height under each sphere leaves out
    the spheres too far above it to touch, which for a capsule standing on the ground is all but
    the lowest; a few left are found one at a time.
    """
    bound = surface.lipschitz
    one = getattr(surface, 'sphere_contact', None)
    if bound is None or one is None:
        return sphere_contacts_by_sliding(surface, centres, radius)
    count = len(centres)
    reach = 2.0 * radius * math.sqrt(1.0 + bound * bound)
    height_at = getattr(surface, 'height_at', None)
    if height_at is not None and count <= _WITH_NUMBERS:
        near = np.array([k for k, (x, y, z) in enumerate(centres.tolist())
                         if y - height_at(x, z) < reach], dtype=np.intp)
    else:
        heights = surface.heights(centres[:, 0], centres[:, 2])
        near = np.flatnonzero(centres[:, 1] - heights < reach)
    touching = np.zeros(count, dtype=bool)
    points = np.zeros((count, 3))
    normals = np.zeros((count, 3))
    normals[:, 1] = 1.0
    depths = np.full(count, -np.inf)
    if len(near) > _ONE_BY_ONE:
        together = getattr(surface, 'sphere_contacts_together', None)
        if together is not None and len(near) <= _WITH_NUMBERS:
            found = together(centres[near], radius)
        else:
            found = sphere_contacts_by_sliding(surface, centres[near], radius)
        touching[near], points[near], normals[near], depths[near] = found
        return touching, points, normals, depths
    for k in near.tolist():
        touching[k], points[k], normals[k], depths[k] = one(
            float(centres[k, 0]), float(centres[k, 1]), float(centres[k, 2]), radius)
    return touching, points, normals, depths


def sphere_contacts_by_sliding(surface: HeightSurface, centres: np.ndarray, radius: float
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spheres of ``radius`` at ``centres`` ``(K, 3)`` against the ground.

    Returns, per sphere, whether it touches, the contact point, the unit normal
    pointing out of the ground and the penetration along it. The nearest
    ground point is found by sliding down the tangent plane: from the point
    under the centre, the foot of the perpendicular from the centre to the
    plane there is the next guess. On a plane the first guess is the answer.
    """
    x = centres[:, 0].copy()
    z = centres[:, 2].copy()
    for _ in range(_SLIDES):
        height, normal = surface.sample(x, z)
        inside = np.isfinite(height)
        ground = np.stack((x, np.where(inside, height, 0.0), z), axis=-1)
        # Written out, not as einsum, whose inner loop may fuse the multiply and
        # the add: written so it rounds as GridSurface.sphere_contacts_together does.
        offset = centres - ground
        distance = (offset[:, 0] * normal[:, 0] + offset[:, 1] * normal[:, 1]
                    + offset[:, 2] * normal[:, 2])
        # The perpendicular foot of the centre on the tangent plane is the next guess.
        foot_x = np.where(inside, centres[:, 0] - distance * normal[:, 0], x)
        foot_z = np.where(inside, centres[:, 2] - distance * normal[:, 2], z)
        settled = max(np.abs(foot_x - x).max(), np.abs(foot_z - z).max()) < _SETTLED
        x, z = foot_x, foot_z
        if settled:
            break
    depth = radius - distance
    touching = inside & (depth > 0.0)
    point = ground - normal * (0.5 * depth)[:, None]
    return touching, point, normal, depth


def vertex_contacts(surface: HeightSurface, vertices: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Points of a shape, ``(N, 3)``, against the ground: those below the surface.

    Returns whether each is below, its contact point (midway between the point
    and the surface above it), the normal and the penetration along that
    normal: the height difference scaled by the normal's vertical component,
    which is the distance to the tangent plane.
    """
    height, normal = surface.sample(vertices[:, 0], vertices[:, 2])
    inside = np.isfinite(height)
    lift = np.where(inside, height - vertices[:, 1], 0.0)
    depth = lift * normal[:, 1]
    below = inside & (depth > 0.0)
    point = vertices + normal * (0.5 * depth)[:, None]
    return below, point, normal, depth


def footprint_contacts(surface: HeightSurface, vertices: np.ndarray,
                       face_normals: np.ndarray, face_offsets: np.ndarray, deeper_than: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground that rises into the interior of a convex shape, past its vertices.

    A peak under the middle of a flat face touches no vertex. The ground is
    sampled across the shape's footprint, a grid of :attr:`HeightSurface.feature`
    spacing over the box of its vertices seen from above, at most 7 samples a
    side. A sample inside the shape, whose faces are the planes
    ``face_normals . x = face_offsets`` with normals pointing out, is in contact:
    the contact point is the ground point and the normal the ground's. The depth
    is how far the shape would have to move along that normal for the point to
    leave through a face looking down.

    Only ground deeper into the shape than ``deeper_than``, the deepest vertex
    contact, is returned, since anything shallower is already a contact.

    Returns the contact points, normals and depths, possibly none.
    """
    empty = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    if not len(face_normals):
        return empty
    low = vertices[:, [0, 2]].min(axis=0)
    high = vertices[:, [0, 2]].max(axis=0)
    counts = np.clip(np.ceil((high - low) / surface.feature).astype(int) + 1, 2,
                     _FOOTPRINT_SAMPLES)
    gx, gz = np.meshgrid(np.linspace(low[0], high[0], counts[0]),
                         np.linspace(low[1], high[1], counts[1]))
    x, z = gx.ravel(), gz.ravel()
    height, normal = surface.sample(x, z)
    keep = np.isfinite(height)
    if not keep.any():
        return empty
    ground = np.column_stack((x[keep], height[keep], z[keep]))
    normal = normal[keep]
    # slack[i, f]: how far inside face f the ground point i is.
    slack = face_offsets[None, :] - ground @ face_normals.T
    inside = np.all(slack >= -1e-9, axis=1)
    if not inside.any():
        return empty
    ground, normal, slack = ground[inside], normal[inside], slack[inside]
    # A face looking down along the ground normal lets the point out: moving the
    # shape up by slack / (-n_face . n_ground) puts the point on that face.
    facing = -(normal @ face_normals.T)
    with np.errstate(divide='ignore', invalid='ignore'):
        escape = np.where(facing > 1e-9, slack / facing, np.inf)
    depth = escape.min(axis=1)
    pressed = np.isfinite(depth) & (depth > max(deeper_than, 0.0) + 1e-6)
    return ground[pressed], normal[pressed], depth[pressed]
