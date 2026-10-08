"""Mass properties of collision shapes: volume, centroid and inertia.

Every shape is taken as a solid of uniform density. A ``convex`` or
``trimesh`` shape is measured as the box its points span, which is what a
hull or a level mesh has without a volume integral over its faces. A
``capsule`` is measured as the cylinder of its height and radius.

A ``compound`` shares its mass among its parts by volume, and its inertia is
the sum of theirs about the body's origin. The world keeps a body's inertia
as a diagonal in the body's own frame, so :func:`inertia_diagonal` answers
the diagonal of that sum; the products of inertia a lopsided compound has are
left out.
"""
import numpy as np

from . import mathutil, model

__all__ = ['centroid', 'inertia_diagonal', 'inertia_tensor', 'volume']

#: Gauss-Legendre nodes and weights on [-1, 1], exact for the degree-4
#: polynomials a narrowing cylinder's integrals are.
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(3)


def volume(shape: model.Shape) -> float:
    """The shape's volume, in cubic metres."""
    kind = shape.type
    if kind == 'sphere':
        return 4.0 / 3.0 * np.pi * shape.radius ** 3
    if kind == 'box':
        x, y, z = shape.size
        return float(x * y * z)
    if kind == 'capsule':
        return float(np.pi * shape.radiusBottom ** 2 * shape.height)
    if kind == 'cylinder':
        bottom, top = shape.radiusBottom, shape.radiusTop
        return float(np.pi * shape.height * (bottom * bottom + bottom * top + top * top) / 3.0)
    if kind == 'compound':
        return float(sum(volume(part.shape) for part in shape.parts or ()))
    extent = _extent(shape)
    return float(np.prod(extent[1] - extent[0])) if extent is not None else 0.0


def centroid(shape: model.Shape) -> np.ndarray:
    """Where the shape's mass is, in its own frame."""
    kind = shape.type
    if kind == 'cylinder':
        offset = _cylinder_moments(shape)[2]
        return np.array([0.0, offset, 0.0])
    if kind == 'compound':
        parts = list(shape.parts or ())
        if not parts:
            return np.zeros(3)
        shares = _shares(parts)
        return np.sum([share * _part_centroid(part) for share, part in zip(shares, parts,
                                                                           strict=True)],
                      axis=0)
    if kind in ('convex', 'trimesh'):
        extent = _extent(shape)
        return np.zeros(3) if extent is None else 0.5 * (extent[0] + extent[1])
    return np.zeros(3)


def inertia_tensor(shape: model.Shape, mass: float) -> np.ndarray:
    """The 3x3 inertia tensor of ``mass`` kilograms spread through the shape.

    About the shape's centroid, except a ``compound``, which is about its
    own origin: the body's origin, where its parts are placed from.
    """
    if shape.type != 'compound':
        return np.diag(_principal(shape, mass))
    parts = list(shape.parts or ())
    total = np.zeros((3, 3))
    for share, part in zip(_shares(parts), parts, strict=True):
        part_mass = mass * share
        turn = mathutil.quat_to_matrix(np.asarray(part.orientation, dtype='d'))
        own = turn @ _about_centroid(part.shape, part_mass) @ turn.T
        total += own + _shifted(_part_centroid(part), part_mass)
    return total


def inertia_diagonal(shape: model.Shape | None, mass: float) -> np.ndarray:
    """The diagonal of the shape's inertia tensor (:func:`inertia_tensor`)."""
    if shape is None:
        return np.zeros(3)
    return np.diag(inertia_tensor(shape, mass)).copy()


def _shifted(offset: np.ndarray, mass: float) -> np.ndarray:
    """What carrying ``mass`` at ``offset`` adds to an inertia tensor (parallel axes)."""
    return mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))


def _about_centroid(shape: model.Shape, mass: float) -> np.ndarray:
    """A shape's inertia tensor about its centroid, in its own frame."""
    if shape.type != 'compound':
        return np.diag(_principal(shape, mass))
    return inertia_tensor(shape, mass) - _shifted(centroid(shape), mass)


def _principal(shape: model.Shape, mass: float) -> np.ndarray:
    """A shape's inertia about its centroid, along its own axes."""
    kind = shape.type
    if kind == 'sphere':
        value = 0.4 * mass * shape.radius ** 2
        return np.array([value, value, value])
    if kind == 'box':
        return _box(np.asarray(shape.size, dtype='d'), mass)
    if kind == 'capsule':
        radius, height = shape.radiusBottom, shape.height
        across = mass * (3 * radius * radius + height * height) / 12.0
        return np.array([across, 0.5 * mass * radius * radius, across])
    if kind == 'cylinder':
        axial, across, offset = _cylinder_moments(shape)
        across -= offset * offset
        return mass * np.array([across, axial, across])
    extent = _extent(shape)
    if extent is None:
        return np.zeros(3)
    return _box(extent[1] - extent[0], mass)


def _box(size: np.ndarray, mass: float) -> np.ndarray:
    """A solid box's inertia about its centre."""
    x, y, z = size
    return mass / 12.0 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])


def _cylinder_moments(shape: model.Shape) -> tuple[float, float, float]:
    """Per kilogram: the axial moment, the moment across the origin, and the centroid height.

    The radius narrows linearly from ``radiusBottom`` at ``-height/2`` to
    ``radiusTop`` at ``+height/2``; each slice is a disc.
    """
    half = 0.5 * shape.height
    heights = _NODES * half
    radii = (shape.radiusBottom + (shape.radiusTop - shape.radiusBottom)
             * (heights + half) / max(shape.height, 1e-12))
    area = radii * radii
    weights = _WEIGHTS * half
    amount = float(np.dot(weights, area))
    if amount <= 0.0:
        return 0.0, 0.0, 0.0
    axial = float(np.dot(weights, area * area / 2.0)) / amount
    across = float(np.dot(weights, area * (area / 4.0 + heights * heights))) / amount
    offset = float(np.dot(weights, area * heights)) / amount
    return axial, across, offset


def _extent(shape: model.Shape) -> tuple[np.ndarray, np.ndarray] | None:
    """The box a point-cloud shape's points span, or None for one with none."""
    if shape.points is None or not len(shape.points):
        return None
    points = np.asarray(shape.points, dtype='d')
    return points.min(axis=0), points.max(axis=0)


def _shares(parts: 'list[model.ShapePart]') -> np.ndarray:
    """Each part's fraction of a compound's mass: by volume, or equal where none has any."""
    volumes = np.array([volume(part.shape) for part in parts], dtype='d')
    total = float(volumes.sum())
    if total <= 0.0:
        return np.full(len(parts), 1.0 / len(parts))
    return volumes / total


def _part_centroid(part: 'model.ShapePart') -> np.ndarray:
    """Where a compound part's mass is, in the compound's frame."""
    turn = mathutil.quat_to_matrix(np.asarray(part.orientation, dtype='d'))
    return np.asarray(part.position, dtype='d') + turn @ centroid(part.shape)
