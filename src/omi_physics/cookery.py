"""Cooking — collision geometry from an arbitrary mesh.

``cook_shape`` turns a loaded ``IndexedFaceSet`` / glTF mesh (a vertex array plus
optional triangle indices) into an OMI :class:`~omi_physics.model.Shape`
(or a compound list for ``decompose``), choosing the cheapest proxy that fits.
Results are cached on the vertex array so a mesh is cooked once.
"""
from typing import Dict, List, Optional, Union
from numpy.typing import ArrayLike
import numpy as np

from . import model
from . import hull

_CACHE: Dict[tuple, tuple] = {}


def cook_shape(points: ArrayLike, indices: Optional[ArrayLike] = None,
               strategy: str = 'auto',
               dynamic: bool = True) -> Union[model.Shape, List[model.Shape]]:
    """Return a ``model.Shape`` (or list of shapes for ``decompose``).

    Strategies: ``primitive`` (best-fit box/sphere), ``convex`` (hull),
    ``decompose`` (compound of convex pieces), ``trimesh`` (static triangle soup),
    ``auto`` (static→trimesh, dynamic→convex, escalating to decompose on high
    concavity).
    """
    pts = np.asarray(points, dtype='d')
    key = (id(points), strategy, dynamic, None if indices is None else id(indices))
    cached = _CACHE.get(key)
    if cached is not None and cached[0] is points:
        return cached[1]

    if strategy == 'auto':
        strategy = _auto_strategy(pts, indices, dynamic)
    result: Union[model.Shape, List[model.Shape]]
    if strategy == 'primitive':
        result = _fit_primitive(pts)
    elif strategy == 'convex':
        result = _cook_convex(pts)
    elif strategy == 'decompose':
        result = [_cook_convex(piece)
                  for piece in hull.approximate_convex_decomposition(pts)]
    elif strategy == 'trimesh':
        result = model.Shape.trimesh(pts, _ensure_indices(pts, indices))
    else:
        raise ValueError('unknown cook strategy %r' % strategy)

    _CACHE[key] = (points, result)
    return result


def _auto_strategy(pts: np.ndarray, indices: Optional[ArrayLike],
                   dynamic: bool) -> str:
    """Pick a cook strategy: static-with-indices→trimesh, concave→decompose, else convex."""
    if not dynamic and indices is not None:
        return 'trimesh'
    if hull.concavity(pts) > 0.4:
        return 'decompose'
    return 'convex'


def _cook_convex(pts: np.ndarray) -> model.Shape:
    """A ``convex`` shape from the point cloud's convex hull (raw points if degenerate)."""
    verts, _ = hull.convex_hull(pts)
    if len(verts) < 4:
        verts = pts
    return model.Shape.convex(verts)


def _fit_primitive(pts: np.ndarray) -> model.Shape:
    """Best-fit primitive: a sphere for near-spherical clouds, else an AABB box."""
    centroid = pts.mean(axis=0)
    radii = np.linalg.norm(pts - centroid, axis=1)
    mean_r = max(radii.mean(), 1e-9)
    if radii.std() / mean_r < 0.1:
        return model.Shape.sphere(float(mean_r))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return model.Shape.box(tuple(hi - lo))


def _ensure_indices(pts: np.ndarray, indices: Optional[ArrayLike]) -> np.ndarray:
    """Triangle indices for a trimesh: the given ones, or the hull's own triangulation."""
    if indices is not None:
        return np.asarray(indices, dtype='i')
    # fall back to the convex hull's own triangulation
    _, faces = hull.convex_hull(pts)
    return faces


def clear_cache() -> None:
    """Drop all cached cooked shapes."""
    _CACHE.clear()
