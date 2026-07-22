"""Convex hull + approximate convex decomposition (load-time cooking helpers).

No scipy dependency: a compact incremental 3D hull builds the face set used for
``convex`` colliders, debug wireframes, and the concavity measure that drives
``auto`` cooking.  The decomposition is a simple recursive split (not full
V-HACD) — enough to turn a concave mesh into a compound of convex pieces that
round-trips as OMI ``convex`` shapes.
"""
from typing import Dict, List, Optional, Tuple
from numpy.typing import ArrayLike
import numpy as np

EPS = 1e-9


def convex_hull(points: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, faces)`` — hull vertex array and triangle indices.

    Degenerate input (fewer than 4 non-coplanar points) returns the unique
    points with an empty face list.
    """
    pts = np.unique(np.asarray(points, dtype='d'), axis=0)
    if len(pts) < 4:
        return pts, np.zeros((0, 3), dtype='i')
    tetra = _initial_tetra(pts)
    if tetra is None:
        return pts, np.zeros((0, 3), dtype='i')
    faces = _incremental(pts, tetra)
    used = np.unique(np.asarray(faces).ravel())
    remap = {old: new for new, old in enumerate(used)}
    verts = pts[used]
    tris = np.array([[remap[i] for i in f] for f in faces], dtype='i')
    return verts, tris


def _initial_tetra(pts: np.ndarray) -> Optional[List[Tuple[int, int, int]]]:
    """Seed four outward-oriented faces from a non-degenerate tetrahedron, or ``None``."""
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(pts - pts[i0], axis=1)))
    if np.linalg.norm(pts[i1] - pts[i0]) < EPS:
        return None
    line = pts[i1] - pts[i0]
    areas = np.linalg.norm(np.cross(pts - pts[i0], line), axis=1)
    i2 = int(np.argmax(areas))
    if areas[i2] < EPS:
        return None
    normal = np.cross(pts[i1] - pts[i0], pts[i2] - pts[i0])
    dists = np.abs((pts - pts[i0]) @ normal)
    i3 = int(np.argmax(dists))
    if dists[i3] < EPS:
        return None
    verts = [i0, i1, i2, i3]
    if np.dot(np.cross(pts[i1] - pts[i0], pts[i2] - pts[i0]), pts[i3] - pts[i0]) > 0:
        verts[1], verts[2] = verts[2], verts[1]
    a, b, c, d = verts
    return [(a, b, c), (a, c, d), (a, d, b), (b, d, c)]


def _incremental(pts: np.ndarray,
                 faces: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Add each point to the hull, replacing the faces it sees with new ones."""
    faces = list(faces)
    for p in range(len(pts)):
        point = pts[p]
        visible: List[Tuple[int, int, int]] = []
        for f in faces:
            n, ref = _face_plane(pts, f)
            if np.dot(n, point - ref) > EPS:
                visible.append(f)
        if not visible:
            continue
        horizon = _horizon_edges(visible)
        faces = [f for f in faces if f not in visible]
        for (i, j) in horizon:
            faces.append((i, j, p))
    return faces


def _face_plane(pts: np.ndarray,
                f: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return a face's ``(unit_normal, reference_point)``."""
    a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    if ln > EPS:
        n = n / ln
    return n, a


def _horizon_edges(visible: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    """Boundary edges of the visible-face patch (each belonging to one visible face)."""
    edge_count: Dict[Tuple[int, ...], int] = {}
    for f in visible:
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = tuple(sorted(e))
            edge_count[key] = edge_count.get(key, 0) + 1
    horizon: List[Tuple[int, int]] = []
    for f in visible:
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if edge_count[tuple(sorted(e))] == 1:
                horizon.append(e)
    return horizon


def hull_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume of a triangulated hull (0 for an empty face list)."""
    if len(faces) == 0:
        return 0.0
    origin = vertices.mean(axis=0)
    vol = 0.0
    for f in faces:
        a, b, c = vertices[f[0]] - origin, vertices[f[1]] - origin, vertices[f[2]] - origin
        vol += abs(np.dot(a, np.cross(b, c))) / 6.0
    return vol


def concavity(points: ArrayLike) -> float:
    """1 − (point-cloud AABB fill) proxy in [0,1): higher = more concave.

    Uses the ratio of the hull volume to its bounding-box volume; a solid convex
    block fills much of its box, an L-shape or shell far less.
    """
    pts = np.asarray(points, dtype='d')
    verts, faces = convex_hull(pts)
    if len(faces) == 0:
        return 0.0
    hv = hull_volume(verts, faces)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    box = np.prod(np.maximum(hi - lo, EPS))
    return float(np.clip(1.0 - hv / box, 0.0, 1.0))


def approximate_convex_decomposition(points: ArrayLike, max_pieces: int = 8,
                                     threshold: float = 0.30,
                                     depth: int = 0) -> List[np.ndarray]:
    """Recursively split a concave cloud into convex-ish pieces.

    Splits on the widest axis at the centroid until each piece is convex enough
    (low concavity) or the piece budget is spent.  Returns a list of point
    arrays, each cooked into an OMI ``convex`` shape by the caller.
    """
    pts = np.asarray(points, dtype='d')
    if depth >= 3 or max_pieces <= 1 or len(pts) <= 8 or concavity(pts) < threshold:
        return [pts]
    axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
    mid = np.median(pts[:, axis])
    left = pts[pts[:, axis] <= mid]
    right = pts[pts[:, axis] > mid]
    if len(left) < 4 or len(right) < 4:
        return [pts]
    budget = max_pieces // 2
    return (approximate_convex_decomposition(left, budget, threshold, depth + 1)
            + approximate_convex_decomposition(right, budget, threshold, depth + 1))
