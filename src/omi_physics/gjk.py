"""GJK distance + EPA penetration for convex proxies — pure CPU numpy.

GJK (Gilbert-Johnson-Keerthi) walks the Minkowski difference A ⊖ B to decide
overlap; EPA (Expanding Polytope Algorithm) then grows the enclosing polytope to
the nearest face to recover the penetration normal, depth, and a witness contact
point.  Used for convex↔convex and convex↔triangle (dynamic-vs-static mesh).

References: van den Bergen, *Collision Detection in Interactive 3D Environments*.
"""
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from . import mathutil
from .collide import Contact

if TYPE_CHECKING:
    from .body import SupportProxy

EPS = 1e-10


class _Vertex:
    """One Minkowski-difference vertex plus its witness points on A and B."""
    __slots__ = ('v', 'a', 'b')

    def __init__(self, v: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
        self.v = v      # Minkowski point a-b
        self.a = a      # witness on A
        self.b = b      # witness on B


def _support(A: 'SupportProxy', B: 'SupportProxy', d: np.ndarray) -> _Vertex:
    """Support vertex of A ⊖ B along ``d`` (farthest on A, opposite on B)."""
    a = A.support(d)
    b = B.support(-d)
    return _Vertex(a - b, a, b)


def gjk_intersect(A: 'SupportProxy', B: 'SupportProxy',
                  max_iter: int = 32) -> Tuple[bool, List[_Vertex]]:
    """Test whether convex proxies A and B overlap; return ``(hit, simplex)``."""
    d = B.support(np.zeros(3)) - A.support(np.zeros(3))
    if np.dot(d, d) < EPS:
        d = np.array([1.0, 0.0, 0.0])
    simplex = [_support(A, B, -d)]
    d = -simplex[0].v
    for _ in range(max_iter):
        if np.dot(d, d) < EPS:
            return True, simplex
        w = _support(A, B, d)
        if np.dot(w.v, d) < 0:
            return False, simplex
        simplex.append(w)
        contains, simplex, d = _do_simplex(simplex)
        if contains:
            return True, simplex
    return False, simplex


def _do_simplex(simplex: List[_Vertex]) -> Tuple[bool, List[_Vertex], np.ndarray]:
    """Advance the simplex toward the origin; return ``(contains_origin, simplex, dir)``."""
    if len(simplex) == 2:
        return _line(simplex)
    if len(simplex) == 3:
        return _triangle(simplex)
    return _tetra(simplex)


def _line(simplex: List[_Vertex]) -> Tuple[bool, List[_Vertex], np.ndarray]:
    """Handle a 2-vertex simplex (edge); return the reduced simplex and search direction."""
    b, a = simplex[0], simplex[1]
    ab = b.v - a.v
    ao = -a.v
    if np.dot(ab, ao) > 0:
        d = np.cross(np.cross(ab, ao), ab)
        if np.dot(d, d) < EPS:
            d = _any_perp(ab)
        return False, [b, a], d
    return False, [a], ao


def _triangle(simplex: List[_Vertex]) -> Tuple[bool, List[_Vertex], np.ndarray]:
    """Handle a 3-vertex simplex (triangle); return the reduced simplex and search direction."""
    c, b, a = simplex
    ab, ac, ao = b.v - a.v, c.v - a.v, -a.v
    abc = np.cross(ab, ac)
    if np.dot(np.cross(abc, ac), ao) > 0:
        if np.dot(ac, ao) > 0:
            return False, [c, a], np.cross(np.cross(ac, ao), ac)
        return _line([b, a])
    if np.dot(np.cross(ab, abc), ao) > 0:
        return _line([b, a])
    if np.dot(abc, ao) > 0:
        return False, [c, b, a], abc
    return False, [b, c, a], -abc


def _tetra(simplex: List[_Vertex]) -> Tuple[bool, List[_Vertex], np.ndarray]:
    """Handle a 4-vertex simplex (tetrahedron); True if it encloses the origin."""
    d, c, b, a = simplex
    ab, ac, ad, ao = b.v - a.v, c.v - a.v, d.v - a.v, -a.v
    abc = np.cross(ab, ac)
    acd = np.cross(ac, ad)
    adb = np.cross(ad, ab)
    if np.dot(abc, ao) > 0:
        return _triangle([c, b, a])
    if np.dot(acd, ao) > 0:
        return _triangle([d, c, a])
    if np.dot(adb, ao) > 0:
        return _triangle([b, d, a])
    return True, simplex, np.zeros(3)


def _any_perp(v: np.ndarray) -> np.ndarray:
    """Any vector perpendicular to ``v`` (for degenerate edge cases)."""
    a = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
    return np.cross(v, a)


def epa(A: 'SupportProxy', B: 'SupportProxy', simplex: List[_Vertex],
        max_iter: int = 64) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """Expand the GJK simplex to the nearest face; return ``(normal, depth, point)``.

    ``simplex`` is the overlapping simplex from :func:`gjk_intersect`.  Returns
    ``None`` if the polytope stays degenerate.  The normal points from A into B
    along the Minkowski face; the caller orients it.
    """
    verts = list(simplex)
    if len(verts) < 4:
        tetra = _expand_to_tetra(A, B, verts)
        if tetra is None:
            return None
        verts = tetra
    faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    for _ in range(max_iter):
        normal, dist, face = _closest_face(verts, faces)
        if normal is None:
            return None
        assert face is not None
        w = _support(A, B, normal)
        d = np.dot(w.v, normal)
        if d - dist < 1e-6:
            return _contact_from_face(verts, face, normal, dist)
        verts.append(w)
        faces = _rebuild(verts, faces, len(verts) - 1)
        if not faces:
            return None
    normal, dist, face = _closest_face(verts, faces)
    if normal is None:
        return None
    assert face is not None
    return _contact_from_face(verts, face, normal, dist)


def _expand_to_tetra(A: 'SupportProxy', B: 'SupportProxy',
                     verts: List[_Vertex]) -> Optional[List[_Vertex]]:
    """Grow a sub-tetrahedral simplex to 4 non-degenerate vertices, or ``None``."""
    dirs = [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
            np.array([0, 1.0, 0]), np.array([0, -1.0, 0]),
            np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]
    pts = list(verts)
    for d in dirs:
        if len(pts) >= 4:
            break
        w = _support(A, B, d)
        if all(np.dot(w.v - p.v, w.v - p.v) > EPS for p in pts):
            pts.append(w)
    if len(pts) < 4:
        return None
    v0, v1, v2, v3 = pts[:4]
    if np.dot(np.cross(v1.v - v0.v, v2.v - v0.v), v3.v - v0.v) > 0:
        pts[1], pts[2] = pts[2], pts[1]
    return pts[:4]


def _closest_face(verts: List[_Vertex], faces: List[Tuple[int, int, int]]
                  ) -> Tuple[Optional[np.ndarray], float, Optional[Tuple[int, int, int]]]:
    """Face nearest the origin as ``(outward_normal, distance, face)``, or all ``None`` if none valid."""
    best_n: Optional[np.ndarray] = None
    best_d: float = np.inf
    best_f: Optional[Tuple[int, int, int]] = None
    for f in faces:
        a, b, c = verts[f[0]].v, verts[f[1]].v, verts[f[2]].v
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < EPS:
            continue
        n = n / ln
        d = np.dot(n, a)
        if d < 0:
            n, d = -n, -d
        if d < best_d:
            best_n, best_d, best_f = n, d, f
    return best_n, best_d, best_f


def _rebuild(verts: List[_Vertex], faces: List[Tuple[int, int, int]],
             new_idx: int) -> List[Tuple[int, int, int]]:
    """Re-triangulate the polytope after adding vertex ``new_idx`` (remove seen faces)."""
    w = verts[new_idx].v
    visible_edges: Dict[Tuple[int, int], bool] = {}
    kept: List[Tuple[int, int, int]] = []
    for f in faces:
        a = verts[f[0]].v
        n = np.cross(verts[f[1]].v - a, verts[f[2]].v - a)
        if np.dot(n, w - a) > 0:               # face sees the new point
            for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                key = (e[1], e[0])
                if key in visible_edges:
                    del visible_edges[key]
                else:
                    visible_edges[e] = True
        else:
            kept.append(f)
    for (i, j) in visible_edges:
        kept.append((i, j, new_idx))
    return kept


def _contact_from_face(verts: List[_Vertex], face: Tuple[int, int, int],
                       normal: np.ndarray, dist: float
                       ) -> Tuple[np.ndarray, float, np.ndarray]:
    """Recover ``(normal, depth, world_point)`` from the nearest EPA face's witnesses."""
    a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
    bary = _barycentric(normal * dist, a.v, b.v, c.v)
    pa = bary[0] * a.a + bary[1] * b.a + bary[2] * c.a
    pb = bary[0] * a.b + bary[1] * b.b + bary[2] * c.b
    point = 0.5 * (pa + pb)
    return normal, dist, point


def _barycentric(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                 c: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of ``p`` in triangle ``abc`` (its own plane projection)."""
    v0, v1, v2 = b - a, c - a, p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < EPS:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - v - w, v, w])


def collide_convex(a: int, b: int, PA: 'SupportProxy',
                   PB: 'SupportProxy') -> List[Contact]:
    """Convex↔convex (and convex↔triangle) via GJK+EPA → a single contact."""
    hit, simplex = gjk_intersect(PA, PB)
    if not hit:
        return []
    result = epa(PA, PB, simplex)
    if result is None:
        return []
    normal, depth, point = result
    # EPA normal points from A into B along the Minkowski face; orient A→B
    if np.dot(normal, PB.center_hint() - PA.center_hint()) < 0:
        normal = -normal
    if depth <= 1e-9:
        return []
    return [Contact(a, b, point, mathutil.normalize(normal), depth)]
