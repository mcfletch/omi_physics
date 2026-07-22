"""Narrow-phase collision math — pure CPU numpy, no GL.

Every routine returns a list of :class:`Contact` with the normal pointing from
body A toward body B and ``depth > 0`` for penetration.  Analytic tests cover the
primitive pairs; oriented box↔box uses SAT plus face clipping to build a stable
multi-point manifold (Ericson, *Real-Time Collision Detection*).  Convex↔convex
via GJK/EPA lands in Phase 3 (:mod:`omi_physics.gjk`).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, cast
import numpy as np

from . import mathutil
from .body import (SphereProxy, BoxProxy, CapsuleProxy, ConvexProxy, TriangleProxy,
                   TriangleMeshProxy, Proxy, SupportProxy)

EPS = 1e-9


@dataclass
class Contact:
    """One contact point between bodies ``a`` and ``b``.

    ``normal`` is unit length and points from ``a`` toward ``b``; ``depth`` is the
    penetration (``> 0`` when overlapping).  ``normal_impulse`` / ``tangent_impulse``
    are scratch the solver fills in and warm-starts from — leave them at their
    defaults when constructing a contact.
    """
    a: int
    b: int
    point: np.ndarray
    normal: np.ndarray          # unit, from A to B
    depth: float
    # solver scratch (filled in solver)
    normal_impulse: float = 0.0
    tangent_impulse: np.ndarray = field(default_factory=lambda: np.zeros(2))


def sphere_sphere(a: int, b: int, A: SphereProxy, B: SphereProxy) -> List[Contact]:
    """Sphere↔sphere contact (empty if the spheres are apart)."""
    d = B.center - A.center
    dist = np.linalg.norm(d)
    r = A.radius + B.radius
    if dist >= r:
        return []
    n = d / dist if dist > EPS else np.array([0.0, 1.0, 0.0])
    depth = r - dist
    point = A.center + n * (A.radius - 0.5 * depth)
    return [Contact(a, b, point, n, depth)]


def sphere_box(a: int, b: int, S: SphereProxy, B: BoxProxy) -> List[Contact]:
    """Sphere↔oriented-box contact, handling the sphere centre inside the box."""
    local = B.R.T @ (S.center - B.center)
    clamped = np.clip(local, -B.half, B.half)
    delta = local - clamped
    dist = np.linalg.norm(delta)
    if dist > S.radius:
        return []
    if dist > EPS:
        n_local = delta / dist
        depth = S.radius - dist
    else:                                   # centre inside the box
        axis = int(np.argmin(B.half - np.abs(local)))
        n_local = np.zeros(3)
        n_local[axis] = 1.0 if local[axis] >= 0 else -1.0
        depth = S.radius + (B.half[axis] - abs(local[axis]))
    normal = B.R @ n_local                  # points box→sphere (A=sphere→B=box: flip)
    world_point = B.center + B.R @ clamped
    return [Contact(a, b, world_point, -normal, depth)]


def _project(half: np.ndarray, R: np.ndarray, center: np.ndarray,
             axis: np.ndarray) -> Tuple[float, float]:
    """Return the ``(min, max)`` extent of an oriented box projected onto ``axis``."""
    r = np.sum(np.abs((R.T @ axis)) * half)
    c = np.dot(center, axis)
    return c - r, c + r


def box_box(a: int, b: int, A: BoxProxy, B: BoxProxy) -> List[Contact]:
    """Oriented box↔box contact via SAT plus face clipping (multi-point manifold)."""
    # Candidate separating axes: A's 3 face normals, B's 3, and the (up to 9)
    # edge-edge cross products. Face normals are already unit (rotation columns).
    axes = [A.R[:, 0], A.R[:, 1], A.R[:, 2], B.R[:, 0], B.R[:, 1], B.R[:, 2]]
    for i in range(3):
        ai = A.R[:, i]
        for j in range(3):
            c = mathutil.cross3(ai, B.R[:, j])
            n2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2]
            if n2 > EPS * EPS:
                axes.append(c / np.sqrt(n2))
    ax = np.array(axes)                          # (K, 3), all unit
    d = B.center - A.center
    # Project both box radii and the centre offset onto every axis at once. The
    # box radius along an axis is |axis in box frame| . half; ax @ R gives each
    # axis's components in the box's local frame for all axes together.
    ra = np.sum(np.abs(ax @ A.R) * A.half, axis=1)
    rb = np.sum(np.abs(ax @ B.R) * B.half, axis=1)
    proj = ax @ d
    overlap = ra + rb - np.abs(proj)
    if np.any(overlap < 0.0):
        return []                                # a separating axis exists
    k = int(np.argmin(overlap))
    best_axis = ax[k] if proj[k] >= 0 else -ax[k]
    return _clip_manifold(a, b, A, B, best_axis, float(overlap[k]))


def _clip_manifold(a: int, b: int, A: BoxProxy, B: BoxProxy, normal: np.ndarray,
                   depth: float) -> List[Contact]:
    """Clip the incident face against the reference face into up to 4 contacts."""
    ref, inc, flip = _pick_reference(A, B, normal)
    ref_n = -normal if flip else normal          # outward from ref toward inc
    inc_face, _ = _incident_face(inc, ref_n)
    ref_face, side_planes, _, _ = _reference_face(ref, ref_n)
    face_point = ref_face[0]

    poly = inc_face
    for plane_n, offset in side_planes:
        poly = _clip_face(poly, plane_n, offset)
        if len(poly) == 0:
            break

    contacts = []
    for p in poly:
        pen = np.dot(face_point - p, ref_n)      # >0 => below ref face (penetrating)
        if pen >= -1e-6:
            proj = p + ref_n * pen               # seat the point on the ref face
            contacts.append(Contact(a, b, proj, normal, max(pen, 0.0)))
    if not contacts:
        contacts = [Contact(a, b, 0.5 * (A.center + B.center), normal, depth)]
    return _reduce_manifold(contacts)


def _pick_reference(A: BoxProxy, B: BoxProxy,
                    normal: np.ndarray) -> Tuple[BoxProxy, BoxProxy, bool]:
    """Choose reference/incident box: the reference face is most aligned with the normal.

    Returns ``(reference, incident, flip)`` where ``flip`` is True when B is the
    reference (so the caller re-orients the outward normal accordingly).
    """
    a_align = np.max(np.abs(A.R.T @ normal))
    b_align = np.max(np.abs(B.R.T @ normal))
    if a_align >= b_align:
        return A, B, False
    return B, A, True


def _face_axis(box: BoxProxy, direction: np.ndarray) -> Tuple[int, float]:
    """Local axis index and sign of the box face most facing ``direction``."""
    dots = box.R.T @ direction
    axis = int(np.argmax(np.abs(dots)))
    sign = 1.0 if dots[axis] >= 0 else -1.0
    return axis, sign


def _incident_face(box: BoxProxy, ref_normal: np.ndarray) -> Tuple[list, int]:
    """The box face most anti-parallel to ``ref_normal``, as ``(vertices, axis)``."""
    axis, sign = _face_axis(box, -ref_normal)
    return _face_vertices(box, axis, sign), axis


def _reference_face(box: BoxProxy, ref_normal: np.ndarray):
    """Reference face vertices, its side clip planes, offsets, and outward normal."""
    axis, sign = _face_axis(box, ref_normal)
    verts = _face_vertices(box, axis, sign)
    n0 = box.R[:, axis] * sign
    side_planes = []
    offsets = []
    for other in range(3):
        if other == axis:
            continue
        for s in (1.0, -1.0):
            plane_n = box.R[:, other] * s
            offset = np.dot(box.center, plane_n) + box.half[other]
            side_planes.append((plane_n, offset))
            offsets.append(offset)
    return verts, side_planes, offsets, box.R[:, axis] * sign


def _face_vertices(box: BoxProxy, axis: int, sign: float) -> list:
    """The four world-space corners of one box face (local ``axis``, ``sign`` side)."""
    other = [i for i in range(3) if i != axis]
    c = box.center + box.R[:, axis] * sign * box.half[axis]
    u = box.R[:, other[0]] * box.half[other[0]]
    v = box.R[:, other[1]] * box.half[other[1]]
    return [c + u + v, c - u + v, c - u - v, c + u - v]


def _clip_face(poly: list, plane_n: np.ndarray, offset: float) -> list:
    """Sutherland-Hodgman clip of polygon ``poly`` against one half-space."""
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        dc = np.dot(plane_n, cur) - offset
        dn = np.dot(plane_n, nxt) - offset
        if dc <= 0:
            out.append(cur)
        if dc * dn < 0:
            t = dc / (dc - dn)
            out.append(cur + t * (nxt - cur))
    return out


def _reduce_manifold(contacts: List[Contact], max_points: int = 4) -> List[Contact]:
    """Keep the ``max_points`` most spread-out contacts (a stable, well-conditioned set)."""
    if len(contacts) <= max_points:
        return contacts
    pts = np.array([c.point for c in contacts])
    keep = [int(np.argmax(pts[:, 0]))]
    d = np.linalg.norm(pts - pts[keep[0]], axis=1)
    keep.append(int(np.argmax(d)))
    d2 = np.linalg.norm(pts - pts[keep[1]], axis=1)
    keep.append(int(np.argmax(d2 + d)))
    d3 = np.linalg.norm(pts - pts[keep[2]], axis=1)
    keep.append(int(np.argmax(d3)))
    seen, uniq = set(), []
    for k in keep:
        if k not in seen:
            seen.add(k)
            uniq.append(contacts[k])
    return uniq


_DISPATCH: Dict[Tuple[str, str], Callable[..., List[Contact]]] = {
    ('sphere', 'sphere'): sphere_sphere,
    ('sphere', 'box'): sphere_box,
    ('box', 'box'): box_box,
}


def collide(a: int, b: int, PA: Proxy, PB: Proxy) -> List[Contact]:
    """Dispatch a proxy pair to the right routine, normalizing order.

    The returned contacts always point from body ``a`` toward body ``b`` regardless
    of the proxies' order.  Any pair not covered by an analytic routine falls back
    to GJK/EPA; a triangle mesh on either side is collided triangle-by-triangle.
    """
    if PA.kind == 'trimesh' or PB.kind == 'trimesh':
        return _collide_mesh(a, b, PA, PB)
    key = (PA.kind, PB.kind)
    if key in _DISPATCH:
        return _DISPATCH[key](a, b, PA, PB)
    rkey = (PB.kind, PA.kind)
    if rkey in _DISPATCH:
        flipped = _DISPATCH[rkey](b, a, PB, PA)
        return [_flip(c) for c in flipped]
    from .gjk import collide_convex
    # The trimesh case returned above, so both proxies answer support() here.
    return collide_convex(a, b, cast(SupportProxy, PA), cast(SupportProxy, PB))


def _collide_mesh(a: int, b: int, PA: Proxy, PB: Proxy) -> List[Contact]:
    """Convex↔triangle-soup: collide the mover against overlapping triangles.

    A capsule mover (the character) uses the analytic capsule↔triangle test — far
    faster and more robust than GJK/EPA, whose polytope expansion stalls on the
    big flat wall/floor triangles of an arbitrary mesh.
    """
    from .gjk import collide_convex
    if PA.kind == 'trimesh':
        mesh, other, mesh_is_a = cast(TriangleMeshProxy, PA), cast(SupportProxy, PB), True
    else:
        mesh, other, mesh_is_a = cast(TriangleMeshProxy, PB), cast(SupportProxy, PA), False
    lo, hi = other.aabb()
    use_capsule = other.kind == 'capsule'
    contacts = []
    for tri in mesh.triangles_overlapping(lo, hi):
        if use_capsule:
            c = capsule_triangle(cast(CapsuleProxy, other), tri)
            if c is None:
                continue
            pt, normal, depth = c
            if mesh_is_a:                       # A is the mesh: normal points A→B
                contacts.append(Contact(a, b, pt, normal, depth))
            else:                               # B is the mesh: flip to A→B
                contacts.append(Contact(a, b, pt, -normal, depth))
        elif mesh_is_a:
            contacts.extend(collide_convex(a, b, tri, other))
        else:
            contacts.extend(collide_convex(a, b, other, tri))
    return contacts


def capsule_triangle(cap: CapsuleProxy,
                     tri: TriangleProxy) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Analytic capsule↔triangle. Returns ``(point, normal, depth)`` with the
    normal pointing from the triangle toward the capsule, or ``None`` if apart."""
    v0, v1, v2 = tri.verts
    best: Tuple[float, Optional[np.ndarray], Optional[np.ndarray]] = (np.inf, None, None)
    for sp in (cap.p0, cap.p1):
        tp = _closest_point_on_triangle(sp, v0, v1, v2)
        d2 = float(np.dot(sp - tp, sp - tp))
        if d2 < best[0]:
            best = (d2, sp, tp)
    for e0, e1 in ((v0, v1), (v1, v2), (v2, v0)):
        sp, tp = _closest_segment_segment(cap.p0, cap.p1, e0, e1)
        d2 = float(np.dot(sp - tp, sp - tp))
        if d2 < best[0]:
            best = (d2, sp, tp)
    best_d2, best_sp, best_tp = best
    r = cap.radius
    if best_d2 >= r * r or best_sp is None or best_tp is None:
        return None
    d = np.sqrt(best_d2)
    if d > EPS:
        normal = (best_sp - best_tp) / d        # triangle -> capsule
    else:
        n = np.cross(v1 - v0, v2 - v0)          # capsule axis through the face
        normal = mathutil.normalize(n)
    return best_tp, normal, r - d


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                               c: np.ndarray) -> np.ndarray:
    """Closest point on triangle ``abc`` to point ``p`` (Ericson's Voronoi test)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def _closest_segment_segment(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray,
                             q2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closest pair of points between segments ``p1q1`` and ``p2q2``."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    aa = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if aa <= EPS and e <= EPS:
        return p1, p2
    if aa <= EPS:
        s = 0.0
        t = np.clip(f / e, 0, 1)
    else:
        cc = np.dot(d1, r)
        if e <= EPS:
            t = 0.0
            s = np.clip(-cc / aa, 0, 1)
        else:
            bb = np.dot(d1, d2)
            denom = aa * e - bb * bb
            s = np.clip((bb * f - cc * e) / denom, 0, 1) if denom > EPS else 0.0
            t = (bb * s + f) / e
            if t < 0:
                t = 0.0
                s = np.clip(-cc / aa, 0, 1)
            elif t > 1:
                t = 1.0
                s = np.clip((bb - cc) / aa, 0, 1)
    return p1 + d1 * s, p2 + d2 * t


def _flip(c: Contact) -> Contact:
    """Return the same contact with A and B swapped (normal negated to stay A→B)."""
    return Contact(c.b, c.a, c.point, -c.normal, c.depth)
