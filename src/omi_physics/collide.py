"""Narrow-phase collision math — pure CPU numpy, no GL.

Every routine returns a list of :class:`Contact` with the normal pointing from
body A toward body B and ``depth > 0`` for penetration.  Analytic tests cover the
primitive pairs; oriented box↔box uses SAT plus face clipping to build a stable
multi-point manifold (Ericson, *Real-Time Collision Detection*).  Convex↔convex
via GJK/EPA lands in Phase 3 (:mod:`omi_physics.gjk`).

**Depth is how far one shape reaches past another, not how near it is to it.**
The distinction only shows up once something is deeply inside: a test that
answers with the distance to the nearest feature starts pointing the wrong way
as soon as the far side is nearer than the near one, and resolving along it then
drives the pair further together instead of apart.  :func:`capsule_triangle` is
where that matters most, because a character landing hard is the common way to
get deeply inside anything.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, cast
import numpy as np

from . import mathutil
from ._accel import accelerators_disabled
from .body import (SphereProxy, BoxProxy, CapsuleProxy, ConvexProxy, TriangleProxy,
                   TriangleMeshProxy, Proxy, SupportProxy)

if accelerators_disabled():
    _native = None
else:
    try:
        from . import _collide_native as _native   # type: ignore[no-redef,attr-defined]
    except ImportError:                # pragma: no cover - pure-Python fallback
        _native = None                 # type: ignore[assignment]

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
    if other.kind == 'capsule':
        # Every candidate in one call. A character controller reaches this nine
        # to twelve times a frame -- three depenetration iterations inside each
        # of four call sites -- so a Python round trip per triangle is the
        # frame at a handful of characters and hopeless at a hundred.
        verts = mesh.candidate_vertices(lo, hi)
        hit, points, normals, depths = capsule_triangle_batch(
            cast(CapsuleProxy, other), verts)
        # A is the mesh: the normal already points A→B. B is the mesh: flip it.
        # Negated once for the batch rather than per contact, which is an array
        # allocated per contact in a list that is usually a dozen long.
        if not mesh_is_a:
            normals = -normals
        return [Contact(a, b, points[i], normals[i], float(depths[i]))
                for i in np.flatnonzero(hit)]
    contacts = []
    for tri in mesh.triangles_overlapping(lo, hi):
        if mesh_is_a:
            contacts.extend(collide_convex(a, b, tri, other))
        else:
            contacts.extend(collide_convex(a, b, other, tri))
    return contacts


def capsule_triangle(cap: CapsuleProxy,
                     tri: TriangleProxy) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Analytic capsule↔triangle. Returns ``(point, normal, depth)`` with the
    normal pointing from the triangle toward the capsule, or ``None`` if apart.

    **Against the face, the capsule is pushed back to the side it is on** --
    the side its axis centre lies on -- along the face's own perpendicular.

    Resolving toward the nearest point on the capsule *axis* instead looks
    right while the capsule is barely touching and is exactly wrong once it is
    not.  A character landing hard puts its lower cap below the floor; that cap
    is then the nearest end, so "push toward the nearest end" drives the
    character **down through the floor it just hit**, and reports no ground
    while doing it.  Sinking into the ground on a fast landing is that, and the
    faster the arrival the deeper it goes.

    The side is taken from the capsule rather than from the winding, because a
    triangle soup does not promise one: the same floor is wound either way in
    different scenes, and a rule that trusted it would eject half of them
    downward.  Taking it from the capsule is also what makes a ceiling work --
    a character jumping into one is below it and is pushed back down.

    Depth is how far the capsule reaches **past** the plane, unbounded, so one
    that has been placed or driven deep inside comes back out instead of being
    pushed further in.  Contacts against an **edge** or a **vertex** keep the
    nearest-point direction: there is no face to be in front of at a rim, and
    that is where a neighbouring triangle has the say.
    """
    v0, v1, v2 = tri.verts
    face = np.cross(v1 - v0, v2 - v0)
    face_len = float(np.linalg.norm(face))
    r = cap.radius

    if face_len > EPS:
        face_n = face / face_len
        centre = (cap.p0 + cap.p1) * 0.5
        if float(np.dot(centre - v0, face_n)) < 0:
            face_n = -face_n                    # the side the capsule is on
        deepest: Optional[Tuple[float, np.ndarray]] = None
        for sp in (cap.p0, cap.p1):
            # Over the face itself rather than off one of its rims: the foot of
            # the perpendicular is then the closest point on the triangle.
            height = float(np.dot(sp - v0, face_n))
            foot = sp - face_n * height
            tp = _closest_point_on_triangle(sp, v0, v1, v2)
            if float(np.dot(foot - tp, foot - tp)) > EPS * EPS:
                continue
            reach = r - height
            if reach > 0 and (deepest is None or reach > deepest[0]):
                deepest = (reach, tp)
        if deepest is not None:
            reach, tp = deepest
            return tp, face_n, reach

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
    if best_d2 >= r * r or best_sp is None or best_tp is None:
        return None
    d = np.sqrt(best_d2)
    if d > EPS:
        normal = (best_sp - best_tp) / d        # triangle -> capsule
    else:
        normal = mathutil.normalize(face)       # capsule axis through the face
    return best_tp, normal, r - d


def capsule_triangles(cap: CapsuleProxy, v0: np.ndarray, v1: np.ndarray,
                      v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """:func:`capsule_triangle` for N triangles at once.

    The compiled accelerator when it is present, the numpy fallback otherwise;
    the two are asserted to agree triangle for triangle. See
    :func:`_capsule_triangles_numpy` for why a batch exists at all.
    """
    if _native is not None:
        v0 = np.ascontiguousarray(v0, dtype='d').reshape(-1, 3)
        v1 = np.ascontiguousarray(v1, dtype='d').reshape(-1, 3)
        v2 = np.ascontiguousarray(v2, dtype='d').reshape(-1, 3)
        verts = np.empty((len(v0), 3, 3), dtype='d')
        verts[:, 0], verts[:, 1], verts[:, 2] = v0, v1, v2
        return capsule_triangle_batch(cap, verts)
    return _capsule_triangles_numpy(cap, v0, v1, v2)


def capsule_mesh_pushes(cap: CapsuleProxy, mesh: TriangleMeshProxy,
                        verts: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Depenetration for a capsule against a mesh, as ``(pushes, depths)``.

    ``pushes`` is ``(N,3)`` unit normals pointing **from the world into the
    capsule** -- the direction the capsule must move -- and ``depths`` is
    ``(N,)``, ordered deepest first, which is the order a sequential-projection
    resolve wants.

    The same contacts :func:`collide` answers with, without building a
    :class:`Contact` for each. A character controller reads only these two
    numbers per contact and asks nine to twelve times a frame; at a hundred
    characters that is twenty thousand objects a frame constructed to have two
    fields read and be dropped.

    ``verts`` lets a caller supply candidates it has already gathered -- a
    controller resolving one position holds the capsule still across several
    calls, and the broad phase does not need asking again for each. A superset
    is safe: the exact test rejects whatever is not really near.
    """
    if verts is None:
        verts = mesh.candidate_vertices(*cap.aabb())
    hit, _points, normals, depths = capsule_triangle_batch(cap, verts)
    where = np.flatnonzero(hit)
    if not len(where):
        return np.zeros((0, 3), dtype='d'), np.zeros(0, dtype='d')
    # The mesh is the world, so the push into the capsule is the normal as the
    # triangle test gives it: from the triangle toward the capsule.
    pushes, found = normals[where], depths[where]
    order = np.argsort(-found, kind='stable')
    return pushes[order], found[order]


def capsule_triangle_batch(cap: CapsuleProxy, verts: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray]:
    """:func:`capsule_triangles` taking the triangles already as ``(N, 3, 3)``.

    The layout :meth:`TriangleMeshProxy.candidate_vertices` gathers into, so the
    mesh path hands its array straight over instead of splitting it into three
    and having it rebuilt.
    """
    verts = np.ascontiguousarray(verts, dtype='d').reshape(-1, 3, 3)
    count = len(verts)
    hit = np.zeros(count, dtype=np.uint8)
    points = np.zeros((count, 3), dtype='d')
    normals = np.zeros((count, 3), dtype='d')
    depths = np.zeros(count, dtype='d')
    if not count:
        return hit.astype(bool), points, normals, depths
    if _native is None:
        return _capsule_triangles_numpy(cap, verts[:, 0], verts[:, 1],
                                        verts[:, 2])
    _native.capsule_triangles(
        np.ascontiguousarray(cap.p0, dtype='d'),
        np.ascontiguousarray(cap.p1, dtype='d'),
        float(cap.radius), verts, hit, points, normals, depths)
    return hit.astype(bool), points, normals, depths


def _capsule_triangles_numpy(cap: CapsuleProxy, v0: np.ndarray, v1: np.ndarray,
                             v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    """:func:`capsule_triangle` for N triangles at once, in numpy.

    Answers ``(hit, points, normals, depths)`` -- an ``(N,)`` bool and three
    arrays of shape ``(N,3)``, ``(N,3)``, ``(N,)`` -- lined up with the input,
    so a caller keeps its own triangle indices.  Entries where ``hit`` is false
    are unspecified rather than meaningful.

    **Identical in answer to the scalar routine, and that is a requirement, not
    a hope**: ``tests/test_collide_batch.py`` asserts it triangle for triangle
    over randomised input as well as against geometry worked out by hand.
    Every rule the scalar version's docstring explains -- the side taken from
    the capsule and not the winding, the unbounded depth, the rim contacts
    keeping the nearest-point direction -- holds here and is not restated.

    This exists because of *dispatch*, not arithmetic.  The scalar routine
    spends about seventy microseconds a triangle, almost all of it numpy
    entering and leaving over three-element vectors; a character controller
    runs three depenetration iterations inside each of four calls a frame, so
    that overhead **is** the frame at a handful of characters and hopeless at a
    hundred.  Done as arrays the same work is one pass of the same operations,
    and the cost stops being per triangle.
    """
    v0 = np.ascontiguousarray(v0, dtype='d').reshape(-1, 3)
    v1 = np.ascontiguousarray(v1, dtype='d').reshape(-1, 3)
    v2 = np.ascontiguousarray(v2, dtype='d').reshape(-1, 3)
    count = len(v0)
    points = np.zeros((count, 3), dtype='d')
    normals = np.zeros((count, 3), dtype='d')
    depths = np.zeros(count, dtype='d')
    if not count:
        return np.zeros(0, dtype=bool), points, normals, depths

    r = float(cap.radius)
    caps = (np.asarray(cap.p0, dtype='d'), np.asarray(cap.p1, dtype='d'))
    centre = (caps[0] + caps[1]) * 0.5

    face = np.cross(v1 - v0, v2 - v0)
    face_len = np.linalg.norm(face, axis=1)
    has_face = face_len > EPS
    # Divide only where there is a face; a degenerate triangle keeps a zero
    # normal and is answered by the edge path below.
    safe_len = np.where(has_face, face_len, 1.0)
    face_n = face / safe_len[:, None]
    # The side the capsule is on, from its axis centre.
    flip = np.einsum('ij,ij->i', centre - v0, face_n) < 0
    face_n = np.where(flip[:, None], -face_n, face_n)

    # -- over the face --------------------------------------------------
    best_reach = np.full(count, -np.inf)
    face_point = np.zeros((count, 3), dtype='d')
    for sp in caps:
        height = np.einsum('ij,ij->i', sp - v0, face_n)
        foot = sp - face_n * height[:, None]
        near = _closest_points_on_triangles(sp, v0, v1, v2)
        gap = foot - near
        # Over the face itself rather than off one of its rims.
        over = np.einsum('ij,ij->i', gap, gap) <= EPS * EPS
        reach = r - height
        take = has_face & over & (reach > 0) & (reach > best_reach)
        best_reach = np.where(take, reach, best_reach)
        face_point = np.where(take[:, None], near, face_point)
    face_hit = best_reach > -np.inf

    # -- off the rim ----------------------------------------------------
    best_d2 = np.full(count, np.inf)
    best_sp = np.zeros((count, 3), dtype='d')
    best_tp = np.zeros((count, 3), dtype='d')
    for sp in caps:
        near = _closest_points_on_triangles(sp, v0, v1, v2)
        delta = sp - near
        d2 = np.einsum('ij,ij->i', delta, delta)
        take = d2 < best_d2
        best_d2 = np.where(take, d2, best_d2)
        best_sp = np.where(take[:, None], np.broadcast_to(sp, (count, 3)), best_sp)
        best_tp = np.where(take[:, None], near, best_tp)
    for e0, e1 in ((v0, v1), (v1, v2), (v2, v0)):
        on_axis, on_edge = _closest_segments_to_segment(caps[0], caps[1], e0, e1)
        delta = on_axis - on_edge
        d2 = np.einsum('ij,ij->i', delta, delta)
        take = d2 < best_d2
        best_d2 = np.where(take, d2, best_d2)
        best_sp = np.where(take[:, None], on_axis, best_sp)
        best_tp = np.where(take[:, None], on_edge, best_tp)

    distance = np.sqrt(best_d2)
    apart = distance > EPS
    # A capsule axis passing exactly through the face has no direction to be
    # pushed along but the face's own.
    rim_n = np.where(apart[:, None],
                     (best_sp - best_tp) / np.where(apart, distance, 1.0)[:, None],
                     _normalize_rows(face))
    rim_hit = best_d2 < r * r

    hit = face_hit | rim_hit
    use_face = face_hit
    points = np.where(use_face[:, None], face_point, best_tp)
    normals = np.where(use_face[:, None], face_n, rim_n)
    depths = np.where(use_face, best_reach, r - distance)
    return hit, points, normals, depths


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Each row scaled to unit length; a zero row stays zero."""
    length = np.linalg.norm(vectors, axis=1)
    return vectors / np.where(length > EPS, length, 1.0)[:, None]


def _closest_points_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                                 c: np.ndarray) -> np.ndarray:
    """:func:`_closest_point_on_triangle` for one point against N triangles.

    Ericson's Voronoi-region test with every branch evaluated and selected by
    mask instead of taken: the regions are mutually exclusive, so computing all
    seven and choosing costs a constant few array passes where branching per
    triangle costs a Python round trip per triangle.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    bp = p - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    cp = p - c
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    # The regions, tested in the scalar routine's order so the same triangle
    # falls in the same one; `np.where` is applied in reverse so the earliest
    # region wins where two masks would both be true on a degenerate triangle.
    with np.errstate(divide='ignore', invalid='ignore'):
        on_ab = a + (d1 / (d1 - d3))[:, None] * ab
        on_ac = a + (d2 / (d2 - d6))[:, None] * ac
        on_bc = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None] * (c - b)
        denom = 1.0 / (va + vb + vc)
        inside = a + ab * (vb * denom)[:, None] + ac * (vc * denom)[:, None]

    found = np.where(np.isfinite(inside), inside, a)
    found = np.where((((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))
                      & np.all(np.isfinite(on_bc), axis=1))[:, None],
                     on_bc, found)
    found = np.where((((vb <= 0) & (d2 >= 0) & (d6 <= 0))
                      & np.all(np.isfinite(on_ac), axis=1))[:, None],
                     on_ac, found)
    found = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, found)
    found = np.where((((vc <= 0) & (d1 >= 0) & (d3 <= 0))
                      & np.all(np.isfinite(on_ab), axis=1))[:, None],
                     on_ab, found)
    found = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, found)
    found = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, found)
    return found


def _closest_segments_to_segment(p1: np.ndarray, q1: np.ndarray,
                                 p2: np.ndarray, q2: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`_closest_segment_segment` for one segment against N segments.

    Answers ``(on_the_single, on_each)``, both ``(N,3)``.  The scalar routine's
    nested branches become masks over the same expressions; the clamped
    re-solve for ``t`` outside ``[0,1]`` is applied last so it overrides the
    unclamped answer exactly as the early return does.
    """
    count = len(p2)
    d1 = np.broadcast_to(q1 - p1, (count, 3))
    d2 = q2 - p2
    r = p1 - p2
    aa = np.einsum('ij,ij->i', d1, d1)
    e = np.einsum('ij,ij->i', d2, d2)
    f = np.einsum('ij,ij->i', d2, r)
    cc = np.einsum('ij,ij->i', d1, r)
    bb = np.einsum('ij,ij->i', d1, d2)

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = aa * e - bb * bb
        s = np.where(denom > EPS,
                     np.clip((bb * f - cc * e) / np.where(denom > EPS, denom, 1.0),
                             0.0, 1.0),
                     0.0)
        t = (bb * s + f) / np.where(e > EPS, e, 1.0)
        # t clamped to its interval, with s re-solved for the clamped t.
        low = t < 0.0
        high = t > 1.0
        s = np.where(low, np.clip(-cc / np.where(aa > EPS, aa, 1.0), 0.0, 1.0), s)
        s = np.where(high, np.clip((bb - cc) / np.where(aa > EPS, aa, 1.0),
                                   0.0, 1.0), s)
        t = np.clip(t, 0.0, 1.0)
        # The degenerate cases the scalar routine returns early for.
        both_points = (aa <= EPS) & (e <= EPS)
        axis_point = (aa <= EPS) & (e > EPS)
        edge_point = (aa > EPS) & (e <= EPS)
        s = np.where(both_points, 0.0, s)
        t = np.where(both_points, 0.0, t)
        s = np.where(axis_point, 0.0, s)
        t = np.where(axis_point,
                     np.clip(f / np.where(e > EPS, e, 1.0), 0.0, 1.0), t)
        t = np.where(edge_point, 0.0, t)
        s = np.where(edge_point,
                     np.clip(-cc / np.where(aa > EPS, aa, 1.0), 0.0, 1.0), s)
    return p1 + d1 * s[:, None], p2 + d2 * t[:, None]


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
