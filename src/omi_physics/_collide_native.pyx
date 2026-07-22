# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Native oriented box↔box contact generation (SAT + face clipping).

A faithful port of :func:`omi_physics.collide.box_box` and its clip
helpers: same 15-axis separating-axis test, same reference/incident face choice,
same Sutherland-Hodgman clip and 4-point manifold reduction -- but on C doubles
with fixed-size scratch, so a pair costs native arithmetic instead of dozens of
tiny numpy calls. Boxes are passed as ``center(3)``, world rotation ``R(3,3)``
(columns are the box axes) and half-extents ``half(3)``; no proxy object needed.
"""
from libc.math cimport fabs, sqrt

DEF EPS = 1e-9


cdef inline double _dot3(double* a, double* b) noexcept nogil:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


cdef int _box_box(double* ca, double* Ra, double* ha,
                  double* cb, double* Rb, double* hb,
                  double* out_normal, double* out_pts, double* out_depths) noexcept nogil:
    """Fill ``out_normal`` (3), up to 4 ``out_pts`` (4x3) and ``out_depths`` (4).

    Returns the contact count (0 = separated). ``Ra``/``Rb`` are row-major 3x3
    with the box axes in the columns, matching numpy ``R[row, col]``.
    """
    cdef double axes[15][3]
    cdef int nax = 0
    cdef int i, j, k
    cdef double cx, cy, cz, ln2, inv
    # A's 3 face axes (columns of Ra) then B's 3.
    for i in range(3):
        axes[nax][0] = Ra[0 * 3 + i]; axes[nax][1] = Ra[1 * 3 + i]; axes[nax][2] = Ra[2 * 3 + i]; nax += 1
    for i in range(3):
        axes[nax][0] = Rb[0 * 3 + i]; axes[nax][1] = Rb[1 * 3 + i]; axes[nax][2] = Rb[2 * 3 + i]; nax += 1
    # Edge-edge cross products of A's columns with B's columns.
    for i in range(3):
        for j in range(3):
            cx = Ra[1 * 3 + i] * Rb[2 * 3 + j] - Ra[2 * 3 + i] * Rb[1 * 3 + j]
            cy = Ra[2 * 3 + i] * Rb[0 * 3 + j] - Ra[0 * 3 + i] * Rb[2 * 3 + j]
            cz = Ra[0 * 3 + i] * Rb[1 * 3 + j] - Ra[1 * 3 + i] * Rb[0 * 3 + j]
            ln2 = cx * cx + cy * cy + cz * cz
            if ln2 > EPS * EPS:
                inv = 1.0 / sqrt(ln2)
                axes[nax][0] = cx * inv; axes[nax][1] = cy * inv; axes[nax][2] = cz * inv; nax += 1

    cdef double d[3]
    d[0] = cb[0] - ca[0]; d[1] = cb[1] - ca[1]; d[2] = cb[2] - ca[2]
    cdef double best_depth = 1e300
    cdef double best_axis[3]
    cdef double axl[3], ra, rb, dist, overlap, proj
    best_axis[0] = 0.0; best_axis[1] = 0.0; best_axis[2] = 0.0
    for k in range(nax):
        axl[0] = axes[k][0]; axl[1] = axes[k][1]; axl[2] = axes[k][2]
        # radius of each box along axl = sum_m |axl . R_col_m| * half_m
        ra = (fabs(axl[0] * Ra[0] + axl[1] * Ra[3] + axl[2] * Ra[6]) * ha[0] +
              fabs(axl[0] * Ra[1] + axl[1] * Ra[4] + axl[2] * Ra[7]) * ha[1] +
              fabs(axl[0] * Ra[2] + axl[1] * Ra[5] + axl[2] * Ra[8]) * ha[2])
        rb = (fabs(axl[0] * Rb[0] + axl[1] * Rb[3] + axl[2] * Rb[6]) * hb[0] +
              fabs(axl[0] * Rb[1] + axl[1] * Rb[4] + axl[2] * Rb[7]) * hb[1] +
              fabs(axl[0] * Rb[2] + axl[1] * Rb[5] + axl[2] * Rb[8]) * hb[2])
        proj = axl[0] * d[0] + axl[1] * d[1] + axl[2] * d[2]
        dist = fabs(proj)
        overlap = ra + rb - dist
        if overlap < 0.0:
            return 0
        if overlap < best_depth:
            best_depth = overlap
            if proj >= 0.0:
                best_axis[0] = axl[0]; best_axis[1] = axl[1]; best_axis[2] = axl[2]
            else:
                best_axis[0] = -axl[0]; best_axis[1] = -axl[1]; best_axis[2] = -axl[2]

    out_normal[0] = best_axis[0]; out_normal[1] = best_axis[1]; out_normal[2] = best_axis[2]
    return _clip(ca, Ra, ha, cb, Rb, hb, best_axis, best_depth, out_pts, out_depths)


cdef inline int _face_axis(double* R, double* direction, double* sign) noexcept nogil:
    """Local axis (0..2) whose column is most parallel to ``direction``; sets ``sign``."""
    cdef double best = -1.0, dv
    cdef int axis = 0, m
    for m in range(3):
        dv = R[0 * 3 + m] * direction[0] + R[1 * 3 + m] * direction[1] + R[2 * 3 + m] * direction[2]
        if fabs(dv) > best:
            best = fabs(dv)
            axis = m
            sign[0] = 1.0 if dv >= 0.0 else -1.0
    return axis


cdef void _face_vertices(double* c, double* R, double* half, int axis, double sign,
                         double* verts) noexcept nogil:
    """Four world corners (4x3) of the box face on local ``axis``/``sign`` side."""
    cdef int o0 = 1 if axis == 0 else 0
    cdef int o1 = 2 if axis != 2 else 1
    if axis == 1:
        o0 = 0; o1 = 2
    cdef double fc[3], u[3], v[3]
    cdef int r
    for r in range(3):
        fc[r] = c[r] + R[r * 3 + axis] * sign * half[axis]
        u[r] = R[r * 3 + o0] * half[o0]
        v[r] = R[r * 3 + o1] * half[o1]
    # order: c+u+v, c-u+v, c-u-v, c+u-v
    for r in range(3):
        verts[0 * 3 + r] = fc[r] + u[r] + v[r]
        verts[1 * 3 + r] = fc[r] - u[r] + v[r]
        verts[2 * 3 + r] = fc[r] - u[r] - v[r]
        verts[3 * 3 + r] = fc[r] + u[r] - v[r]


cdef int _clip(double* ca, double* Ra, double* ha,
               double* cb, double* Rb, double* hb,
               double* normal, double depth,
               double* out_pts, double* out_depths) noexcept nogil:
    # Pick reference: the box whose face is most aligned with the normal.
    cdef double a_align = -1.0, b_align = -1.0, dv
    cdef int m
    for m in range(3):
        dv = fabs(Ra[0 * 3 + m] * normal[0] + Ra[1 * 3 + m] * normal[1] + Ra[2 * 3 + m] * normal[2])
        if dv > a_align:
            a_align = dv
        dv = fabs(Rb[0 * 3 + m] * normal[0] + Rb[1 * 3 + m] * normal[1] + Rb[2 * 3 + m] * normal[2])
        if dv > b_align:
            b_align = dv
    cdef double* rc
    cdef double* rR
    cdef double* rh
    cdef double* ic
    cdef double* iR
    cdef double* ih
    cdef double ref_n[3]
    cdef int flip
    if a_align >= b_align:
        rc = ca; rR = Ra; rh = ha; ic = cb; iR = Rb; ih = hb; flip = 0
    else:
        rc = cb; rR = Rb; rh = hb; ic = ca; iR = Ra; ih = ha; flip = 1
    ref_n[0] = -normal[0] if flip else normal[0]
    ref_n[1] = -normal[1] if flip else normal[1]
    ref_n[2] = -normal[2] if flip else normal[2]

    # Incident face: most anti-parallel to ref_n.
    cdef double neg[3]
    neg[0] = -ref_n[0]; neg[1] = -ref_n[1]; neg[2] = -ref_n[2]
    cdef double isign, rsign
    cdef int iaxis = _face_axis(iR, neg, &isign)
    cdef double poly[16][3]
    cdef int npoly = 4
    _face_vertices(ic, iR, ih, iaxis, isign, &poly[0][0])

    # Reference face point and its side clip planes.
    cdef int raxis = _face_axis(rR, ref_n, &rsign)
    cdef double refv[4][3]
    _face_vertices(rc, rR, rh, raxis, rsign, &refv[0][0])
    cdef double face_point[3]
    face_point[0] = refv[0][0]; face_point[1] = refv[0][1]; face_point[2] = refv[0][2]

    # Sutherland-Hodgman clip against the up-to-4 side planes.
    cdef double buf[16][3]
    cdef int other, si, nb, i, r
    cdef double plane_n[3], offset, s, dc, dn, t
    for other in range(3):
        if other == raxis:
            continue
        for si in range(2):
            s = 1.0 if si == 0 else -1.0
            plane_n[0] = rR[0 * 3 + other] * s
            plane_n[1] = rR[1 * 3 + other] * s
            plane_n[2] = rR[2 * 3 + other] * s
            offset = (rc[0] * plane_n[0] + rc[1] * plane_n[1] + rc[2] * plane_n[2]) + rh[other]
            nb = 0
            for i in range(npoly):
                dc = plane_n[0] * poly[i][0] + plane_n[1] * poly[i][1] + plane_n[2] * poly[i][2] - offset
                dn = (plane_n[0] * poly[(i + 1) % npoly][0] + plane_n[1] * poly[(i + 1) % npoly][1]
                      + plane_n[2] * poly[(i + 1) % npoly][2] - offset)
                if dc <= 0.0:
                    buf[nb][0] = poly[i][0]; buf[nb][1] = poly[i][1]; buf[nb][2] = poly[i][2]; nb += 1
                if dc * dn < 0.0:
                    t = dc / (dc - dn)
                    for r in range(3):
                        buf[nb][r] = poly[i][r] + t * (poly[(i + 1) % npoly][r] - poly[i][r])
                    nb += 1
            npoly = nb
            for i in range(npoly):
                poly[i][0] = buf[i][0]; poly[i][1] = buf[i][1]; poly[i][2] = buf[i][2]
            if npoly == 0:
                break
        if npoly == 0:
            break

    # Keep points below the reference face; seat them on it.
    cdef double allpts[16 * 3]
    cdef double alldep[16]
    cdef int nc = 0
    cdef double pen
    for i in range(npoly):
        pen = ((face_point[0] - poly[i][0]) * ref_n[0] + (face_point[1] - poly[i][1]) * ref_n[1]
               + (face_point[2] - poly[i][2]) * ref_n[2])
        if pen >= -1e-6:
            for r in range(3):
                allpts[nc * 3 + r] = poly[i][r] + ref_n[r] * pen
            alldep[nc] = pen if pen > 0.0 else 0.0
            nc += 1
    if nc == 0:
        out_pts[0] = 0.5 * (ca[0] + cb[0]); out_pts[1] = 0.5 * (ca[1] + cb[1]); out_pts[2] = 0.5 * (ca[2] + cb[2])
        out_depths[0] = depth
        return 1
    return _reduce(allpts, alldep, nc, out_pts, out_depths)


cdef int _reduce(double* allpts, double* alldep, int nc,
                 double* out_pts, double* out_depths) noexcept nogil:
    """Keep up to 4 most spread-out points (matches the Python selection order)."""
    cdef int keep[4]
    cdef int i, r, j, nkeep, dup
    cdef double best, dd, d0, dsum
    if nc <= 4:
        for i in range(nc):
            for r in range(3):
                out_pts[i * 3 + r] = allpts[i * 3 + r]
            out_depths[i] = alldep[i]
        return nc
    keep[0] = 0
    best = allpts[0]
    for i in range(1, nc):
        if allpts[i * 3] > best:
            best = allpts[i * 3]; keep[0] = i
    keep[1] = 0; best = -1.0
    for i in range(nc):
        dd = 0.0
        for r in range(3):
            d0 = allpts[i * 3 + r] - allpts[keep[0] * 3 + r]; dd += d0 * d0
        if dd > best:
            best = dd; keep[1] = i
    keep[2] = 0; best = -1.0
    for i in range(nc):
        dsum = 0.0; dd = 0.0
        for r in range(3):
            d0 = allpts[i * 3 + r] - allpts[keep[1] * 3 + r]; dsum += d0 * d0
            d0 = allpts[i * 3 + r] - allpts[keep[0] * 3 + r]; dd += d0 * d0
        if sqrt(dsum) + sqrt(dd) > best:
            best = sqrt(dsum) + sqrt(dd); keep[2] = i
    keep[3] = 0; best = -1.0
    for i in range(nc):
        dd = 0.0
        for r in range(3):
            d0 = allpts[i * 3 + r] - allpts[keep[2] * 3 + r]; dd += d0 * d0
        if sqrt(dd) > best:
            best = sqrt(dd); keep[3] = i
    nkeep = 0
    for i in range(4):
        dup = 0
        for j in range(nkeep):
            if keep[i] == keep[j]:
                dup = 1; break
        if not dup:
            keep[nkeep] = keep[i]; nkeep += 1
    for i in range(nkeep):
        for r in range(3):
            out_pts[i * 3 + r] = allpts[keep[i] * 3 + r]
        out_depths[i] = alldep[keep[i]]
    return nkeep


def box_box(double[::1] ca, double[:, ::1] Ra, double[::1] ha,
            double[::1] cb, double[:, ::1] Rb, double[::1] hb):
    """Return ``(count, normal, points, depths)`` for the oriented box pair.

    ``normal`` is a length-3 tuple, ``points`` a list of ``count`` (x, y, z)
    tuples and ``depths`` a list of ``count`` penetration depths.
    """
    cdef double normal[3]
    cdef double pts[4][3]
    cdef double dep[4]
    normal[0] = 0.0; normal[1] = 0.0; normal[2] = 0.0
    cdef int n = _box_box(&ca[0], &Ra[0, 0], &ha[0], &cb[0], &Rb[0, 0], &hb[0],
                          normal, &pts[0][0], dep)
    cdef int i
    points = [(pts[i][0], pts[i][1], pts[i][2]) for i in range(n)]
    depths = [dep[i] for i in range(n)]
    return n, (normal[0], normal[1], normal[2]), points, depths


def sphere_sphere(double[::1] ca, double ra, double[::1] cb, double rb):
    """Sphere↔sphere: ``(count, normal a→b, [point], [depth])``."""
    cdef double dx = cb[0] - ca[0], dy = cb[1] - ca[1], dz = cb[2] - ca[2]
    cdef double dist = sqrt(dx * dx + dy * dy + dz * dz)
    cdef double r = ra + rb
    if dist >= r:
        return 0, (0.0, 0.0, 0.0), [], []
    cdef double nx, ny, nz, depth, f
    if dist > EPS:
        nx = dx / dist; ny = dy / dist; nz = dz / dist
    else:
        nx = 0.0; ny = 1.0; nz = 0.0
    depth = r - dist
    f = ra - 0.5 * depth
    return 1, (nx, ny, nz), [(ca[0] + nx * f, ca[1] + ny * f, ca[2] + nz * f)], [depth]


def sphere_box(double[::1] cs, double rs, double[::1] cb, double[:, ::1] Rb, double[::1] hb):
    """Sphere↔oriented-box: ``(count, normal sphere→box, [point], [depth])``."""
    cdef double sx = cs[0] - cb[0], sy = cs[1] - cb[1], sz = cs[2] - cb[2]
    # local = Rb.T @ (cs - cb)
    cdef double lx = Rb[0, 0] * sx + Rb[1, 0] * sy + Rb[2, 0] * sz
    cdef double ly = Rb[0, 1] * sx + Rb[1, 1] * sy + Rb[2, 1] * sz
    cdef double lz = Rb[0, 2] * sx + Rb[1, 2] * sy + Rb[2, 2] * sz
    cdef double cx = lx, cy = ly, cz = lz
    if cx > hb[0]: cx = hb[0]
    elif cx < -hb[0]: cx = -hb[0]
    if cy > hb[1]: cy = hb[1]
    elif cy < -hb[1]: cy = -hb[1]
    if cz > hb[2]: cz = hb[2]
    elif cz < -hb[2]: cz = -hb[2]
    cdef double dx = lx - cx, dy = ly - cy, dz = lz - cz
    cdef double dist = sqrt(dx * dx + dy * dy + dz * dz)
    if dist > rs:
        return 0, (0.0, 0.0, 0.0), [], []
    cdef double nlx, nly, nlz, depth, m0, m1, m2, mm
    cdef int axis
    if dist > EPS:
        nlx = dx / dist; nly = dy / dist; nlz = dz / dist
        depth = rs - dist
    else:
        m0 = hb[0] - (lx if lx >= 0 else -lx)
        m1 = hb[1] - (ly if ly >= 0 else -ly)
        m2 = hb[2] - (lz if lz >= 0 else -lz)
        axis = 0; mm = m0
        if m1 < mm: axis = 1; mm = m1
        if m2 < mm: axis = 2; mm = m2
        nlx = 0.0; nly = 0.0; nlz = 0.0
        if axis == 0: nlx = 1.0 if lx >= 0 else -1.0
        elif axis == 1: nly = 1.0 if ly >= 0 else -1.0
        else: nlz = 1.0 if lz >= 0 else -1.0
        depth = rs + mm
    # normal (box→sphere) = Rb @ n_local; a→b (sphere→box) is its negation.
    cdef double nx = Rb[0, 0] * nlx + Rb[0, 1] * nly + Rb[0, 2] * nlz
    cdef double ny = Rb[1, 0] * nlx + Rb[1, 1] * nly + Rb[1, 2] * nlz
    cdef double nz = Rb[2, 0] * nlx + Rb[2, 1] * nly + Rb[2, 2] * nlz
    cdef double wx = cb[0] + Rb[0, 0] * cx + Rb[0, 1] * cy + Rb[0, 2] * cz
    cdef double wy = cb[1] + Rb[1, 0] * cx + Rb[1, 1] * cy + Rb[1, 2] * cz
    cdef double wz = cb[2] + Rb[2, 0] * cx + Rb[2, 1] * cy + Rb[2, 2] * cz
    return 1, (-nx, -ny, -nz), [(wx, wy, wz)], [depth]
