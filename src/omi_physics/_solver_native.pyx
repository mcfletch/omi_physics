# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Native inner loops for the sequential-impulse contact solver.

These reproduce the pure-Python solver's algorithm exactly -- the same
Gauss-Seidel per-contact update order, the same normal-then-friction math -- but
run the velocity and position iterations in C over pre-assembled SoA arrays, so
the solver's dominant per-contact cost is no longer Python. The loops hold no
Python objects and release the GIL, so the physics step can run off the render
thread.
"""
import numpy as np
from libc.math cimport sqrt


cdef inline void _apply_w(double[:, ::1] lv, double[:, ::1] av,
                          Py_ssize_t a, Py_ssize_t b, double ima, double imb,
                          double[:, :, ::1] invIw, double* rAk, double* rBk,
                          double Px, double Py, double Pz) noexcept nogil:
    """Impulse apply using the per-body world inverse-inertia tensor ``invIw``."""
    cdef double crx, cry, crz
    lv[a, 0] -= ima * Px; lv[a, 1] -= ima * Py; lv[a, 2] -= ima * Pz
    lv[b, 0] += imb * Px; lv[b, 1] += imb * Py; lv[b, 2] += imb * Pz
    crx = rAk[1] * Pz - rAk[2] * Py
    cry = rAk[2] * Px - rAk[0] * Pz
    crz = rAk[0] * Py - rAk[1] * Px
    av[a, 0] -= invIw[a, 0, 0] * crx + invIw[a, 0, 1] * cry + invIw[a, 0, 2] * crz
    av[a, 1] -= invIw[a, 1, 0] * crx + invIw[a, 1, 1] * cry + invIw[a, 1, 2] * crz
    av[a, 2] -= invIw[a, 2, 0] * crx + invIw[a, 2, 1] * cry + invIw[a, 2, 2] * crz
    crx = rBk[1] * Pz - rBk[2] * Py
    cry = rBk[2] * Px - rBk[0] * Pz
    crz = rBk[0] * Py - rBk[1] * Px
    av[b, 0] += invIw[b, 0, 0] * crx + invIw[b, 0, 1] * cry + invIw[b, 0, 2] * crz
    av[b, 1] += invIw[b, 1, 0] * crx + invIw[b, 1, 1] * cry + invIw[b, 1, 2] * crz
    av[b, 2] += invIw[b, 2, 0] * crx + invIw[b, 2, 1] * cry + invIw[b, 2, 2] * crz


cdef inline double _effmass(double* r_a, double* r_b, double dx, double dy, double dz,
                            double[:, :, ::1] invIw, Py_ssize_t a, Py_ssize_t b,
                            double ima, double imb, double softness) noexcept nogil:
    """Reciprocal of the pair's inverse mass along (dx,dy,dz), with ``softness`` added."""
    cdef double ax, ay, az, bx, by, bz, wx, wy, wz, kk
    ax = r_a[1] * dz - r_a[2] * dy
    ay = r_a[2] * dx - r_a[0] * dz
    az = r_a[0] * dy - r_a[1] * dx
    bx = r_b[1] * dz - r_b[2] * dy
    by = r_b[2] * dx - r_b[0] * dz
    bz = r_b[0] * dy - r_b[1] * dx
    kk = ima + imb
    wx = invIw[a, 0, 0] * ax + invIw[a, 0, 1] * ay + invIw[a, 0, 2] * az
    wy = invIw[a, 1, 0] * ax + invIw[a, 1, 1] * ay + invIw[a, 1, 2] * az
    wz = invIw[a, 2, 0] * ax + invIw[a, 2, 1] * ay + invIw[a, 2, 2] * az
    kk += ax * wx + ay * wy + az * wz
    wx = invIw[b, 0, 0] * bx + invIw[b, 0, 1] * by + invIw[b, 0, 2] * bz
    wy = invIw[b, 1, 0] * bx + invIw[b, 1, 1] * by + invIw[b, 1, 2] * bz
    wz = invIw[b, 2, 0] * bx + invIw[b, 2, 1] * by + invIw[b, 2, 2] * bz
    kk += bx * wx + by * wy + bz * wz
    kk += softness
    if kk > 1e-12:
        return 1.0 / kk
    return 0.0


def prepare_and_solve(Py_ssize_t[::1] a_idx, Py_ssize_t[::1] b_idx,
                      double[:, ::1] point, double[:, ::1] normal, double[::1] depth,
                      double[:, ::1] pos, double[:, ::1] lv, double[:, ::1] av,
                      double[::1] inv_mass, double[:, :, ::1] invIw,
                      double[::1] restitution, double[::1] threshold,
                      double[:, ::1] friction, double[:, ::1] direction,
                      double[:, ::1] surface, double[:, ::1] softness,
                      double[::1] correction,
                      double[::1] nImp, double[:, ::1] tImp, double[:, ::1] t1,
                      int iters, bint warm_start, double slop, double max_correction):
    """Build constraints from contacts, then solve velocity + position, natively.

    Mirrors the pure-Python solver's ``_prepare`` (arms, tangent basis, effective
    masses, restitution bias) followed by warm start, ``iters`` velocity
    iterations and one split-impulse position pass. ``lv``/``av``/``pos`` are
    mutated in place; ``nImp``/``tImp`` carry impulses in (warm start) and out,
    and ``t1`` receives each contact's first friction direction.

    Per contact: ``threshold`` is the closing speed below which nothing
    bounces; ``friction`` the two coefficients; ``direction`` the first
    friction direction, or zero; ``surface`` the target sliding speed along
    each friction direction; ``softness`` the softness of the normal row and
    of each friction row, already divided by the step; ``correction`` the
    fraction of an overlap the position pass takes out.
    """
    cdef Py_ssize_t K = a_idx.shape[0]
    cdef double[:, ::1] rA = np.empty((K, 3))
    cdef double[:, ::1] rB = np.empty((K, 3))
    cdef double[:, ::1] t2 = np.empty((K, 3))
    cdef double[::1] nMass = np.empty(K)
    cdef double[::1] t1Mass = np.empty(K)
    cdef double[::1] t2Mass = np.empty(K)
    cdef double[::1] vBias = np.empty(K)
    cdef double[::1] ima = np.empty(K)
    cdef double[::1] imb = np.empty(K)
    # Positions as they stand before the position pass, so each contact there
    # can be corrected against the movement the earlier ones already applied.
    # The velocity iterations in between move nothing, so this is still what
    # the pass starts from when it reaches them.
    cdef double[:, ::1] startpos = np.array(pos, dtype=np.float64, copy=True)
    cdef Py_ssize_t k, it, a, b
    cdef double nx, ny, nz, axx, ayy, azz, ln, inv, dvx, dvy, dvz, vn0, along
    cdef double t1x, t1y, t1z, imak, imbk, maxf, vn, new, dL, vt, newt
    with nogil:
        for k in range(K):
            a = a_idx[k]; b = b_idx[k]
            imak = inv_mass[a]; imbk = inv_mass[b]
            ima[k] = imak; imb[k] = imbk
            rA[k, 0] = point[k, 0] - pos[a, 0]
            rA[k, 1] = point[k, 1] - pos[a, 1]
            rA[k, 2] = point[k, 2] - pos[a, 2]
            rB[k, 0] = point[k, 0] - pos[b, 0]
            rB[k, 1] = point[k, 1] - pos[b, 1]
            rB[k, 2] = point[k, 2] - pos[b, 2]
            nx = normal[k, 0]; ny = normal[k, 1]; nz = normal[k, 2]
            ln = 0.0
            if direction[k, 0] != 0.0 or direction[k, 1] != 0.0 or direction[k, 2] != 0.0:
                # the given direction, taken across the normal
                along = direction[k, 0] * nx + direction[k, 1] * ny + direction[k, 2] * nz
                t1x = direction[k, 0] - along * nx
                t1y = direction[k, 1] - along * ny
                t1z = direction[k, 2] - along * nz
                ln = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
            if ln > 1e-9:
                t1x = t1x / ln; t1y = t1y / ln; t1z = t1z / ln
            else:
                # tangent basis: a = x-axis unless n is near-parallel to it, else y-axis
                if (nx if nx >= 0 else -nx) < 0.9:
                    axx = 1.0; ayy = 0.0; azz = 0.0
                else:
                    axx = 0.0; ayy = 1.0; azz = 0.0
                # t1 = normalize(a x n)
                t1x = ayy * nz - azz * ny
                t1y = azz * nx - axx * nz
                t1z = axx * ny - ayy * nx
                ln = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
                inv = 1.0 / ln if ln > 1e-12 else 0.0
                t1x *= inv; t1y *= inv; t1z *= inv
            t1[k, 0] = t1x; t1[k, 1] = t1y; t1[k, 2] = t1z
            # t2 = n x t1
            t2[k, 0] = ny * t1z - nz * t1y
            t2[k, 1] = nz * t1x - nx * t1z
            t2[k, 2] = nx * t1y - ny * t1x
            nMass[k] = _effmass(&rA[k, 0], &rB[k, 0], nx, ny, nz, invIw, a, b, imak, imbk,
                                softness[k, 0])
            t1Mass[k] = _effmass(&rA[k, 0], &rB[k, 0], t1x, t1y, t1z, invIw, a, b, imak, imbk,
                                 softness[k, 1])
            t2Mass[k] = _effmass(&rA[k, 0], &rB[k, 0], t2[k, 0], t2[k, 1], t2[k, 2], invIw,
                                 a, b, imak, imbk, softness[k, 2])
            # restitution bias from the approach velocity
            dvx = (lv[b, 0] + av[b, 1] * rB[k, 2] - av[b, 2] * rB[k, 1]) - (lv[a, 0] + av[a, 1] * rA[k, 2] - av[a, 2] * rA[k, 1])
            dvy = (lv[b, 1] + av[b, 2] * rB[k, 0] - av[b, 0] * rB[k, 2]) - (lv[a, 1] + av[a, 2] * rA[k, 0] - av[a, 0] * rA[k, 2])
            dvz = (lv[b, 2] + av[b, 0] * rB[k, 1] - av[b, 1] * rB[k, 0]) - (lv[a, 2] + av[a, 0] * rA[k, 1] - av[a, 1] * rA[k, 0])
            vn0 = dvx * nx + dvy * ny + dvz * nz
            if vn0 < -threshold[k]:
                vBias[k] = -restitution[k] * vn0
            else:
                vBias[k] = 0.0

        if warm_start:
            for k in range(K):
                a = a_idx[k]; b = b_idx[k]
                _apply_w(lv, av, a, b, ima[k], imb[k], invIw, &rA[k, 0], &rB[k, 0],
                         nImp[k] * normal[k, 0] + tImp[k, 0] * t1[k, 0] + tImp[k, 1] * t2[k, 0],
                         nImp[k] * normal[k, 1] + tImp[k, 0] * t1[k, 1] + tImp[k, 1] * t2[k, 1],
                         nImp[k] * normal[k, 2] + tImp[k, 0] * t1[k, 2] + tImp[k, 1] * t2[k, 2])

        for it in range(iters):
            for k in range(K):
                a = a_idx[k]; b = b_idx[k]
                nx = normal[k, 0]; ny = normal[k, 1]; nz = normal[k, 2]
                dvx = (lv[b, 0] + av[b, 1] * rB[k, 2] - av[b, 2] * rB[k, 1]) - (lv[a, 0] + av[a, 1] * rA[k, 2] - av[a, 2] * rA[k, 1])
                dvy = (lv[b, 1] + av[b, 2] * rB[k, 0] - av[b, 0] * rB[k, 2]) - (lv[a, 1] + av[a, 2] * rA[k, 0] - av[a, 0] * rA[k, 2])
                dvz = (lv[b, 2] + av[b, 0] * rB[k, 1] - av[b, 1] * rB[k, 0]) - (lv[a, 2] + av[a, 0] * rA[k, 1] - av[a, 1] * rA[k, 0])
                vn = dvx * nx + dvy * ny + dvz * nz
                new = nImp[k] - nMass[k] * (vn - vBias[k] + softness[k, 0] * nImp[k])
                if new < 0.0:
                    new = 0.0
                dL = new - nImp[k]; nImp[k] = new
                _apply_w(lv, av, a, b, ima[k], imb[k], invIw, &rA[k, 0], &rB[k, 0],
                         dL * nx, dL * ny, dL * nz)

                dvx = (lv[b, 0] + av[b, 1] * rB[k, 2] - av[b, 2] * rB[k, 1]) - (lv[a, 0] + av[a, 1] * rA[k, 2] - av[a, 2] * rA[k, 1])
                dvy = (lv[b, 1] + av[b, 2] * rB[k, 0] - av[b, 0] * rB[k, 2]) - (lv[a, 1] + av[a, 2] * rA[k, 0] - av[a, 0] * rA[k, 2])
                dvz = (lv[b, 2] + av[b, 0] * rB[k, 1] - av[b, 1] * rB[k, 0]) - (lv[a, 2] + av[a, 0] * rA[k, 1] - av[a, 1] * rA[k, 0])
                maxf = friction[k, 0] * nImp[k]
                vt = dvx * t1[k, 0] + dvy * t1[k, 1] + dvz * t1[k, 2]
                newt = tImp[k, 0] - t1Mass[k] * (vt - surface[k, 0] + softness[k, 1] * tImp[k, 0])
                if newt > maxf:
                    newt = maxf
                elif newt < -maxf:
                    newt = -maxf
                dL = newt - tImp[k, 0]; tImp[k, 0] = newt
                _apply_w(lv, av, a, b, ima[k], imb[k], invIw, &rA[k, 0], &rB[k, 0],
                         dL * t1[k, 0], dL * t1[k, 1], dL * t1[k, 2])

                maxf = friction[k, 1] * nImp[k]
                vt = dvx * t2[k, 0] + dvy * t2[k, 1] + dvz * t2[k, 2]
                newt = tImp[k, 1] - t2Mass[k] * (vt - surface[k, 1] + softness[k, 2] * tImp[k, 1])
                if newt > maxf:
                    newt = maxf
                elif newt < -maxf:
                    newt = -maxf
                dL = newt - tImp[k, 1]; tImp[k, 1] = newt
                _apply_w(lv, av, a, b, ima[k], imb[k], invIw, &rA[k, 0], &rB[k, 0],
                         dL * t2[k, 0], dL * t2[k, 1], dL * t2[k, 2])

        # Split-impulse position pass, each contact corrected against what the
        # pass has already done: four points of one manifold acting on the same
        # narrow-phase depth would push the pair four times as far as it is in.
        for k in range(K):
            a = a_idx[k]; b = b_idx[k]
            vn = (((pos[b, 0] - startpos[b, 0]) - (pos[a, 0] - startpos[a, 0])) * normal[k, 0]
                  + ((pos[b, 1] - startpos[b, 1]) - (pos[a, 1] - startpos[a, 1])) * normal[k, 1]
                  + ((pos[b, 2] - startpos[b, 2]) - (pos[a, 2] - startpos[a, 2])) * normal[k, 2])
            vn = depth[k] - vn - slop
            if vn <= 0.0:
                continue
            imak = ima[k]; imbk = imb[k]
            if imak + imbk <= 1e-12:
                continue
            vn = correction[k] * vn
            if vn > max_correction:
                vn = max_correction
            dL = vn / (imak + imbk)
            pos[a, 0] -= imak * dL * normal[k, 0]
            pos[a, 1] -= imak * dL * normal[k, 1]
            pos[a, 2] -= imak * dL * normal[k, 2]
            pos[b, 0] += imbk * dL * normal[k, 0]
            pos[b, 1] += imbk * dL * normal[k, 1]
            pos[b, 2] += imbk * dL * normal[k, 2]


cdef inline void _apply_row(Py_ssize_t a, Py_ssize_t b, double amount, double* lin,
                            double* turn_a, double* turn_b, double[:, ::1] lv,
                            double[:, ::1] av, double[::1] inv_mass) noexcept nogil:
    """Apply ``amount`` of a row's impulse: ``b`` along it, ``a`` against it."""
    cdef int i
    cdef double scale
    if amount == 0.0:
        return
    if a >= 0:
        scale = inv_mass[a] * amount
        for i in range(3):
            lv[a, i] = lv[a, i] - scale * lin[i]
            av[a, i] = av[a, i] - turn_a[i] * amount
    if b >= 0:
        scale = inv_mass[b] * amount
        for i in range(3):
            lv[b, i] = lv[b, i] + scale * lin[i]
            av[b, i] = av[b, i] + turn_b[i] * amount


def solve_rows(Py_ssize_t[::1] a_idx, Py_ssize_t[::1] b_idx,
               double[:, ::1] lin, double[:, ::1] ang_a, double[:, ::1] ang_b,
               double[::1] bias, double[::1] low, double[::1] high, double[::1] softness,
               double[::1] impulse, double[:, ::1] lv, double[:, ::1] av,
               double[::1] inv_mass, double[:, :, ::1] invIw, int iters):
    """Solve joint rows by Gauss-Seidel, warm started from ``impulse``.

    The twin of ``jointrows._solve_rows``, operation for operation; see
    :mod:`omi_physics.jointrows` for what a row is. A body index below zero
    is the world. ``impulse`` carries each row's accumulated impulse in and out.
    """
    cdef Py_ssize_t K = a_idx.shape[0]
    cdef double[::1] mass = np.zeros(K)
    cdef double[:, ::1] turn_a = np.zeros((K, 3))
    cdef double[:, ::1] turn_b = np.zeros((K, 3))
    cdef Py_ssize_t k, a, b, it
    cdef int i
    cdef double effective, lx, ly, lz, jv, old, new
    with nogil:
        for k in range(K):
            a = a_idx[k]; b = b_idx[k]
            lx = lin[k, 0]; ly = lin[k, 1]; lz = lin[k, 2]
            effective = 0.0
            if a >= 0:
                effective += inv_mass[a] * (lx * lx + ly * ly + lz * lz)
                for i in range(3):
                    turn_a[k, i] = (invIw[a, i, 0] * ang_a[k, 0]
                                    + invIw[a, i, 1] * ang_a[k, 1]
                                    + invIw[a, i, 2] * ang_a[k, 2])
                effective += (ang_a[k, 0] * turn_a[k, 0] + ang_a[k, 1] * turn_a[k, 1]
                              + ang_a[k, 2] * turn_a[k, 2])
            if b >= 0:
                effective += inv_mass[b] * (lx * lx + ly * ly + lz * lz)
                for i in range(3):
                    turn_b[k, i] = (invIw[b, i, 0] * ang_b[k, 0]
                                    + invIw[b, i, 1] * ang_b[k, 1]
                                    + invIw[b, i, 2] * ang_b[k, 2])
                effective += (ang_b[k, 0] * turn_b[k, 0] + ang_b[k, 1] * turn_b[k, 1]
                              + ang_b[k, 2] * turn_b[k, 2])
            effective += softness[k]
            if effective > 1e-12:
                mass[k] = 1.0 / effective
            else:
                mass[k] = 0.0
            _apply_row(a, b, impulse[k], &lin[k, 0], &turn_a[k, 0], &turn_b[k, 0],
                       lv, av, inv_mass)
        for it in range(iters):
            for k in range(K):
                a = a_idx[k]; b = b_idx[k]
                jv = 0.0
                if b >= 0:
                    jv += (lin[k, 0] * lv[b, 0] + lin[k, 1] * lv[b, 1]
                           + lin[k, 2] * lv[b, 2])
                    jv += (ang_b[k, 0] * av[b, 0] + ang_b[k, 1] * av[b, 1]
                           + ang_b[k, 2] * av[b, 2])
                if a >= 0:
                    jv -= (lin[k, 0] * lv[a, 0] + lin[k, 1] * lv[a, 1]
                           + lin[k, 2] * lv[a, 2])
                    jv -= (ang_a[k, 0] * av[a, 0] + ang_a[k, 1] * av[a, 1]
                           + ang_a[k, 2] * av[a, 2])
                old = impulse[k]
                new = old + mass[k] * (bias[k] - jv - softness[k] * old)
                if new < low[k]:
                    new = low[k]
                elif new > high[k]:
                    new = high[k]
                impulse[k] = new
                _apply_row(a, b, new - old, &lin[k, 0], &turn_a[k, 0], &turn_b[k, 0],
                           lv, av, inv_mass)
