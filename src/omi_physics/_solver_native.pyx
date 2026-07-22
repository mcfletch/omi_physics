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


cdef inline void _apply(double[:, ::1] lv, double[:, ::1] av,
                        Py_ssize_t a, Py_ssize_t b, double ima, double imb,
                        double[:, :, ::1] invIa, double[:, :, ::1] invIb,
                        double[:, ::1] rA, double[:, ::1] rB, Py_ssize_t k,
                        double Px, double Py, double Pz) noexcept nogil:
    """Apply impulse ``P`` to body ``b`` and ``-P`` to ``a`` (linear + angular)."""
    cdef double crx, cry, crz
    lv[a, 0] -= ima * Px; lv[a, 1] -= ima * Py; lv[a, 2] -= ima * Pz
    lv[b, 0] += imb * Px; lv[b, 1] += imb * Py; lv[b, 2] += imb * Pz
    crx = rA[k, 1] * Pz - rA[k, 2] * Py
    cry = rA[k, 2] * Px - rA[k, 0] * Pz
    crz = rA[k, 0] * Py - rA[k, 1] * Px
    av[a, 0] -= invIa[k, 0, 0] * crx + invIa[k, 0, 1] * cry + invIa[k, 0, 2] * crz
    av[a, 1] -= invIa[k, 1, 0] * crx + invIa[k, 1, 1] * cry + invIa[k, 1, 2] * crz
    av[a, 2] -= invIa[k, 2, 0] * crx + invIa[k, 2, 1] * cry + invIa[k, 2, 2] * crz
    crx = rB[k, 1] * Pz - rB[k, 2] * Py
    cry = rB[k, 2] * Px - rB[k, 0] * Pz
    crz = rB[k, 0] * Py - rB[k, 1] * Px
    av[b, 0] += invIb[k, 0, 0] * crx + invIb[k, 0, 1] * cry + invIb[k, 0, 2] * crz
    av[b, 1] += invIb[k, 1, 0] * crx + invIb[k, 1, 1] * cry + invIb[k, 1, 2] * crz
    av[b, 2] += invIb[k, 2, 0] * crx + invIb[k, 2, 1] * cry + invIb[k, 2, 2] * crz


def solve_velocity(Py_ssize_t[::1] a_idx, Py_ssize_t[::1] b_idx,
                   double[:, ::1] rA, double[:, ::1] rB,
                   double[:, ::1] n, double[:, ::1] t1, double[:, ::1] t2,
                   double[::1] nMass, double[::1] t1Mass, double[::1] t2Mass,
                   double[::1] friction, double[::1] vBias,
                   double[:, :, ::1] invIa, double[:, :, ::1] invIb,
                   double[::1] ima, double[::1] imb,
                   double[::1] nImp, double[:, ::1] tImp,
                   double[:, ::1] lv, double[:, ::1] av,
                   int iters, bint warm_start):
    """Warm-start then run ``iters`` velocity iterations, mutating ``lv``/``av``.

    ``nImp``/``tImp`` carry the accumulated normal/tangent impulses in and out
    (for warm starting the next step).
    """
    cdef Py_ssize_t K = a_idx.shape[0]
    cdef Py_ssize_t it, k, a, b
    cdef double dvx, dvy, dvz, vn, new, dL, maxf, vt, newt, nx, ny, nz
    with nogil:
        if warm_start:
            for k in range(K):
                a = a_idx[k]; b = b_idx[k]
                _apply(lv, av, a, b, ima[k], imb[k], invIa, invIb, rA, rB, k,
                       nImp[k] * n[k, 0] + tImp[k, 0] * t1[k, 0] + tImp[k, 1] * t2[k, 0],
                       nImp[k] * n[k, 1] + tImp[k, 0] * t1[k, 1] + tImp[k, 1] * t2[k, 1],
                       nImp[k] * n[k, 2] + tImp[k, 0] * t1[k, 2] + tImp[k, 1] * t2[k, 2])
        for it in range(iters):
            for k in range(K):
                a = a_idx[k]; b = b_idx[k]
                nx = n[k, 0]; ny = n[k, 1]; nz = n[k, 2]
                dvx = (lv[b, 0] + av[b, 1] * rB[k, 2] - av[b, 2] * rB[k, 1]) - (lv[a, 0] + av[a, 1] * rA[k, 2] - av[a, 2] * rA[k, 1])
                dvy = (lv[b, 1] + av[b, 2] * rB[k, 0] - av[b, 0] * rB[k, 2]) - (lv[a, 1] + av[a, 2] * rA[k, 0] - av[a, 0] * rA[k, 2])
                dvz = (lv[b, 2] + av[b, 0] * rB[k, 1] - av[b, 1] * rB[k, 0]) - (lv[a, 2] + av[a, 0] * rA[k, 1] - av[a, 1] * rA[k, 0])
                vn = dvx * nx + dvy * ny + dvz * nz
                new = nImp[k] - nMass[k] * (vn - vBias[k])
                if new < 0.0:
                    new = 0.0
                dL = new - nImp[k]; nImp[k] = new
                _apply(lv, av, a, b, ima[k], imb[k], invIa, invIb, rA, rB, k,
                       dL * nx, dL * ny, dL * nz)

                maxf = friction[k] * nImp[k]
                # Relative velocity is recomputed once; t1 and t2 share it, matching
                # the reference solver's friction pass.
                dvx = (lv[b, 0] + av[b, 1] * rB[k, 2] - av[b, 2] * rB[k, 1]) - (lv[a, 0] + av[a, 1] * rA[k, 2] - av[a, 2] * rA[k, 1])
                dvy = (lv[b, 1] + av[b, 2] * rB[k, 0] - av[b, 0] * rB[k, 2]) - (lv[a, 1] + av[a, 2] * rA[k, 0] - av[a, 0] * rA[k, 2])
                dvz = (lv[b, 2] + av[b, 0] * rB[k, 1] - av[b, 1] * rB[k, 0]) - (lv[a, 2] + av[a, 0] * rA[k, 1] - av[a, 1] * rA[k, 0])
                vt = dvx * t1[k, 0] + dvy * t1[k, 1] + dvz * t1[k, 2]
                newt = tImp[k, 0] - t1Mass[k] * vt
                if newt > maxf:
                    newt = maxf
                elif newt < -maxf:
                    newt = -maxf
                dL = newt - tImp[k, 0]; tImp[k, 0] = newt
                _apply(lv, av, a, b, ima[k], imb[k], invIa, invIb, rA, rB, k,
                       dL * t1[k, 0], dL * t1[k, 1], dL * t1[k, 2])

                vt = dvx * t2[k, 0] + dvy * t2[k, 1] + dvz * t2[k, 2]
                newt = tImp[k, 1] - t2Mass[k] * vt
                if newt > maxf:
                    newt = maxf
                elif newt < -maxf:
                    newt = -maxf
                dL = newt - tImp[k, 1]; tImp[k, 1] = newt
                _apply(lv, av, a, b, ima[k], imb[k], invIa, invIb, rA, rB, k,
                       dL * t2[k, 0], dL * t2[k, 1], dL * t2[k, 2])


def solve_positions(Py_ssize_t[::1] a_idx, Py_ssize_t[::1] b_idx,
                    double[:, ::1] n, double[::1] depth,
                    double[::1] ima, double[::1] imb,
                    double[:, ::1] position, double correction, double slop):
    """One split-impulse position pass: push overlapping bodies apart along ``n``."""
    cdef Py_ssize_t K = a_idx.shape[0]
    cdef Py_ssize_t k, a, b
    cdef double corr, inv_sum, mv
    with nogil:
        for k in range(K):
            corr = depth[k] - slop
            if corr <= 0.0:
                continue
            inv_sum = ima[k] + imb[k]
            if inv_sum <= 1e-12:
                continue
            mv = correction * corr / inv_sum
            a = a_idx[k]; b = b_idx[k]
            position[a, 0] -= ima[k] * mv * n[k, 0]
            position[a, 1] -= ima[k] * mv * n[k, 1]
            position[a, 2] -= ima[k] * mv * n[k, 2]
            position[b, 0] += imb[k] * mv * n[k, 0]
            position[b, 1] += imb[k] * mv * n[k, 1]
            position[b, 2] += imb[k] * mv * n[k, 2]


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
                            double ima, double imb) noexcept nogil:
    """Reciprocal inverse mass of the pair along direction (dx,dy,dz)."""
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
    if kk > 1e-12:
        return 1.0 / kk
    return 0.0


def prepare_and_solve(Py_ssize_t[::1] a_idx, Py_ssize_t[::1] b_idx,
                      double[:, ::1] point, double[:, ::1] normal, double[::1] depth,
                      double[:, ::1] pos, double[:, ::1] lv, double[:, ::1] av,
                      double[::1] inv_mass, double[:, :, ::1] invIw,
                      double[::1] restitution, double[::1] friction,
                      double[::1] nImp, double[:, ::1] tImp,
                      double restitution_threshold, int iters, bint warm_start,
                      double correction, double slop):
    """Build constraints from contacts, then solve velocity + position, natively.

    Mirrors the pure-Python solver's ``_prepare`` (arms, tangent basis, effective
    masses, restitution bias) followed by warm start, ``iters`` velocity
    iterations and one split-impulse position pass. ``lv``/``av``/``pos`` are
    mutated in place; ``nImp``/``tImp`` carry impulses in (warm start) and out.
    """
    cdef Py_ssize_t K = a_idx.shape[0]
    cdef double[:, ::1] rA = np.empty((K, 3))
    cdef double[:, ::1] rB = np.empty((K, 3))
    cdef double[:, ::1] t1 = np.empty((K, 3))
    cdef double[:, ::1] t2 = np.empty((K, 3))
    cdef double[::1] nMass = np.empty(K)
    cdef double[::1] t1Mass = np.empty(K)
    cdef double[::1] t2Mass = np.empty(K)
    cdef double[::1] vBias = np.empty(K)
    cdef double[::1] ima = np.empty(K)
    cdef double[::1] imb = np.empty(K)
    cdef Py_ssize_t k, it, a, b
    cdef double nx, ny, nz, axx, ayy, azz, ln, inv, dvx, dvy, dvz, vn0
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
            nMass[k] = _effmass(&rA[k, 0], &rB[k, 0], nx, ny, nz, invIw, a, b, imak, imbk)
            t1Mass[k] = _effmass(&rA[k, 0], &rB[k, 0], t1x, t1y, t1z, invIw, a, b, imak, imbk)
            t2Mass[k] = _effmass(&rA[k, 0], &rB[k, 0], t2[k, 0], t2[k, 1], t2[k, 2], invIw, a, b, imak, imbk)
            # restitution bias from the approach velocity
            dvx = (lv[b, 0] + av[b, 1] * rB[k, 2] - av[b, 2] * rB[k, 1]) - (lv[a, 0] + av[a, 1] * rA[k, 2] - av[a, 2] * rA[k, 1])
            dvy = (lv[b, 1] + av[b, 2] * rB[k, 0] - av[b, 0] * rB[k, 2]) - (lv[a, 1] + av[a, 2] * rA[k, 0] - av[a, 0] * rA[k, 2])
            dvz = (lv[b, 2] + av[b, 0] * rB[k, 1] - av[b, 1] * rB[k, 0]) - (lv[a, 2] + av[a, 0] * rA[k, 1] - av[a, 1] * rA[k, 0])
            vn0 = dvx * nx + dvy * ny + dvz * nz
            if vn0 < -restitution_threshold:
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
                new = nImp[k] - nMass[k] * (vn - vBias[k])
                if new < 0.0:
                    new = 0.0
                dL = new - nImp[k]; nImp[k] = new
                _apply_w(lv, av, a, b, ima[k], imb[k], invIw, &rA[k, 0], &rB[k, 0],
                         dL * nx, dL * ny, dL * nz)

                maxf = friction[k] * nImp[k]
                dvx = (lv[b, 0] + av[b, 1] * rB[k, 2] - av[b, 2] * rB[k, 1]) - (lv[a, 0] + av[a, 1] * rA[k, 2] - av[a, 2] * rA[k, 1])
                dvy = (lv[b, 1] + av[b, 2] * rB[k, 0] - av[b, 0] * rB[k, 2]) - (lv[a, 1] + av[a, 2] * rA[k, 0] - av[a, 0] * rA[k, 2])
                dvz = (lv[b, 2] + av[b, 0] * rB[k, 1] - av[b, 1] * rB[k, 0]) - (lv[a, 2] + av[a, 0] * rA[k, 1] - av[a, 1] * rA[k, 0])
                vt = dvx * t1[k, 0] + dvy * t1[k, 1] + dvz * t1[k, 2]
                newt = tImp[k, 0] - t1Mass[k] * vt
                if newt > maxf:
                    newt = maxf
                elif newt < -maxf:
                    newt = -maxf
                dL = newt - tImp[k, 0]; tImp[k, 0] = newt
                _apply_w(lv, av, a, b, ima[k], imb[k], invIw, &rA[k, 0], &rB[k, 0],
                         dL * t1[k, 0], dL * t1[k, 1], dL * t1[k, 2])

                vt = dvx * t2[k, 0] + dvy * t2[k, 1] + dvz * t2[k, 2]
                newt = tImp[k, 1] - t2Mass[k] * vt
                if newt > maxf:
                    newt = maxf
                elif newt < -maxf:
                    newt = -maxf
                dL = newt - tImp[k, 1]; tImp[k, 1] = newt
                _apply_w(lv, av, a, b, ima[k], imb[k], invIw, &rA[k, 0], &rB[k, 0],
                         dL * t2[k, 0], dL * t2[k, 1], dL * t2[k, 2])

        # split-impulse position pass
        for k in range(K):
            vn = depth[k] - slop
            if vn <= 0.0:
                continue
            imak = ima[k]; imbk = imb[k]
            if imak + imbk <= 1e-12:
                continue
            dL = correction * vn / (imak + imbk)
            a = a_idx[k]; b = b_idx[k]
            pos[a, 0] -= imak * dL * normal[k, 0]
            pos[a, 1] -= imak * dL * normal[k, 1]
            pos[a, 2] -= imak * dL * normal[k, 2]
            pos[b, 0] += imbk * dL * normal[k, 0]
            pos[b, 1] += imbk * dL * normal[k, 1]
            pos[b, 2] += imbk * dL * normal[k, 2]
