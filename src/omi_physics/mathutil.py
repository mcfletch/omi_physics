"""Vectorized quaternion and vector helpers for the physics core.

Quaternions are stored in glTF/OMI order ``(x, y, z, w)`` as ``(...,4)`` arrays so
the whole world integrates in one array op.  These are deliberately small and
dependency-free (numpy only) — no GL, no scenegraph.

Every routine broadcasts over leading axes, so a single call handles one vector or
a whole ``(N, ...)`` batch.  Callers pass numpy arrays (or array-likes that
:func:`numpy.asarray` accepts); the return is always a fresh array.
"""
from typing import Sequence, Union
import numpy as np

# A flat vector argument: a sequence of floats or an ndarray.  numpy cannot check
# a fixed length (e.g. exactly 3) statically — its shape type parameter is only
# ``tuple[int, ...]`` — but this excludes the scalars/strings that ``ArrayLike``
# admits, so unpacking and ``tuple(...)`` on such a parameter type-check.
Vec = Union[Sequence[float], np.ndarray]


def normalize(v: Vec, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Unit-length ``v`` along ``axis``; a shorter-than-``eps`` vector is left near zero."""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def quat_normalize(q: Vec) -> np.ndarray:
    """Return ``q`` scaled to unit length (a valid rotation quaternion)."""
    return normalize(q, axis=-1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` — the rotation ``a`` after ``b`` (both ``(...,4)``)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by quaternion(s) ``q`` (both broadcastable)."""
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return the inverse rotation of unit quaternion(s) ``q`` (negated vector part)."""
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Return (...,3,3) rotation matrices for quaternion(s) ``q``."""
    q = quat_normalize(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    m[..., 0, 0] = 1 - 2 * (yy + zz)
    m[..., 0, 1] = 2 * (xy - wz)
    m[..., 0, 2] = 2 * (xz + wy)
    m[..., 1, 0] = 2 * (xy + wz)
    m[..., 1, 1] = 1 - 2 * (xx + zz)
    m[..., 1, 2] = 2 * (yz - wx)
    m[..., 2, 0] = 2 * (xz - wy)
    m[..., 2, 1] = 2 * (yz + wx)
    m[..., 2, 2] = 1 - 2 * (xx + yy)
    return m


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two single 3-vectors, ~12x faster than :func:`numpy.cross`.

    numpy.cross pays for axis-moving and broadcasting; on the fixed 3-vectors of
    the collision/solver hot paths that overhead dwarfs the six multiplies.
    """
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Quaternion(s) ``(...,4)`` xyzw → VRML axis-angle ``(...,4)`` ``(x, y, z, angle)``.

    Batched form of :func:`physicsbody.quat_to_vrml_rotation`; a near-identity
    rotation (sin(angle/2) ~ 0) yields the canonical ``(0, 1, 0, 0)``.
    """
    q = quat_normalize(q)
    w = np.clip(q[..., 3], -1.0, 1.0)
    s = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    small = s < 1e-9
    s_safe = np.where(small, 1.0, s)
    axis = q[..., :3] / s_safe[..., None]
    out = np.empty(q.shape[:-1] + (4,), dtype=q.dtype)
    out[..., :3] = np.where(small[..., None], np.array([0.0, 1.0, 0.0]), axis)
    out[..., 3] = np.where(small, 0.0, 2.0 * np.arccos(w))
    return out


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Advance orientation ``q`` by body angular velocity ``omega`` over ``dt``."""
    w = np.zeros(q.shape, dtype=q.dtype)
    w[..., :3] = omega
    dq = 0.5 * quat_mul(w, q) * dt
    return quat_normalize(q + dq)


def quat_from_axis_angle(axis: Vec, angle: float) -> np.ndarray:
    """Unit quaternion for a rotation of ``angle`` radians about ``axis``."""
    axis = normalize(np.asarray(axis, dtype='d'))
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)])
