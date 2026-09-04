"""Vectorized quaternion and vector helpers for the physics core.

Quaternions are stored in glTF/OMI order ``(x, y, z, w)`` as ``(...,4)`` arrays so
the whole world integrates in one array op.  These are deliberately small and
dependency-free (numpy only) — no GL, no scenegraph.

Every routine broadcasts over leading axes, so a single call handles one vector or
a whole ``(N, ...)`` batch.  Callers pass numpy arrays (or array-likes that
:func:`numpy.asarray` accepts); the return is always a fresh array.
"""
import math
from collections.abc import Sequence

import numpy as np

# A flat vector argument: a sequence of floats or an ndarray.  numpy cannot check
# a fixed length (e.g. exactly 3) statically — its shape type parameter is only
# ``tuple[int, ...]`` — but this excludes the scalars/strings that ``ArrayLike``
# admits, so unpacking and ``tuple(...)`` on such a parameter type-check.
Vec = Sequence[float] | np.ndarray


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


def length(v: np.ndarray) -> float:
    """Length of a single 3-vector.

    ``numpy.linalg.norm`` is general -- an axis, a keepdims, a choice of order,
    a dispatch through ``__array_function__`` -- and on one 3-vector all of that
    costs several times the arithmetic. The hot paths here ask for the length of
    three floats, millions of times.
    """
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def flat_length(v: np.ndarray) -> float:
    """Length of a 3-vector's horizontal part, across x and z.

    Spelled out rather than ``norm(v[[0, 2]])``: the fancy index builds a
    two-element array to throw away, and how far something went along the
    ground is asked several times per movement step.
    """
    return math.sqrt(v[0] * v[0] + v[2] * v[2])


#: The rotation the identity quaternion names. Copied rather than handed out,
#: so a caller that writes into the matrix it was given cannot reach every
#: other caller's.
_IDENTITY = np.eye(3)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Return (...,3,3) rotation matrices for quaternion(s) ``q``."""
    # The identity is most of what this is asked for: a character's capsule
    # proxy is rebuilt several times a frame at a new position and an unchanged
    # orientation, and normalising a quaternion that is already unit and then
    # multiplying out nine entries that are all 0 or 1 is the whole cost of it.
    if q.shape == (4,) and q[3] == 1.0 and not (q[0] or q[1] or q[2]):
        return (_IDENTITY.copy() if q.dtype == _IDENTITY.dtype
                else _IDENTITY.astype(q.dtype))
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

    The axis comes from the quaternion's own vector part and the angle from
    ``arctan2``, rather than either being recovered through ``sqrt(1 - w*w)`` and
    ``arccos(w)``. Both of those lose most of their digits as ``w`` approaches
    one -- the subtraction cancels -- and a body turning slowly, which is a body
    at rest as much as anything else, is exactly where ``w`` is near one.
    """
    q = quat_normalize(q)
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1)
    small = s < 1e-9
    axis = v / np.where(small, 1.0, s)[..., None]
    out = np.empty(q.shape[:-1] + (4,), dtype=q.dtype)
    out[..., :3] = np.where(small[..., None], np.array([0.0, 1.0, 0.0]), axis)
    out[..., 3] = np.where(small, 0.0, 2.0 * np.arctan2(s, q[..., 3]))
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
