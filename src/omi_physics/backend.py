"""Compute-backend surface for the simulation pipeline.

Each ``world.step`` stage is a pure array-in/array-out kernel so it can later map
onto a GPU compute dispatch without changing simulation logic (see the GPGPU
section of ``docs/PIPELINE.md``).  ``NumpyBackend`` is the v1 CPU path;
a future ``GLComputeBackend`` drops in behind the same surface.

The backend operates on the :class:`~omi_physics.world.PhysicsWorld`'s
structure-of-arrays state; it never touches scenegraph objects.
"""
from typing import Any, TYPE_CHECKING
import numpy as np

from . import model
from . import mathutil

if TYPE_CHECKING:
    from .world import PhysicsWorld


class NumpyBackend:
    """Vectorized CPU kernels over the world's awake dynamic bodies."""

    name = 'numpy'

    def integrate_forces(self, world: "PhysicsWorld", dt: float) -> None:
        """Apply gravity, damping, and quadratic drag to dynamic bodies over ``dt`` seconds."""
        dyn = world.dynamic_mask()
        if not dyn.any():
            return
        g = world.resolve_gravity()
        accel = g * world.gravity_factor[:, None]
        v = world.linear_velocity
        lin_damp = world.linear_damping[dyn, None] + world.default_linear_damping
        ang_damp = world.angular_damping[dyn, None] + world.default_angular_damping
        v[dyn] += accel[dyn] * dt
        v[dyn] *= np.maximum(1.0 - lin_damp * dt, 0.0)
        speed = np.linalg.norm(v[dyn], axis=1, keepdims=True)
        v[dyn] -= world.quadratic_drag[dyn, None] * speed * v[dyn] * dt
        world.angular_velocity[dyn] *= np.maximum(1.0 - ang_damp * dt, 0.0)

    def integrate_positions(self, world: "PhysicsWorld", dt: float) -> None:
        """Advance position and orientation of moving bodies over ``dt`` seconds."""
        moving = world.moving_mask()
        if not moving.any():
            return
        world.position[moving] += world.linear_velocity[moving] * dt
        world.orientation[moving] = mathutil.quat_integrate(
            world.orientation[moving], world.angular_velocity[moving], dt)

    def refit_aabbs(self, world: "PhysicsWorld") -> None:
        """Recompute each body's world-space AABB after integration."""
        world.refit_aabbs()


def __getattr__(name: str) -> Any:
    # GLComputeBackend lives in ``glcompute`` (it imports OpenGL); expose it here
    # lazily so ``import omi_physics`` stays GL-free.
    if name == 'GLComputeBackend':
        from .glcompute import GLComputeBackend
        return GLComputeBackend
    raise AttributeError(name)


def _has_gl_compute() -> bool:
    """True if a current GL context reports version 4.3 or newer (compute shaders)."""
    try:
        from OpenGL.GL import glGetString, GL_VERSION
        version = glGetString(GL_VERSION)
    except Exception:
        return False
    if not version:
        return False
    try:
        text = version.decode() if isinstance(version, bytes) else str(version)
        major, minor = (int(x) for x in text.split()[0].split('.')[:2])
    except Exception:
        return False
    return (major, minor) >= (4, 3)


def select_backend(prefer: str = 'auto') -> Any:
    """Pick a backend, falling back to numpy when GL compute is unavailable.

    Mirrors the instancing capability detection already in the codebase: a
    ``GLComputeBackend`` is used only where GL 4.3 compute exists, else numpy.

    ``prefer`` is ``numpy``/``cpu``, ``glcompute``/``gpu``, or ``auto``.  ``auto``
    uses the GPU compute backend when GL 4.3 compute is present and falls back to
    numpy otherwise — the default is the accelerated path, degrading gracefully
    where the hardware/driver can't run it.  An explicit ``gpu`` request raises if
    compute is genuinely unavailable so the caller learns, rather than silently
    getting the CPU path it asked against.
    """
    if prefer in ('numpy', 'cpu'):
        return NumpyBackend()
    if prefer in ('glcompute', 'gpu'):
        from .glcompute import GLComputeBackend
        return GLComputeBackend()
    # auto: prefer GPU, fall back to numpy when compute is missing or fails.
    if _has_gl_compute():
        try:
            from .glcompute import GLComputeBackend
            return GLComputeBackend()
        except Exception:
            pass
    return NumpyBackend()
