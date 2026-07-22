"""Drive a kinematic body along an authored pose over time.

A kinematic body ignores gravity and contact response; the integrator advances it
purely from its ``linear_velocity`` / ``angular_velocity`` (see
:meth:`PhysicsWorld.moving_mask` and ``backend.integrate_positions``).  To make
such a body *follow* a path — an elevator's vertical travel, a rotating arm's
sweep — something must set those velocities each frame.

:class:`KinematicAnimator` does exactly that from a pose function ``pose(t) ->
(position, orientation)``.  Each :meth:`update` it looks one frame ahead, then
sets the velocity that carries the body from where it *is* to where the pose says
it should be next.  Two consequences matter:

* Because the velocity is derived from the body's *current* position toward the
  next target (not open-loop), any small integration drift self-corrects.
* Because it is a real velocity, the contact solver **carries riders**: a marble
  resting on a rising platform is pushed up with it, exactly as a player expects.
"""
from typing import Any, Callable, Tuple, TYPE_CHECKING
import numpy as np

from . import mathutil
from .mathutil import Vec

if TYPE_CHECKING:
    from .world import PhysicsWorld


class KinematicAnimator:
    """Steer one kinematic body so it tracks ``pose_fn(t)``.

    ``pose_fn`` receives the animator's accumulated time and returns either a
    3-vector position (orientation held identity) or a ``(position, quaternion)``
    pair.  Call :meth:`update` once per frame *before* stepping the world.
    """

    def __init__(self, world: "PhysicsWorld", index: int,
                 pose_fn: Callable[[float], Any], time: float = 0.0):
        self.world = world
        self.index = index
        self.pose_fn = pose_fn
        self.time = time

    @staticmethod
    def _split_pose(pose: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize a pose return value into ``(position(3,), quaternion(4,))``."""
        if len(pose) == 2 and np.ndim(pose[0]) == 1:
            position, orientation = pose
        else:
            position, orientation = pose, (0.0, 0.0, 0.0, 1.0)
        return (np.asarray(position, dtype='d'),
                np.asarray(orientation, dtype='d'))

    def update(self, dt: float) -> None:
        """Set the body's velocity so it reaches ``pose_fn(time + dt)`` this frame."""
        if dt <= 0:
            return
        target_pos, target_quat = self._split_pose(self.pose_fn(self.time + dt))
        world, i = self.world, self.index

        world.linear_velocity[i] = (target_pos - world.position[i]) / dt
        world.angular_velocity[i] = self._angular_velocity(
            world.orientation[i], target_quat, dt)
        world.wake(i)
        self.time += dt

    @staticmethod
    def _angular_velocity(current_quat: Vec, target_quat: Vec, dt: float) -> np.ndarray:
        """World-frame angular velocity rotating ``current`` onto ``target`` in ``dt``."""
        q0 = mathutil.quat_normalize(np.asarray(current_quat, dtype='d'))
        q1 = mathutil.quat_normalize(target_quat)
        delta = mathutil.quat_mul(q1, mathutil.quat_conjugate(q0))
        if delta[3] < 0:                    # shortest arc
            delta = -delta
        axis = delta[:3]
        sin_half = np.linalg.norm(axis)
        if sin_half < 1e-9:
            return np.zeros(3)
        angle = 2.0 * np.arctan2(sin_half, delta[3])
        return (axis / sin_half) * (angle / dt)
