"""Drive a kinematic body along an authored pose over time, or to a place.

:class:`KinematicMover` is the simpler of the two: it sends a body to a target
position at a set speed and stops it there, which is a door, a lift or a
returning platform. The rest of this describes :class:`KinematicAnimator`.

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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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
    def _split_pose(pose: Any) -> tuple[np.ndarray, np.ndarray]:
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


class KinematicMover:
    """Move one kinematic body to :attr:`target` at :attr:`speed`, and stop on it.

    What a door, a lift or a returning platform does: the game says where the
    body should be and how fast it goes there, in metres per second, and
    calls :meth:`update` once a frame before advancing the world. The body
    travels at that speed as a real velocity, so the solver carries what
    rides on it. On the frame it would reach the target it is placed there
    exactly and its velocity is set to zero.

    ``target`` starts where the body is, so a new mover leaves it standing.
    """

    def __init__(self, world: "PhysicsWorld", index: int, speed: float,
                 target: Vec | None = None) -> None:
        if not speed > 0.0:
            raise ValueError('a mover needs a speed above nought, not %r' % (speed,))
        self.world = world
        self.index = index
        self.speed = float(speed)
        self._target = np.array(world.position[index] if target is None else target,
                                dtype='d')

    @property
    def target(self) -> np.ndarray:
        """Where the body is going, in world coordinates."""
        return self._target

    @target.setter
    def target(self, position: Vec) -> None:
        self._target = np.array(position, dtype='d')

    @property
    def arrived(self) -> bool:
        """Whether the body is standing on :attr:`target`."""
        return bool(np.array_equal(self.world.position[self.index], self.target)
                    and not self.world.linear_velocity[self.index].any())

    def update(self, dt: float) -> None:
        """Set this frame's velocity toward :attr:`target`, or place the body on it."""
        world, i = self.world, self.index
        target = self._target
        gap = target - world.position[i]
        distance = float(np.linalg.norm(gap))
        if distance <= self.speed * max(dt, 0.0):
            if distance > 0.0:
                world.place_body(i, position=target)
            world.linear_velocity[i] = 0.0
            return
        world.linear_velocity[i] = gap * (self.speed / distance)
        world.wake(i)
