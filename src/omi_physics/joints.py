"""Joint constraints (``OMI_physics_joint``) solved as velocity constraints.

Implemented as constraint rows solved in the same Gauss-Seidel style as contacts
(Catto): a point (ball) constraint locks an anchor, a distance constraint holds a
length, and a motor drive reaches a target velocity within a force budget.  A
body index of ``-1`` means the world (an infinite-mass anchor).  These map onto
the OMI limits/drives vocabulary (locked linear axes → point; a linear limit with
``min==max`` → distance; a drive → motor).

The articulated joints -- :class:`BallJoint`, :class:`HingeJoint`,
:class:`SliderJoint`, :class:`UniversalJoint`, :class:`TwoAxisHingeJoint` and
:class:`MotorJoint` -- are built from rows (:mod:`omi_physics.jointrows`) and
solved together. Each is given its anchor and axes in world coordinates as
the bodies stand when it is added to the world, which fixes them in each
body's frame; each answers its angles, rates and anchors as the bodies move.
Angles are in radians, within (-π, π], measured from where the joint was
added; stops and motors are optional on every axis that turns or slides.
"""
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from . import mathutil
from .jointrows import RowJoint, Rows, spring
from .mathutil import Vec, cross

if TYPE_CHECKING:
    from .world import PhysicsWorld


def _skew(r: np.ndarray) -> np.ndarray:
    """The skew-symmetric matrix so that ``_skew(r) @ v == cross(r, v)``."""
    x, y, z = r
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype='d')


class _JointBase:
    """Shared per-body inertia/velocity helpers for the constraint solvers.

    Body index ``-1`` denotes the world: infinite mass and zero inverse inertia,
    so applying an impulse to it does nothing.
    """
    baumgarte = 0.2
    _invI: dict[int, np.ndarray]

    def _inv_inertia_world(self, world: "PhysicsWorld", i: int) -> np.ndarray:
        """World-space inverse inertia tensor of body ``i`` (zero for the world)."""
        if i < 0:
            return np.zeros((3, 3))
        R = mathutil.quat_to_matrix(world.orientation[i])
        return (R * world.inv_inertia[i]) @ R.T

    def _inv_mass(self, world: "PhysicsWorld", i: int) -> float:
        """Inverse mass of body ``i`` (0 for the world / infinite mass)."""
        return 0.0 if i < 0 else world.inv_mass[i]

    def _point_velocity(self, world: "PhysicsWorld", i: int, r: np.ndarray) -> np.ndarray:
        """Velocity of the point at body-relative offset ``r`` on body ``i``."""
        if i < 0:
            return np.zeros(3)
        return world.linear_velocity[i] + cross(world.angular_velocity[i], r)

    def _apply(self, world: "PhysicsWorld", i: int, r: np.ndarray, P: np.ndarray) -> None:
        """Apply impulse ``P`` at offset ``r`` to body ``i`` (no-op for the world)."""
        if i < 0:
            return
        world.linear_velocity[i] += world.inv_mass[i] * P
        world.angular_velocity[i] += self._invI[i] @ cross(r, P)


class PointConstraint(_JointBase):
    """Ball / point-to-point: hold two body anchors coincident (pendulum, hinge).

    ``a`` or ``b`` may be ``-1`` to pin against the world at ``anchor``.
    """

    def __init__(self, a: int, b: int, anchor: Vec,
                 local_a: np.ndarray | None = None,
                 local_b: np.ndarray | None = None):
        self.a, self.b = a, b
        self.anchor = np.asarray(anchor, dtype='d')
        self.local_a = local_a
        self.local_b = local_b
        self._invI = {}

    def _world_anchor(self, world: "PhysicsWorld", i: int,
                      local: np.ndarray | None) -> np.ndarray:
        """World position of body ``i``'s anchor (``anchor`` itself for the world)."""
        if i < 0:
            return self.anchor
        assert local is not None
        R = mathutil.quat_to_matrix(world.orientation[i])
        return world.position[i] + R @ local

    def prepare(self, world: "PhysicsWorld") -> None:
        """Cache inertia, anchor offsets, the effective-mass matrix, and the bias."""
        for i in (self.a, self.b):
            self._invI[i] = self._inv_inertia_world(world, i)
        if self.local_a is None:
            self.local_a = self._to_local(world, self.a, self.anchor)
        if self.local_b is None:
            self.local_b = self._to_local(world, self.b, self.anchor)
        self.rA = self._world_anchor(world, self.a, self.local_a) - self._pos(world, self.a)
        self.rB = self._world_anchor(world, self.b, self.local_b) - self._pos(world, self.b)
        k = (self._inv_mass(world, self.a) + self._inv_mass(world, self.b)) * np.eye(3)
        k -= _skew(self.rA) @ self._invI[self.a] @ _skew(self.rA)
        k -= _skew(self.rB) @ self._invI[self.b] @ _skew(self.rB)
        self.K = k
        pa = self._pos(world, self.a) + self.rA
        pb = self._pos(world, self.b) + self.rB
        self.bias = self.baumgarte * (pb - pa)

    def _pos(self, world: "PhysicsWorld", i: int) -> np.ndarray:
        """Centre-of-mass position of body ``i`` (``anchor`` for the world)."""
        return self.anchor if i < 0 else world.position[i]

    def _to_local(self, world: "PhysicsWorld", i: int,
                  world_pt: np.ndarray) -> np.ndarray | None:
        """World point expressed in body ``i``'s local frame (``None`` for the world)."""
        if i < 0:
            return None
        R = mathutil.quat_to_matrix(world.orientation[i])
        return R.T @ (world_pt - world.position[i])

    def solve(self, world: "PhysicsWorld", dt: float) -> None:
        """Apply one impulse that drives the anchor separation velocity to zero."""
        cdot = (self._point_velocity(world, self.b, self.rB)
                - self._point_velocity(world, self.a, self.rA))
        rhs = -(cdot + self.bias / dt)
        P = np.linalg.solve(self.K, rhs)
        self._apply(world, self.a, self.rA, -P)
        self._apply(world, self.b, self.rB, P)


class DistanceConstraint(_JointBase):
    """Hold ``|anchorB − anchorA| == length`` (a rigid rod / rope at full extent).

    ``length`` defaults to the anchors' separation at :meth:`prepare`.  ``a`` or
    ``b`` may be ``-1`` to anchor against the world.
    """

    def __init__(self, a: int, b: int, anchor_a: Vec, anchor_b: Vec,
                 length: float | None = None):
        self.a, self.b = a, b
        self.anchor_a = np.asarray(anchor_a, dtype='d')
        self.anchor_b = np.asarray(anchor_b, dtype='d')
        self.length = length
        self._invI = {}

    def prepare(self, world: "PhysicsWorld") -> None:
        """Cache inertia, the constraint axis, effective mass, and the bias."""
        for i in (self.a, self.b):
            self._invI[i] = self._inv_inertia_world(world, i)
        self.pa = self._anchor(world, self.a, self.anchor_a)
        self.pb = self._anchor(world, self.b, self.anchor_b)
        d = self.pb - self.pa
        dist = np.linalg.norm(d)
        self.n = d / dist if dist > 1e-9 else np.array([0.0, 1.0, 0.0])
        if self.length is None:
            self.length = float(dist)
        self.rA = self.pa - self._pos(world, self.a)
        self.rB = self.pb - self._pos(world, self.b)
        raxn = cross(self.rA, self.n)
        rbxn = cross(self.rB, self.n)
        k = self._inv_mass(world, self.a) + self._inv_mass(world, self.b)
        k += raxn @ self._invI[self.a] @ raxn
        k += rbxn @ self._invI[self.b] @ rbxn
        self.mass = 1.0 / k if k > 1e-12 else 0.0
        self.bias = self.baumgarte * (dist - self.length)

    def _anchor(self, world: "PhysicsWorld", i: int, local: np.ndarray) -> np.ndarray:
        """World position of body ``i``'s local anchor (``local`` itself for the world)."""
        if i < 0:
            return local
        R = mathutil.quat_to_matrix(world.orientation[i])
        return world.position[i] + R @ local

    def _pos(self, world: "PhysicsWorld", i: int) -> np.ndarray:
        """Centre-of-mass position of body ``i`` (``anchor_a`` for the world)."""
        return self.anchor_a if i < 0 else world.position[i]

    def solve(self, world: "PhysicsWorld", dt: float) -> None:
        """Apply one impulse along the axis that holds the pair at ``length``."""
        cdot = np.dot(self._point_velocity(world, self.b, self.rB)
                      - self._point_velocity(world, self.a, self.rA), self.n)
        impulse = -self.mass * (cdot + self.bias / dt)
        P = impulse * self.n
        self._apply(world, self.a, self.rA, -P)
        self._apply(world, self.b, self.rB, P)


class AngularMotor(_JointBase):
    """A drive: reach ``target`` angular velocity about ``axis`` within ``max_force``.

    ``max_force`` bounds the accumulated impulse per step (``max_force * dt``);
    the default is unlimited.
    """

    def __init__(self, body: int, axis: Vec, target: float,
                 max_force: float = np.inf):
        self.body = body
        self.axis = mathutil.normalize(np.asarray(axis, dtype='d'))
        self.target = float(target)
        self.max_force = max_force
        self.accum = 0.0

    def prepare(self, world: "PhysicsWorld") -> None:
        """Cache the body's inverse inertia about ``axis`` and reset the accumulator."""
        self._invI = {self.body: self._inv_inertia_world(world, self.body)}
        eff = self.axis @ self._invI[self.body] @ self.axis
        self.mass = 1.0 / eff if eff > 1e-12 else 0.0
        self.accum = 0.0

    def solve(self, world: "PhysicsWorld", dt: float) -> None:
        """Apply one clamped impulse toward the target spin about ``axis``."""
        w = np.dot(world.angular_velocity[self.body], self.axis)
        impulse = self.mass * (self.target - w)
        limit = self.max_force * dt
        new = np.clip(self.accum + impulse, -limit, limit)
        impulse = new - self.accum
        self.accum = new
        world.angular_velocity[self.body] += (self._invI[self.body]
                                              @ (self.axis * impulse))


def _unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` made unit length; zero stays zero."""
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-12 else np.zeros(3)


def _perpendicular(axis: np.ndarray) -> np.ndarray:
    """A unit vector at right angles to the unit ``axis``."""
    other = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(axis, other))


def _angle_about(axis: np.ndarray, start: np.ndarray, now: np.ndarray) -> float:
    """The angle about ``axis`` from ``start`` to ``now``, both taken across the axis."""
    start = start - np.dot(start, axis) * axis
    now = now - np.dot(now, axis) * axis
    return math.atan2(float(np.dot(np.cross(start, now), axis)), float(np.dot(start, now)))


def _orientation(world: "PhysicsWorld", body: int) -> np.ndarray:
    """The unit quaternion of ``body``; no turn for the world."""
    if body < 0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.asarray(world.orientation[body], dtype='d')


def _relative_turn(world: "PhysicsWorld", a: int, b: int) -> np.ndarray:
    """The quaternion that turns ``a`` into ``b``, in ``a``'s frame."""
    return mathutil.quat_mul(mathutil.quat_conjugate(_orientation(world, a)),
                             _orientation(world, b))


def _drift(world: "PhysicsWorld", a: int, b: int, turn: np.ndarray) -> np.ndarray:
    """How far ``b`` has turned from where ``turn`` (:func:`_relative_turn`) put it on ``a``.

    A world-frame quaternion with a non-negative scalar part: for a small
    drift its vector part is half the rotation vector.
    """
    held = mathutil.quat_mul(_orientation(world, a), turn)
    drift = mathutil.quat_mul(_orientation(world, b), mathutil.quat_conjugate(held))
    return -drift if drift[3] < 0.0 else drift


class Stops:
    """A pair of stops on one axis of a joint, with how hard they push back.

    ``lower`` and ``upper`` bound the joint's angle (radians) or separation
    (metres); infinite stops are no stop. ``bounce`` is the fraction of the
    speed the joint arrives with that a stop throws back, and ``erp`` the
    fraction of an overshoot it takes out each step.
    """

    def __init__(self, lower: float = -np.inf, upper: float = np.inf, bounce: float = 0.0,
                 erp: float = 0.2, softness: float = 0.0) -> None:
        self.lower = float(lower)
        self.upper = float(upper)
        self.bounce = float(bounce)
        self.erp = float(erp)
        self.softness = float(softness)

    def rows(self, joint: RowJoint, rows: Rows, tag: str, lin: np.ndarray,
             ang_a: np.ndarray, ang_b: np.ndarray, value: float, rate: float,
             dt: float) -> None:
        """Add the stop's row where ``value`` has reached either stop."""
        if self.lower == -np.inf and self.upper == np.inf:
            return
        joint.stop_rows(rows, tag, lin, ang_a, ang_b, value, rate, self.lower, self.upper,
                        dt, self.bounce, self.erp, self.softness / dt)


class BallJoint(RowJoint):
    """Two bodies that turn freely about a shared point, ``anchor``."""

    def __init__(self, a: int, b: int, anchor: Vec) -> None:
        super().__init__(a, b)
        self.anchor = np.asarray(anchor, dtype='d')

    def bind(self, world: "PhysicsWorld") -> None:
        """Fix the anchor in each body's frame."""
        self.local_a = self.to_local(world, self.a, self.anchor)
        self.local_b = self.to_local(world, self.b, self.anchor)
        super().bind(world)

    def anchor_a(self, world: "PhysicsWorld") -> np.ndarray:
        """Where the anchor is as ``a`` carries it, in world coordinates."""
        position, rotation = self.frame(world, self.a)
        return position + rotation @ self.local_a

    def anchor_b(self, world: "PhysicsWorld") -> np.ndarray:
        """Where the anchor is as ``b`` carries it, in world coordinates."""
        position, rotation = self.frame(world, self.b)
        return position + rotation @ self.local_b

    def build(self, world: "PhysicsWorld", dt: float, rows: Rows) -> None:
        """Hold the two anchors together."""
        self.point_rows(world, dt, rows, 'p', self.anchor_a(world), self.anchor_b(world))


class HingeJoint(BallJoint):
    """A door hinge: ``b`` turns against ``a`` about ``axis`` through ``anchor``, and no other way.

    ``lower`` and ``upper`` stop the angle, ``stop_bounce`` and ``stop_erp``
    say how the stops push back (:class:`Stops`). A motor turns the hinge at
    ``motor_speed`` radians a second with at most ``motor_torque`` newton-metres.
    """

    def __init__(self, a: int, b: int, anchor: Vec, axis: Vec, lower: float = -np.inf,
                 upper: float = np.inf, stop_bounce: float = 0.0, stop_erp: float = 0.2,
                 motor_speed: float = 0.0, motor_torque: float = 0.0) -> None:
        super().__init__(a, b, anchor)
        self.axis = mathutil.normalize(np.asarray(axis, dtype='d'))
        self.stops = Stops(lower, upper, stop_bounce, stop_erp)
        self.motor_speed = float(motor_speed)
        self.motor_torque = float(motor_torque)

    def bind(self, world: "PhysicsWorld") -> None:
        """Fix the anchor, the axis and a reference across it in each body's frame."""
        reference = _perpendicular(self.axis)
        self.axis_a = self.direction_to_local(world, self.a, self.axis)
        self.axis_b = self.direction_to_local(world, self.b, self.axis)
        self.reference_a = self.direction_to_local(world, self.a, reference)
        self.reference_b = self.direction_to_local(world, self.b, reference)
        super().bind(world)

    def hinge_axis(self, world: "PhysicsWorld") -> np.ndarray:
        """The hinge's axis as ``a`` carries it, in world coordinates."""
        return self.frame(world, self.a)[1] @ self.axis_a

    def angle(self, world: "PhysicsWorld") -> float:
        """How far ``b`` has turned against ``a`` about the axis since the joint was added."""
        return _angle_about(self.hinge_axis(world),
                            self.frame(world, self.a)[1] @ self.reference_a,
                            self.frame(world, self.b)[1] @ self.reference_b)

    def angle_rate(self, world: "PhysicsWorld") -> float:
        """How fast the angle is changing, in radians a second."""
        return float(np.dot(self.relative_spin(world), self.hinge_axis(world)))

    def build(self, world: "PhysicsWorld", dt: float, rows: Rows) -> None:
        """Hold the anchor and the axis; add the stops and the motor."""
        super().build(world, dt, rows)
        axis = self.hinge_axis(world)
        carried = self.frame(world, self.b)[1] @ self.axis_b
        tilt = np.cross(axis, carried)
        across = _perpendicular(axis)
        for index, direction in enumerate((across, np.cross(axis, across))):
            self.angular_row(rows, 'a%d' % index, direction,
                             -self.erp * float(np.dot(tilt, direction)) / dt,
                             softness=self.cfm / dt)
        self.stops.rows(self, rows, 'stop', np.zeros(3), axis, axis, self.angle(world),
                        self.angle_rate(world), dt)
        self.motor_row(rows, 'motor', np.zeros(3), axis, axis, self.motor_speed,
                       self.motor_torque, dt)


class SliderJoint(RowJoint):
    """``b`` slides against ``a`` along ``axis``, and neither turns against the other.

    The separation is how far ``b`` has moved along the axis since the joint
    was added, in metres; ``lower`` and ``upper`` stop it. ``force`` pushes
    the bodies apart along the axis, in newtons, each step; a motor slides
    them at ``motor_speed`` metres a second with at most ``motor_force``.
    """

    def __init__(self, a: int, b: int, axis: Vec, lower: float = -np.inf,
                 upper: float = np.inf, stop_bounce: float = 0.0, stop_erp: float = 0.2,
                 force: float = 0.0, motor_speed: float = 0.0,
                 motor_force: float = 0.0) -> None:
        super().__init__(a, b)
        self.axis = mathutil.normalize(np.asarray(axis, dtype='d'))
        self.stops = Stops(lower, upper, stop_bounce, stop_erp)
        self.force = float(force)
        self.motor_speed = float(motor_speed)
        self.motor_force = float(motor_force)

    def bind(self, world: "PhysicsWorld") -> None:
        """Fix the axis in ``a``'s frame, and where ``b`` starts from."""
        self.axis_a = self.direction_to_local(world, self.a, self.axis)
        self.start_a = self.to_local(world, self.a, self.frame(world, self.b)[0])
        self.turn = _relative_turn(world, self.a, self.b)
        super().bind(world)

    def slide_axis(self, world: "PhysicsWorld") -> np.ndarray:
        """The axis as ``a`` carries it, in world coordinates."""
        return self.frame(world, self.a)[1] @ self.axis_a

    def _offset(self, world: "PhysicsWorld") -> np.ndarray:
        """How far ``b`` is from where it started, as ``a`` carries that place."""
        position, rotation = self.frame(world, self.a)
        return self.frame(world, self.b)[0] - (position + rotation @ self.start_a)

    def separation(self, world: "PhysicsWorld") -> float:
        """How far ``b`` has slid along the axis since the joint was added, in metres."""
        return float(np.dot(self._offset(world), self.slide_axis(world)))

    def separation_rate(self, world: "PhysicsWorld") -> float:
        """How fast the separation is changing, in metres a second."""
        b_at = self.frame(world, self.b)[0]
        return float(np.dot(self.velocity_at(world, self.b, b_at)
                            - self.velocity_at(world, self.a, b_at), self.slide_axis(world)))

    def drive(self, world: "PhysicsWorld", dt: float) -> None:
        """Push the bodies apart along the axis with :attr:`force`."""
        if self.force and self.bound:
            self.push(world, self.frame(world, self.b)[0],
                      self.force * dt * self.slide_axis(world))

    def build(self, world: "PhysicsWorld", dt: float, rows: Rows) -> None:
        """Hold the turn and the line; add the stops and the motor."""
        drift = _drift(world, self.a, self.b, self.turn)
        for index in range(3):
            direction = np.eye(3)[index]
            self.angular_row(rows, 'a%d' % index, direction,
                             -self.erp * 2.0 * float(drift[index]) / dt,
                             softness=self.cfm / dt)
        axis = self.slide_axis(world)
        arm = self.frame(world, self.b)[0] - self.frame(world, self.a)[0]
        offset = self._offset(world)
        across = _perpendicular(axis)
        for index, direction in enumerate((across, np.cross(axis, across))):
            rows.add(self, 'l%d' % index, direction, np.cross(arm, direction), np.zeros(3),
                     -self.erp * float(np.dot(offset, direction)) / dt,
                     softness=self.cfm / dt)
        turn_a = np.cross(arm, axis)
        self.stops.rows(self, rows, 'stop', axis, turn_a, np.zeros(3), self.separation(world),
                        self.separation_rate(world), dt)
        self.motor_row(rows, 'motor', axis, turn_a, np.zeros(3), self.motor_speed,
                       self.motor_force, dt)


class UniversalJoint(BallJoint):
    """A cardan joint: ``axis1`` on ``a`` and ``axis2`` on ``b`` stay at the angle they start at.

    ``b`` turns against ``a`` about either axis, so a spin about the line at
    right angles to both is carried across the joint.
    """

    def __init__(self, a: int, b: int, anchor: Vec, axis1: Vec, axis2: Vec) -> None:
        super().__init__(a, b, anchor)
        self.axis1_world = mathutil.normalize(np.asarray(axis1, dtype='d'))
        self.axis2_world = mathutil.normalize(np.asarray(axis2, dtype='d'))

    def bind(self, world: "PhysicsWorld") -> None:
        """Fix ``axis1`` in ``a``'s frame and ``axis2`` in ``b``'s."""
        self.axis1_a = self.direction_to_local(world, self.a, self.axis1_world)
        self.axis2_b = self.direction_to_local(world, self.b, self.axis2_world)
        self.cosine = float(np.dot(self.axis1_world, self.axis2_world))
        super().bind(world)

    def axis1(self, world: "PhysicsWorld") -> np.ndarray:
        """The first axis as ``a`` carries it, in world coordinates."""
        return self.frame(world, self.a)[1] @ self.axis1_a

    def axis2(self, world: "PhysicsWorld") -> np.ndarray:
        """The second axis as ``b`` carries it, in world coordinates."""
        return self.frame(world, self.b)[1] @ self.axis2_b

    def cross_rows(self, world: "PhysicsWorld", dt: float, rows: Rows) -> None:
        """The row that keeps the two axes at the angle they started at."""
        first, second = self.axis1(world), self.axis2(world)
        across = np.cross(first, second)
        length = float(np.linalg.norm(across))
        if length < 1e-9:
            return
        # The cosine falls as b turns about first × second.
        self.angular_row(rows, 'x', across / length,
                         self.erp * (float(np.dot(first, second)) - self.cosine) / (length * dt),
                         softness=self.cfm / dt)

    def build(self, world: "PhysicsWorld", dt: float, rows: Rows) -> None:
        """Hold the anchor and the angle between the axes."""
        super().build(world, dt, rows)
        self.cross_rows(world, dt, rows)


class TwoAxisHingeJoint(UniversalJoint):
    """A steered wheel: ``b`` turns about ``axis1`` on ``a`` and about ``axis2`` on itself.

    ``axis1`` is the steering axis and ``axis2`` the axle. ``lower1`` and
    ``upper1`` stop the steering angle (equal stops lock it); motors turn
    either axis at a speed, in radians a second, with at most a torque. A
    suspension lets the anchor give along ``axis1`` as a spring of
    ``suspension_stiffness`` newtons per metre and ``suspension_damping``
    newton-seconds per metre; without one the anchor is held.
    """

    def __init__(self, a: int, b: int, anchor: Vec, axis1: Vec, axis2: Vec,
                 lower1: float = -np.inf, upper1: float = np.inf, stop1_bounce: float = 0.0,
                 stop1_erp: float = 0.2, motor1_speed: float = 0.0,
                 motor1_torque: float = 0.0, motor2_speed: float = 0.0,
                 motor2_torque: float = 0.0, suspension_stiffness: float = 0.0,
                 suspension_damping: float = 0.0) -> None:
        super().__init__(a, b, anchor, axis1, axis2)
        self.stops1 = Stops(lower1, upper1, stop1_bounce, stop1_erp)
        self.motor1_speed = float(motor1_speed)
        self.motor1_torque = float(motor1_torque)
        self.motor2_speed = float(motor2_speed)
        self.motor2_torque = float(motor2_torque)
        self.suspension_stiffness = float(suspension_stiffness)
        self.suspension_damping = float(suspension_damping)

    def bind(self, world: "PhysicsWorld") -> None:
        """Fix the axle's starting direction in ``a``'s frame and the steering axis in ``b``'s."""
        self.axle_start_a = self.direction_to_local(world, self.a, self.axis2_world)
        self.steering_b = self.direction_to_local(world, self.b, self.axis1_world)
        super().bind(world)

    def angle1(self, world: "PhysicsWorld") -> float:
        """The steering angle: how far the axle has turned about ``axis1``."""
        return _angle_about(self.axis1(world),
                            self.frame(world, self.a)[1] @ self.axle_start_a,
                            self.axis2(world))

    def angle1_rate(self, world: "PhysicsWorld") -> float:
        """How fast the steering angle is changing, in radians a second."""
        return float(np.dot(self.relative_spin(world), self.axis1(world)))

    def angle2(self, world: "PhysicsWorld") -> float:
        """How far ``b`` has turned about its axle since the joint was added."""
        return _angle_about(self.axis2(world), self.axis1(world),
                            self.frame(world, self.b)[1] @ self.steering_b)

    def angle2_rate(self, world: "PhysicsWorld") -> float:
        """How fast ``b`` turns about its axle, in radians a second."""
        return float(np.dot(self.relative_spin(world), self.axis2(world)))

    def build(self, world: "PhysicsWorld", dt: float, rows: Rows) -> None:
        """The anchor (sprung along the steering axis), the axes, stops and motors."""
        first = self.axis1(world)
        across = _perpendicular(first)
        directions = (first, across, np.cross(first, across))
        softened = None
        if self.suspension_stiffness > 0.0 or self.suspension_damping > 0.0:
            erp, softness = spring(self.suspension_stiffness, self.suspension_damping, dt)
            softened = {0: (erp, softness / dt)}
        self.point_rows(world, dt, rows, 'p', self.anchor_a(world), self.anchor_b(world),
                        directions, softened)
        self.cross_rows(world, dt, rows)
        self.stops1.rows(self, rows, 'stop1', np.zeros(3), first, first, self.angle1(world),
                         self.angle1_rate(world), dt)
        self.motor_row(rows, 'motor1', np.zeros(3), first, first, self.motor1_speed,
                       self.motor1_torque, dt)
        axle = self.axis2(world)
        self.motor_row(rows, 'motor2', np.zeros(3), axle, axle, self.motor2_speed,
                       self.motor2_torque, dt)


class MotorJoint(RowJoint):
    """Torques between two bodies about up to three axes, each step.

    ``axes`` are directions in world coordinates as the bodies stand when the
    joint is added; ``a`` carries them from then on (the world, where ``a``
    is ``-1``). ``torques`` are in newton-metres: ``b`` is turned forward
    about each axis and ``a`` back. The angle about each axis is how far
    ``b`` has turned against ``a`` since the joint was added.
    """

    def __init__(self, a: int, b: int, axes: Sequence[Vec], torques: Sequence[float]) -> None:
        super().__init__(a, b)
        self.axes_world = [mathutil.normalize(np.asarray(axis, dtype='d')) for axis in axes]
        self.torques = [float(torque) for torque in torques]

    def bind(self, world: "PhysicsWorld") -> None:
        """Fix the axes in ``a``'s frame and the starting turn between the bodies."""
        self.axes_a = [self.direction_to_local(world, self.a, axis) for axis in self.axes_world]
        self.turn = _relative_turn(world, self.a, self.b)
        super().bind(world)

    def axis(self, world: "PhysicsWorld", index: int) -> np.ndarray:
        """Axis ``index`` as ``a`` carries it, in world coordinates."""
        return self.frame(world, self.a)[1] @ self.axes_a[index]

    def angle(self, world: "PhysicsWorld", index: int) -> float:
        """How far ``b`` has turned against ``a`` about axis ``index``."""
        drift = _drift(world, self.a, self.b, self.turn)
        along = float(np.dot(drift[:3], self.axis(world, index)))
        angle = 2.0 * math.atan2(along, float(drift[3]))
        return math.remainder(angle, 2.0 * math.pi)

    def angle_rate(self, world: "PhysicsWorld", index: int) -> float:
        """How fast ``b`` turns against ``a`` about axis ``index``, in radians a second."""
        return float(np.dot(self.relative_spin(world), self.axis(world, index)))

    def drive(self, world: "PhysicsWorld", dt: float) -> None:
        """Turn the bodies against each other by the torques for one step."""
        if not self.bound:
            return
        torque = np.zeros(3)
        for index, amount in enumerate(self.torques[:len(self.axes_a)]):
            torque += amount * self.axis(world, index)
        if torque.any():
            self.twist(world, torque * dt)

    def build(self, world: "PhysicsWorld", dt: float, rows: Rows) -> None:
        """A motor joint holds nothing; it only turns the bodies (:meth:`drive`)."""
