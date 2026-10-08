"""Joints as rows of velocity constraints, solved together each step.

A row asks one thing of the relative motion of two bodies: that a point on
one keeps pace with a point on the other along a direction, or that they turn
at the same rate about an axis. With ``J`` the row's direction and ``v`` the
two bodies' velocities, the solver drives ``J v`` to the row's ``bias`` with
an impulse kept between the row's ``low`` and ``high``:

- a locked degree of freedom has an unbounded impulse, and a bias that takes
  out a fraction (``erp``, error reduction per step) of how far the joint has
  drifted;
- a stop is a row that pushes one way only, present while the joint is at or
  past it, with a bias that can also throw the motion back (``bounce``);
- a motor is a row whose bias is the speed wanted and whose impulse is
  bounded by the force or torque the motor has, times the step.

A row's ``softness`` lets it give: a constraint-force mixing term in seconds
per kilogram, so a soft row trades stiffness for stability. A spring of
stiffness ``k`` and damping ``c`` is a row with ``erp = dt k / (dt k + c)`` and
``softness = 1 / (dt k + c)`` (:func:`spring`).

Every joint's rows go to one Gauss-Seidel solve, compiled where
:mod:`omi_physics._solver_native` is built and in Python otherwise, the two
computing the same numbers in the same order. Each row's impulse is kept by
its joint for the next step and applied first, so a chain carrying a load
starts each step from the impulses that held it last time.
"""
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from . import mathutil
from ._accel import accelerators_disabled

if TYPE_CHECKING:
    from .world import PhysicsWorld

#: The compiled row solver, or None where it was not built or was switched off.
_native: Any
if accelerators_disabled():
    _native = None
else:
    try:
        from . import _solver_native as _native  # type: ignore[no-redef,attr-defined]
    except ImportError:                            # pragma: no cover - pure-Python fallback
        _native = None
    if _native is not None and not hasattr(_native, 'solve_rows'):
        _native = None                             # pragma: no cover - an older build

__all__ = ['RowJoint', 'Rows', 'drive', 'solve', 'spring']

_IDENTITY = np.identity(3)
_ORIGIN = np.zeros(3)


def spring(stiffness: float, damping: float, dt: float) -> tuple[float, float]:
    """The ``(erp, softness)`` of a row that acts as a spring over a step of ``dt``.

    ``stiffness`` is in newtons per metre (or newton-metres per radian for an
    angular row) and ``damping`` in newton-seconds per metre.
    """
    total = dt * stiffness + damping
    if total <= 0.0:
        return 0.0, 0.0
    return dt * stiffness / total, 1.0 / total


class Rows:
    """The rows of one step's joints, gathered for the solver."""

    def __init__(self) -> None:
        self.owner: list[RowJoint] = []
        self.tag: list[str] = []
        self.a: list[int] = []
        self.b: list[int] = []
        self.lin: list[np.ndarray] = []
        self.ang_a: list[np.ndarray] = []
        self.ang_b: list[np.ndarray] = []
        self.bias: list[float] = []
        self.low: list[float] = []
        self.high: list[float] = []
        self.softness: list[float] = []

    def __len__(self) -> int:
        return len(self.tag)

    def add(self, owner: 'RowJoint', tag: str, lin: np.ndarray, ang_a: np.ndarray,
            ang_b: np.ndarray, bias: float, low: float = -np.inf, high: float = np.inf,
            softness: float = 0.0) -> None:
        """Add one row: ``J v = lin·(vB - vA) + ang_b·ωB - ang_a·ωA`` driven to ``bias``.

        ``tag`` names the row within its joint, so its impulse can be carried
        to the next step; ``softness`` is the row's constraint-force mixing
        divided by the step, in seconds per kilogram.
        """
        self.owner.append(owner)
        self.tag.append(tag)
        self.a.append(owner.a)
        self.b.append(owner.b)
        self.lin.append(lin)
        self.ang_a.append(ang_a)
        self.ang_b.append(ang_b)
        self.bias.append(float(bias))
        self.low.append(float(low))
        self.high.append(float(high))
        self.softness.append(float(softness))


class RowJoint:
    """A joint between bodies ``a`` and ``b`` that the row solver holds.

    Either body may be ``-1``, the world. Subclasses say what they hold in
    :meth:`build`; one that pushes the bodies with a force of its own does it
    in :meth:`drive`. ``erp`` is the fraction of a joint's drift taken out
    each step and ``softness`` the constraint-force mixing of its locked rows
    (:mod:`omi_physics.jointrows`).
    """

    #: Fraction of the joint's drift corrected each step.
    erp = 0.2
    #: Constraint-force mixing of the locked rows, in metres per newton-second.
    cfm = 0.0
    #: Whether the two bodies still collide with each other. A joint's bodies
    #: usually meet at the joint, so by default they do not.
    collide_connected = False

    def __init__(self, a: int, b: int) -> None:
        self.a = int(a)
        self.b = int(b)
        #: Each row's impulse at the end of the last step, by tag.
        self.impulses: dict[str, float] = {}
        self.bound = False

    # -- what a subclass provides ---------------------------------------
    def bind(self, world: 'PhysicsWorld') -> None:  # noqa: ARG002 subclasses read the world
        """Fix the joint's anchors and axes in each body's frame, where the bodies are now."""
        self.bound = True

    def build(self, world: 'PhysicsWorld', dt: float, rows: Rows) -> None:
        """Add this step's rows to ``rows``."""
        raise NotImplementedError

    def drive(self, world: 'PhysicsWorld', dt: float) -> None:
        """Push the bodies with the joint's own force for one step; none by default."""

    # -- the bodies -------------------------------------------------------
    def frame(self, world: 'PhysicsWorld', body: int) -> tuple[np.ndarray, np.ndarray]:
        """``(position, rotation matrix)`` of ``body``; the world's origin for ``-1``."""
        if body < 0:
            return _ORIGIN, _IDENTITY
        return (np.asarray(world.position[body], dtype='d'),
                mathutil.quat_to_matrix(np.asarray(world.orientation[body], dtype='d')))

    def spin(self, world: 'PhysicsWorld', body: int) -> np.ndarray:
        """Angular velocity of ``body``; nothing for the world."""
        return _ORIGIN if body < 0 else np.asarray(world.angular_velocity[body], dtype='d')

    def velocity_at(self, world: 'PhysicsWorld', body: int, point: np.ndarray) -> np.ndarray:
        """Velocity of the point of ``body`` at ``point``, in world coordinates."""
        if body < 0:
            return _ORIGIN
        return (np.asarray(world.linear_velocity[body], dtype='d')
                + np.cross(world.angular_velocity[body], point - world.position[body]))

    def to_local(self, world: 'PhysicsWorld', body: int, point: Iterable[float]) -> np.ndarray:
        """A world point in ``body``'s frame."""
        position, rotation = self.frame(world, body)
        return rotation.T @ (np.asarray(point, dtype='d') - position)

    def direction_to_local(self, world: 'PhysicsWorld', body: int,
                           direction: Iterable[float]) -> np.ndarray:
        """A world direction, made unit length, in ``body``'s frame."""
        return self.frame(world, body)[1].T @ mathutil.normalize(
            np.asarray(direction, dtype='d'))

    def relative_spin(self, world: 'PhysicsWorld') -> np.ndarray:
        """How fast ``b`` turns against ``a``, in world coordinates."""
        return self.spin(world, self.b) - self.spin(world, self.a)

    def push(self, world: 'PhysicsWorld', point: np.ndarray, impulse: np.ndarray) -> None:
        """Push ``b`` by ``impulse`` at ``point`` and ``a`` by its opposite, in world terms."""
        for body, sign in ((self.a, -1.0), (self.b, 1.0)):
            if body >= 0 and world.inv_mass[body] > 0.0:
                world.linear_velocity[body] += sign * world.inv_mass[body] * impulse
                world.angular_velocity[body] += sign * _inv_inertia(world, body) @ np.cross(
                    point - world.position[body], impulse)
                world.wake(body)

    def twist(self, world: 'PhysicsWorld', torque: np.ndarray) -> None:
        """Turn ``b`` by the angular impulse ``torque`` and ``a`` by its opposite."""
        for body, sign in ((self.a, -1.0), (self.b, 1.0)):
            if body >= 0 and world.inv_mass[body] > 0.0:
                world.angular_velocity[body] += sign * _inv_inertia(world, body) @ torque
                world.wake(body)

    # -- rows every joint builds from ------------------------------------
    def point_rows(self, world: 'PhysicsWorld', dt: float, rows: Rows, tag: str,
                   on_a: np.ndarray, on_b: np.ndarray,
                   directions: 'Iterable[np.ndarray]' = (_IDENTITY[0], _IDENTITY[1],
                                                         _IDENTITY[2]),
                   softened: 'dict[int, tuple[float, float]] | None' = None) -> None:
        """Rows holding the world points ``on_a`` and ``on_b`` together along ``directions``.

        ``softened`` gives an ``(erp, softness)`` in place of the joint's own
        for the directions it names by index.
        """
        arm_a = on_a - self.frame(world, self.a)[0]
        arm_b = on_b - self.frame(world, self.b)[0]
        apart = on_b - on_a
        for index, direction in enumerate(directions):
            erp, softness = (softened or {}).get(index, (self.erp, self.cfm / dt))
            rows.add(self, '%s%d' % (tag, index), direction, np.cross(arm_a, direction),
                     np.cross(arm_b, direction), -erp * float(np.dot(apart, direction)) / dt,
                     softness=softness)

    def angular_row(self, rows: Rows, tag: str, axis: np.ndarray, bias: float,
                    low: float = -np.inf, high: float = np.inf, softness: float = 0.0) -> None:
        """A row on how fast ``b`` turns against ``a`` about ``axis``."""
        rows.add(self, tag, _ORIGIN, axis, axis, bias, low, high, softness)

    def stop_rows(self, rows: Rows, tag: str, lin: np.ndarray, ang_a: np.ndarray,
                  ang_b: np.ndarray, value: float, rate: float, lower: float,
                  upper: float, dt: float, bounce: float, erp: float,
                  softness: float = 0.0) -> None:
        """The row a pair of stops adds, where ``value`` has reached either of them.

        ``rate`` is how fast ``value`` is changing; a stop moving the joint
        back throws it at ``bounce`` times the speed it arrived with. Equal
        stops lock the joint where they are.
        """
        if lower == upper:
            rows.add(self, tag, lin, ang_a, ang_b, -erp * (value - lower) / dt,
                     softness=softness)
        elif value <= lower:
            bias = -erp * (value - lower) / dt
            if rate < 0.0:
                bias = max(bias, -bounce * rate)
            rows.add(self, tag, lin, ang_a, ang_b, bias, 0.0, np.inf, softness)
        elif value >= upper:
            bias = -erp * (value - upper) / dt
            if rate > 0.0:
                bias = min(bias, -bounce * rate)
            rows.add(self, tag, lin, ang_a, ang_b, bias, -np.inf, 0.0, softness)

    def motor_row(self, rows: Rows, tag: str, lin: np.ndarray, ang_a: np.ndarray,
                  ang_b: np.ndarray, speed: float, strength: float, dt: float) -> None:
        """A motor reaching ``speed`` with at most ``strength`` newtons or newton-metres."""
        if strength > 0.0:
            limit = strength * dt
            rows.add(self, tag, lin, ang_a, ang_b, speed, -limit, limit)


def _inv_inertia(world: 'PhysicsWorld', body: int) -> np.ndarray:
    """World-space inverse inertia tensor of ``body``."""
    rotation = mathutil.quat_to_matrix(np.asarray(world.orientation[body], dtype='d'))
    return (rotation * world.inv_inertia[body]) @ rotation.T


def drive(world: 'PhysicsWorld', joints: 'Iterable[RowJoint]', dt: float) -> None:
    """Let every joint push its bodies with its own force for one step."""
    for joint in joints:
        joint.drive(world, dt)


def solve(world: 'PhysicsWorld', joints: 'list[RowJoint]', dt: float,
          iterations: int) -> None:
    """Build every joint's rows and solve them together, ``iterations`` times over."""
    rows = Rows()
    for joint in joints:
        if not joint.bound:
            joint.bind(world)
        joint.build(world, dt, rows)
    count = len(rows)
    if not count:
        for joint in joints:
            joint.impulses = {}
        return
    a_idx = np.array(rows.a, dtype=np.intp)
    b_idx = np.array(rows.b, dtype=np.intp)
    lin = np.ascontiguousarray(rows.lin, dtype='d')
    ang_a = np.ascontiguousarray(rows.ang_a, dtype='d')
    ang_b = np.ascontiguousarray(rows.ang_b, dtype='d')
    bias = np.array(rows.bias, dtype='d')
    low = np.array(rows.low, dtype='d')
    high = np.array(rows.high, dtype='d')
    softness = np.array(rows.softness, dtype='d')
    impulse = np.array([min(max(owner.impulses.get(tag, 0.0), lo), hi)
                        for owner, tag, lo, hi in zip(rows.owner, rows.tag, rows.low,
                                                      rows.high, strict=True)], dtype='d')
    inv_inertia = _inv_inertia_all(world)
    if _native is not None:
        _native.solve_rows(a_idx, b_idx, lin, ang_a, ang_b, bias, low, high, softness,
                           impulse, world.linear_velocity, world.angular_velocity,
                           world.inv_mass, inv_inertia, int(iterations))
    else:
        _solve_rows(a_idx, b_idx, lin, ang_a, ang_b, bias, low, high, softness, impulse,
                    world.linear_velocity, world.angular_velocity, world.inv_mass,
                    inv_inertia, int(iterations))
    kept: dict[int, dict[str, float]] = {id(joint): {} for joint in joints}
    for owner, tag, value in zip(rows.owner, rows.tag, impulse.tolist(), strict=True):
        kept[id(owner)][tag] = value
    for joint in joints:
        joint.impulses = kept[id(joint)]


def _inv_inertia_all(world: 'PhysicsWorld') -> np.ndarray:
    """World inverse inertia tensor of every body, ``(n, 3, 3)``."""
    count = world.body_count
    rotation = mathutil.quat_to_matrix(np.asarray(world.orientation[:count], dtype='d'))
    scaled = rotation * world.inv_inertia[:count][:, None, :]
    return np.ascontiguousarray(np.einsum('nij,nkj->nik', scaled, rotation))


def _solve_rows(a_idx: np.ndarray, b_idx: np.ndarray, lin: np.ndarray, ang_a: np.ndarray,
                ang_b: np.ndarray, bias: np.ndarray, low: np.ndarray, high: np.ndarray,
                softness: np.ndarray, impulse: np.ndarray, lv: np.ndarray, av: np.ndarray,
                inv_mass: np.ndarray, inv_inertia: np.ndarray, iterations: int) -> None:
    """Python twin of ``_solver_native.solve_rows``, operation for operation."""
    count = len(a_idx)
    mass = [0.0] * count
    # Each row's angular terms turned by the bodies' inverse inertia, once.
    turn_a: list[list[float]] = []
    turn_b: list[list[float]] = []
    for k in range(count):
        a, b = int(a_idx[k]), int(b_idx[k])
        lx, ly, lz = float(lin[k, 0]), float(lin[k, 1]), float(lin[k, 2])
        effective = 0.0
        ta = [0.0, 0.0, 0.0]
        tb = [0.0, 0.0, 0.0]
        if a >= 0:
            effective += float(inv_mass[a]) * (lx * lx + ly * ly + lz * lz)
            for i in range(3):
                ta[i] = (float(inv_inertia[a, i, 0]) * float(ang_a[k, 0])
                         + float(inv_inertia[a, i, 1]) * float(ang_a[k, 1])
                         + float(inv_inertia[a, i, 2]) * float(ang_a[k, 2]))
            effective += (float(ang_a[k, 0]) * ta[0] + float(ang_a[k, 1]) * ta[1]
                          + float(ang_a[k, 2]) * ta[2])
        if b >= 0:
            effective += float(inv_mass[b]) * (lx * lx + ly * ly + lz * lz)
            for i in range(3):
                tb[i] = (float(inv_inertia[b, i, 0]) * float(ang_b[k, 0])
                         + float(inv_inertia[b, i, 1]) * float(ang_b[k, 1])
                         + float(inv_inertia[b, i, 2]) * float(ang_b[k, 2]))
            effective += (float(ang_b[k, 0]) * tb[0] + float(ang_b[k, 1]) * tb[1]
                          + float(ang_b[k, 2]) * tb[2])
        effective += float(softness[k])
        mass[k] = 1.0 / effective if effective > 1e-12 else 0.0
        turn_a.append(ta)
        turn_b.append(tb)
        _apply_row(k, a, b, float(impulse[k]), lin, ta, tb, lv, av, inv_mass)
    for _ in range(iterations):
        for k in range(count):
            a, b = int(a_idx[k]), int(b_idx[k])
            jv = 0.0
            if b >= 0:
                jv += (float(lin[k, 0]) * float(lv[b, 0]) + float(lin[k, 1]) * float(lv[b, 1])
                       + float(lin[k, 2]) * float(lv[b, 2]))
                jv += (float(ang_b[k, 0]) * float(av[b, 0]) + float(ang_b[k, 1]) * float(av[b, 1])
                       + float(ang_b[k, 2]) * float(av[b, 2]))
            if a >= 0:
                jv -= (float(lin[k, 0]) * float(lv[a, 0]) + float(lin[k, 1]) * float(lv[a, 1])
                       + float(lin[k, 2]) * float(lv[a, 2]))
                jv -= (float(ang_a[k, 0]) * float(av[a, 0]) + float(ang_a[k, 1]) * float(av[a, 1])
                       + float(ang_a[k, 2]) * float(av[a, 2]))
            old = float(impulse[k])
            new = old + mass[k] * (float(bias[k]) - jv - float(softness[k]) * old)
            if new < float(low[k]):
                new = float(low[k])
            elif new > float(high[k]):
                new = float(high[k])
            impulse[k] = new
            _apply_row(k, a, b, new - old, lin, turn_a[k], turn_b[k], lv, av, inv_mass)


def _apply_row(k: int, a: int, b: int, amount: float, lin: np.ndarray, turn_a: list[float],
               turn_b: list[float], lv: np.ndarray, av: np.ndarray,
               inv_mass: np.ndarray) -> None:
    """Apply ``amount`` of row ``k``'s impulse: ``b`` along it, ``a`` against it."""
    if amount == 0.0:
        return
    if a >= 0:
        scale = float(inv_mass[a]) * amount
        for i in range(3):
            lv[a, i] = float(lv[a, i]) - scale * float(lin[k, i])
            av[a, i] = float(av[a, i]) - turn_a[i] * amount
    if b >= 0:
        scale = float(inv_mass[b]) * amount
        for i in range(3):
            lv[b, i] = float(lv[b, i]) + scale * float(lin[k, i])
            av[b, i] = float(av[b, i]) + turn_b[i] * amount
