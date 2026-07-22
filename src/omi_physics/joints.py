"""Joint constraints (``OMI_physics_joint``) solved as velocity constraints.

Implemented as constraint rows solved in the same Gauss-Seidel style as contacts
(Catto): a point (ball) constraint locks an anchor, a distance constraint holds a
length, and a motor drive reaches a target velocity within a force budget.  A
body index of ``-1`` means the world (an infinite-mass anchor).  These map onto
the OMI limits/drives vocabulary (locked linear axes → point; a linear limit with
``min==max`` → distance; a drive → motor).
"""
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

from . import mathutil
from .mathutil import Vec

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
    _invI: Dict[int, np.ndarray]

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
        return world.linear_velocity[i] + np.cross(world.angular_velocity[i], r)

    def _apply(self, world: "PhysicsWorld", i: int, r: np.ndarray, P: np.ndarray) -> None:
        """Apply impulse ``P`` at offset ``r`` to body ``i`` (no-op for the world)."""
        if i < 0:
            return
        world.linear_velocity[i] += world.inv_mass[i] * P
        world.angular_velocity[i] += self._invI[i] @ np.cross(r, P)


class PointConstraint(_JointBase):
    """Ball / point-to-point: hold two body anchors coincident (pendulum, hinge).

    ``a`` or ``b`` may be ``-1`` to pin against the world at ``anchor``.
    """

    def __init__(self, a: int, b: int, anchor: Vec,
                 local_a: Optional[np.ndarray] = None,
                 local_b: Optional[np.ndarray] = None):
        self.a, self.b = a, b
        self.anchor = np.asarray(anchor, dtype='d')
        self.local_a = local_a
        self.local_b = local_b
        self._invI = {}

    def _world_anchor(self, world: "PhysicsWorld", i: int,
                      local: Optional[np.ndarray]) -> np.ndarray:
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
                  world_pt: np.ndarray) -> Optional[np.ndarray]:
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
                 length: Optional[float] = None):
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
        raxn = np.cross(self.rA, self.n)
        rbxn = np.cross(self.rB, self.n)
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
