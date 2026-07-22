"""Contact solver — sequential impulses (projected Gauss-Seidel).

Removes approach velocity along the contact normal (with restitution), applies
Coulomb friction on two tangents, and corrects penetration with a split-impulse
(NGS) position pass so restitution is not polluted by Baumgarte energy (Catto).
Material friction/restitution combine per the OMI combine modes.  Contacts
partition into islands so disjoint groups solve independently.
"""
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from . import model, mathutil

try:
    from . import _solver_native as _native   # compiled Cython inner loops
except ImportError:                            # pragma: no cover - pure-Python fallback
    _native = None

if TYPE_CHECKING:
    from .world import PhysicsWorld
    from .collide import Contact


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors.

    numpy.cross is overhead-bound on a single 3-vector -- its axis-moving and
    broadcast machinery dwarfs the six multiplies -- and the contact solver calls
    it on the order of a hundred times per contact per step. Explicit components
    are ~12x faster and are the dominant cost saving for small/medium scenes.
    """
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def _skew_apply(inv_inertia_world: np.ndarray, r: np.ndarray,
                impulse: np.ndarray) -> np.ndarray:
    """Angular velocity change from an impulse ``P`` at arm ``r``: ``I⁻¹ (r × P)``."""
    return inv_inertia_world @ _cross(r, impulse)


def _basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal tangents spanning the plane perpendicular to unit normal ``n``."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = mathutil.normalize(_cross(a, n))
    t2 = _cross(n, t1)
    return t1, t2


class _ContactConstraint:
    """Solver scratch for one contact: contact arms, normal/tangent basis, and masses."""
    __slots__ = ('c', 'rA', 'rB', 'n', 't1', 't2', 'nMass', 't1Mass', 't2Mass',
                 'restitution', 'friction', 'vBias')

    c: 'Contact'
    rA: np.ndarray
    rB: np.ndarray
    n: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    nMass: float
    t1Mass: float
    t2Mass: float
    restitution: float
    friction: float
    vBias: float

    def __init__(self, contact: 'Contact') -> None:
        self.c = contact


class SequentialImpulseSolver:
    """Projected Gauss-Seidel contact solver: normal, friction, then position pass."""

    def __init__(self, velocity_iterations: int = 10, position_iterations: int = 3,
                 baumgarte: float = 0.2, slop: float = 0.005,
                 restitution_threshold: float = 0.5,
                 warm_start: bool = True) -> None:
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations
        self.baumgarte = baumgarte
        self.slop = slop
        self.restitution_threshold = restitution_threshold
        self.warm_start = warm_start
        # (a,b) -> [(point, nImpulse, t0, t1)] carried to the next step for warm start
        self._cache: Dict[Tuple[int, int], List[Tuple[np.ndarray, float, float, float]]] = {}

    # -- setup -----------------------------------------------------------
    def _inv_inertia_world(self, world: "PhysicsWorld", i: int) -> np.ndarray:
        """World-space inverse inertia tensor of body ``i`` (rotate the body-space diagonal)."""
        R = mathutil.quat_to_matrix(world.orientation[i])
        return (R * world.inv_inertia[i]) @ R.T

    def _material(self, world: "PhysicsWorld", i: int) -> model.Material:
        """Physics material of body ``i`` (the default material when it has none)."""
        return world.material_for(world.collider_material[i])

    def _prepare(self, world: "PhysicsWorld", contacts: List["Contact"],
                 ) -> Tuple[List[_ContactConstraint], Dict[int, np.ndarray]]:
        """Build a constraint per contact and cache each body's world inverse inertia.

        Returns the constraints and the ``body index -> inverse inertia`` map the
        velocity iterations reuse.  Seeds warm-start impulses from the last step.
        """
        cons = []
        invI: Dict[int, np.ndarray] = {}
        for c in contacts:
            a, b = c.a, c.b
            if a not in invI:
                invI[a] = self._inv_inertia_world(world, a)
            if b not in invI:
                invI[b] = self._inv_inertia_world(world, b)
            k = _ContactConstraint(c)
            k.rA = c.point - world.position[a]
            k.rB = c.point - world.position[b]
            k.n = c.normal
            k.t1, k.t2 = _basis(c.normal)
            k.nMass = self._effective_mass(world, invI, a, b, k.rA, k.rB, k.n)
            k.t1Mass = self._effective_mass(world, invI, a, b, k.rA, k.rB, k.t1)
            k.t2Mass = self._effective_mass(world, invI, a, b, k.rA, k.rB, k.t2)
            mat_a, mat_b = self._material(world, a), self._material(world, b)
            k.restitution = model.combine(mat_a.restitution, mat_b.restitution,
                                          mat_a.restitutionCombine)
            override = world.pair_friction_for(world.collider_material[a],
                                               world.collider_material[b])
            if override is not None:
                k.friction = override[1]                # dynamic mu
            else:
                k.friction = model.combine(mat_a.dynamicFriction, mat_b.dynamicFriction,
                                           mat_a.frictionCombine)
            vn0 = np.dot(self._rel_vel(world, a, b, k.rA, k.rB), k.n)
            k.vBias = -k.restitution * vn0 if vn0 < -self.restitution_threshold else 0.0
            if self.warm_start:
                self._seed_from_cache(k)
            cons.append(k)
        return cons, invI

    def _seed_from_cache(self, k: _ContactConstraint) -> None:
        """Copy the nearest cached impulses into contact ``k`` so it resumes converged."""
        cached = self._cache.get((k.c.a, k.c.b))
        if not cached:
            return
        best: Optional[Tuple[float, float, float]] = None
        best_d = 0.04                       # (0.2 m)² match tolerance
        for point, nImp, t0, t1 in cached:
            d = float(np.dot(k.c.point - point, k.c.point - point))
            if d < best_d:
                best, best_d = (nImp, t0, t1), d
        if best is not None:
            k.c.normal_impulse, k.c.tangent_impulse[0], k.c.tangent_impulse[1] = best

    def _apply_warm_start(self, world: "PhysicsWorld", cons: List[_ContactConstraint],
                          invI: Dict[int, np.ndarray]) -> None:
        """Apply each contact's seeded impulse so the solver starts near the solution."""
        for k in cons:
            P = k.c.normal_impulse * k.n + k.c.tangent_impulse[0] * k.t1 \
                + k.c.tangent_impulse[1] * k.t2
            self._apply_impulse(world, invI, k.c.a, k.c.b, k.rA, k.rB, P)

    def _store_cache(self, cons: List[_ContactConstraint]) -> None:
        """Save converged impulses keyed by body pair for next step's warm start."""
        cache: Dict[Tuple[int, int], List[Tuple[np.ndarray, float, float, float]]] = {}
        for k in cons:
            cache.setdefault((k.c.a, k.c.b), []).append(
                (k.c.point.copy(), k.c.normal_impulse,
                 k.c.tangent_impulse[0], k.c.tangent_impulse[1]))
        self._cache = cache

    def _effective_mass(self, world: "PhysicsWorld", invI: Dict[int, np.ndarray],
                        a: int, b: int, rA: np.ndarray, rB: np.ndarray,
                        dir: np.ndarray) -> float:
        """Reciprocal of the pair's inverse mass along ``dir`` (0 for two immovable bodies)."""
        raxn = _cross(rA, dir)
        rbxn = _cross(rB, dir)
        k = world.inv_mass[a] + world.inv_mass[b]
        k += np.dot(raxn, invI[a] @ raxn)
        k += np.dot(rbxn, invI[b] @ rbxn)
        return 1.0 / k if k > 1e-12 else 0.0

    def _rel_vel(self, world: "PhysicsWorld", a: int, b: int, rA: np.ndarray,
                 rB: np.ndarray) -> np.ndarray:
        """Velocity of body ``b``'s contact point relative to body ``a``'s."""
        return (world.linear_velocity[b] + _cross(world.angular_velocity[b], rB)
                - world.linear_velocity[a] - _cross(world.angular_velocity[a], rA))

    # -- solve -----------------------------------------------------------
    def solve(self, world: "PhysicsWorld", contacts: List["Contact"], dt: float) -> None:
        """Resolve ``contacts`` in place: velocity iterations then a position pass.

        Mutates the world's velocities and positions.  Empty ``contacts`` is a no-op.
        """
        if not contacts:
            return
        self._wake_sleepers(world, contacts)
        if _native is not None and hasattr(_native, 'prepare_and_solve'):
            self._solve_native_full(world, contacts)
            return
        cons, invI = self._prepare(world, contacts)
        if _native is not None:
            self._solve_native(world, cons, invI)
        else:
            if self.warm_start:
                self._apply_warm_start(world, cons, invI)
            for _ in range(self.velocity_iterations):
                self._solve_velocity(world, cons, invI)
            self._solve_positions(world, contacts)
        if self.warm_start:
            self._store_cache(cons)

    def _inv_inertia_world_all(self, world: "PhysicsWorld") -> np.ndarray:
        """World inverse-inertia tensor ``R diag(I⁻¹) Rᵀ`` for every body, batched."""
        n = world.body_count
        R = mathutil.quat_to_matrix(world.orientation[:n])          # (n, 3, 3)
        scaled = R * world.inv_inertia[:n][:, None, :]              # scale columns
        return np.ascontiguousarray(np.einsum('nij,nkj->nik', scaled, R))

    def _seed_pair(self, c: "Contact"):
        """Nearest cached ``(nImpulse, t0, t1)`` for this contact's pair, or None."""
        cached = self._cache.get((c.a, c.b))
        if not cached:
            return None
        best, best_d = None, 0.04
        for point, nImp, t0, t1 in cached:
            diff = c.point - point
            d = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            if d < best_d:
                best, best_d = (nImp, t0, t1), d
        return best

    def _solve_native_full(self, world: "PhysicsWorld", contacts: List["Contact"]) -> None:
        """Assemble contact arrays and run prep + velocity + position in the kernel.

        Only the material combine and warm-start seeding stay in Python (both cheap
        per contact); the geometric constraint setup and all iterations are native.
        """
        K = len(contacts)
        a_idx = np.empty(K, dtype=np.intp); b_idx = np.empty(K, dtype=np.intp)
        point = np.empty((K, 3)); normal = np.empty((K, 3)); depth = np.empty(K)
        restitution = np.empty(K); friction = np.empty(K)
        nImp = np.zeros(K); tImp = np.zeros((K, 2))
        warm = self.warm_start
        combine = model.combine
        for i, c in enumerate(contacts):
            a, b = c.a, c.b
            a_idx[i] = a; b_idx[i] = b
            point[i] = c.point; normal[i] = c.normal; depth[i] = c.depth
            mat_a, mat_b = self._material(world, a), self._material(world, b)
            restitution[i] = combine(mat_a.restitution, mat_b.restitution,
                                     mat_a.restitutionCombine)
            override = world.pair_friction_for(world.collider_material[a],
                                               world.collider_material[b])
            friction[i] = (override[1] if override is not None else
                           combine(mat_a.dynamicFriction, mat_b.dynamicFriction,
                                   mat_a.frictionCombine))
            if warm:
                seed = self._seed_pair(c)
                if seed is not None:
                    nImp[i], tImp[i, 0], tImp[i, 1] = seed
        invIw = self._inv_inertia_world_all(world)
        _native.prepare_and_solve(
            a_idx, b_idx, point, normal, depth,
            world.position, world.linear_velocity, world.angular_velocity,
            world.inv_mass, invIw, restitution, friction, nImp, tImp,
            self.restitution_threshold, self.velocity_iterations, warm,
            0.8, self.slop)
        if warm:
            cache: Dict[Tuple[int, int], List[Tuple[np.ndarray, float, float, float]]] = {}
            for i, c in enumerate(contacts):
                c.normal_impulse = nImp[i]
                c.tangent_impulse[0] = tImp[i, 0]; c.tangent_impulse[1] = tImp[i, 1]
                cache.setdefault((c.a, c.b), []).append(
                    (c.point.copy(), nImp[i], tImp[i, 0], tImp[i, 1]))
            self._cache = cache

    def _solve_native(self, world: "PhysicsWorld", cons: List[_ContactConstraint],
                      invI: Dict[int, np.ndarray]) -> None:
        """Run the velocity and position iterations in the Cython kernel.

        Packs the per-contact constraint data into SoA arrays once, then the
        native loops do all ``velocity_iterations`` (+ warm start) and the split-
        impulse position pass without returning to Python. Accumulated impulses
        are written back onto the contacts so the warm-start cache still works.
        """
        K = len(cons)
        a_idx = np.empty(K, dtype=np.intp)
        b_idx = np.empty(K, dtype=np.intp)
        rA = np.empty((K, 3)); rB = np.empty((K, 3)); nrm = np.empty((K, 3))
        t1 = np.empty((K, 3)); t2 = np.empty((K, 3))
        nMass = np.empty(K); t1Mass = np.empty(K); t2Mass = np.empty(K)
        friction = np.empty(K); vBias = np.empty(K); depth = np.empty(K)
        invIa = np.empty((K, 3, 3)); invIb = np.empty((K, 3, 3))
        nImp = np.empty(K); tImp = np.empty((K, 2))
        for i, k in enumerate(cons):
            c = k.c
            a_idx[i] = c.a; b_idx[i] = c.b
            rA[i] = k.rA; rB[i] = k.rB; nrm[i] = k.n; t1[i] = k.t1; t2[i] = k.t2
            nMass[i] = k.nMass; t1Mass[i] = k.t1Mass; t2Mass[i] = k.t2Mass
            friction[i] = k.friction; vBias[i] = k.vBias; depth[i] = c.depth
            invIa[i] = invI[c.a]; invIb[i] = invI[c.b]
            nImp[i] = c.normal_impulse
            tImp[i, 0] = c.tangent_impulse[0]; tImp[i, 1] = c.tangent_impulse[1]
        ima = np.ascontiguousarray(world.inv_mass[a_idx])
        imb = np.ascontiguousarray(world.inv_mass[b_idx])
        _native.solve_velocity(a_idx, b_idx, rA, rB, nrm, t1, t2,
                               nMass, t1Mass, t2Mass, friction, vBias,
                               invIa, invIb, ima, imb, nImp, tImp,
                               world.linear_velocity, world.angular_velocity,
                               self.velocity_iterations, self.warm_start)
        for i, k in enumerate(cons):
            k.c.normal_impulse = nImp[i]
            k.c.tangent_impulse[0] = tImp[i, 0]
            k.c.tangent_impulse[1] = tImp[i, 1]
        _native.solve_positions(a_idx, b_idx, nrm, depth, ima, imb,
                                world.position, 0.8, self.slop)

    def _wake_sleepers(self, world: "PhysicsWorld", contacts: List["Contact"],
                       wake_speed: float = 0.15) -> None:
        """A body wakes a sleeping neighbour only on a real impact (relative speed
        above ``wake_speed``) — a body merely resting against a sleeper leaves it
        asleep, so settled piles stay settled.  Kinematic movers always wake."""
        v2 = np.sum(world.linear_velocity ** 2, axis=1)

        def wakes(i: int) -> bool:
            t = world.motion_type[i]
            if t == 1:
                return True
            return t == 2 and world.awake[i] and v2[i] > wake_speed ** 2
        for c in contacts:
            if not world.awake[c.b] and wakes(c.a):
                world.wake(c.b)
            if not world.awake[c.a] and wakes(c.b):
                world.wake(c.a)

    def _apply_impulse(self, world: "PhysicsWorld", invI: Dict[int, np.ndarray],
                       a: int, b: int, rA: np.ndarray, rB: np.ndarray,
                       P: np.ndarray) -> None:
        """Apply impulse ``P`` to body ``b`` and ``-P`` to body ``a`` (linear and angular)."""
        world.linear_velocity[a] -= world.inv_mass[a] * P
        world.angular_velocity[a] -= _skew_apply(invI[a], rA, P)
        world.linear_velocity[b] += world.inv_mass[b] * P
        world.angular_velocity[b] += _skew_apply(invI[b], rB, P)

    def _solve_velocity(self, world: "PhysicsWorld", cons: List[_ContactConstraint],
                        invI: Dict[int, np.ndarray]) -> None:
        """One velocity iteration: clamp the normal impulse, then Coulomb friction.

        The world velocity/mass arrays are bound once (in-place writes still hit
        the world through the view) and the relative-velocity / impulse math is
        inlined, so this per-contact loop -- run ``velocity_iterations`` times per
        step and the solver's dominant cost -- avoids a property lookup and a
        function call on every one of its inner operations.
        """
        lv = world.linear_velocity          # views: `lv[a] -= ...` mutates the world
        av = world.angular_velocity
        im = world.inv_mass
        for k in cons:
            c = k.c
            a, b = c.a, c.b
            rA, rB, nrm = k.rA, k.rB, k.n
            iIa, iIb, ima, imb = invI[a], invI[b], im[a], im[b]

            dv = lv[b] + _cross(av[b], rB) - lv[a] - _cross(av[a], rA)
            vn = dv[0] * nrm[0] + dv[1] * nrm[1] + dv[2] * nrm[2]
            new_impulse = c.normal_impulse - k.nMass * (vn - k.vBias)
            if new_impulse < 0.0:
                new_impulse = 0.0
            P = (new_impulse - c.normal_impulse) * nrm
            c.normal_impulse = new_impulse
            lv[a] -= ima * P
            av[a] -= iIa @ _cross(rA, P)
            lv[b] += imb * P
            av[b] += iIb @ _cross(rB, P)

            max_friction = k.friction * c.normal_impulse
            dv = lv[b] + _cross(av[b], rB) - lv[a] - _cross(av[a], rA)
            for t, tMass, comp in ((k.t1, k.t1Mass, 0), (k.t2, k.t2Mass, 1)):
                vt = dv[0] * t[0] + dv[1] * t[1] + dv[2] * t[2]
                old = c.tangent_impulse[comp]
                new = old - tMass * vt
                if new > max_friction:
                    new = max_friction
                elif new < -max_friction:
                    new = -max_friction
                c.tangent_impulse[comp] = new
                Pt = (new - old) * t
                lv[a] -= ima * Pt
                av[a] -= iIa @ _cross(rA, Pt)
                lv[b] += imb * Pt
                av[b] += iIb @ _cross(rB, Pt)

    def _solve_positions(self, world: "PhysicsWorld", contacts: List["Contact"],
                         correction: float = 0.8) -> None:
        """One split-impulse pass: push overlapping bodies apart along the normal."""
        for c in contacts:
            a, b = c.a, c.b
            corr = correction * max(c.depth - self.slop, 0.0)
            if corr <= 0:
                continue
            inv_sum = world.inv_mass[a] + world.inv_mass[b]
            if inv_sum <= 1e-12:
                continue
            move = corr / inv_sum * c.normal
            world.position[a] -= world.inv_mass[a] * move
            world.position[b] += world.inv_mass[b] * move


def build_islands(world: "PhysicsWorld", contacts: List["Contact"]) -> List[List["Contact"]]:
    """Union-find over dynamic bodies sharing contacts → list of contact lists.

    Each returned list is an island that can be solved independently; contacts
    touching no dynamic body are gathered into one final island.
    """
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    dyn = world.motion_type == 2
    for c in contacts:
        if dyn[c.a] and dyn[c.b]:
            union(c.a, c.b)
    groups: Dict[int, List["Contact"]] = {}
    singles: List["Contact"] = []
    for c in contacts:
        anchor: Optional[int] = c.a if dyn[c.a] else (c.b if dyn[c.b] else None)
        if anchor is None:
            singles.append(c)
            continue
        groups.setdefault(find(anchor), []).append(c)
    islands = list(groups.values())
    if singles:
        islands.append(singles)
    return islands
