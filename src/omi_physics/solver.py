"""Contact solver — sequential impulses (projected Gauss-Seidel).

Removes approach velocity along the contact normal (with restitution), applies
Coulomb friction on two tangents, and corrects penetration with a split-impulse
(NGS) position pass so restitution is not polluted by Baumgarte energy (Catto).
Material friction/restitution combine per the OMI combine modes.  Contacts
partition into islands so disjoint groups solve independently.
"""
from typing import TYPE_CHECKING, Any

import numpy as np

from . import mathutil, model
from ._accel import accelerators_disabled
from .contactevents import ContactBatch, relative_motion, world_tangent

#: The compiled accelerator, or None where it was not built or was switched
#: off. Declared, because what is called on it lives in the extension module
#: rather than in anything a checker can read.
_native: Any
if accelerators_disabled():
    _native = None
else:
    try:
        from . import _solver_native as _native  # type: ignore[no-redef,attr-defined]
    except ImportError:                            # pragma: no cover - pure-Python fallback
        _native = None

if TYPE_CHECKING:
    from .collide import Contact
    from .world import PhysicsWorld


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


def _basis(n: np.ndarray, direction: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal tangents spanning the plane perpendicular to unit normal ``n``.

    The first lies along ``direction`` taken across the normal, where one is
    given and is not along the normal itself.
    """
    if direction is not None:
        across = direction - float(np.dot(direction, n)) * n
        length = float(np.sqrt(across[0] * across[0] + across[1] * across[1]
                               + across[2] * across[2]))
        if length > 1e-9:
            t1 = across / length
            return t1, _cross(n, t1)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = mathutil.normalize(_cross(a, n))
    t2 = _cross(n, t1)
    return t1, t2


class _Terms:
    """What each contact of a solve is to do, as arrays a row per contact.

    From the pair's :class:`~omi_physics.model.ContactParameters` where a
    pair rule gave it some, and from the bodies' materials and the solver's
    own settings where not.
    """
    __slots__ = ('restitution', 'threshold', 'friction', 'direction', 'surface',
                 'softness', 'correction')

    def __init__(self, count: int) -> None:
        self.restitution = np.empty(count)
        self.threshold = np.empty(count)
        #: The two friction coefficients.
        self.friction = np.empty((count, 2))
        #: The first friction direction, or zero for the solver's own choice.
        self.direction = np.zeros((count, 3))
        #: How fast the second body is carried across the first, per direction.
        self.surface = np.zeros((count, 2))
        #: Softness of the normal row and of each friction row, divided by the step.
        self.softness = np.zeros((count, 3))
        self.correction = np.empty(count)


class _ContactConstraint:
    """Solver scratch for one contact: contact arms, normal/tangent basis, and masses."""
    __slots__ = ('c', 'rA', 'rB', 'n', 't1', 't2', 'nMass', 't1Mass', 't2Mass',
                 'restitution', 'friction', 'friction2', 'vBias', 'surface', 'softness')

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
    friction2: float
    vBias: float
    surface: tuple[float, float]
    softness: tuple[float, float, float]

    def __init__(self, contact: 'Contact') -> None:
        self.c = contact


class SequentialImpulseSolver:
    """Projected Gauss-Seidel contact solver: normal, friction, then position pass."""

    def __init__(self, velocity_iterations: int = 10, slop: float = 0.005,
                 restitution_threshold: float = 0.5,
                 warm_start: bool = True,
                 max_correction: float = 0.2, correction: float = 0.8,
                 softness: float = 0.0) -> None:
        self.velocity_iterations = velocity_iterations
        #: How far, in metres, a pair may overlap before the position pass
        #: pushes it apart.
        self.slop = slop
        #: The closing speed, in metres a second, below which nothing bounces.
        self.restitution_threshold = restitution_threshold
        self.warm_start = warm_start
        #: The furthest, in metres, the position pass will push a pair apart in
        #: one step.  A body that arrives deep inside another -- placed there,
        #: or driven there at speed -- then climbs out over a few frames rather
        #: than being flung out of the level in one.
        self.max_correction = max_correction
        #: The fraction of an overlap the position pass takes out each step.
        self.correction = correction
        #: Constraint-force mixing of every contact a pair rule says nothing
        #: about, in metres per newton-second: zero is a hard contact.
        self.softness = softness
        # (a,b) -> [(point, nImpulse, t0, t1)] carried to the next step for warm start
        self._cache: dict[tuple[int, int], list[tuple[np.ndarray, float, float, float]]] = {}
        #: The last solve's contacts as arrays, impulses included, where the
        #: compiled path built them anyway; None where it did not. The contact
        #: tracker reads it rather than gathering the same values again.
        self.last_batch: ContactBatch | None = None
        #: The world's :attr:`~omi_physics.world.PhysicsWorld.step_count` on
        #: the step :attr:`last_batch` was built for.
        self.last_batch_step: int | None = None

    def forget(self, body: int) -> None:
        """Drop the warm-start impulses of every pair ``body`` was in.

        Called when the body is removed, so the next body in its slot does not
        start from impulses that belonged to the last one.
        """
        self._cache = {pair: cached for pair, cached in self._cache.items()
                       if body not in pair}

    # -- setup -----------------------------------------------------------
    def _inv_inertia_world(self, world: "PhysicsWorld", i: int) -> np.ndarray:
        """World-space inverse inertia tensor of body ``i`` (rotate the body-space diagonal)."""
        R = mathutil.quat_to_matrix(world.orientation[i])
        return (R * world.inv_inertia[i]) @ R.T

    def _material(self, world: "PhysicsWorld", i: int) -> model.Material:
        """Physics material of body ``i`` (the default material when it has none)."""
        return world.material_for(world.collider_material[i])

    def _terms(self, world: "PhysicsWorld", contacts: list["Contact"], dt: float) -> _Terms:
        """Each contact's bounce, friction, give and correction (:class:`_Terms`)."""
        terms = _Terms(len(contacts))
        combine = model.combine
        softness = self.softness / dt if dt > 0.0 else 0.0
        rules = world.pair_parameters
        for i, c in enumerate(contacts):
            a, b = c.a, c.b
            found = rules.get((a, b)) if rules else None
            if found is None and rules:
                found = rules.get((b, a))
                flipped = True
            else:
                flipped = False
            if found is None:
                mat_a, mat_b = self._material(world, a), self._material(world, b)
                terms.restitution[i] = combine(mat_a.restitution, mat_b.restitution,
                                               mat_a.restitutionCombine)
                override = world.pair_friction_for(world.collider_material[a],
                                                   world.collider_material[b])
                mu = (override[1] if override is not None else
                      combine(mat_a.dynamicFriction, mat_b.dynamicFriction,
                              mat_a.frictionCombine))
                terms.friction[i, 0] = mu
                terms.friction[i, 1] = mu
                terms.threshold[i] = self.restitution_threshold
                terms.softness[i, 0] = softness
                terms.correction[i] = self.correction
                continue
            terms.restitution[i] = found.restitution
            terms.threshold[i] = (self.restitution_threshold if found.bounce_speed is None
                                  else found.bounce_speed)
            terms.friction[i] = found.frictions
            if found.friction_direction is not None:
                terms.direction[i] = found.friction_direction
            sign = -1.0 if flipped else 1.0
            terms.surface[i] = (sign * found.surface_speed[0], sign * found.surface_speed[1])
            if dt > 0.0:
                give = self.softness if found.softness is None else found.softness
                terms.softness[i] = (give / dt, found.slip[0] / dt, found.slip[1] / dt)
            terms.correction[i] = (self.correction if found.error_correction is None
                                   else found.error_correction)
        return terms

    def _prepare(self, world: "PhysicsWorld", contacts: list["Contact"], terms: _Terms,
                 ) -> tuple[list[_ContactConstraint], dict[int, np.ndarray]]:
        """Build a constraint per contact and cache each body's world inverse inertia.

        Returns the constraints and the ``body index -> inverse inertia`` map the
        velocity iterations reuse.  Seeds warm-start impulses from the last step,
        and records how hard each pair met, which is readable here and nowhere
        after (:attr:`~omi_physics.collide.Contact.approach`).
        """
        cons = []
        invI: dict[int, np.ndarray] = {}
        for index, c in enumerate(contacts):
            a, b = c.a, c.b
            if a not in invI:
                invI[a] = self._inv_inertia_world(world, a)
            if b not in invI:
                invI[b] = self._inv_inertia_world(world, b)
            k = _ContactConstraint(c)
            k.rA = c.point - world.position[a]
            k.rB = c.point - world.position[b]
            k.n = c.normal
            direction = terms.direction[index]
            k.t1, k.t2 = _basis(c.normal, direction if direction.any() else None)
            c.tangent = k.t1
            soft_n, soft_1, soft_2 = (float(value) for value in terms.softness[index])
            k.softness = (soft_n, soft_1, soft_2)
            k.nMass = self._effective_mass(world, invI, a, b, k.rA, k.rB, k.n, soft_n)
            k.t1Mass = self._effective_mass(world, invI, a, b, k.rA, k.rB, k.t1, soft_1)
            k.t2Mass = self._effective_mass(world, invI, a, b, k.rA, k.rB, k.t2, soft_2)
            k.restitution = float(terms.restitution[index])
            k.friction = float(terms.friction[index, 0])
            k.friction2 = float(terms.friction[index, 1])
            k.surface = (float(terms.surface[index, 0]), float(terms.surface[index, 1]))
            # How hard they met, which restitution is a fraction of and
            # impact_on answers from. Taken before anything is solved because
            # the iterations below are precisely the removal of it.
            relative = (world.linear_velocity[b] + _cross(world.angular_velocity[b], k.rB)
                        - world.linear_velocity[a] - _cross(world.angular_velocity[a], k.rA))
            along = float(np.dot(relative, k.n))
            c.approach = -along
            c.slip = float(np.linalg.norm(relative - along * k.n))
            k.vBias = (k.restitution * c.approach
                       if c.approach > float(terms.threshold[index]) else 0.0)
            if self.warm_start:
                self._seed_from_cache(k)
            cons.append(k)
        return cons, invI

    def _seed_from_cache(self, k: _ContactConstraint) -> None:
        """Copy the nearest cached impulses into contact ``k`` so it resumes converged."""
        cached = self._cache.get((k.c.a, k.c.b))
        if not cached:
            return
        best: tuple[float, float, float] | None = None
        best_d = 0.04                       # (0.2 m)² match tolerance
        for point, nImp, t0, t1 in cached:
            d = float(np.dot(k.c.point - point, k.c.point - point))
            if d < best_d:
                best, best_d = (nImp, t0, t1), d
        if best is not None:
            k.c.normal_impulse, k.c.tangent_impulse[0], k.c.tangent_impulse[1] = best

    def _apply_warm_start(self, world: "PhysicsWorld", cons: list[_ContactConstraint],
                          invI: dict[int, np.ndarray]) -> None:
        """Apply each contact's seeded impulse so the solver starts near the solution."""
        for k in cons:
            P = k.c.normal_impulse * k.n + k.c.tangent_impulse[0] * k.t1 \
                + k.c.tangent_impulse[1] * k.t2
            self._apply_impulse(world, invI, k.c.a, k.c.b, k.rA, k.rB, P)

    def _store_cache(self, cons: list[_ContactConstraint]) -> None:
        """Save converged impulses keyed by body pair for next step's warm start."""
        cache: dict[tuple[int, int], list[tuple[np.ndarray, float, float, float]]] = {}
        for k in cons:
            cache.setdefault((k.c.a, k.c.b), []).append(
                (k.c.point.copy(), k.c.normal_impulse,
                 k.c.tangent_impulse[0], k.c.tangent_impulse[1]))
        self._cache = cache

    def _effective_mass(self, world: "PhysicsWorld", invI: dict[int, np.ndarray],
                        a: int, b: int, rA: np.ndarray, rB: np.ndarray,
                        dir: np.ndarray, softness: float = 0.0) -> float:
        """Reciprocal of the pair's inverse mass along ``dir``, with ``softness`` added.

        Zero for two immovable bodies and a hard row.
        """
        raxn = _cross(rA, dir)
        rbxn = _cross(rB, dir)
        k = world.inv_mass[a] + world.inv_mass[b]
        k += np.dot(raxn, invI[a] @ raxn)
        k += np.dot(rbxn, invI[b] @ rbxn)
        k += softness
        return 1.0 / k if k > 1e-12 else 0.0

    # -- solve -----------------------------------------------------------
    def solve(self, world: "PhysicsWorld", contacts: list["Contact"],
              dt: float) -> None:
        """Resolve ``contacts`` in place: velocity iterations then a position pass.

        Mutates the world's velocities and positions.  Empty ``contacts`` is a no-op.
        ``dt`` turns a contact's softness, a constraint-force mixing, into the
        give of its rows; the position pass is a split impulse rather than a
        Baumgarte bias, so how far it moves a pair apart does not depend on
        how long the step was.
        """
        self.last_batch = None
        if not contacts:
            return
        contacts = self._static_last(world, contacts)
        self._wake_sleepers(world, contacts)
        terms = self._terms(world, contacts, dt)
        if _native is not None:
            self._solve_native_full(world, contacts, terms)
            return
        cons, invI = self._prepare(world, contacts, terms)
        if self.warm_start:
            self._apply_warm_start(world, cons, invI)
        for _ in range(self.velocity_iterations):
            self._solve_velocity(world, cons, invI)
        self._solve_positions(world, contacts, terms.correction)
        if self.warm_start:
            self._store_cache(cons)

    @staticmethod
    def _static_last(world: "PhysicsWorld",
                     contacts: list["Contact"]) -> list["Contact"]:
        """``contacts`` reordered so those against a static body come last.

        Both passes here are Gauss-Seidel, and what one corrects last is what
        ends up satisfied.  Where a body is held between two contacts that ask
        opposite things, only one of them can be, and it has to be the one
        against the static body.

        A tenth of a gramme under a stack is the case that shows it.  The floor
        pushes it up and the box above pushes it down, and both corrections
        move only the light body, since only it has any inverse mass to speak
        of; with the floor corrected first, the box above undoes it and the
        light body sinks a little every step.  A body left overlapping another
        *dynamic* body is pushed apart over the next few steps and no harm is
        done -- but a body left inside the level's own geometry is thrown out
        of the far side of it the moment its centre passes the middle, because
        the nearest separating axis is then the wrong one.  It fell to
        y = -43 in ``test_a_crushed_body_does_not_go_through_the_floor``.

        A stable sort, so contacts otherwise keep the order the narrow phase
        found them in and a step stays reproducible.
        """
        return sorted(contacts, key=lambda c: (world.inv_mass[c.a] == 0.0
                                               or world.inv_mass[c.b] == 0.0))

    def _inv_inertia_world_all(self, world: "PhysicsWorld") -> np.ndarray:
        """World inverse-inertia tensor ``R diag(I⁻¹) Rᵀ`` for every body, batched."""
        n = world.body_count
        R = mathutil.quat_to_matrix(world.orientation[:n])          # (n, 3, 3)
        scaled = R * world.inv_inertia[:n][:, None, :]              # scale columns
        return np.ascontiguousarray(np.einsum('nij,nkj->nik', scaled, R))

    def _seed_pair(self, c: "Contact") -> tuple[float, float, float] | None:
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

    def _solve_native_full(self, world: "PhysicsWorld", contacts: list["Contact"],
                           terms: _Terms) -> None:
        """Assemble contact arrays and run prep + velocity + position in the kernel.

        Only the warm-start seeding and the record of how hard each pair met
        stay in Python (both cheap per contact); the geometric constraint setup
        and all iterations are native.
        """
        K = len(contacts)
        a_idx = np.empty(K, dtype=np.intp); b_idx = np.empty(K, dtype=np.intp)
        point = np.empty((K, 3)); normal = np.empty((K, 3)); depth = np.empty(K)
        nImp = np.zeros(K); tImp = np.zeros((K, 2)); first = np.empty((K, 3))
        warm = self.warm_start
        for i, c in enumerate(contacts):
            a_idx[i] = c.a; b_idx[i] = c.b
            point[i] = c.point; normal[i] = c.normal; depth[i] = c.depth
            if warm:
                seed = self._seed_pair(c)
                if seed is not None:
                    nImp[i], tImp[i, 0], tImp[i, 1] = seed
        # How hard each pair met, off the arrays just assembled and before the
        # kernel resolves anything: the iterations it runs are the removal of
        # exactly this velocity, so there is no reading it afterwards.
        approach, slip = relative_motion(world, a_idx, b_idx, point, normal)
        for c, closing, sliding in zip(contacts, approach.tolist(), slip.tolist(),
                                       strict=True):
            c.approach = closing
            c.slip = sliding
        invIw = self._inv_inertia_world_all(world)
        _native.prepare_and_solve(
            a_idx, b_idx, point, normal, depth,
            world.position, world.linear_velocity, world.angular_velocity,
            world.inv_mass, invIw, terms.restitution, terms.threshold,
            np.ascontiguousarray(terms.friction), terms.direction, terms.surface,
            terms.softness, terms.correction, nImp, tImp, first,
            self.velocity_iterations, warm, self.slop, self.max_correction)
        cache: dict[tuple[int, int], list[tuple[np.ndarray, float, float, float]]] = {}
        for i, c in enumerate(contacts):
            c.normal_impulse = float(nImp[i])
            c.tangent_impulse[0] = tImp[i, 0]; c.tangent_impulse[1] = tImp[i, 1]
            c.tangent = first[i]
            if warm:
                cache.setdefault((c.a, c.b), []).append(
                    (c.point.copy(), nImp[i], tImp[i, 0], tImp[i, 1]))
        if warm:
            self._cache = cache
        self.last_batch = ContactBatch(
            a_idx, b_idx, point, normal, depth, approach, slip, nImp,
            world_tangent(normal, tImp, first), np.ones(K, dtype=bool))
        self.last_batch_step = world.step_count

    def _wake_sleepers(self, world: "PhysicsWorld", contacts: list["Contact"],
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

    def _apply_impulse(self, world: "PhysicsWorld", invI: dict[int, np.ndarray],
                       a: int, b: int, rA: np.ndarray, rB: np.ndarray,
                       P: np.ndarray) -> None:
        """Apply impulse ``P`` to body ``b`` and ``-P`` to body ``a`` (linear and angular)."""
        world.linear_velocity[a] -= world.inv_mass[a] * P
        world.angular_velocity[a] -= _skew_apply(invI[a], rA, P)
        world.linear_velocity[b] += world.inv_mass[b] * P
        world.angular_velocity[b] += _skew_apply(invI[b], rB, P)

    def _solve_velocity(self, world: "PhysicsWorld", cons: list[_ContactConstraint],
                        invI: dict[int, np.ndarray]) -> None:
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
            soft_n, soft_1, soft_2 = k.softness

            dv = lv[b] + _cross(av[b], rB) - lv[a] - _cross(av[a], rA)
            vn = dv[0] * nrm[0] + dv[1] * nrm[1] + dv[2] * nrm[2]
            new_impulse = c.normal_impulse - k.nMass * (vn - k.vBias
                                                        + soft_n * c.normal_impulse)
            if new_impulse < 0.0:
                new_impulse = 0.0
            P = (new_impulse - c.normal_impulse) * nrm
            c.normal_impulse = new_impulse
            lv[a] -= ima * P
            av[a] -= iIa @ _cross(rA, P)
            lv[b] += imb * P
            av[b] += iIb @ _cross(rB, P)

            dv = lv[b] + _cross(av[b], rB) - lv[a] - _cross(av[a], rA)
            for t, tMass, comp, mu, speed, soft in (
                    (k.t1, k.t1Mass, 0, k.friction, k.surface[0], soft_1),
                    (k.t2, k.t2Mass, 1, k.friction2, k.surface[1], soft_2)):
                max_friction = mu * c.normal_impulse
                vt = dv[0] * t[0] + dv[1] * t[1] + dv[2] * t[2]
                old = c.tangent_impulse[comp]
                new = old - tMass * (vt - speed + soft * old)
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

    def _solve_positions(self, world: "PhysicsWorld", contacts: list["Contact"],
                         correction: np.ndarray) -> None:
        """One split-impulse pass: push overlapping bodies apart along the normal.

        Each contact is corrected against what the pass has **already** done,
        not against the overlap the narrow phase measured.  A box resting on a
        floor meets it at four points that all report the same overlap; four
        full corrections push it four times as far as it is in, the pair comes
        back the next step further out of place than it started, and the
        overshoot grows from there rather than settling.  Subtracting the
        movement already applied leaves the deepest point of a manifold in
        charge and the rest of it with little left to do.

        ``correction`` is the fraction of each contact's overlap taken out.
        The correction is bounded by :attr:`max_correction` as well, so a body
        that arrives deep inside another comes out over a few frames.
        """
        moved: dict[int, np.ndarray] = {}
        unmoved = np.zeros(3)
        for index, c in enumerate(contacts):
            a, b = c.a, c.b
            done = float(np.dot(moved.get(b, unmoved) - moved.get(a, unmoved), c.normal))
            corr = float(correction[index]) * max(c.depth - done - self.slop, 0.0)
            if corr <= 0:
                continue
            inv_sum = world.inv_mass[a] + world.inv_mass[b]
            if inv_sum <= 1e-12:
                continue
            move = min(corr, self.max_correction) / inv_sum * c.normal
            world.position[a] -= world.inv_mass[a] * move
            world.position[b] += world.inv_mass[b] * move
            moved[a] = moved.get(a, unmoved) - world.inv_mass[a] * move
            moved[b] = moved.get(b, unmoved) + world.inv_mass[b] * move


def build_islands(world: "PhysicsWorld", contacts: list["Contact"]) -> list[list["Contact"]]:
    """Union-find over dynamic bodies sharing contacts → list of contact lists.

    Each returned list is an island that can be solved independently; contacts
    touching no dynamic body are gathered into one final island.
    """
    parent: dict[int, int] = {}

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
    groups: dict[int, list[Contact]] = {}
    singles: list[Contact] = []
    for c in contacts:
        anchor: int | None = c.a if dyn[c.a] else (c.b if dyn[c.b] else None)
        if anchor is None:
            singles.append(c)
            continue
        groups.setdefault(find(anchor), []).append(c)
    islands = list(groups.values())
    if singles:
        islands.append(singles)
    return islands
