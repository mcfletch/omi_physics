"""``PhysicsWorld`` — the simulation state, columnar, beside the scenegraph.

State is kept in a flat structure-of-arrays (the OMI model in columnar form) and
synced to/from the render tree only at step boundaries.  The arrays are
contiguous numpy, so integration and AABB refit are single array ops over all
awake bodies — fast on the CPU today and GPU-shaped for later (see
``docs/PIPELINE.md``).
"""
from typing import (Any, Callable, Dict, List, Optional, Set, Tuple,
                    TYPE_CHECKING)
import os

import numpy as np

from . import model
from . import mathutil
from .mathutil import Vec
from .backend import NumpyBackend, select_backend

if TYPE_CHECKING:
    from .collide import Contact

_TYPE_INT = {model.STATIC: 0, model.KINEMATIC: 1, model.DYNAMIC: 2}
_STATIC, _KINEMATIC, _DYNAMIC = 0, 1, 2


class PhysicsWorld:
    """A rigid-body world.  Add shapes/materials/filters, then bodies, then step."""

    def __init__(self, gravity: Optional[model.Gravity] = None,
                 fixed_dt: float = 1.0 / 60.0, max_frame: float = 0.25,
                 backend: Any = None, sleep_enabled: bool = True,
                 default_linear_damping: float = 0.0,
                 default_angular_damping: float = 0.0,
                 gpu_threshold: int = 10000) -> None:
        self.gravity = gravity if gravity is not None else model.Gravity()
        self.gravity_volumes: List[Any] = []
        self.fixed_dt = fixed_dt
        self.max_frame = max_frame
        self._init_backend(backend, gpu_threshold)
        self.sleep_enabled = sleep_enabled
        # Baseline resistance applied to every dynamic body on top of its own
        # damping, so no motion persists forever and scenes settle to sleep.
        # Zero on the bare world (deterministic integrator tests); gameplay
        # (PhysicsManager / demos) sets sensible non-zero defaults.
        self.default_linear_damping = default_linear_damping
        self.default_angular_damping = default_angular_damping

        self.shapes: List[model.Shape] = []
        self.materials: List[model.Material] = []
        self.filters: List[model.CollisionFilter] = []
        self.joints: List[model.Joint] = []
        self.joint_attachments: List[model.JointAttach] = []
        # Per-material-pair friction overrides, keyed by the sorted (a, b) index
        # tuple -> (static_mu, dynamic_mu).  Consulted by the solver ahead of the
        # material-combine rule, so a specific pairing (e.g. rubber-on-ice) can be
        # tuned without disturbing any other pairing.
        self.pair_friction: Dict[Tuple[int, int], Tuple[float, float]] = {}

        self._n = 0
        self._cap = 0
        self._accumulator = 0.0
        self.time = 0.0
        self.bodies: List[Any] = []            # optional per-index user handle

        self._alloc(8)
        self._collision: Optional["_CollisionStages"] = None  # built when a collider exists
        self.contacts: List["Contact"] = []
        self.triggers_overlaps: Set[Any] = set()
        self.trigger_listeners: List[Callable[..., None]] = []
        self.joint_constraints: List[Any] = []
        self.joint_iterations = 10

    # -- storage ---------------------------------------------------------
    _VEC = ('position', 'prev_position', 'orientation', 'prev_orientation',
            'linear_velocity', 'angular_velocity', 'inv_inertia',
            'aabb_min', 'aabb_max')
    _SCALAR_F = ('inv_mass', 'mass', 'gravity_factor', 'linear_damping',
                 'angular_damping', 'quadratic_drag', 'sleep_timer',
                 'restitution_pad')
    _SCALAR_I = ('motion_type', 'collider_shape', 'collider_material',
                 'collider_filter', 'trigger_shape', 'island')

    # Columnar backing arrays, built by ``_alloc``/``_grow`` via ``setattr``.
    _position: np.ndarray
    _prev_position: np.ndarray
    _orientation: np.ndarray
    _prev_orientation: np.ndarray
    _linear_velocity: np.ndarray
    _angular_velocity: np.ndarray
    _inv_inertia: np.ndarray
    _aabb_min: np.ndarray
    _aabb_max: np.ndarray
    _inv_mass: np.ndarray
    _mass: np.ndarray
    _gravity_factor: np.ndarray
    _linear_damping: np.ndarray
    _angular_damping: np.ndarray
    _quadratic_drag: np.ndarray
    _sleep_timer: np.ndarray
    _restitution_pad: np.ndarray
    _motion_type: np.ndarray
    _collider_shape: np.ndarray
    _collider_material: np.ndarray
    _collider_filter: np.ndarray
    _trigger_shape: np.ndarray
    _island: np.ndarray
    _awake: np.ndarray

    def _alloc(self, cap: int) -> None:
        """Allocate the columnar arrays for ``cap`` bodies (all fields zeroed)."""
        self._cap = cap
        for name in self._VEC:
            width = 4 if 'orientation' in name else 3
            setattr(self, '_' + name, np.zeros((cap, width), dtype='d'))
        for name in self._SCALAR_F:
            setattr(self, '_' + name, np.zeros(cap, dtype='d'))
        for name in self._SCALAR_I:
            setattr(self, '_' + name, np.full(cap, -1, dtype='i4'))
        self._awake = np.ones(cap, dtype=bool)
        self._orientation[:, 3] = 1.0
        self._prev_orientation[:, 3] = 1.0

    def _grow(self) -> None:
        """Double the array capacity, preserving existing body state."""
        old = self._cap
        new = max(8, old * 2)
        for name in self._VEC + self._SCALAR_F + self._SCALAR_I + ('awake',):
            arr = getattr(self, '_' + name)
            grown = np.zeros((new,) + arr.shape[1:], dtype=arr.dtype)
            if name in self._SCALAR_I:
                grown.fill(-1)
            grown[:old] = arr
            setattr(self, '_' + name, grown)
        self._orientation[old:, 3] = 1.0
        self._prev_orientation[old:, 3] = 1.0
        self._awake[old:] = True
        self._cap = new

    # -- views (only the used prefix) ------------------------------------
    def _view(self, name: str) -> np.ndarray:
        """A view of column ``name`` over the live bodies only (the first ``_n`` rows)."""
        return getattr(self, '_' + name)[:self._n]

    position = property(lambda s: s._view('position'))
    prev_position = property(lambda s: s._view('prev_position'))
    orientation = property(lambda s: s._view('orientation'))
    prev_orientation = property(lambda s: s._view('prev_orientation'))
    linear_velocity = property(lambda s: s._view('linear_velocity'))
    angular_velocity = property(lambda s: s._view('angular_velocity'))
    inv_mass = property(lambda s: s._view('inv_mass'))
    mass = property(lambda s: s._view('mass'))
    inv_inertia = property(lambda s: s._view('inv_inertia'))
    gravity_factor = property(lambda s: s._view('gravity_factor'))
    linear_damping = property(lambda s: s._view('linear_damping'))
    angular_damping = property(lambda s: s._view('angular_damping'))
    quadratic_drag = property(lambda s: s._view('quadratic_drag'))
    motion_type = property(lambda s: s._view('motion_type'))
    collider_shape = property(lambda s: s._view('collider_shape'))
    collider_material = property(lambda s: s._view('collider_material'))
    collider_filter = property(lambda s: s._view('collider_filter'))
    trigger_shape = property(lambda s: s._view('trigger_shape'))
    aabb_min = property(lambda s: s._view('aabb_min'))
    aabb_max = property(lambda s: s._view('aabb_max'))
    awake = property(lambda s: s._view('awake'))
    island = property(lambda s: s._view('island'))
    sleep_timer = property(lambda s: s._view('sleep_timer'))

    @property
    def body_count(self) -> int:
        """The number of live bodies."""
        return self._n

    # -- document tables -------------------------------------------------
    def add_shape(self, shape: model.Shape) -> int:
        """Register a collision shape; return its index for use by colliders/triggers."""
        self.shapes.append(shape)
        return len(self.shapes) - 1

    def add_material(self, mat: model.Material) -> int:
        """Register a physics material; return its index."""
        self.materials.append(mat)
        return len(self.materials) - 1

    def add_filter(self, filt: model.CollisionFilter) -> int:
        """Register a collision filter; return its index."""
        self.filters.append(filt)
        return len(self.filters) - 1

    def add_joint(self, joint: model.Joint) -> int:
        """Register a joint definition; return its index."""
        self.joints.append(joint)
        return len(self.joints) - 1

    def material_for(self, index: int) -> model.Material:
        """The material at ``index``, or the default material if out of range."""
        if 0 <= index < len(self.materials):
            return self.materials[index]
        return model.DEFAULT_MATERIAL

    def filter_for(self, index: int) -> model.CollisionFilter:
        """The filter at ``index``, or the default filter if out of range."""
        if 0 <= index < len(self.filters):
            return self.filters[index]
        return model.DEFAULT_FILTER

    def set_pair_friction(self, material_a: int, material_b: int,
                          static_mu: float, dynamic_mu: float) -> None:
        """Override the contact friction for one ordered-independent material pair.

        ``material_a``/``material_b`` are indices into ``materials`` (as returned
        by :meth:`add_material`).  When two bodies with these materials touch, the
        solver uses ``dynamic_mu`` instead of combining the two materials'
        per-material coefficients.  Symmetric: the pair ``(a, b)`` and ``(b, a)``
        are the same key.
        """
        self.pair_friction[self._pair_key(material_a, material_b)] = (
            float(static_mu), float(dynamic_mu))

    def pair_friction_for(self, material_a: int,
                          material_b: int) -> Optional[Tuple[float, float]]:
        """Return ``(static_mu, dynamic_mu)`` for the pair, or ``None`` if unset."""
        return self.pair_friction.get(self._pair_key(material_a, material_b))

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        """The order-independent key for a material pair (smaller index first)."""
        return (a, b) if a <= b else (b, a)

    # -- game-facing impulses -------------------------------------------
    def apply_impulse(self, i: int, impulse: Vec) -> None:
        """Add a linear impulse ``J`` (N·s) to body ``i``: ``v += J / m``.

        A no-op for static/kinematic bodies (``inv_mass == 0``).  Wakes the body
        so a resting marble responds to steering, boosts, and spring traps.
        """
        self._linear_velocity[i] += self._inv_mass[i] * np.asarray(impulse, dtype='d')
        self.wake(i)

    def apply_angular_impulse(self, i: int, impulse: Vec) -> None:
        """Add an angular impulse ``L`` (N·m·s) to body ``i``: ``w += I⁻¹ L``.

        ``I⁻¹`` is the world-space inverse inertia (the body's local diagonal
        rotated by its current orientation), so the spin is applied about world
        axes.  A no-op for static/kinematic bodies.  Wakes the body.
        """
        R = mathutil.quat_to_matrix(self._orientation[i])
        inv_inertia_world = (R * self._inv_inertia[i]) @ R.T
        self._angular_velocity[i] += inv_inertia_world @ np.asarray(impulse, dtype='d')
        self.wake(i)

    # -- bodies ----------------------------------------------------------
    def add_body(self, motion: Optional[model.Motion] = None,
                 collider: Optional[model.Collider] = None,
                 trigger: Optional[model.Trigger] = None,
                 position: Vec = (0, 0, 0),
                 orientation: Vec = (0, 0, 0, 1),
                 handle: Any = None) -> int:
        """Create a body from its motion/collider/trigger; return its index.

        ``handle`` is an opaque per-body value the caller can retrieve later (the
        scenegraph node the body drives, for instance).  Grows the columnar arrays
        if needed.
        """
        if motion is None:
            motion = model.Motion()
        if self._n >= self._cap:
            self._grow()
        i = self._n
        self._n += 1

        self._position[i] = position
        self._prev_position[i] = position
        self._orientation[i] = orientation
        self._prev_orientation[i] = orientation
        self._linear_velocity[i] = motion.linearVelocity
        self._angular_velocity[i] = motion.angularVelocity
        self._motion_type[i] = _TYPE_INT[motion.type]
        self._gravity_factor[i] = motion.gravityFactor
        self._linear_damping[i] = motion.linearDamping
        self._angular_damping[i] = motion.angularDamping
        self._quadratic_drag[i] = motion.quadraticDrag
        self._awake[i] = True
        self._sleep_timer[i] = 0.0

        shape: Optional[model.Shape] = None
        if collider is not None and 0 <= collider.shape < len(self.shapes):
            shape = self.shapes[collider.shape]
            self._collider_shape[i] = collider.shape
            self._collider_material[i] = collider.physicsMaterial
            self._collider_filter[i] = collider.collisionFilter
            self._collision = self._collision or _CollisionStages()
        if trigger is not None and 0 <= trigger.shape < len(self.shapes):
            self._trigger_shape[i] = trigger.shape
            self._collision = self._collision or _CollisionStages()

        self._set_mass_properties(i, motion, shape)
        self.bodies.append(handle)
        return i

    def _set_mass_properties(self, i: int, motion: model.Motion,
                             shape: Optional[model.Shape]) -> None:
        """Fill body ``i``'s mass and inverse inertia; static/kinematic bodies get zero."""
        if motion.type != model.DYNAMIC:
            self._inv_mass[i] = 0.0
            self._mass[i] = 0.0
            self._inv_inertia[i] = 0.0
            return
        m = max(motion.mass, 1e-9)
        self._mass[i] = m
        self._inv_mass[i] = 1.0 / m
        diag = np.asarray(motion.inertiaDiagonal, dtype='d')
        if not diag.any():
            diag = inertia_diagonal(shape, m) if shape is not None else np.zeros(3)
        with np.errstate(divide='ignore'):
            self._inv_inertia[i] = np.where(diag > 0, 1.0 / np.maximum(diag, 1e-12), 0.0)

    # -- backend policy --------------------------------------------------
    def _init_backend(self, backend: Any, gpu_threshold: int) -> None:
        """Choose the compute backend.

        An explicit backend object is used as-is.  Otherwise the mode
        (``OPENGLCONTEXT_PHYSICS_BACKEND`` or the ``backend`` string, default
        ``auto``) decides: ``numpy``/``gpu`` are fixed, while ``auto`` runs on
        numpy and hands off to the GPU compute backend only once the awake-body
        count crosses ``gpu_threshold`` — below it numpy is faster, since GPU
        transfer overhead outweighs the tiny per-body integrate (see
        ``glcompute.py``).  The GPU backend is built lazily on the first crossing
        (needs a live GL context) and cached.
        """
        self.gpu_threshold = gpu_threshold
        self._cpu_backend: Any = None
        self._gpu_backend: Any = None
        self._gpu_failed = False
        if backend is not None and not isinstance(backend, str):
            self._backend_mode = 'fixed'
            self.backend: Any = backend
            return
        mode = backend or os.environ.get('OPENGLCONTEXT_PHYSICS_BACKEND', 'auto')
        self._cpu_backend = NumpyBackend()
        if mode in ('numpy', 'cpu'):
            self._backend_mode = 'fixed'
            self.backend = self._cpu_backend
        elif mode in ('gpu', 'glcompute'):
            self._backend_mode = 'fixed'
            self.backend = select_backend('gpu')       # raises if unavailable
            self._gpu_backend = self.backend
        else:
            self._backend_mode = 'auto'
            self.backend = self._cpu_backend           # upgraded past threshold

    def _select_auto_backend(self) -> None:
        """In ``auto`` mode, swap CPU↔GPU backend by awake-body count (with hysteresis)."""
        awake = int(np.count_nonzero(self.dynamic_mask()))
        if self.backend is self._cpu_backend:
            if awake >= self.gpu_threshold:
                gpu = self._ensure_gpu_backend()
                if gpu is not None:
                    self.backend = gpu
        elif awake < int(self.gpu_threshold * 0.8):     # hysteresis, avoid thrash
            self.backend = self._cpu_backend

    def _ensure_gpu_backend(self) -> Any:
        """Build and cache the GPU backend, or ``None`` if it can't be created."""
        if self._gpu_backend is None and not self._gpu_failed:
            try:
                self._gpu_backend = select_backend('gpu')
            except Exception:
                self._gpu_failed = True                 # no context/compute
        return self._gpu_backend

    # -- masks & gravity -------------------------------------------------
    def dynamic_mask(self) -> np.ndarray:
        """Boolean mask of the awake dynamic bodies."""
        return (self.motion_type == _DYNAMIC) & self.awake

    def moving_mask(self) -> np.ndarray:
        """Boolean mask of the awake dynamic or kinematic bodies."""
        return ((self.motion_type == _DYNAMIC) | (self.motion_type == _KINEMATIC)) & self.awake

    def is_awake_mover(self, i: int) -> bool:
        """True if body ``i`` is an awake dynamic body or any kinematic body."""
        t = self._motion_type[i]
        return (t == _DYNAMIC and self._awake[i]) or t == _KINEMATIC

    def resolve_gravity(self) -> np.ndarray:
        """Per-body gravity acceleration ``(N, 3)``, including any gravity volumes."""
        base = np.asarray(self.gravity.direction, dtype='d')
        norm = np.linalg.norm(base)
        base = base / norm if norm else base
        g = base * self.gravity.gravity
        out = np.tile(g, (self._n, 1))
        if self.gravity_volumes:
            from .gravity import apply_volumes
            apply_volumes(self.gravity_volumes, self.position, out)
        return out

    # -- the step --------------------------------------------------------
    def advance(self, real_dt: float) -> float:
        """Fixed-timestep accumulator: advance by real elapsed time.

        Returns the interpolation alpha for the render writeback.
        """
        self._accumulator = min(self._accumulator + real_dt, self.max_frame)
        while self._accumulator >= self.fixed_dt:
            self.step(self.fixed_dt)
            self._accumulator -= self.fixed_dt
        return self._accumulator / self.fixed_dt

    def step(self, dt: float) -> None:
        """Advance the whole world by one fixed step ``dt``: forces, collision, joints, sleep."""
        if self._backend_mode == 'auto':
            self._select_auto_backend()
        self._prev_position[:self._n] = self.position
        self._prev_orientation[:self._n] = self.orientation

        self.backend.integrate_forces(self, dt)

        if self._collision is not None:
            self.backend.refit_aabbs(self)
            self._collision.run(self, dt)

        if self.joint_constraints:
            self._solve_joints(dt)

        self.backend.integrate_positions(self, dt)

        if self.sleep_enabled:
            self._update_sleep(dt)
        self.time += dt

    def writeback(self, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolated poses for rendering: lerp prev→current by ``alpha``."""
        pos = self.prev_position * (1 - alpha) + self.position * alpha
        quat = mathutil.quat_normalize(
            self.prev_orientation * (1 - alpha) + self.orientation * alpha)
        return pos, quat

    # -- AABBs & sleeping ------------------------------------------------
    #: refit kinds -- 0 no shape, 1 sphere, 2 box, 3 other (per-body fallback)
    _RFIT_NONE, _RFIT_SPHERE, _RFIT_BOX, _RFIT_OTHER = 0, 1, 2, 3

    def _build_refit_cache(self) -> None:
        """Per-body shape kind and local half-extents for the vectorized AABB refit.

        Shapes are immutable once created, so this only rebuilds when the body
        count changes (a wave added/removed) -- not every step.
        """
        n = self._n
        kind = np.zeros(n, dtype=np.int8)
        half = np.zeros((n, 3), dtype='d')
        for i in range(n):
            si = int(self._collider_shape[i])
            ti = int(self._trigger_shape[i])
            idx = si if si >= 0 else ti
            if idx < 0:
                continue
            sh = self.shapes[idx]
            if sh.type == 'sphere':
                kind[i] = self._RFIT_SPHERE
                half[i] = sh.radius
            elif sh.type == 'box':
                kind[i] = self._RFIT_BOX
                half[i] = 0.5 * np.asarray(sh.size, dtype='d')
            else:
                kind[i] = self._RFIT_OTHER
        self._refit_kind = kind
        self._refit_half = half
        self._refit_cache_n = n

    def refit_aabbs(self, margin: float = 0.05) -> None:
        """Recompute each body's world AABB from its shape and pose, expanded by ``margin``.

        Boxes and spheres -- the overwhelming majority -- are batched: a box's
        world half-extent is ``|R| @ half`` (equivalent to the min/max over its
        eight rotated corners), a sphere's is its radius on every axis. Rarer
        shapes (capsule/cylinder/convex/trimesh) fall back to the exact per-body
        proxy AABB.
        """
        n = self._n
        if n == 0:
            return
        if getattr(self, '_refit_cache_n', -1) != n:
            self._build_refit_cache()
        kind = self._refit_kind
        half = self._refit_half
        pos = self._position[:n]
        absR = np.abs(mathutil.quat_to_matrix(self._orientation[:n]))
        hw = np.einsum('nij,nj->ni', absR, half)      # rotated box half-extents
        sph = kind == self._RFIT_SPHERE
        if sph.any():
            hw[sph] = half[sph]                        # sphere: isotropic, unrotated
        self._aabb_min[:n] = pos - hw - margin
        self._aabb_max[:n] = pos + hw + margin
        none = np.where(kind == self._RFIT_NONE)[0]
        for i in none:                                 # point AABB, no margin
            self._aabb_min[i] = self._position[i]
            self._aabb_max[i] = self._position[i]
        other = np.where(kind == self._RFIT_OTHER)[0]
        if len(other):
            from .body import world_aabb
            for i in other:
                si = int(self._collider_shape[i])
                idx = si if si >= 0 else int(self._trigger_shape[i])
                lo, hi = world_aabb(self.shapes[idx], self._position[i],
                                    self._orientation[i])
                self._aabb_min[i] = lo - margin
                self._aabb_max[i] = hi + margin

    def _update_sleep(self, dt: float, lin_thresh: float = 0.08,
                      ang_thresh: float = 0.25, t_sleep: float = 0.4) -> None:
        """Island-aware sleeping: a group of touching bodies sleeps only when
        *all* of them have been below the threshold long enough, so one jittering
        body can't keep the whole pile awake (and can't be re-woken by a resting
        neighbour — only a real impact wakes it, see the solver)."""
        dyn = self.motion_type == _DYNAMIC
        speed2 = np.sum(self.linear_velocity ** 2, axis=1)
        spin2 = np.sum(self.angular_velocity ** 2, axis=1)
        slow = dyn & (speed2 < lin_thresh ** 2) & (spin2 < ang_thresh ** 2)
        self._sleep_timer[:self._n][slow] += dt
        self._sleep_timer[:self._n][~slow] = 0.0
        ready = slow & (self.sleep_timer > t_sleep)

        to_sleep = np.zeros(self._n, dtype=bool)
        islands, contacted = self._contact_islands(dyn)
        for members in islands:
            if all(ready[m] for m in members):
                for m in members:
                    to_sleep[m] = True
        loners = np.where(dyn & ready)[0]
        for i in loners:
            if i not in contacted:
                to_sleep[i] = True

        self._awake[:self._n][to_sleep] = False
        self._awake[:self._n][dyn & ~slow] = True   # anything moving is awake
        self._linear_velocity[:self._n][~self.awake & dyn] = 0.0
        self._angular_velocity[:self._n][~self.awake & dyn] = 0.0

    def _contact_islands(self, dyn: np.ndarray) -> Tuple[List[List[int]], Set[int]]:
        """Union-find the touching dynamic bodies into islands.

        Returns the list of islands (each a list of body indices) and the set of
        dynamic bodies that are in contact with anything.
        """
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        contacted: Set[int] = set()
        for c in self.contacts:
            a, b = c.a, c.b
            if dyn[a]:
                contacted.add(a)
            if dyn[b]:
                contacted.add(b)
            if dyn[a] and dyn[b]:
                parent[find(a)] = find(b)
        groups: Dict[int, List[int]] = {}
        for i in contacted:
            groups.setdefault(find(i), []).append(i)
        return list(groups.values()), contacted

    def wake(self, i: int) -> None:
        """Wake body ``i`` and reset its sleep timer."""
        self._awake[i] = True
        self._sleep_timer[i] = 0.0

    # -- triggers & gravity zones ---------------------------------------
    def add_gravity_volume(self, volume: Any) -> Any:
        """Register a gravity zone; return it."""
        self.gravity_volumes.append(volume)
        return volume

    def add_trigger_listener(self, fn: Callable[..., None]) -> None:
        """Register ``fn(event_type, trigger_index, other_index)``."""
        self.trigger_listeners.append(fn)

    def dispatch_trigger(self, event_type: str, trigger_index: int,
                         other_index: int) -> None:
        """Notify every registered trigger listener of one overlap event."""
        for fn in self.trigger_listeners:
            fn(event_type, trigger_index, other_index)

    def add_joint_constraint(self, constraint: Any) -> Any:
        """Register a joint constraint, waking the bodies it touches; return it."""
        self.joint_constraints.append(constraint)
        for i in (getattr(constraint, 'a', None), getattr(constraint, 'b', None),
                  getattr(constraint, 'body', None)):
            if i is not None and i >= 0:
                self.wake(i)
        return constraint

    def _solve_joints(self, dt: float) -> None:
        """Prepare and iteratively solve every joint constraint, waking their bodies."""
        for c in self.joint_constraints:
            i = getattr(c, 'body', None)
            if i is not None and i >= 0:
                self.wake(i)
            for i in (getattr(c, 'a', -1), getattr(c, 'b', -1)):
                if i >= 0:
                    self.wake(i)
            c.prepare(self)
        for _ in range(self.joint_iterations):
            for c in self.joint_constraints:
                c.solve(self, dt)


def inertia_diagonal(shape: Optional[model.Shape], mass: float) -> np.ndarray:
    """Solid-body inertia diagonal for a primitive shape about its centre."""
    if shape is None:
        return np.zeros(3)
    t = shape.type
    if t == 'sphere':
        v = 0.4 * mass * shape.radius ** 2
        return np.array([v, v, v])
    if t == 'box':
        x, y, z = shape.size
        k = mass / 12.0
        return k * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
    if t in ('capsule', 'cylinder'):
        r = shape.radiusBottom
        h = shape.height
        ix = mass * (3 * r * r + h * h) / 12.0
        iy = 0.5 * mass * r * r
        return np.array([ix, iy, ix])
    # convex / trimesh: approximate with the AABB box of the points
    if shape.points is not None and len(shape.points):
        lo = shape.points.min(axis=0)
        hi = shape.points.max(axis=0)
        x, y, z = hi - lo
        k = mass / 12.0
        return k * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
    return np.zeros(3)


class _CollisionStages:
    """Broad phase → narrow phase → contact solve → trigger events, per step."""

    def __init__(self, solver: Any = None, velocity_iterations: int = 10) -> None:
        """Build the broad/narrow phases, the contact solver, and the trigger system."""
        from .broadphase import BroadPhase
        from .narrowphase import NarrowPhase
        from .solver import SequentialImpulseSolver
        from .triggers import TriggerSystem
        self.broadphase = BroadPhase()
        self.narrowphase = NarrowPhase()
        self.solver = solver or SequentialImpulseSolver(
            velocity_iterations=velocity_iterations)
        self.triggers = TriggerSystem()
        self.contacts: List["Contact"] = []

    def run(self, world: "PhysicsWorld", dt: float) -> None:
        """Run one collision pass: build pairs, solve solid contacts, fire triggers."""
        pairs = self.broadphase.pairs(world)
        solid: List[Tuple[int, int]] = []
        has_trigger = False
        for i, j in pairs:
            if world.trigger_shape[i] >= 0 or world.trigger_shape[j] >= 0:
                has_trigger = True
            elif world.collider_shape[i] >= 0 and world.collider_shape[j] >= 0:
                # A pair at rest (both asleep/static) needs no work; it is revived
                # when a mover touches one of them (solver wakes the sleeper).
                if world.is_awake_mover(i) or world.is_awake_mover(j):
                    solid.append((i, j))
        self.contacts = self.narrowphase.generate(world, solid)
        world.contacts = self.contacts
        self.solver.solve(world, self.contacts, dt)
        if has_trigger or self.triggers.overlaps:
            for event in self.triggers.update(world, pairs):
                world.dispatch_trigger(*event)
