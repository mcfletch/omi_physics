"""``PhysicsWorld`` — the simulation state, columnar, beside the scenegraph.

State is kept in a flat structure-of-arrays (the OMI model in columnar form) and
synced to/from the render tree only at step boundaries.  The arrays are
contiguous numpy, so integration and AABB refit are single array ops over all
awake bodies — fast on the CPU today and GPU-shaped for later (see
``docs/PIPELINE.md``).
"""
import logging
import os
from collections import deque
from collections.abc import Callable, Container, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from . import jointrows, mathutil, model
from .backend import NumpyBackend, select_backend
from .body import world_aabb
from .broadphase import BroadPhase
from .contactevents import (
    REPORTING_LEVELS,
    BodyRef,
    ContactBatch,
    ContactEvent,
    ContactLog,
    ContactTracker,
    PairPreview,
    Reporting,
    TriggerEvent,
    measure,
)
from .gravity import apply_volumes
from .jointrows import RowJoint
from .massprops import inertia_diagonal
from .mathutil import Vec
from .narrowphase import NarrowPhase
from .solver import SequentialImpulseSolver
from .triggers import TriggerSystem

if TYPE_CHECKING:
    from .collide import Contact

log = logging.getLogger(__name__)

_TYPE_INT = {model.STATIC: 0, model.KINEMATIC: 1, model.DYNAMIC: 2}
_STATIC, _KINEMATIC, _DYNAMIC = 0, 1, 2



def _column(name: str) -> property:
    """A read-only property answering column ``name`` over the live bodies."""
    def rows(self: 'PhysicsWorld') -> np.ndarray:
        return self._view(name)
    return property(rows)

class PhysicsWorld:
    """A rigid-body world.  Add shapes/materials/filters, then bodies, then step."""

    def __init__(self, gravity: model.Gravity | None = None,
                 fixed_dt: float = 1.0 / 60.0, max_frame: float = 0.25,
                 backend: Any = None, sleep_enabled: bool = True,
                 default_linear_damping: float = 0.0,
                 default_angular_damping: float = 0.0,
                 gpu_threshold: int = 10000) -> None:
        self.gravity = gravity if gravity is not None else model.Gravity()
        self.gravity_volumes: list[Any] = []
        #: Exact AABBs of the shapes that need a proxy to measure, and the pose
        #: each was measured at, so a body that has not moved is not measured
        #: again -- nor even asked. See :meth:`_refit_measured`.
        self._aabb_cache: dict[int, tuple] = {}
        self._refit_lo: np.ndarray = np.zeros((0, 3), dtype='d')
        self._refit_hi: np.ndarray = np.zeros((0, 3), dtype='d')
        self._refit_pose: np.ndarray = np.zeros((0, 3), dtype='d')
        self._refit_quat: np.ndarray = np.zeros((0, 4), dtype='d')
        self.fixed_dt = fixed_dt
        self.max_frame = max_frame
        self._init_backend(backend, gpu_threshold)
        self.sleep_enabled = sleep_enabled
        #: A dynamic body slower than this, in metres a second, and turning
        #: slower than :attr:`sleep_angular_speed` radians a second, for
        #: :attr:`sleep_time` seconds, sleeps (:meth:`set_sleep` for one body).
        self.sleep_linear_speed = 0.08
        self.sleep_angular_speed = 0.25
        self.sleep_time = 0.4
        # Baseline resistance applied to every dynamic body on top of its own
        # damping, so no motion persists forever and scenes settle to sleep.
        # Zero on the bare world (deterministic integrator tests); gameplay
        # (PhysicsManager / demos) sets sensible non-zero defaults.
        self.default_linear_damping = default_linear_damping
        self.default_angular_damping = default_angular_damping

        self.shapes: list[model.Shape] = []
        self.materials: list[model.Material] = []
        self.filters: list[model.CollisionFilter] = []
        self.joints: list[model.Joint] = []
        self.joint_attachments: list[model.JointAttach] = []
        # Per-material-pair friction overrides, keyed by the sorted (a, b) index
        # tuple -> (static_mu, dynamic_mu).  Consulted by the solver ahead of the
        # material-combine rule, so a specific pairing (e.g. rubber-on-ice) can be
        # tuned without disturbing any other pairing.
        self.pair_friction: dict[tuple[int, int], tuple[float, float]] = {}
        #: Asked of each pair that might collide, each step (:meth:`set_pair_rule`).
        self._pair_rule: Callable[[int, int], Any] | None = None
        #: What the pair rule said this step about the pairs it gave
        #: parameters for, keyed by the pair as it was asked.
        self.pair_parameters: dict[tuple[int, int], model.ContactParameters] = {}

        self._n = 0
        self._cap = 0
        #: Slots whose body has been removed, ready for the next one. A
        #: streaming world adds and removes a body per tile for as long as it
        #: runs, and without reuse the arrays only ever grow.
        self._free: list[int] = []
        self._accumulator = 0.0
        self.time = 0.0
        self.bodies: list[Any] = []            # optional per-index user handle

        self._alloc(8)
        self._collision: _CollisionStages | None = None  # built when a collider exists
        self.contacts: list[Contact] = []
        self.triggers_overlaps: set[Any] = set()
        self.trigger_listeners: list[Callable[..., None]] = []
        #: Steps run so far; the ``step`` every event carries.
        self.step_count = 0
        self._reporting: Reporting = 'off'
        #: Whether ``'persist'`` contact events and ``'stay'`` trigger events
        #: are recorded. Off, a touching pair is still tracked, and only its
        #: ``'begin'`` and ``'end'`` are reported.
        self.report_persist = False
        #: Contact and trigger events of every step, in order, until drained.
        #: Filled while :attr:`contact_reporting` is not ``'off'`` and
        #: :attr:`log_events` is on.
        self.contact_log = ContactLog()
        #: Whether events are kept in :attr:`contact_log`. A world whose events
        #: go only to its listeners turns it off, and the log stays empty.
        self.log_events = True
        self.contact_listeners: list[Callable[[ContactEvent], None]] = []
        self._tracker = ContactTracker()
        self._contact_filter: Callable[[PairPreview], Any] | None = None
        self._undelivered: deque[ContactEvent] = deque()
        self._delivering = False
        #: The first exception a listener raised during the current step or
        #: removal, raised once that has finished.
        self._listener_error: Exception | None = None
        self._listener_depth = 0
        self.joint_constraints: list[Any] = []
        #: The joints of :attr:`joint_constraints` built from rows.
        self.row_joints: list[RowJoint] = []
        self.joint_iterations = 10

    # -- storage ---------------------------------------------------------
    _VEC = ('position', 'prev_position', 'orientation', 'prev_orientation',
            'linear_velocity', 'angular_velocity', 'inv_inertia',
            'aabb_min', 'aabb_max', 'force', 'torque')
    _SCALAR_F = ('inv_mass', 'mass', 'gravity_factor', 'linear_damping',
                 'angular_damping', 'quadratic_drag', 'sleep_timer',
                 'restitution_pad', 'sleep_linear', 'sleep_angular', 'sleep_after')
    _SCALAR_I = ('motion_type', 'collider_shape', 'collider_material',
                 'collider_filter', 'trigger_shape', 'trigger_filter', 'island')
    #: The remaining per-body columns, each with a type of its own.
    _SLOT = ('awake', 'generation', 'report', 'sensor', 'vacant', 'exact_rotation')

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
    _force: np.ndarray
    _torque: np.ndarray
    _sleep_linear: np.ndarray
    _sleep_angular: np.ndarray
    _sleep_after: np.ndarray
    _exact_rotation: np.ndarray
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
    _trigger_filter: np.ndarray
    _island: np.ndarray
    _awake: np.ndarray
    _generation: np.ndarray
    _report: np.ndarray
    _sensor: np.ndarray
    _vacant: np.ndarray

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
        self._generation = np.zeros(cap, dtype=np.int64)
        self._report = np.zeros(cap, dtype=bool)
        self._sensor = np.zeros(cap, dtype=bool)
        #: True for a slot whose body has been removed and not yet replaced.
        self._vacant = np.zeros(cap, dtype=bool)
        #: True for a body whose turning is integrated exactly (:meth:`set_exact_rotation`).
        self._exact_rotation = np.zeros(cap, dtype=bool)
        # Below zero: the world's own thresholds (:meth:`set_sleep`).
        self._sleep_linear[:] = -1.0
        self._sleep_angular[:] = -1.0
        self._sleep_after[:] = -1.0
        self._orientation[:, 3] = 1.0
        self._prev_orientation[:, 3] = 1.0

    def _grow(self) -> None:
        """Double the array capacity, preserving existing body state."""
        old = self._cap
        new = max(8, old * 2)
        for name in self._VEC + self._SCALAR_F + self._SCALAR_I + self._SLOT:
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

    position = _column('position')
    prev_position = _column('prev_position')
    orientation = _column('orientation')
    prev_orientation = _column('prev_orientation')
    linear_velocity = _column('linear_velocity')
    angular_velocity = _column('angular_velocity')
    inv_mass = _column('inv_mass')
    mass = _column('mass')
    inv_inertia = _column('inv_inertia')
    gravity_factor = _column('gravity_factor')
    linear_damping = _column('linear_damping')
    angular_damping = _column('angular_damping')
    quadratic_drag = _column('quadratic_drag')
    motion_type = _column('motion_type')
    collider_shape = _column('collider_shape')
    collider_material = _column('collider_material')
    collider_filter = _column('collider_filter')
    trigger_shape = _column('trigger_shape')
    trigger_filter = _column('trigger_filter')
    aabb_min = _column('aabb_min')
    aabb_max = _column('aabb_max')
    awake = _column('awake')
    island = _column('island')
    sleep_timer = _column('sleep_timer')
    #: The steady force and torque on each body (:meth:`set_force`, :meth:`set_torque`).
    force = _column('force')
    torque = _column('torque')

    @property
    def body_count(self) -> int:
        """How many rows the columnar arrays hold.

        What anything walking the columns wants -- the solver, the broadphase,
        a backend uploading them. An emptied slot keeps its row, because other
        code holds body indices as handles and compacting would renumber every
        body above it, so this is not the number of bodies that are *there*:
        see :attr:`live_body_count`.
        """
        return self._n

    @property
    def live_body_count(self) -> int:
        """How many of the rows hold a body.

        What a caller streaming a world in and out asks when it wants to know
        whether what it added has come out again.
        """
        return self._n - len(self._free)

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
                          material_b: int) -> tuple[float, float] | None:
        """Return ``(static_mu, dynamic_mu)`` for the pair, or ``None`` if unset."""
        return self.pair_friction.get(self._pair_key(material_a, material_b))

    def set_pair_rule(self, rule: 'Callable[[int, int], Any] | None') -> None:
        """Ask ``rule(a, b)`` whether, and how, two bodies collide, each step they might.

        The rule is asked about every pair the broad phase finds that the
        world would otherwise solve, once a step, before the narrow phase.
        It answers False to let the pair pass through each other, None to
        leave the bodies' materials in charge, or a
        :class:`~omi_physics.model.ContactParameters` saying how they meet.
        A surface speed in the parameters carries ``b`` across ``a``. None
        removes the rule.
        """
        self._pair_rule = rule
        if rule is None:
            self.pair_parameters = {}

    @property
    def contact_solver(self) -> Any:
        """The contact solver; its iterations, slop and corrections are settable."""
        self._collision = self._collision or _CollisionStages()
        return self._collision.solver

    @staticmethod
    def _pair_key(a: int, b: int) -> tuple[int, int]:
        """The order-independent key for a material pair (smaller index first)."""
        return (a, b) if a <= b else (b, a)

    # -- game-facing impacts ---------------------------------------------
    def impact_on(self, i: int, above: float = 0.0, skip_static: bool = False,
                  among: Container[int] | None = None
                  ) -> tuple[int, float] | None:
        """The heaviest blow body ``i`` took in the step just run.

        Returns ``(other body, closing speed)`` -- how fast the two were coming
        together along the contact normal at the moment they met, in metres per
        second -- or ``None`` if nothing struck it that hard.

        What a game asks to decide damage, a crash, an impact sound or a jolt of
        force feedback.  Read it after :meth:`step` and before the next one:
        contacts are rebuilt every step, and the speed on them is the one taken
        before the solve, since resolving a contact is exactly cancelling the
        velocity that measures it.  A caller sampling once a frame rather than
        once a step sees only the last step of the frame, and an impact resolved
        in an earlier one has already been answered for.  Such a caller wants
        :meth:`add_contact_listener` or :attr:`contact_log`, which record every
        step: a :class:`~omi_physics.contactevents.ContactEvent`'s ``approach``
        is this same speed.

        ``above`` is the speed below which a contact is not a blow.  **A body
        held against something is closing on it slightly on every step**: over
        one step gravity gives it ``g·dt`` to be taken away again, so a box
        resting on the floor at 120 Hz reads about 0.08 m/s for as long as it
        sits there.  Anything telling a landing from a rest wants ``above`` set
        past that; anything about a horizontal contact, which gravity does not
        push along, can leave it at zero.

        ``skip_static`` ignores blows from static bodies, for a caller that
        cares what it hit rather than what it landed on -- the ground is a
        contact on every step of every fall.

        ``among`` narrows the answer to a set of bodies, for a caller with one
        question rather than a general one -- "did I hit one of *these*".
        Without it the heaviest blow is the only answer, and a heavier one from
        something the caller does not care about hides the one it does: a car
        that clips a bollard and a rival in the same step is told about the
        bollard and drives on.
        """
        found: tuple[int, float] | None = None
        floor = max(float(above), 0.0)
        for contact in self.contacts:
            if contact.a == i:
                other = contact.b
            elif contact.b == i:
                other = contact.a
            else:
                continue
            if skip_static and self._motion_type[other] == _STATIC:
                continue
            if among is not None and other not in among:
                continue
            if contact.approach > floor and (found is None
                                             or contact.approach > found[1]):
                found = (int(other), float(contact.approach))
        return found

    # -- contact events --------------------------------------------------
    @property
    def contact_reporting(self) -> Reporting:
        """Which pairs' contacts are recorded as events: ``'off'``, ``'flagged'`` or ``'all'``.

        ``'off'``, the default, records nothing and runs nothing. ``'flagged'``
        records the pairs in which at least one body has been flagged with
        :meth:`report_contacts`. ``'all'`` records every pair, and costs a
        sort of the step's contacts plus an object per event delivered.
        Adding a contact listener switches ``'off'`` to ``'all'``.
        """
        return self._reporting

    @contact_reporting.setter
    def contact_reporting(self, level: Reporting) -> None:
        if level not in REPORTING_LEVELS:
            raise ValueError('contact_reporting is one of %s, not %r'
                             % (', '.join(REPORTING_LEVELS), level))
        if level == 'off':
            self._tracker.reset()
        self._reporting = level

    def is_sensor(self, i: int) -> bool:
        """Whether body ``i``'s collider only senses.

        A sensor enters trigger volumes, and its touches with dynamic bodies
        are reported as contact events with ``solved`` False, but nothing is
        ever solved against it: it pushes nothing and nothing pushes it. Rays
        pass through it. A walking character's body is one
        (:class:`~omi_physics.character.CharacterController`), since the
        controller moves it and collides it itself.
        """
        return bool(self._sensor[i])

    def report_contacts(self, i: int, on: bool = True) -> None:
        """Flag body ``i`` so its pairs are recorded under ``'flagged'`` reporting."""
        self._report[i] = on

    def add_contact_listener(self, fn: Callable[[ContactEvent], None]
                             ) -> Callable[[ContactEvent], None]:
        """Call ``fn(event)`` with each contact event, inside the step; return ``fn``.

        ``event`` is a :class:`~omi_physics.contactevents.ContactEvent`.

        ``fn`` runs inside :meth:`step`, after the solve, on whichever thread
        is stepping, so it can act before the next step: remove a projectile
        that struck something, wake a mechanism. It may read and change the
        world, including removing bodies; the ``'end'`` events a removal
        produces are delivered after the current one. It must not touch
        anything owned by another thread, such as a scenegraph or a GL object.
        Anything for the main thread belongs in :attr:`contact_log` instead.

        Switches :attr:`contact_reporting` from ``'off'`` to ``'all'``, since a
        listener on a world that records nothing would never be called.
        """
        self.contact_listeners.append(fn)
        if self._reporting == 'off':
            self.contact_reporting = 'all'
        return fn

    def remove_contact_listener(self, fn: Callable[[ContactEvent], None]) -> None:
        """Stop calling ``fn``; a listener that was never added is ignored."""
        if fn in self.contact_listeners:
            self.contact_listeners.remove(fn)

    def set_contact_filter(self, fn: Callable[[PairPreview], Any] | None) -> None:
        """Ask ``fn(pair)`` whether to solve a reported pair, on the step it would begin.

        ``fn`` receives a :class:`~omi_physics.contactevents.PairPreview` --
        the pair, how hard it is about to meet and its reduced mass -- and
        returns a :class:`~omi_physics.contactevents.Verdict`: ``SOLVE`` (or
        None), ``IGNORE_STEP`` to be asked again on the next step, or
        ``IGNORE_PAIR`` to let the two pass through each other until they
        part. A glass pane that breaks under the blow ignores the pair, so the
        ball carries on with its momentum rather than bouncing off first.

        Asked only about pairs :attr:`contact_reporting` records, and only
        once per pair: an ignored pair still produces its ``'begin'`` and
        ``'end'``, with ``solved`` False. It runs inside :meth:`step`, under
        the same rules as a contact listener. None removes the filter.
        """
        self._contact_filter = fn

    def ref(self, i: int) -> BodyRef:
        """A :class:`~omi_physics.contactevents.BodyRef` to the body now at index ``i``."""
        return BodyRef(int(i), int(self._generation[i]))

    def alive(self, ref: BodyRef) -> bool:
        """Whether the body ``ref`` names is still in the world."""
        i = ref.index
        return (0 <= i < self._n and int(self._generation[i]) == ref.generation
                and not self._vacant[i])

    def handle_of(self, ref: BodyRef) -> Any:
        """The ``handle`` given to :meth:`add_body` for ``ref``; None once it is removed."""
        if not self.alive(ref) or ref.index >= len(self.bodies):
            return None
        return self.bodies[ref.index]

    def reports_pair(self, a: int, b: int) -> bool:
        """Whether :attr:`contact_reporting` records the pair ``(a, b)``."""
        return self._reporting == 'all' or bool(self._report[a] or self._report[b])

    def _record_contacts(self, solver: Any, contacts: 'list[Contact]',
                         excluded: 'list[Contact]') -> None:
        """Run the contact tracker over the step just solved, and deliver its events."""
        batch = getattr(solver, 'last_batch', None)
        if batch is None or getattr(solver, 'last_batch_step', None) != self.step_count:
            batch = ContactBatch.from_contacts(contacts)
        if excluded:
            batch = batch.concatenate(ContactBatch.from_contacts(excluded, solved=False))
        flags = None
        if self._reporting == 'flagged':
            flags = self._report
            keep = flags[batch.a] | flags[batch.b]
            if not keep.all():
                batch = ContactBatch(*(column[keep] for column in batch))
        self._deliver(self._tracker.update(self, batch, flags))

    def report_contact_events(self, events: list[ContactEvent]) -> None:
        """Log and deliver contact events the step did not produce itself.

        For a body something other than the solver collides: a character
        controller reports its capsule meeting the static world through this,
        so its touches reach the same listeners and log as every other.
        A listener that raises is raised from here once every listener has
        been called.
        """
        with self._listener_errors_raised_after():
            self._deliver(events)

    def _deliver(self, events: list[ContactEvent]) -> None:
        """Log ``events`` and call the contact listeners with each, in order.

        A listener that removes a body produces more events while this is
        running. They are queued behind the ones already being delivered, so
        every listener sees one event at a time and in step order.
        """
        if not events:
            return
        if self.log_events:
            self.contact_log.extend(events)
        if not self.contact_listeners:
            return
        self._undelivered.extend(events)
        if self._delivering:
            return
        self._delivering = True
        try:
            while self._undelivered:
                event = self._undelivered.popleft()
                self._call_listeners(self.contact_listeners, event)
        except BaseException:
            self._undelivered.clear()
            raise
        finally:
            self._delivering = False

    def _call_listeners(self, listeners: list[Callable[..., None]],
                        *args: Any) -> None:
        """Call each of ``listeners`` with ``args``, holding back what they raise.

        The first exception is kept for :meth:`_listener_errors_raised_after`
        to raise; any after it in the same step are logged.
        """
        for fn in list(listeners):
            try:
                fn(*args)
            except Exception as exc:
                if self._listener_error is None:
                    self._listener_error = exc
                else:
                    log.exception('Listener %r also raised', fn)

    @contextmanager
    def _listener_errors_raised_after(self) -> Iterator[None]:
        """Run the block to the end, then raise the first error a listener raised.

        A listener runs part-way through a step or a removal. Raising from it
        there would leave the world half-way through the change, so the error
        waits until the outermost such block has finished. An exception the
        block raises itself propagates at once, and a held listener error is
        then logged.
        """
        self._listener_depth += 1
        try:
            yield
        except BaseException:
            if self._listener_depth == 1 and self._listener_error is not None:
                log.error(  # noqa: TRY400 logs the held listener error, not the one handled here
                    'Listener raised during a failed step',
                    exc_info=self._listener_error)
                self._listener_error = None
            raise
        finally:
            self._listener_depth -= 1
        if self._listener_depth == 0 and self._listener_error is not None:
            failure, self._listener_error = self._listener_error, None
            raise failure

    def _log_triggers(self, fired: list[tuple[str, int, int]]) -> None:
        """Record this step's trigger events in :attr:`contact_log`."""
        if self._reporting == 'off' or not fired or not self.log_events:
            return
        flagged = self._reporting == 'flagged'
        self.contact_log.extend(
            TriggerEvent(phase, self.ref(trigger), self.ref(other),
                         self.step_count, self.time, self._position[other].copy())
            for phase, trigger, other in fired
            if (phase != 'stay' or self.report_persist)
            and (not flagged or self._report[trigger] or self._report[other]))

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

    def set_force(self, i: int, force: Vec) -> None:
        """Push body ``i`` with ``force`` newtons through its centre, every step until changed.

        A steady thrust, wind or a tow; zero takes it away. A static or
        kinematic body is not moved by it. Wakes the body.
        """
        self._force[i] = np.asarray(force, dtype='d')
        self.wake(i)

    def set_torque(self, i: int, torque: Vec) -> None:
        """Turn body ``i`` by ``torque`` newton-metres about world axes, each step until changed."""
        self._torque[i] = np.asarray(torque, dtype='d')
        self.wake(i)

    def set_exact_rotation(self, i: int, exact: bool = True) -> None:
        """Integrate body ``i``'s turning exactly rather than to first order.

        The first-order step turns a body a little less than its spin asks
        each step, which a wheel or a propeller spinning many times a second
        shows as a lag; the exact step turns it through the whole angle.
        """
        self._exact_rotation[i] = bool(exact)

    def set_sleep(self, i: int, linear: float | None = None, angular: float | None = None,
                  after: float | None = None) -> None:
        """Body ``i``'s own sleep: below ``linear`` m/s and ``angular`` rad/s for ``after`` seconds.

        None leaves that threshold to the world's (:attr:`sleep_linear_speed`,
        :attr:`sleep_angular_speed`, :attr:`sleep_time`); an infinite
        ``after`` keeps the body awake.
        """
        self._sleep_linear[i] = -1.0 if linear is None else float(linear)
        self._sleep_angular[i] = -1.0 if angular is None else float(angular)
        self._sleep_after[i] = -1.0 if after is None else float(after)

    def _apply_steady_forces(self, dt: float) -> None:
        """Add a step of each dynamic body's steady force and torque to its velocities."""
        n = self._n
        pushed = (np.any(self._force[:n] != 0.0, axis=1)
                  | np.any(self._torque[:n] != 0.0, axis=1))
        pushed &= self._motion_type[:n] == _DYNAMIC
        if not pushed.any():
            return
        rows = np.nonzero(pushed)[0]
        self._awake[rows] = True
        self._sleep_timer[rows] = 0.0
        self._linear_velocity[rows] += (self._inv_mass[rows, None] * self._force[rows]) * dt
        rotation = mathutil.quat_to_matrix(self._orientation[rows])
        local = np.einsum('nji,nj->ni', rotation, self._torque[rows])
        self._angular_velocity[rows] += np.einsum(
            'nij,nj->ni', rotation, self._inv_inertia[rows] * local) * dt

    def _turn_exactly(self, turned: np.ndarray, before: np.ndarray, dt: float) -> None:
        """Set the orientations of ``turned`` bodies from ``before`` by their whole spin."""
        spin = self._angular_velocity[turned]
        rate = np.linalg.norm(spin, axis=1)
        angle = rate * dt
        axis = np.divide(spin, rate[:, None], out=np.zeros_like(spin),
                         where=rate[:, None] > 1e-12)
        half = 0.5 * angle
        step = np.concatenate((axis * np.sin(half)[:, None], np.cos(half)[:, None]), axis=1)
        self._orientation[turned] = mathutil.quat_normalize(mathutil.quat_mul(step, before))

    # -- bodies ----------------------------------------------------------
    def add_body(self, motion: model.Motion | None = None,
                 collider: model.Collider | None = None,
                 trigger: model.Trigger | None = None,
                 position: Vec = (0, 0, 0),
                 orientation: Vec = (0, 0, 0, 1),
                 handle: Any = None, report_contacts: bool = False,
                 sensor: bool = False) -> int:
        """Create a body from its motion/collider/trigger; return its index.

        ``handle`` is an opaque per-body value the caller can retrieve later (the
        scenegraph node the body drives, for instance).  ``report_contacts``
        flags it for ``'flagged'`` contact reporting (:meth:`report_contacts`).
        ``sensor`` makes its collider a sensor (:meth:`is_sensor`).
        Grows the columnar arrays if needed.
        """
        if motion is None:
            motion = model.Motion()
        if self._free:
            i = self._free.pop()
            self._reset_body(i)
            self._vacant[i] = False
        else:
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
        self._report[i] = report_contacts
        self._sensor[i] = sensor
        self._force[i] = 0.0
        self._torque[i] = 0.0
        self._exact_rotation[i] = False
        self._sleep_linear[i] = -1.0
        self._sleep_angular[i] = -1.0
        self._sleep_after[i] = -1.0

        shape: model.Shape | None = None
        if collider is not None and 0 <= collider.shape < len(self.shapes):
            shape = self.shapes[collider.shape]
            self._collider_shape[i] = collider.shape
            self._collider_material[i] = collider.physicsMaterial
            self._collider_filter[i] = collider.collisionFilter
            self._collision = self._collision or _CollisionStages()
        if trigger is not None and 0 <= trigger.shape < len(self.shapes):
            self._trigger_shape[i] = trigger.shape
            self._trigger_filter[i] = trigger.collisionFilter
            self._collision = self._collision or _CollisionStages()

        self._set_mass_properties(i, motion, shape)
        while len(self.bodies) <= i:
            self.bodies.append(None)
        self.bodies[i] = handle
        self._fit_body(i)
        return i

    def _fit_body(self, i: int, margin: float = 0.05) -> None:
        """Give one body its world AABB, without refitting the rest.

        A body arrives with a box, so a world that has been built but never
        stepped can still be culled against and cast at. The alternative --
        boxes that are all zero until the first step -- makes a query before
        that step quietly answer nothing.
        """
        shape_index = int(self._collider_shape[i])
        if shape_index < 0:
            shape_index = int(self._trigger_shape[i])
        if shape_index < 0:
            self._aabb_min[i] = self._position[i]
            self._aabb_max[i] = self._position[i]
            return
        lo, hi = world_aabb(self.shapes[shape_index], self._position[i],
                            self._orientation[i])
        self._aabb_min[i] = lo - margin
        self._aabb_max[i] = hi + margin

    def place_body(self, i: int, position: Vec | None = None,
                   orientation: Vec | None = None,
                   margin: float = 0.05) -> None:
        """Put body ``i`` where the caller says it is, between steps.

        For a body the caller moves itself rather than one the solver moves: an
        animated platform, a lift, a character proxy, a body staged for one
        query and taken away again.

        A bare write to :attr:`position` is not enough on its own. The arrays
        are views, so nothing sees the write, and a body's **world AABB** is
        what the broadphase rejects against -- a ray, an overlap query or a
        contact pair against a body whose box is still where it used to be is
        answered about where it used to be. This writes the pose and refits
        that one box, which is what keeps the cost of moving a handful of
        bodies proportional to the handful rather than to the world.

        The previous pose is set with the new one, so the body **arrives**
        rather than travelling: an interpolating writeback would otherwise
        smear it across the gap, and a solver reading a velocity from the two
        would read the jump as one.

        Either argument may be left out, which moves or turns the body without
        touching the other.
        """
        if position is not None:
            self._position[i] = np.asarray(position, dtype='d')
            self._prev_position[i] = self._position[i]
        if orientation is not None:
            self._orientation[i] = np.asarray(orientation, dtype='d')
            self._prev_orientation[i] = self._orientation[i]
        self._fit_body(i, margin)

    def remove_body(self, i: int) -> None:
        """Take body ``i`` out of the world, freeing its slot for the next one.

        What a streaming world does with the ground behind it. The body stops
        colliding, stops moving and stops being found by a ray, and the slot is
        handed to the next :meth:`add_body`.

        **The index is not renumbered, and it is not yours afterwards.** Other
        code holds body indices as handles; compacting the arrays would
        renumber every body above this one and invalidate all of those at once,
        so the slot is emptied in place. A caller that keeps using the index
        after removing it is talking about whatever body was put there next.
        A :class:`~omi_physics.contactevents.BodyRef` taken with :meth:`ref`
        is not: removal moves the slot to its next generation.

        Every pair the body was touching ends, with ``reason='removed'``, and
        every trigger it was inside reports an ``'exit'``, now rather than on
        the next step. A listener that raises on one of those is raised from
        here once the body is out of the world, as :meth:`step` does.
        """
        if not 0 <= i < self._n:
            raise IndexError("no body %r to remove" % (i,))
        if self._vacant[i]:
            return
        with self._listener_errors_raised_after():
            self._retire(i)
            self._generation[i] += 1
            self._reset_body(i)
            self._vacant[i] = True
            self._free.append(i)
            if i < len(self.bodies):
                self.bodies[i] = None
            # Anything cached against this slot is about a body that is gone.
            # Left behind it is both a leak -- a world that streams for an hour
            # keeps every mesh it ever held -- and, once the slot is handed out
            # again, an answer about the wrong geometry.
            self._forget_body(i)

    def _retire(self, i: int) -> None:
        """End body ``i``'s contacts and trigger overlaps, and drop its warm start."""
        if self._collision is None:
            return
        forget = getattr(self._collision.solver, 'forget', None)
        if forget is not None:
            forget(i)
        overlaps = self._collision.triggers.overlaps
        inside = sorted(pair for pair in overlaps if i in pair)
        if inside:
            overlaps.difference_update(inside)
            exits = [('exit', trigger, other) for trigger, other in inside]
            self._log_triggers(exits)
            for event in exits:
                self.dispatch_trigger(*event)
        if self._reporting != 'off':
            self._deliver(self._tracker.retire(self, i))

    def _forget_body(self, i: int) -> None:
        """Drop per-body caches other modules keep on the world."""
        meshes = getattr(self, '_raycast_meshes', None)
        if meshes is not None:
            # A compound's meshes are kept under the body's index and the part's.
            for key in [key for key in meshes
                        if key == i or (isinstance(key, tuple) and key[0] == i)]:
                del meshes[key]

    def _reset_body(self, i: int) -> None:
        """Empty a slot: no collider, no motion, and nowhere the broad phase looks."""
        self._collider_shape[i] = -1
        self._collider_material[i] = -1
        self._collider_filter[i] = -1
        self._trigger_shape[i] = -1
        self._trigger_filter[i] = -1
        self._report[i] = False
        self._sensor[i] = False
        self._motion_type[i] = _TYPE_INT[model.STATIC]
        self._linear_velocity[i] = 0.0
        self._angular_velocity[i] = 0.0
        self._force[i] = 0.0
        self._torque[i] = 0.0
        self._exact_rotation[i] = False
        self._inv_mass[i] = 0.0
        self._mass[i] = 0.0
        self._inv_inertia[i] = 0.0
        self._awake[i] = False
        self._sleep_timer[i] = 0.0
        self._island[i] = -1
        # An AABB that overlaps nothing, so the broad phase never pairs it and
        # a refit that skips empty slots leaves it that way.
        self._aabb_min[i] = np.inf
        self._aabb_max[i] = -np.inf
        self._aabb_cache.pop(i, None)
        # A slot handed to another body must be measured for it, not trusted
        # with what the last one's bounds were.
        if i < len(self._refit_pose):
            self._refit_pose[i] = np.nan
            self._refit_quat[i] = np.nan
        # The refit cache is keyed by body count and shape, and this slot's
        # shape has changed; rebuilding it is what makes the emptiness stick.
        self._refit_cache_n = -1

    def _set_mass_properties(self, i: int, motion: model.Motion,
                             shape: model.Shape | None) -> None:
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
        """Build and cache the GPU backend, or ``None`` if it can't be created.

        ``select_backend('auto')`` answers numpy where there is no GL compute,
        and logs why where the GPU backend failed to build.
        """
        if self._gpu_backend is None and not self._gpu_failed:
            gpu = select_backend('auto')
            if type(gpu) is NumpyBackend:               # GLComputeBackend subclasses it
                self._gpu_failed = True
            else:
                self._gpu_backend = gpu
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
        """Advance the whole world by one fixed step ``dt``: forces, collision, joints, sleep.

        Contact and trigger listeners run inside the step. One that raises does
        not stop it: every other listener is still called, the step runs to
        the end, and the first exception is raised from here once it has.
        """
        with self._listener_errors_raised_after():
            self._step(dt)

    def _step(self, dt: float) -> None:
        """The body of :meth:`step`."""
        if self._backend_mode == 'auto':
            self._select_auto_backend()
        self._prev_position[:self._n] = self.position
        self._prev_orientation[:self._n] = self.orientation

        self.backend.integrate_forces(self, dt)
        self._apply_steady_forces(dt)
        if self.row_joints:
            jointrows.drive(self, self.row_joints, dt)

        if self._collision is not None:
            self.backend.refit_aabbs(self)
            self._collide(self._collision, dt)

        if self.joint_constraints:
            self._solve_joints(dt)

        exact = np.nonzero(self._exact_rotation[:self._n] & self.moving_mask())[0]
        before = self._orientation[exact].copy() if len(exact) else None
        self.backend.integrate_positions(self, dt)
        if before is not None:
            self._turn_exactly(exact, before, dt)

        if self.sleep_enabled:
            self._update_sleep(dt)
        self.time += dt
        self.step_count += 1


    def _jointed(self) -> set[tuple[int, int]]:
        """The body pairs a row joint holds and keeps from colliding, smaller index first."""
        return {(min(joint.a, joint.b), max(joint.a, joint.b)) for joint in self.row_joints
                if not joint.collide_connected and joint.a >= 0 and joint.b >= 0}

    def _collide(self, stages: '_CollisionStages', dt: float) -> None:
        """Run one collision pass: build pairs, solve solid contacts, fire triggers."""
        pairs = stages.broadphase.pairs(self)
        jointed = self._jointed() if self.row_joints else None
        rule = self._pair_rule
        if rule is not None:
            self.pair_parameters = {}
        solid: list[tuple[int, int]] = []
        sensed: list[tuple[int, int]] = []
        reporting = self.contact_reporting != 'off'
        sensors = self._sensor
        has_trigger = False
        for i, j in pairs:
            if self.trigger_shape[i] >= 0 or self.trigger_shape[j] >= 0:
                has_trigger = True
            elif sensors[i] or sensors[j]:
                # Touches a sensor has are only ever reported, never solved.
                if (reporting and self.collider_shape[i] >= 0
                        and self.collider_shape[j] >= 0):
                    sensed.append((i, j))
            # A pair at rest (both asleep/static) needs no work; it is revived
            # when a mover touches one of them (solver wakes the sleeper).
            # Neither does a pair with no dynamic body, which nothing can move.
            elif (self.collider_shape[i] >= 0 and self.collider_shape[j] >= 0
                    and (self.is_awake_mover(i) or self.is_awake_mover(j))
                    and (self.motion_type[i] == _DYNAMIC
                         or self.motion_type[j] == _DYNAMIC)
                    and not (jointed and (min(i, j), max(i, j)) in jointed)):
                if rule is not None:
                    verdict = rule(i, j)
                    if verdict is False:
                        continue
                    if verdict is not None:
                        self.pair_parameters[(i, j)] = verdict
                solid.append((i, j))
        contacts = stages.narrowphase.generate(self, solid)
        excluded: list[Contact] = []
        if reporting and contacts and self._contact_filter is not None:
            contacts, excluded = self._tracker.presolve(
                self, contacts, self._contact_filter, self.reports_pair)
        if sensed:
            touched = stages.narrowphase.generate(self, sensed)
            if touched:
                measure(self, touched)
                excluded += touched
        stages.contacts = contacts
        self.contacts = contacts
        stages.solver.solve(self, contacts, dt)
        if reporting:
            self._record_contacts(stages.solver, contacts, excluded)
        if has_trigger or stages.triggers.overlaps:
            fired = stages.triggers.update(self, pairs)
            self._log_triggers(fired)
            for event in fired:
                self.dispatch_trigger(*event)

    def writeback(self, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated poses for rendering: lerp prev→current by ``alpha``."""
        pos = self.prev_position * (1 - alpha) + self.position * alpha
        quat = mathutil.quat_normalize(
            self.prev_orientation * (1 - alpha) + self.orientation * alpha)
        return pos, quat

    # -- AABBs & sleeping ------------------------------------------------
    #: refit kinds -- 0 no shape, 1 sphere, 2 box, 3 other (per-body fallback)
    _RFIT_NONE, _RFIT_SPHERE, _RFIT_BOX, _RFIT_OTHER = 0, 1, 2, 3

    def _refit_shapes(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-body shape kind (``_RFIT_*``) and local half-extents, ``(n,)`` and ``(n, 3)``.

        Shapes are immutable once created, so the arrays are rebuilt only when
        the body count changes (a wave added/removed) or a slot is emptied --
        not every step.
        """
        if getattr(self, '_refit_cache_n', -1) != self._n:
            self._build_refit_cache()
        return self._refit_kind, self._refit_half

    def _build_refit_cache(self) -> None:
        """Rebuild the arrays :meth:`_refit_shapes` answers."""
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
        kind, half = self._refit_shapes()
        pos = self._position[:n]
        absR = np.abs(mathutil.quat_to_matrix(self._orientation[:n]))
        hw = np.einsum('nij,nj->ni', absR, half)      # rotated box half-extents
        sph = kind == self._RFIT_SPHERE
        if sph.any():
            hw[sph] = half[sph]                        # sphere: isotropic, unrotated
        self._aabb_min[:n] = pos - hw - margin
        self._aabb_max[:n] = pos + hw + margin
        none = np.where(kind == self._RFIT_NONE)[0]
        if len(none):                                  # point AABB, no margin
            self._aabb_min[none] = self._position[none]
            self._aabb_max[none] = self._position[none]
        other = np.where(kind == self._RFIT_OTHER)[0]
        if len(other):
            self._refit_measured(other, margin)

    def _refit_measured(self, other: np.ndarray, margin: float) -> None:
        """Refit the bodies whose AABB has to be measured rather than derived.

        The exact AABB of a capsule, cylinder, hull or mesh means building its
        world proxy, which for a terrain tile is thousands of triangles -- so it
        is remembered, and remembering it is not the whole saving. A landscape
        is dozens of bodies that never move, and *asking* each of them whether
        it has moved is itself the cost once the answer is always no. Which of
        them moved is one array comparison, and the ones that did not are
        written out of the remembered bounds in one go.
        """
        position, orientation = self._position, self._orientation
        if len(self._refit_pose) < len(position):
            self._grow_refit_bounds()
        moved = other[
            np.any(position[other] != self._refit_pose[other], axis=1)
            | np.any(orientation[other] != self._refit_quat[other], axis=1)]
        for i in moved:
            si = int(self._collider_shape[i])
            idx = si if si >= 0 else int(self._trigger_shape[i])
            lo, hi = world_aabb(self.shapes[idx], position[i], orientation[i])
            self._refit_lo[i], self._refit_hi[i] = lo, hi
        if len(moved):
            self._refit_pose[moved] = position[moved]
            self._refit_quat[moved] = orientation[moved]
        self._aabb_min[other] = self._refit_lo[other] - margin
        self._aabb_max[other] = self._refit_hi[other] + margin

    def _grow_refit_bounds(self) -> None:
        """Make room to remember every body's measured bounds and its pose.

        What is already remembered is carried over; the new room starts as a
        pose no body can be at, so a body that has just appeared is measured
        rather than trusted with whatever was in its slot.
        """
        count = len(self._position)
        kept = len(self._refit_pose)
        lo = np.zeros((count, 3), dtype='d')
        hi = np.zeros((count, 3), dtype='d')
        pose = np.full((count, 3), np.nan, dtype='d')
        quat = np.full((count, 4), np.nan, dtype='d')
        if kept:
            lo[:kept], hi[:kept] = self._refit_lo, self._refit_hi
            pose[:kept], quat[:kept] = self._refit_pose, self._refit_quat
        self._refit_lo, self._refit_hi = lo, hi
        self._refit_pose, self._refit_quat = pose, quat

    def _update_sleep(self, dt: float) -> None:
        """Island-aware sleeping: a group of touching bodies sleeps only when
        *all* of them have been below the threshold long enough, so one jittering
        body can't keep the whole pile awake (and can't be re-woken by a resting
        neighbour — only a real impact wakes it, see the solver).

        Each body's thresholds are its own where :meth:`set_sleep` gave them
        and the world's otherwise."""
        n = self._n
        dyn = self.motion_type == _DYNAMIC
        lin_thresh = np.where(self._sleep_linear[:n] < 0.0, self.sleep_linear_speed,
                              self._sleep_linear[:n])
        ang_thresh = np.where(self._sleep_angular[:n] < 0.0, self.sleep_angular_speed,
                              self._sleep_angular[:n])
        t_sleep = np.where(self._sleep_after[:n] < 0.0, self.sleep_time, self._sleep_after[:n])
        speed2 = np.sum(self.linear_velocity ** 2, axis=1)
        spin2 = np.sum(self.angular_velocity ** 2, axis=1)
        slow = dyn & (speed2 < lin_thresh ** 2) & (spin2 < ang_thresh ** 2)
        self._sleep_timer[:n][slow] += dt
        self._sleep_timer[:n][~slow] = 0.0
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

    def _contact_islands(self, dyn: np.ndarray) -> tuple[list[list[int]], set[int]]:
        """Union-find the touching dynamic bodies into islands.

        Returns the list of islands (each a list of body indices) and the set of
        dynamic bodies that are in contact with anything.
        """
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        contacted: set[int] = set()
        for c in self.contacts:
            a, b = c.a, c.b
            if dyn[a]:
                contacted.add(a)
            if dyn[b]:
                contacted.add(b)
            if dyn[a] and dyn[b]:
                parent[find(a)] = find(b)
        groups: dict[int, list[int]] = {}
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
        """Notify every registered trigger listener of one overlap event.

        Every listener is called; the first exception one raises is raised
        after the last, or after the step when this runs inside one.
        """
        with self._listener_errors_raised_after():
            self._call_listeners(self.trigger_listeners, event_type,
                                 trigger_index, other_index)

    def add_joint_constraint(self, constraint: Any) -> Any:
        """Register a joint constraint, waking the bodies it touches; return it.

        A joint built from rows (:class:`~omi_physics.jointrows.RowJoint`)
        fixes its anchors and axes in its bodies' frames here, where the
        bodies are now.
        """
        self.joint_constraints.append(constraint)
        if isinstance(constraint, RowJoint):
            constraint.bind(self)
            self.row_joints.append(constraint)
        for i in (getattr(constraint, 'a', None), getattr(constraint, 'b', None),
                  getattr(constraint, 'body', None)):
            if i is not None and i >= 0:
                self.wake(i)
        return constraint

    def remove_joint_constraint(self, constraint: Any) -> None:
        """Stop solving ``constraint``; one the world does not hold is ignored."""
        if constraint in self.joint_constraints:
            self.joint_constraints.remove(constraint)
        if constraint in self.row_joints:
            self.row_joints.remove(constraint)
        for i in (getattr(constraint, 'a', None), getattr(constraint, 'b', None),
                  getattr(constraint, 'body', None)):
            if i is not None and 0 <= i < self._n:
                self.wake(i)

    def _solve_joints(self, dt: float) -> None:
        """Prepare and iteratively solve every joint constraint, waking their bodies.

        The joints built from rows are solved together after the others
        (:func:`omi_physics.jointrows.solve`).
        """
        for c in self.joint_constraints:
            i = getattr(c, 'body', None)
            if i is not None and i >= 0:
                self.wake(i)
            for i in (getattr(c, 'a', -1), getattr(c, 'b', -1)):
                if i >= 0:
                    self.wake(i)
            if not isinstance(c, RowJoint):
                c.prepare(self)
        legacy = [c for c in self.joint_constraints if not isinstance(c, RowJoint)]
        for _ in range(self.joint_iterations):
            for c in legacy:
                c.solve(self, dt)
        if self.row_joints:
            jointrows.solve(self, self.row_joints, dt, self.joint_iterations)


class _CollisionStages:
    """The broad phase, narrow phase, contact solver and trigger system of a world.

    :meth:`PhysicsWorld._collide` runs them, in that order, each step.
    """

    def __init__(self, solver: Any = None, velocity_iterations: int = 10) -> None:
        """Build the broad/narrow phases, the contact solver, and the trigger system."""
        self.broadphase = BroadPhase()
        self.narrowphase = NarrowPhase()
        self.solver = solver or SequentialImpulseSolver(
            velocity_iterations=velocity_iterations)
        self.triggers = TriggerSystem()
        self.contacts: list[Contact] = []
