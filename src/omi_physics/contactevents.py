"""Contact events: which pairs started, kept or stopped touching, on every step.

:attr:`PhysicsWorld.contacts <omi_physics.world.PhysicsWorld.contacts>` is the
narrow phase's output for one step, one :class:`~omi_physics.collide.Contact`
per manifold point, and it is rebuilt on the next. A game advancing the world
once a frame at a fixed step runs several steps per frame, and a question asked
of the contact list after :meth:`~omi_physics.world.PhysicsWorld.advance` is
answered about the last of them only.

:class:`ContactTracker` runs after the solve on every step and turns that list
into one :class:`ContactEvent` per touching pair: ``'begin'`` on the step two
bodies meet, ``'persist'`` on each step they stay in contact, ``'end'`` on the
step they part. Events go two ways:

* to the world's contact listeners, called inside the step
  (:meth:`~omi_physics.world.PhysicsWorld.add_contact_listener`);
* into :class:`ContactLog`, which a consumer drains when it chooses -- once a
  frame, on another thread (:attr:`PhysicsWorld.contact_log
  <omi_physics.world.PhysicsWorld.contact_log>`).

Trigger overlaps go into the same log as :class:`TriggerEvent`, so the two
arrive in the order they happened.

Bodies are named by :class:`BodyRef`, an index and the generation of the body in
that slot. A removed body's slot is handed to the next one added, and a
reference to the removed body never matches the new one.

The tracker works over arrays: the compiled solver hands it the contact arrays
it has already built, the grouping by pair is a sort, and Python objects are
built only for the events a consumer is going to receive.
"""
import threading
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np

from .mathutil import cross

if TYPE_CHECKING:
    from .collide import Contact
    from .world import PhysicsWorld

__all__ = [
    'BodyRef', 'ContactBatch', 'ContactEvent', 'ContactLog', 'ContactTracker',
    'PairPreview', 'Reporting', 'TriggerEvent', 'Verdict',
]

#: What a world records: nothing, the pairs of flagged bodies, or every pair.
Reporting = Literal['off', 'flagged', 'all']
REPORTING_LEVELS: tuple[str, ...] = ('off', 'flagged', 'all')

_LOW = np.int64(0xFFFFFFFF)


@dataclass(frozen=True, slots=True)
class BodyRef:
    """A body, by slot and by which body has occupied that slot.

    ``index`` is the body index every other world method takes. ``generation``
    counts how many bodies have been removed from that slot, so a reference
    taken before :meth:`~omi_physics.world.PhysicsWorld.remove_body` compares
    unequal to one taken after it, and
    :meth:`~omi_physics.world.PhysicsWorld.alive` answers False for it.
    """
    index: int
    generation: int


@dataclass(frozen=True, slots=True)
class ContactEvent:
    """One pair of bodies starting, keeping or ending contact on one step.

    ``a.index < b.index`` always, and ``normal`` points from ``a`` to ``b``.

    ``approach``, ``impulse``, ``friction_impulse``, ``slip`` and ``depth`` are
    zero on an ``'end'``, since the pair is no longer touching. ``point`` and
    ``normal`` on an ``'end'`` are where the pair last touched.
    """
    #: ``'begin'``, ``'persist'`` or ``'end'``.
    phase: str
    a: BodyRef
    b: BodyRef
    #: World-space contact point, the manifold's points weighted by the
    #: impulse each carried (their mean where none carried any).
    point: np.ndarray
    #: Unit vector from ``a`` to ``b``, weighted the same way.
    normal: np.ndarray
    #: Largest closing speed along the normal over the pair's points, taken
    #: before the solve, in m/s. The value
    #: :meth:`~omi_physics.world.PhysicsWorld.impact_on` compares.
    approach: float
    #: Sum of the points' normal impulses, in N·s. Includes both masses, which
    #: ``approach`` does not: the value to compare against a breaking strength.
    impulse: float
    #: Magnitude of the pair's summed friction impulse, in N·s.
    friction_impulse: float
    #: Largest sliding speed across the contact before the solve, in m/s.
    slip: float
    #: Deepest penetration of the pair's points, in metres.
    depth: float
    #: How many manifold points the pair had this step.
    points: int
    #: The world's step counter (:attr:`PhysicsWorld.step_count`) and its
    #: simulation time, in seconds, at the start of the step.
    step: int
    time: float
    #: On an ``'end'``: ``'separated'``, or ``'removed'`` where one body was
    #: taken out of the world. None otherwise.
    reason: str | None = None
    #: False where a contact filter answered :attr:`Verdict.IGNORE_PAIR`: the
    #: bodies overlap and the solver let them.
    solved: bool = True

    kind = 'contact'


@dataclass(frozen=True, slots=True)
class TriggerEvent:
    """A body entering, staying inside or leaving a trigger volume on one step."""
    #: ``'enter'``, ``'stay'`` or ``'exit'``.
    phase: str
    trigger: BodyRef
    other: BodyRef
    step: int
    time: float
    #: Where ``other`` was when the overlap was tested, in world coordinates:
    #: its position at the start of the step, or where it was taken out of the
    #: world, for an ``'exit'`` on removal.
    point: np.ndarray = field(default_factory=lambda: np.zeros(3), compare=False)

    kind = 'trigger'


class Verdict(Enum):
    """A contact filter's answer about a pair on the step it first touches."""
    #: Solve it as usual.
    SOLVE = 'solve'
    #: Leave it out of this step's solve and ask again on the next.
    IGNORE_STEP = 'ignore_step'
    #: Leave it out of every solve until the two part.
    IGNORE_PAIR = 'ignore_pair'


@dataclass(frozen=True, slots=True)
class PairPreview:
    """A pair about to be solved for the first time, for a contact filter.

    Taken after the narrow phase and before the solver has changed anything,
    so ``approach`` is the blow about to be delivered.
    """
    a: BodyRef
    b: BodyRef
    point: np.ndarray
    #: Unit vector from ``a`` to ``b``.
    normal: np.ndarray
    #: Closing speed along ``normal``, in m/s.
    approach: float
    #: The pair's reduced mass, ``1 / (1/m_a + 1/m_b)``, in kg; the moving
    #: body's own mass where the other is static or kinematic.
    mass: float
    points: int
    step: int
    time: float

    @property
    def impulse(self) -> float:
        """The momentum the blow carries, ``approach × mass``, in N·s.

        What stopping the pair dead along the normal takes. A bouncing contact
        delivers more than this by its restitution; this is the figure a
        breaking strength compares against before any bounce is decided.
        """
        return max(self.approach, 0.0) * self.mass


class ContactLog:
    """Events in step order, held until a consumer drains them.

    Bounded: a consumer that stops draining loses the oldest events rather than
    growing without limit, and :attr:`dropped` counts how many it lost.
    Appending and draining may happen on different threads; each takes the
    log's lock, so :attr:`dropped` counts exactly the events discarded.
    """

    def __init__(self, maxlen: int = 65536) -> None:
        self._events: deque[Any] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.maxlen = maxlen
        #: Events discarded because the log was full.
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._events)

    def extend(self, events: Iterable[Any]) -> None:
        """Append ``events``, discarding the oldest past :attr:`maxlen`."""
        events = list(events)
        with self._lock:
            overflow = len(self._events) + len(events) - self.maxlen
            if overflow > 0:
                self.dropped += overflow
            self._events.extend(events)

    def drain(self) -> list[Any]:
        """Every event held, oldest first, leaving the log empty."""
        with self._lock:
            out = list(self._events)
            self._events.clear()
        return out

    def clear(self) -> None:
        """Discard every event held."""
        with self._lock:
            self._events.clear()


class ContactBatch(NamedTuple):
    """One step's contact points as parallel arrays, the tracker's input.

    ``tangent`` is each point's friction impulse as a world-space vector.
    """
    a: np.ndarray
    b: np.ndarray
    point: np.ndarray
    normal: np.ndarray
    depth: np.ndarray
    approach: np.ndarray
    slip: np.ndarray
    normal_impulse: np.ndarray
    tangent: np.ndarray
    solved: np.ndarray

    @classmethod
    def empty(cls) -> 'ContactBatch':
        """A batch of no contacts."""
        v = np.zeros((0, 3))
        s = np.zeros(0)
        i = np.zeros(0, dtype=np.intp)
        return cls(i, i, v, v, s, s, s, s, v, np.zeros(0, dtype=bool))

    @classmethod
    def from_contacts(cls, contacts: Sequence['Contact'],
                      solved: bool = True) -> 'ContactBatch':
        """Gather ``contacts``, whose impulses the solver has filled in."""
        if not contacts:
            return cls.empty()
        normal = np.array([c.normal for c in contacts], dtype='d')
        tangent_impulse = np.array([c.tangent_impulse for c in contacts], dtype='d')
        return cls(
            np.fromiter((c.a for c in contacts), dtype=np.intp, count=len(contacts)),
            np.fromiter((c.b for c in contacts), dtype=np.intp, count=len(contacts)),
            np.array([c.point for c in contacts], dtype='d'),
            normal,
            np.fromiter((c.depth for c in contacts), dtype='d', count=len(contacts)),
            np.fromiter((c.approach for c in contacts), dtype='d', count=len(contacts)),
            np.fromiter((c.slip for c in contacts), dtype='d', count=len(contacts)),
            np.fromiter((c.normal_impulse for c in contacts), dtype='d',
                        count=len(contacts)),
            world_tangent(normal, tangent_impulse),
            np.full(len(contacts), solved, dtype=bool))

    def concatenate(self, other: 'ContactBatch') -> 'ContactBatch':
        """This batch followed by ``other``."""
        if not len(other.a):
            return self
        if not len(self.a):
            return other
        return ContactBatch(*(np.concatenate((mine, theirs))
                              for mine, theirs in zip(self, other, strict=True)))


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The solver's two friction directions for each row of ``normal``.

    The same construction as :func:`omi_physics.solver._basis` and the compiled
    kernel, so a tangential impulse the solver accumulated in that basis turns
    back into the world-space vector it applied.
    """
    seed = np.zeros_like(normal)
    near_x = np.abs(normal[:, 0]) >= 0.9
    seed[~near_x, 0] = 1.0
    seed[near_x, 1] = 1.0
    t1 = cross(seed, normal)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, cross(normal, t1)


def world_tangent(normal: np.ndarray, tangent_impulse: np.ndarray) -> np.ndarray:
    """Each point's friction impulse, from the solver's basis into world space."""
    t1, t2 = tangent_basis(normal)
    return tangent_impulse[:, 0:1] * t1 + tangent_impulse[:, 1:2] * t2


def relative_motion(world: 'PhysicsWorld', a: np.ndarray, b: np.ndarray,
                    point: np.ndarray, normal: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Closing speed along ``normal`` and sliding speed across it, per point.

    Read before anything is solved: resolving a contact is cancelling exactly
    this velocity. The normal points from A to B, so closing is the negative
    of B receding along it.
    """
    at_a = (world.linear_velocity[a]
            + cross(world.angular_velocity[a], point - world.position[a]))
    at_b = (world.linear_velocity[b]
            + cross(world.angular_velocity[b], point - world.position[b]))
    relative = at_b - at_a
    along = np.einsum('ij,ij->i', relative, normal)
    across = relative - along[:, None] * normal
    return -along, np.linalg.norm(across, axis=1)


def measure(world: 'PhysicsWorld', contacts: Sequence['Contact']) -> None:
    """Record approach and slip on contacts the solver will not see.

    The solver records both on what it solves; a contact left out of the
    solve -- a pair a contact filter ignored, or a sensor's touch -- is
    measured here instead, at the same moment of the step.
    """
    a = np.fromiter((c.a for c in contacts), dtype=np.intp, count=len(contacts))
    b = np.fromiter((c.b for c in contacts), dtype=np.intp, count=len(contacts))
    point = np.array([c.point for c in contacts], dtype='d')
    normal = np.array([c.normal for c in contacts], dtype='d')
    approach, slip = relative_motion(world, a, b, point, normal)
    for c, closing, sliding in zip(contacts, approach.tolist(), slip.tolist(),
                                   strict=True):
        c.approach, c.slip = closing, sliding


def pair_keys(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One int64 per ``(a, b)`` pair, sortable and comparable in bulk."""
    return (np.asarray(a, dtype=np.int64) << 32) | np.asarray(b, dtype=np.int64)


def split_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ``(a, b)`` index arrays of ``keys``."""
    return (keys >> 32).astype(np.intp), (keys & _LOW).astype(np.intp)


class _Pairs(NamedTuple):
    """Per-pair aggregates of one step's contact points, sorted by key."""
    keys: np.ndarray
    point: np.ndarray
    normal: np.ndarray
    approach: np.ndarray
    impulse: np.ndarray
    friction: np.ndarray
    slip: np.ndarray
    depth: np.ndarray
    points: np.ndarray
    solved: np.ndarray


def _group(batch: ContactBatch) -> _Pairs:
    """Reduce a batch of points to one row per pair."""
    keys = pair_keys(batch.a, batch.b)
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    count = np.diff(np.r_[starts, len(keys)])

    def total(values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(values[order], starts, axis=0)

    def largest(values: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(values[order], starts, axis=0)

    weight = np.maximum(batch.normal_impulse, 0.0)
    weight_sum = total(weight)
    weighted = weight_sum > 0.0
    point = total(batch.point) / count[:, None]
    normal = total(batch.normal)
    if weighted.any():
        point[weighted] = (total(batch.point * weight[:, None])[weighted]
                           / weight_sum[weighted, None])
        normal[weighted] = total(batch.normal * weight[:, None])[weighted]
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 0.0)
    return _Pairs(
        keys[starts], point, normal, largest(batch.approach),
        total(batch.normal_impulse), np.linalg.norm(total(batch.tangent), axis=1),
        largest(batch.slip), largest(batch.depth), count,
        np.logical_and.reduceat(batch.solved[order], starts))


class ContactTracker:
    """Turns each step's contacts into begin/persist/end events per pair.

    Held by the world; a caller reaches it through the world's reporting
    settings and listeners rather than directly.

    A pair whose two bodies are both at rest -- asleep, or asleep against a
    static body -- is not sent to the narrow phase, so it has no contacts. It
    is carried forward as touching while their boxes still overlap, and ends
    only when one of them moves off or is removed: a crate that settles and
    sleeps on the floor has begun touching it and has not stopped.
    """

    def __init__(self) -> None:
        #: Keys of the pairs touching after the last step, sorted, and where
        #: each last touched.
        self._keys = np.zeros(0, dtype=np.int64)
        self._point = np.zeros((0, 3))
        self._normal = np.zeros((0, 3))
        #: Keys of the pairs a contact filter has excluded until they part.
        self.ignored: set[int] = set()

    def touching(self, key: int) -> bool:
        """Whether the pair with ``key`` was touching after the last step."""
        at = int(np.searchsorted(self._keys, key))
        return at < len(self._keys) and int(self._keys[at]) == key

    def reset(self) -> None:
        """Forget every pair, as when reporting is switched off."""
        self._keys = np.zeros(0, dtype=np.int64)
        self._point = np.zeros((0, 3))
        self._normal = np.zeros((0, 3))
        self.ignored.clear()

    def update(self, world: 'PhysicsWorld', batch: ContactBatch,
               flags: np.ndarray | None = None) -> list[ContactEvent]:
        """Diff this step's contacts against the last step's; return the events.

        ``batch`` holds only the pairs the world reports. ``flags`` is the
        per-body flag array under ``'flagged'`` reporting: a pair that stopped
        being reported because neither body is flagged any longer is dropped
        without an ``'end'``, since nothing says the two parted, and a contact
        filter's verdict on it is dropped with it.
        """
        step, time = world.step_count, world.time
        now = _group(batch) if len(batch.a) else None
        now_keys = now.keys if now is not None else np.zeros(0, dtype=np.int64)

        was = self._keys
        kept = np.isin(was, now_keys, assume_unique=True)
        missing = np.flatnonzero(~kept)
        carried = missing[self._at_rest(world, was[missing])]
        ended = np.setdiff1d(missing, carried, assume_unique=True)
        if flags is not None and len(ended):
            a_idx, b_idx = split_keys(was[ended])
            still_flagged = flags[a_idx] | flags[b_idx]
            # An untracked pair's filter verdict goes with it: flagged again,
            # the two meet afresh and the filter is asked afresh.
            self.ignored.difference_update(was[ended[~still_flagged]].tolist())
            ended = ended[still_flagged]

        events: list[ContactEvent] = []
        if now is not None:
            fresh = ~np.isin(now_keys, was, assume_unique=True)
            wanted = np.ones_like(fresh) if world.report_persist else fresh
            generation = world._generation
            a_idx, b_idx = split_keys(now_keys)
            for row in np.flatnonzero(wanted).tolist():
                a, b = int(a_idx[row]), int(b_idx[row])
                events.append(ContactEvent(
                    'begin' if fresh[row] else 'persist',
                    BodyRef(a, int(generation[a])), BodyRef(b, int(generation[b])),
                    now.point[row].copy(), now.normal[row].copy(),
                    float(now.approach[row]),
                    float(now.impulse[row]), float(now.friction[row]),
                    float(now.slip[row]), float(now.depth[row]),
                    int(now.points[row]), step, time,
                    solved=bool(now.solved[row])))
        events += self._ends(world, ended, 'separated', step, time)

        keys = np.concatenate((now_keys, was[carried]))
        point = np.concatenate((now.point if now is not None else np.zeros((0, 3)),
                                self._point[carried]))
        normal = np.concatenate((now.normal if now is not None else np.zeros((0, 3)),
                                 self._normal[carried]))
        order = np.argsort(keys, kind='stable')
        self._keys, self._point, self._normal = keys[order], point[order], normal[order]
        return events

    def retire(self, world: 'PhysicsWorld', index: int) -> list[ContactEvent]:
        """End every pair body ``index`` is in, before its slot is emptied."""
        a_idx, b_idx = split_keys(self._keys)
        rows = np.flatnonzero((a_idx == index) | (b_idx == index))
        if not len(rows):
            return []
        events = self._ends(world, rows, 'removed', world.step_count, world.time)
        keep = np.ones(len(self._keys), dtype=bool)
        keep[rows] = False
        self._keys, self._point, self._normal = (
            self._keys[keep], self._point[keep], self._normal[keep])
        return events

    def _ends(self, world: 'PhysicsWorld', rows: np.ndarray, reason: str,
              step: int, time: float) -> list[ContactEvent]:
        """``'end'`` events for the stored pairs at ``rows``."""
        events = []
        generation = world._generation
        a_idx, b_idx = split_keys(self._keys[rows])
        for row, a, b in zip(rows.tolist(), a_idx.tolist(), b_idx.tolist(), strict=True):
            key = int(self._keys[row])
            solved = key not in self.ignored
            self.ignored.discard(key)
            events.append(ContactEvent(
                'end', BodyRef(a, int(generation[a])), BodyRef(b, int(generation[b])),
                self._point[row].copy(), self._normal[row].copy(),
                0.0, 0.0, 0.0, 0.0, 0.0, 0,
                step, time, reason=reason, solved=solved))
        return events

    @staticmethod
    def _at_rest(world: 'PhysicsWorld', keys: np.ndarray) -> np.ndarray:
        """Which of ``keys`` were not tested because both bodies are at rest.

        Such a pair is still touching if their boxes still overlap; a body
        moved by hand while asleep has boxes that no longer do.
        """
        if not len(keys):
            return np.zeros(0, dtype=bool)
        a, b = split_keys(keys)
        motion = world._motion_type
        awake = world._awake
        mover_a = ((motion[a] == 2) & awake[a]) | (motion[a] == 1)
        mover_b = ((motion[b] == 2) & awake[b]) | (motion[b] == 1)
        lo, hi = world._aabb_min, world._aabb_max
        overlap = np.all((hi[a] >= lo[b]) & (hi[b] >= lo[a]), axis=1)
        return ~mover_a & ~mover_b & overlap

    # -- before the solve ------------------------------------------------
    def presolve(self, world: 'PhysicsWorld', contacts: list['Contact'],
                 decide: Callable[[PairPreview], Any],
                 reported: Callable[[int, int], bool]
                 ) -> tuple[list['Contact'], list['Contact']]:
        """Split ``contacts`` into those to solve and those a filter excluded.

        ``decide`` is asked once about each reported pair on the step it would
        begin. A pair it answers :attr:`Verdict.IGNORE_PAIR` for is left out
        of every solve until it ends; :attr:`Verdict.IGNORE_STEP` leaves it out
        of this step only, and the pair is neither tracked nor reported until a
        step it is solved on. Any other answer, None included, solves it.
        """
        by_pair: dict[tuple[int, int], list[Contact]] = {}
        for c in contacts:
            by_pair.setdefault((c.a, c.b), []).append(c)
        solve: list[Contact] = []
        excluded: list[Contact] = []
        for (a, b), points in by_pair.items():
            key = (a << 32) | b
            if key in self.ignored:
                excluded += points
                continue
            if self.touching(key) or not reported(a, b):
                solve += points
                continue
            verdict = decide(self._preview(world, a, b, points))
            if verdict is Verdict.IGNORE_PAIR:
                self.ignored.add(key)
                excluded += points
            elif verdict is not Verdict.IGNORE_STEP:
                solve += points
        if excluded:
            measure(world, excluded)
        return solve, excluded

    @staticmethod
    def _preview(world: 'PhysicsWorld', a: int, b: int,
                 points: list['Contact']) -> PairPreview:
        """What a contact filter is shown about a pair."""
        idx_a = np.full(len(points), a, dtype=np.intp)
        idx_b = np.full(len(points), b, dtype=np.intp)
        point = np.array([c.point for c in points], dtype='d')
        normal = np.array([c.normal for c in points], dtype='d')
        approach, _slip = relative_motion(world, idx_a, idx_b, point, normal)
        mean_normal = normal.sum(axis=0)
        mean_normal /= np.linalg.norm(mean_normal) or 1.0
        inverse = float(world._inv_mass[a] + world._inv_mass[b])
        generation = world._generation
        return PairPreview(
            BodyRef(a, int(generation[a])), BodyRef(b, int(generation[b])),
            point.mean(axis=0), mean_normal, float(approach.max()),
            1.0 / inverse if inverse > 0.0 else 0.0, len(points),
            world.step_count, world.time)
