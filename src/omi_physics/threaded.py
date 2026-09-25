"""Background-threaded simulation: step the world off the consumer's thread.

:class:`ThreadedSimulation` runs a :class:`~omi_physics.world.PhysicsWorld`'s
fixed-timestep step on its own daemon thread and publishes an immutable pose
snapshot after every tick. A consumer (a renderer, a network sender, ...) reads
the most recent snapshot with :meth:`latest` each frame; it never blocks waiting
for a step. Because the native solver and narrow phase release the GIL, the
physics tick and the consumer's work genuinely overlap, so a render loop can hold
60 fps while the simulation advances as fast as it can underneath.

Only the physics thread touches the world's mutable state; the consumer reads a
copied snapshot under a short lock, so there is no tearing. Structural changes to
the world (adding/removing bodies) must be made with the loop stopped
(:meth:`stop`) or while holding :meth:`with_world`.

A snapshot is the tuple ``(position, axis_angle, awake, dynamic)``:

* ``position``   -- ``(n, 3)`` body positions,
* ``axis_angle`` -- ``(n, 4)`` orientations as VRML-style ``(x, y, z, angle)``,
* ``awake``      -- ``(n,)`` bool, False for sleeping bodies,
* ``dynamic``    -- ``(n,)`` bool, True for dynamic bodies.

Contact and trigger events travel with the snapshots. Each publish moves what
the world's :attr:`~omi_physics.world.PhysicsWorld.contact_log` holds into the
simulation's own :attr:`~ThreadedSimulation.events` log
(:meth:`~ThreadedSimulation.flush_events` does the same between ticks, for a
removal made under :meth:`~ThreadedSimulation.with_world`), and
:meth:`~ThreadedSimulation.latest_and_events` hands a consumer the newest
snapshot together with every event up to it, so a handler reads the scene in
the pose of the step that produced the event or a later one, never an earlier
one. Contact listeners on the world run on the physics thread.

The asked-for tick rate is a request, not a promise. A tick that overruns its
budget resets the schedule instead of trying to run the missed ones back to back,
because catching up on a machine that is already behind is how a simulation
spirals. That is the right behaviour and it is completely invisible: a starved
simulation and a healthy one differ only in that everything in the world moves
slowly, which reads as a physics bug rather than as a scheduling one. So the
thread also counts what it settled for -- :meth:`~ThreadedSimulation.rate`, the
ticks per second actually achieved over the recent past, and
:attr:`~ThreadedSimulation.dropped`, how many ticks were abandoned to stay out of
the spiral. A consumer showing those two beside its own frame rate can tell the
two apart at a glance.
"""
import threading
import time
from typing import Any

import numpy as np

from . import mathutil
from .contactevents import ContactLog

Snapshot = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def pace(next_t: float, now: float, dt: float) -> tuple[float, float, bool]:
    """Schedule the tick after the one that just finished.

    next_t -- when the finished tick was due to be over
    now -- when it was
    dt -- the interval between ticks

    Returns how long to wait before starting the next one, when that one is due
    to be over, and whether the tick that just finished overran its budget.

    A tick with time to spare waits out the remainder and leaves the cadence
    alone: deadlines advance by exactly *dt*, so arriving a little early or a
    little late does not accumulate into drift. One that overruns waits not at
    all and moves the deadline to the present, giving up the ticks it missed
    rather than running them back to back -- a machine already too slow cannot
    pay that debt back, and the attempt is the spiral.
    """
    due = next_t + dt
    delay = due - now
    if delay > 0:
        return delay, due, False
    return 0.0, now, True


class ThreadedSimulation:
    """Steps a :class:`PhysicsWorld` on a background thread, publishing snapshots."""

    #: Tick timestamps kept for :meth:`rate`. A second or so of a healthy
    #: simulation: long enough to be steady, short enough to describe *now*
    #: rather than an average across a stall that has already passed.
    RATE_WINDOW = 120

    def __init__(self, world: Any, sim_hz: float = 120.0) -> None:
        self.world = world
        self._sim_dt = 1.0 / sim_hz
        self.sim_hz = sim_hz
        self._lock = threading.Lock()
        self._world_lock = threading.Lock()
        self._snap: Snapshot | None = None
        self._version = 0                        # bumped each publish
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._steps = 0
        self._dropped = 0
        self._tick_times: list[float] = []
        #: Events published with the snapshots and not yet taken.
        self.events = ContactLog()

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        """Begin simulating on a daemon thread (idempotent)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._publish()
        self._thread = threading.Thread(target=self._loop, name='physics', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the simulation thread and wait for it to exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def with_world(self) -> threading.Lock:
        """Context manager giving exclusive access to the world (pauses stepping)."""
        return self._world_lock

    @property
    def steps(self) -> int:
        """Total simulation ticks executed since the thread started."""
        return self._steps

    @property
    def dropped(self) -> int:
        """Ticks abandoned because the previous one overran its budget.

        Zero on a simulation that is keeping up. Anything else is the world
        running slower than wall clock -- too many bodies, or a consumer thread
        holding the GIL -- and is the number that separates "the physics is
        wrong" from "the physics is not getting a turn".
        """
        return self._dropped

    def rate(self) -> float:
        """Ticks per second actually achieved over the recent past.

        Compare against ``sim_hz``: the two agreeing means the thread is
        getting the turns it asked for, and a large gap means it is not,
        whatever the consumer's own frame rate says.

        Zero until there are two ticks to measure an interval between.
        """
        with self._lock:
            times = list(self._tick_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    # -- thread ----------------------------------------------------------
    def _loop(self) -> None:
        clock = time.perf_counter
        dt = self._sim_dt
        next_t = clock()
        while not self._stop.is_set():
            with self._world_lock:
                self.world.step(dt)
                self._steps += 1
                self._publish()
            now = clock()
            self._record_tick(now)
            delay, next_t, dropped = pace(next_t, now, dt)
            if dropped:
                self._dropped += 1
            else:
                self._stop.wait(delay)          # interruptible sleep

    def _record_tick(self, now: float) -> None:
        """Note that a tick finished at ``now``, for :meth:`rate`'s window."""
        with self._lock:
            times = self._tick_times
            times.append(now)
            if len(times) > self.RATE_WINDOW:
                del times[:-self.RATE_WINDOW]

    def _publish(self) -> None:
        w = self.world
        n = w.body_count
        snap = (w.position[:n].copy(),
                mathutil.quat_to_axis_angle(w.orientation[:n]),
                w.awake[:n].copy(),
                (w.motion_type[:n] == 2))
        events = w.contact_log.drain()
        with self._lock:
            self._snap = snap
            self._version += 1
            self.events.extend(events)

    def flush_events(self) -> None:
        """Publish what the world has logged since the last tick, without a step.

        For a change made between ticks under :meth:`with_world`: removing a
        body logs its ``'end'`` and ``'exit'`` events at once, and this hands
        them to the consumer on its next read rather than with the next tick.
        They follow every event already published, so they still arrive in
        the order they happened. Call it holding :meth:`with_world`, or with
        the loop stopped.
        """
        events = self.world.contact_log.drain()
        with self._lock:
            self.events.extend(events)

    # -- consumer thread -------------------------------------------------
    def latest(self) -> tuple[Snapshot | None, int]:
        """The most recent published snapshot and its version.

        Returns ``(None, version)`` before the first publish. The version lets a
        consumer skip work when nothing new has been simulated since its last read.
        """
        with self._lock:
            return self._snap, self._version

    def latest_and_events(self) -> tuple[Snapshot | None, int, list[Any]]:
        """The newest snapshot, its version, and every event published up to it.

        Taken together under one lock, so no event belongs to a step the
        snapshot has not reached. The events are removed from :attr:`events`.
        """
        with self._lock:
            return self._snap, self._version, self.events.drain()

    def drain_events(self) -> list[Any]:
        """Every event published so far and not yet taken, oldest first."""
        with self._lock:
            return self.events.drain()
