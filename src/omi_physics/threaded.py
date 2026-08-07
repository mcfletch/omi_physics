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
from typing import Any, Optional, Tuple

import numpy as np

from . import mathutil

Snapshot = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


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
        self._snap: Optional[Snapshot] = None
        self._version = 0                        # bumped each publish
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._steps = 0
        self._dropped = 0
        self._tick_times: list[float] = []

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
            next_t += dt
            delay = next_t - now
            if delay > 0:
                self._stop.wait(delay)          # interruptible sleep
            else:
                self._dropped += 1
                next_t = now                    # fell behind: reset, don't spiral

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
        with self._lock:
            self._snap = snap
            self._version += 1

    # -- consumer thread -------------------------------------------------
    def latest(self) -> Tuple[Optional[Snapshot], int]:
        """The most recent published snapshot and its version.

        Returns ``(None, version)`` before the first publish. The version lets a
        consumer skip work when nothing new has been simulated since its last read.
        """
        with self._lock:
            return self._snap, self._version
