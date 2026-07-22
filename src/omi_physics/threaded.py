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
"""
import threading
import time
from typing import Any, Optional, Tuple

import numpy as np

from . import mathutil

Snapshot = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ThreadedSimulation:
    """Steps a :class:`PhysicsWorld` on a background thread, publishing snapshots."""

    def __init__(self, world: Any, sim_hz: float = 120.0) -> None:
        self.world = world
        self._sim_dt = 1.0 / sim_hz
        self._lock = threading.Lock()
        self._world_lock = threading.Lock()
        self._snap: Optional[Snapshot] = None
        self._version = 0                        # bumped each publish
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._steps = 0

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
            next_t += dt
            delay = next_t - clock()
            if delay > 0:
                self._stop.wait(delay)          # interruptible sleep
            else:
                next_t = clock()                # fell behind: reset, don't spiral

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
