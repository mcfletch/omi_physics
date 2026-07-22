"""Trigger sensors (``OMI_physics_body.trigger``).

A trigger detects overlap but generates **no** impulse.  Each step it diffs the
current overlap set against the previous one and emits ``enter`` / ``stay`` /
``exit`` events, which the world dispatches to listeners (a natural fit for
``TouchSensor``-style routing).  Pickups, pressure plates, and gravity-zone
bounds all use triggers.
"""
from typing import List, Optional, Set, Tuple, TYPE_CHECKING
from .body import make_proxy
from . import collide

if TYPE_CHECKING:
    from .world import PhysicsWorld


class TriggerSystem:
    """Tracks sensor overlaps across steps and emits enter/stay/exit events."""

    def __init__(self) -> None:
        """Start with no recorded overlaps."""
        self.overlaps: Set[Tuple[int, int]] = set()   # {(trigger_body, other_body)}

    def update(self, world: "PhysicsWorld",
               pairs: List[Tuple[int, int]]) -> List[Tuple[str, int, int]]:
        """Diff this step's overlaps against the last; return ``(event, trigger_body, other_body)``.

        ``event`` is ``'enter'``, ``'stay'``, or ``'exit'``.  Mutates the stored
        overlap set to the current one.
        """
        new: Set[Tuple[int, int]] = set()
        for i, j in pairs:
            pair = self._sensor_pair(world, i, j)
            if pair is None:
                continue
            trg, tshape, other, oshape = pair
            if self._overlapping(world, trg, tshape, other, oshape):
                new.add((trg, other))
        events: List[Tuple[str, int, int]] = []
        events += [('enter', t, o) for (t, o) in new - self.overlaps]
        events += [('stay', t, o) for (t, o) in new & self.overlaps]
        events += [('exit', t, o) for (t, o) in self.overlaps - new]
        self.overlaps = new
        return events

    def _sensor_pair(self, world: "PhysicsWorld", i: int,
                     j: int) -> Optional[Tuple[int, int, int, int]]:
        """Order a pair as ``(trigger, trigger_shape, other, other_shape)``, or None if neither is a sensor."""
        ti = world.trigger_shape[i] >= 0
        tj = world.trigger_shape[j] >= 0
        if not (ti or tj):
            return None
        if ti:
            trg, other = i, j
        else:
            trg, other = j, i
        tshape = world.trigger_shape[trg]
        oshape = world.collider_shape[other]
        if oshape < 0:
            oshape = world.trigger_shape[other]
        if oshape < 0:
            return None
        return trg, tshape, other, oshape

    def _overlapping(self, world: "PhysicsWorld", trg: int, tshape: int,
                     other: int, oshape: int) -> bool:
        """True if the sensor shape and the other body's shape currently intersect."""
        PA = make_proxy(world.shapes[tshape], world.position[trg], world.orientation[trg])
        PB = make_proxy(world.shapes[oshape], world.position[other], world.orientation[other])
        return len(collide.collide(trg, other, PA, PB)) > 0
