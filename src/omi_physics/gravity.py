"""Gravity volumes (``OMI_physics_gravity`` zones).

A volume is a :class:`~omi_physics.model.Gravity` field plus a region
that decides which bodies it affects.  ``resolve_gravity`` applies volumes in
ascending ``priority`` (highest wins), honouring ``replace`` (override) and
``stop`` (halt accumulation).
"""
from typing import List, Optional, Union
import numpy as np

from . import model, mathutil
from .mathutil import Vec


class SphereRegion:
    """A ball region: every point within ``radius`` of ``center``."""

    def __init__(self, center: Vec, radius: float):
        self.center = np.asarray(center, dtype='d')
        self.radius = float(radius)

    def contains(self, pos: np.ndarray) -> bool:
        """True if world point ``pos`` lies inside the sphere."""
        return bool(np.dot(pos - self.center, pos - self.center) <= self.radius ** 2)


class BoxRegion:
    """An axis-aligned box region centred at ``center`` with the given half-extents."""

    def __init__(self, center: Vec, half: Vec):
        self.center = np.asarray(center, dtype='d')
        self.half = np.asarray(half, dtype='d')

    def contains(self, pos: np.ndarray) -> bool:
        """True if world point ``pos`` lies inside the box."""
        return bool(np.all(np.abs(pos - self.center) <= self.half))


class InfiniteRegion:
    """A region covering all of space (the default for a global field)."""

    def contains(self, pos: np.ndarray) -> bool:
        """Always True: every point is inside."""
        return True


Region = Union[SphereRegion, BoxRegion, InfiniteRegion]


class GravityVolume:
    """A gravity field with a region of influence."""

    def __init__(self, field: model.Gravity, region: Optional[Region] = None):
        self.field = field
        self.region = region or InfiniteRegion()

    @property
    def priority(self) -> int:
        """The field's priority; higher priority volumes override lower ones."""
        return self.field.priority

    def contains(self, pos: np.ndarray) -> bool:
        """True if world point ``pos`` is within this volume's region."""
        return self.region.contains(pos)

    def field_at(self, pos: np.ndarray) -> np.ndarray:
        """Gravity acceleration (m/s²) this volume applies at ``pos``.

        A ``point`` field pulls toward its ``center``; a directional field is
        uniform along ``direction``.
        """
        f = self.field
        if f.type == model.POINT:
            d = np.asarray(f.center, dtype='d') - pos
            return mathutil.normalize(d) * f.gravity
        return mathutil.normalize(np.asarray(f.direction, dtype='d')) * f.gravity


def apply_volumes(volumes: List[GravityVolume], positions: np.ndarray,
                  out: np.ndarray) -> np.ndarray:
    """Resolve gravity per body given a list of volumes (in place on ``out``).

    ``out`` holds each body's starting gravity vector and is overwritten with the
    resolved value, then returned.  Volumes apply in ascending ``priority``; a
    ``replace`` volume overrides the accumulation, a ``stop`` volume zeroes it,
    otherwise fields add.
    """
    ordered = sorted(volumes, key=lambda v: v.priority)
    for i in range(len(positions)):
        pos = positions[i]
        g = out[i]
        for v in ordered:
            if not v.contains(pos):
                continue
            if v.field.replace:
                g = v.field_at(pos)
            elif v.field.stop:
                g = np.zeros(3)
            else:
                g = g + v.field_at(pos)
        out[i] = g
    return out
