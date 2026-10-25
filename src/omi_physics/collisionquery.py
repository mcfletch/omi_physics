"""What a character asks of the world around it.

:class:`CollisionQuery` is the set of questions :class:`~omi_physics.character
.CharacterController` puts to the world it walks in: which bodies are solid and
fixed, where their collision shapes are in world space, which of them are
near a box, which moving bodies are near a box, and what a ray meets. Any
object that answers them can stand in for the world, among them an immutable
copy of the colliders published by a physics thread, so a controller can step
on a thread that does not own the world.

:class:`WorldQuery` answers them from a live
:class:`~omi_physics.world.PhysicsWorld`. A controller given a world builds one.
"""
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

import numpy as np

from . import model
from .body import Proxy, make_proxy
from .mathutil import Vec
from .raycast import DEFAULT_RANGE, RayHit, raycast
from .staticindex import StaticGrid

if TYPE_CHECKING:
    from .world import PhysicsWorld

__all__ = ['CollisionQuery', 'DynamicBody', 'WorldQuery']

#: World-space proxies for each world's static colliders, keyed by world and
#: then by body index. Weak, so an unloaded level takes its collision copy with
#: it. A trimesh proxy is a transformed copy of the geometry with a grid over
#: it, so one per controller is one copy of the level per character in it;
#: sharing is correct because these bodies never move.
STATIC_PROXIES: "WeakKeyDictionary[Any, dict[int, Proxy]]" = WeakKeyDictionary()


#: The grid over each world's fixed bodies, rebuilt when the world's layout changes.
STATIC_GRIDS: "WeakKeyDictionary[Any, StaticGrid]" = WeakKeyDictionary()


def static_grid(world: "PhysicsWorld") -> StaticGrid:
    """The grid over ``world``'s fixed bodies, as they are now."""
    grid = STATIC_GRIDS.get(world)
    if grid is None or grid.epoch != world.layout_epoch:
        grid = STATIC_GRIDS[world] = StaticGrid(world)
    return grid


def static_proxies(world: "PhysicsWorld") -> dict[int, Proxy]:
    """The proxy cache belonging to ``world``, made on first use."""
    found = STATIC_PROXIES.get(world)
    if found is None:
        found = STATIC_PROXIES[world] = {}
    return found


@dataclass(frozen=True, slots=True)
class DynamicBody:
    """A moving body as a :class:`CollisionQuery` reports it.

    ``position`` and ``orientation`` (an ``x, y, z, w`` quaternion) place
    ``shape`` in world space, ``velocity`` is in metres a second and ``mass`` in
    kilograms. The arrays are copies, so a later step of the world does not
    change what was reported.
    """

    body: int
    shape: model.Shape
    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    mass: float


@runtime_checkable
class CollisionQuery(Protocol):
    """The world as a character controller reads it."""

    def static_bodies(self) -> Sequence[int]:
        """Ids of the solid bodies that do not move under the solver."""
        ...

    def static_near(self, lo: np.ndarray, hi: np.ndarray) -> Sequence[int]:
        """Ids of the solid fixed bodies whose bounding boxes overlap ``lo`` to ``hi``.

        Ascending, and a subset of :meth:`static_bodies`. A coordinate of ``lo``
        or ``hi`` may be infinite.
        """
        ...

    def static_proxy(self, body: int) -> Proxy:
        """The collision proxy of static body ``body``, in world space."""
        ...

    def dynamic_near(self, lo: np.ndarray, hi: np.ndarray) -> list[DynamicBody]:
        """The solid moving bodies whose bounding box overlaps ``lo`` to ``hi``."""
        ...

    def raycast(self, origin: Vec, direction: Vec, max_distance: float = DEFAULT_RANGE,
                skip: Iterable[int] = ()) -> RayHit | None:
        """The nearest hit along the ray, ignoring the bodies in ``skip``."""
        ...


class WorldQuery:
    """A :class:`CollisionQuery` over a live :class:`~omi_physics.world.PhysicsWorld`.

    Moving bodies are read at the pose they have when asked. Static proxies are
    built on first use and shared by every query over the same world.
    """

    def __init__(self, world: "PhysicsWorld") -> None:
        self.world = world
        self._proxies = static_proxies(world)

    def static_bodies(self) -> list[int]:
        """Ids of the solid bodies that are not dynamic and not sensors."""
        world = self.world
        solid = (world.collider_shape >= 0) & (world.motion_type != 2) & ~world.sensor
        return np.flatnonzero(solid).tolist()

    def static_near(self, lo: np.ndarray, hi: np.ndarray) -> list[int]:
        """The solid fixed bodies whose boxes overlap ``lo`` to ``hi``.

        Answered from a grid that every query on the world shares.
        """
        return static_grid(self.world).near(lo, hi)

    def static_proxy(self, body: int) -> Proxy:
        """The cached world-space proxy of static body ``body``."""
        proxy = self._proxies.get(body)
        if proxy is None:
            world = self.world
            proxy = make_proxy(world.shapes[world.collider_shape[body]],
                               world.position[body], world.orientation[body])
            self._proxies[body] = proxy
        return proxy

    def dynamic_near(self, lo: np.ndarray, hi: np.ndarray) -> list[DynamicBody]:
        """The dynamic bodies with a collider whose box overlaps ``lo`` to ``hi``."""
        world = self.world
        near = ((world.motion_type == 2) & (world.collider_shape >= 0)
                & np.all(world.aabb_max >= lo, axis=1)
                & np.all(world.aabb_min <= hi, axis=1))
        return [DynamicBody(body=j, shape=world.shapes[world.collider_shape[j]],
                            position=world.position[j].copy(),
                            orientation=world.orientation[j].copy(),
                            velocity=world.linear_velocity[j].copy(),
                            mass=float(world.mass[j]))
                for j in np.flatnonzero(near).tolist() if not world.is_sensor(j)]

    def raycast(self, origin: Vec, direction: Vec, max_distance: float = DEFAULT_RANGE,
                skip: Iterable[int] = ()) -> RayHit | None:
        """The nearest hit along the ray, as :func:`omi_physics.raycast.raycast` finds it."""
        return raycast(self.world, origin, direction, max_distance, skip)
