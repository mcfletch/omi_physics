"""The questions a character asks of a world, answered by a world or by a copy of one."""
from collections.abc import Iterable, Sequence

import numpy as np
import pytest

from omi_physics import model, raycast
from omi_physics.body import Proxy, make_proxy, world_aabb
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.collisionquery import CollisionQuery, DynamicBody, WorldQuery
from omi_physics.world import PhysicsWorld

DT = 1.0 / 60.0


def level() -> tuple[PhysicsWorld, dict[str, int]]:
    """A floor, a far wall, a crate near the origin, a crate far away and a sensor."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    ids: dict[str, int] = {}
    floor = world.add_shape(model.Shape.box((100, 1, 100)))
    ids['floor'] = world.add_body(model.Motion(type=model.STATIC),
                                  collider=model.Collider(shape=floor),
                                  position=(0, -0.5, 0))
    crate = world.add_shape(model.Shape.box((0.5, 0.5, 0.5)))
    ids['near'] = world.add_body(model.Motion(type=model.DYNAMIC, mass=20.0,
                                              linearVelocity=(1.0, 0.0, 0.0)),
                                 collider=model.Collider(shape=crate),
                                 position=(2.0, 0.25, 0.0))
    ids['far'] = world.add_body(model.Motion(type=model.DYNAMIC, mass=5.0),
                                collider=model.Collider(shape=crate),
                                position=(40.0, 0.25, 0.0))
    ids['sensor'] = world.add_body(model.Motion(type=model.STATIC),
                                   collider=model.Collider(shape=crate),
                                   position=(5.0, 0.25, 0.0), sensor=True)
    return world, ids


class FrozenQuery:
    """A copy of a world's colliders taken at one moment, as a published view would be."""

    def __init__(self, world: PhysicsWorld) -> None:
        live = WorldQuery(world)
        self.statics = {body: live.static_proxy(body) for body in live.static_bodies()}
        self.dynamics = live.dynamic_near(np.full(3, -1e9), np.full(3, 1e9))
        self.world = world

    def static_bodies(self) -> Sequence[int]:
        return list(self.statics)

    def static_near(self, lo: np.ndarray, hi: np.ndarray) -> Sequence[int]:
        return [body for body, proxy in self.statics.items()
                if np.all(proxy.aabb()[1] >= lo) and np.all(proxy.aabb()[0] <= hi)]

    def static_proxy(self, body: int) -> Proxy:
        return self.statics[body]

    def dynamic_near(self, lo: np.ndarray, hi: np.ndarray) -> list[DynamicBody]:
        found = []
        for body in self.dynamics:
            low, high = world_aabb(body.shape, body.position, body.orientation)
            if np.all(high >= lo) and np.all(low <= hi):
                found.append(body)
        return found

    def raycast(self, origin: Sequence[float], direction: Sequence[float],
                max_distance: float = 1000.0, skip: Iterable[int] = ()
                ) -> raycast.RayHit | None:
        return raycast.raycast(self.world, origin, direction, max_distance, skip)


class TestWorldQuery:
    def test_it_is_a_collision_query(self) -> None:
        world, _ = level()
        assert isinstance(WorldQuery(world), CollisionQuery)
        assert isinstance(FrozenQuery(world), CollisionQuery)

    def test_static_bodies_are_the_solid_ones_that_do_not_move(self) -> None:
        world, ids = level()
        assert list(WorldQuery(world).static_bodies()) == [ids['floor']]

    def test_a_static_proxy_is_in_world_space_and_shared(self) -> None:
        world, ids = level()
        query = WorldQuery(world)
        proxy = query.static_proxy(ids['floor'])
        lo, hi = proxy.aabb()
        assert hi[1] == pytest.approx(0.0)
        assert lo[1] == pytest.approx(-1.0)
        assert WorldQuery(world).static_proxy(ids['floor']) is proxy

    def test_dynamic_bodies_near_a_box(self) -> None:
        world, ids = level()
        found = WorldQuery(world).dynamic_near(np.array([1.0, 0.0, -1.0]),
                                               np.array([3.0, 1.0, 1.0]))
        assert [d.body for d in found] == [ids['near']]

    def test_a_dynamic_body_carries_its_shape_pose_velocity_and_mass(self) -> None:
        world, ids = level()
        crate = WorldQuery(world).dynamic_near(np.array([1.0, 0.0, -1.0]),
                                               np.array([3.0, 1.0, 1.0]))[0]
        assert crate.shape.type == 'box'
        assert crate.position == pytest.approx([2.0, 0.25, 0.0])
        assert crate.orientation == pytest.approx([0.0, 0.0, 0.0, 1.0])
        assert crate.velocity == pytest.approx([1.0, 0.0, 0.0])
        assert crate.mass == pytest.approx(20.0)
        assert crate.body == ids['near']

    def test_a_dynamic_body_reports_its_pose_as_of_the_query(self) -> None:
        world, _ = level()
        query = WorldQuery(world)
        before = query.dynamic_near(np.array([-1e9] * 3), np.array([1e9] * 3))[0]
        world.step(DT)
        again = query.dynamic_near(np.array([-1e9] * 3), np.array([1e9] * 3))[0]
        assert before.position[0] == pytest.approx(2.0)
        assert again.position[0] > before.position[0]

    def test_a_ray_meets_what_raycast_meets(self) -> None:
        world, ids = level()
        hit = WorldQuery(world).raycast((0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
        assert hit is not None
        assert hit.body == ids['floor']
        assert hit.distance == pytest.approx(1.0)

    def test_a_ray_skips_the_bodies_it_is_told_to(self) -> None:
        world, ids = level()
        query = WorldQuery(world)
        hit = query.raycast((-1.0, 0.25, 0.0), (1.0, 0.0, 0.0))
        assert hit is not None and hit.body == ids['near']
        assert query.raycast((-1.0, 0.25, 0.0), (1.0, 0.0, 0.0), 10.0,
                             skip=[ids['near']]) is None


class TestACharacterOnAView:
    """The controller reads only the protocol, so a frozen copy gives the same walk."""

    def walk(self, source: PhysicsWorld | CollisionQuery) -> np.ndarray:
        character = CharacterController(source, CharacterCapabilities(),
                                        position=(0.0, 2.0, 0.0), gravity=9.81)
        character.set_move((1, 0, 0), 'run')
        for _ in range(90):
            character.update(DT)
        assert character.grounded
        return character.position

    def test_it_ends_where_it_would_on_the_world(self) -> None:
        world, _ = level()
        live = self.walk(world)
        frozen = self.walk(FrozenQuery(world))
        assert frozen == pytest.approx(live, abs=1e-12)

    def test_it_stands_on_a_floor_it_was_given_no_world_for(self) -> None:
        world, _ = level()
        position = self.walk(FrozenQuery(world))
        assert position[1] == pytest.approx(0.9, abs=0.01)

    def test_its_body_needs_a_world(self) -> None:
        world, _ = level()
        with pytest.raises(ValueError, match='PhysicsWorld'):
            CharacterController(FrozenQuery(world), CharacterCapabilities(), body=True)


def test_a_proxy_for_a_dynamic_body_is_made_from_its_pose() -> None:
    world, _ = level()
    crate = WorldQuery(world).dynamic_near(np.array([1.0, 0.0, -1.0]),
                                           np.array([3.0, 1.0, 1.0]))[0]
    lo, hi = make_proxy(crate.shape, crate.position, crate.orientation).aabb()
    assert lo == pytest.approx([1.75, 0.0, -0.25])
    assert hi == pytest.approx([2.25, 0.5, 0.25])


# -- the moving bodies a query keeps ------------------------------------------------

def test_the_moving_bodies_of_a_world_are_found_once_while_none_is_added_or_removed():
    from omi_physics.collisionquery import dynamic_rows
    world, ids = level()
    first = dynamic_rows(world)
    world.step(DT)
    assert dynamic_rows(world) is first
    assert sorted(first.tolist()) == sorted([ids['near'], ids['far']])


def test_the_moving_bodies_follow_a_body_added_and_one_removed():
    from omi_physics.collisionquery import dynamic_rows
    world, ids = level()
    dynamic_rows(world)
    crate = world.add_shape(model.Shape.box((0.5, 0.5, 0.5)))
    added = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                           collider=model.Collider(shape=crate), position=(2.0, 0.25, 1.0))
    world.remove_body(ids['far'])
    assert sorted(dynamic_rows(world).tolist()) == sorted([ids['near'], added])
    found = WorldQuery(world).dynamic_near(np.array([1.0, -1.0, -1.0]), np.array([3.0, 1.0, 2.0]))
    assert sorted(body.body for body in found) == sorted([ids['near'], added])
