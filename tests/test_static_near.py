"""The static bodies near a box, and a character that reads only those.

A character used to test its capsule against every static body in the world on
each depenetration, so a forest of trunks made each step cost a trunk's worth of
shape tests per tree in the world.  ``static_near`` answers from a grid over the
bodies' boxes, and the controller asks it for the neighbourhood of its capsule.
"""
import gc
import weakref
from collections.abc import Iterable

import numpy as np

from omi_physics import model
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.collisionquery import CollisionQuery, WorldQuery
from omi_physics.world import PhysicsWorld

DT = 1.0 / 60.0


def world_with_floor() -> tuple[PhysicsWorld, int]:
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    floor = world.add_shape(model.Shape.box((2000, 1, 2000)))
    body = world.add_body(model.Motion(type=model.STATIC),
                          collider=model.Collider(shape=floor), position=(0, -0.5, 0))
    return world, body


def add_posts(world: PhysicsWorld, positions: Iterable[tuple[float, float, float]]) -> np.ndarray:
    post = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.4))
    return world.add_bodies(post, np.asarray(list(positions), dtype='d'),
                            collider=model.Collider(shape=post))


def posts_in_a_row(world: PhysicsWorld, count: int, spacing: float = 3.0) -> np.ndarray:
    return add_posts(world, [(x * spacing, 2.0, 40.0) for x in range(count)])


BOX_LO = np.array([-1.0, -1.0, 39.0])
BOX_HI = np.array([4.0, 5.0, 41.0])


class TestStaticNear:
    def test_it_lists_the_static_bodies_whose_boxes_overlap_the_box(self) -> None:
        world, floor = world_with_floor()
        posts = posts_in_a_row(world, 100)
        found = set(WorldQuery(world).static_near(BOX_LO, BOX_HI))
        assert found == {floor, int(posts[0]), int(posts[1])}

    def test_a_body_larger_than_the_grid_cell_is_found_wherever_it_reaches(self) -> None:
        world, floor = world_with_floor()
        far = np.array([900.0, -1.0, -900.0])
        assert floor in set(WorldQuery(world).static_near(far, far + 1.0))

    def test_it_leaves_out_dynamic_bodies_and_sensors(self) -> None:
        world, floor = world_with_floor()
        crate = world.add_shape(model.Shape.box((0.5, 0.5, 0.5)))
        world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                       collider=model.Collider(shape=crate), position=(0.0, 0.25, 40.0))
        world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=crate),
                       position=(1.0, 0.25, 40.0), sensor=True)
        assert set(WorldQuery(world).static_near(BOX_LO, BOX_HI)) == {floor}

    def test_a_box_with_no_bottom_reaches_what_is_below_it(self) -> None:
        world, floor = world_with_floor()
        low = np.array([-1.0, -np.inf, 39.0])
        assert floor in set(WorldQuery(world).static_near(low, np.array([1.0, 0.0, 41.0])))

    def test_it_agrees_with_a_scan_of_every_static_body(self) -> None:
        world, _ = world_with_floor()
        rng = np.random.default_rng(3)
        add_posts(world, [(x, 2.0, z) for x, z in rng.uniform(-300, 300, (400, 2))])
        query = WorldQuery(world)
        for _ in range(40):
            centre = rng.uniform(-300, 300, 3) * (1, 0.01, 1)
            size = rng.uniform(0.5, 30.0, 3)
            lo, hi = centre - size, centre + size
            expected = {j for j in query.static_bodies()
                        if np.all(world.aabb_max[j] >= lo) and np.all(world.aabb_min[j] <= hi)}
            assert set(query.static_near(lo, hi)) == expected


class TestTheIndexFollowsTheWorld:
    def test_a_body_added_afterwards_is_found(self) -> None:
        world, _ = world_with_floor()
        query = WorldQuery(world)
        query.static_near(BOX_LO, BOX_HI)
        added = posts_in_a_row(world, 1)
        assert int(added[0]) in set(query.static_near(BOX_LO, BOX_HI))

    def test_a_body_removed_afterwards_is_gone(self) -> None:
        world, _ = world_with_floor()
        posts = posts_in_a_row(world, 3)
        query = WorldQuery(world)
        assert int(posts[1]) in set(query.static_near(BOX_LO, BOX_HI))
        world.remove_body(int(posts[1]))
        assert int(posts[1]) not in set(query.static_near(BOX_LO, BOX_HI))

    def test_a_static_body_placed_elsewhere_is_found_there(self) -> None:
        world, _ = world_with_floor()
        post = int(posts_in_a_row(world, 1)[0])
        query = WorldQuery(world)
        query.static_near(BOX_LO, BOX_HI)
        world.place_body(post, (500.0, 2.0, 500.0))
        assert post not in set(query.static_near(BOX_LO, BOX_HI))
        assert post in set(query.static_near(np.array([499.0, 0.0, 499.0]),
                                             np.array([501.0, 3.0, 501.0])))

    def test_moving_a_dynamic_body_does_not_rebuild_it(self) -> None:
        world, _ = world_with_floor()
        crate = world.add_shape(model.Shape.box((0.5, 0.5, 0.5)))
        mover = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                               collider=model.Collider(shape=crate), position=(0.0, 0.25, 0.0))
        before = world.layout_epoch
        world.place_body(mover, (1.0, 0.25, 0.0))
        assert world.layout_epoch == before


def test_the_grid_does_not_keep_its_world_alive() -> None:
    world, _ = world_with_floor()
    WorldQuery(world).static_near(BOX_LO, BOX_HI)
    gone = weakref.ref(world)
    del world
    gc.collect()
    assert gone() is None


class CountingQuery(WorldQuery):
    """A world query that counts the shape proxies a character asks for."""

    asked = 0

    def static_proxy(self, body):
        self.asked += 1
        return super().static_proxy(body)


def walk(post_count: int) -> tuple[CharacterController, CountingQuery]:
    world, _ = world_with_floor()
    rng = np.random.default_rng(7)
    spots = rng.uniform(-900, 900, (post_count, 2))
    keep = np.hypot(spots[:, 0] - 30.0, spots[:, 1]) > 8.0
    add_posts(world, [(x, 2.0, z) for x, z in spots[keep]])
    add_posts(world, [(30.0, 2.0, 0.0)])
    query = CountingQuery(world)
    me = CharacterController(query, CharacterCapabilities(), position=(0.0, 1.0, 0.0),
                             gravity=9.81)
    return me, query


class TestACharacterAmongManyBodies:
    def test_it_is_stopped_by_the_post_in_its_way(self) -> None:
        me, _ = walk(2000)
        me.set_move((1, 0, 0), 'run')
        for _ in range(60 * 5 + 20):
            me.update(DT)
        stopped = 30.0 - 0.4 - me.caps.radius
        assert stopped - 0.05 < me.position[0] < stopped + 0.05

    def test_a_frame_asks_for_the_same_proxies_whatever_the_world_holds(self) -> None:
        asked = []
        for count in (500, 4000):
            me, query = walk(count)
            for _ in range(30):
                me.update(DT)
            query.asked = 0
            for _ in range(30):
                me.update(DT)
            asked.append(query.asked)
        assert asked[1] <= asked[0] * 1.5 + 10
        assert asked[1] < 4000

    def test_a_frozen_view_without_static_near_is_refused_by_the_protocol(self) -> None:
        class NoNear:
            def static_bodies(self):
                return []

        assert not isinstance(NoNear(), CollisionQuery)
