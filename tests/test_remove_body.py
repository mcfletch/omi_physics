"""Taking a body out again.

A game that streams its world adds a collider for every piece of ground that
arrives and has to take one away for every piece that leaves. Without that the
world only ever grows: a car driving for ten minutes accumulates every tile it
has ever seen, and pays for all of them on every step.

Removal frees the slot rather than compacting the arrays, because a body index
is a handle other code is holding. Compaction would renumber every body above
the removed one and quietly invalidate all of those handles at once.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.raycast import raycast
from omi_physics.world import PhysicsWorld

STEP = 1.0 / 120.0


def _world():
    world = PhysicsWorld()
    floor = world.add_shape(model.Shape.box((100.0, 2.0, 100.0)))
    ground = world.add_body(model.Motion(type=model.STATIC),
                            collider=model.Collider(shape=floor),
                            position=(0.0, -1.0, 0.0))
    return world, floor, ground


def _falling_box(world, shape, position=(0.0, 5.0, 0.0)):
    return world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=shape),
                          position=position)


class TestRemoving:
    def test_a_removed_body_no_longer_collides(self) -> None:
        world, _floor, ground = _world()
        box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
        falling = _falling_box(world, box)
        world.remove_body(ground)
        for _ in range(240):                     # two seconds of falling
            world.step(STEP)
        assert float(world.position[falling][1]) < -5.0, "it landed on nothing"

    def test_the_others_keep_their_indices(self) -> None:
        """A body index is a handle; removing one must not renumber the rest."""
        world, _floor, _ground = _world()
        box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
        first = _falling_box(world, box, (0.0, 5.0, 0.0))
        second = _falling_box(world, box, (10.0, 5.0, 0.0))
        world.remove_body(first)
        world.step(STEP)
        assert float(world.position[second][0]) == pytest.approx(10.0)

    def test_it_stops_moving(self) -> None:
        world, _floor, _ground = _world()
        box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
        falling = _falling_box(world, box)
        world.remove_body(falling)
        before = world.position[falling].copy()
        for _ in range(60):
            world.step(STEP)
        assert np.allclose(world.position[falling], before)

    def test_a_ray_passes_through_it(self) -> None:
        world, _floor, ground = _world()
        assert raycast(world, (0.0, 10.0, 0.0), (0, -1, 0)) is not None
        world.remove_body(ground)
        assert raycast(world, (0.0, 10.0, 0.0), (0, -1, 0)) is None

    def test_removing_twice_is_harmless(self) -> None:
        world, _floor, ground = _world()
        world.remove_body(ground)
        world.remove_body(ground)
        world.step(STEP)

    def test_an_index_that_was_never_a_body_is_refused(self) -> None:
        world, _floor, _ground = _world()
        with pytest.raises(IndexError):
            world.remove_body(99)


class TestTheSlotComesBack:
    def test_the_next_body_reuses_it(self) -> None:
        """Otherwise a streaming world grows a slot per tile it ever loaded."""
        world, _floor, _ground = _world()
        box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
        first = _falling_box(world, box)
        world.remove_body(first)
        assert _falling_box(world, box) == first

    def test_the_count_stays_where_it_was(self) -> None:
        world, _floor, _ground = _world()
        box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
        before = world.body_count
        for _ in range(50):
            world.remove_body(_falling_box(world, box))
        assert world.body_count <= before + 1

    def test_a_reused_slot_is_a_clean_body(self) -> None:
        world, _floor, _ground = _world()
        box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
        first = _falling_box(world, box, (0.0, 20.0, 0.0))
        for _ in range(60):
            world.step(STEP)
        world.remove_body(first)
        again = _falling_box(world, box, (3.0, 4.0, 0.0))
        assert again == first
        assert np.allclose(world.position[again], (3.0, 4.0, 0.0))
        assert not world.linear_velocity[again].any()

    def test_a_reused_slot_collides_again(self) -> None:
        world, _floor, _ground = _world()
        box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
        world.remove_body(_falling_box(world, box))
        landing = _falling_box(world, box, (0.0, 3.0, 0.0))
        for _ in range(180):
            world.step(STEP)
        assert float(world.position[landing][1]) > 0.0, "it fell through"

    def test_the_handle_is_forgotten_with_the_body(self) -> None:
        world, _floor, _ground = _world()
        box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
        first = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                               collider=model.Collider(shape=box),
                               handle='the old one')
        world.remove_body(first)
        again = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                               collider=model.Collider(shape=box),
                               handle='the new one')
        assert world.bodies[again] == 'the new one'


class TestAStreamedWorldStaysBounded:
    def test_tiles_coming_and_going_do_not_pile_up(self) -> None:
        """The shape of a streaming session: a tile arrives, an old one leaves."""
        world = PhysicsWorld()
        points = np.array([(0, 0, 0), (10, 0, 0), (0, 0, 10), (10, 0, 10)], 'd')
        faces = np.array([(0, 1, 2), (1, 3, 2)], 'i')
        resident: list[int] = []
        for step in range(60):
            shape = world.add_shape(model.Shape.trimesh(
                points + np.array([step * 10.0, 0, 0]), faces))
            resident.append(world.add_body(
                model.Motion(type=model.STATIC),
                collider=model.Collider(shape=shape)))
            if len(resident) > 8:
                world.remove_body(resident.pop(0))
            world.step(STEP)
        assert world.body_count <= 10
