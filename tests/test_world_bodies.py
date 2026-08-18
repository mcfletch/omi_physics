"""How many bodies a world is carrying, and how many rows it iterates.

Those are two different numbers once anything streams. A removed body's slot is
emptied in place rather than compacted out, because other code holds body
indices as handles and renumbering would invalidate every one of them at once.
So the columns keep a row for it, and the next body added takes that row back.

``body_count`` is the number of rows -- what the solver, the broadphase and
anything else walking the columns needs. ``live_body_count`` is how many of them
hold a body, which is what a caller streaming a landscape in and out asks when
it wants to know whether it is leaking.
"""
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld


def _world():
    world = PhysicsWorld()
    return world, world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))


def _static(world, shape, at=(0.0, 0.0, 0.0)):
    return world.add_body(model.Motion(type=model.STATIC),
                          collider=model.Collider(shape=shape), position=at)


class TestHowManyAreThere:
    def test_an_empty_world_carries_none(self) -> None:
        assert PhysicsWorld().live_body_count == 0

    def test_adding_one_counts_it(self) -> None:
        world, shape = _world()
        _static(world, shape)
        assert world.live_body_count == 1

    def test_removing_it_uncounts_it(self) -> None:
        world, shape = _world()
        world.remove_body(_static(world, shape))
        assert world.live_body_count == 0

    def test_a_world_that_streams_does_not_grow_for_ever(self) -> None:
        world, shape = _world()
        for _round in range(50):
            held = [_static(world, shape, (float(i), 0.0, 0.0))
                    for i in range(4)]
            for one in held:
                world.remove_body(one)
        assert world.live_body_count == 0

    def test_the_slot_comes_back(self) -> None:
        world, shape = _world()
        world.remove_body(_static(world, shape))
        _static(world, shape)
        assert world.live_body_count == 1


class TestHowManyRowsThereAre:
    """The solver and the broadphase walk the columns, so what they want is the
    number of rows -- emptied slots included, because the row is still there."""

    def test_it_counts_every_slot(self) -> None:
        world, shape = _world()
        held = [_static(world, shape, (float(i), 0.0, 0.0)) for i in range(3)]
        world.remove_body(held[1])
        assert world.body_count == 3
        assert world.live_body_count == 2

    def test_the_columns_are_that_long(self) -> None:
        world, shape = _world()
        for index in range(3):
            _static(world, shape, (float(index), 0.0, 0.0))
        assert len(world.position) == world.body_count

    def test_an_emptied_slot_does_not_move(self) -> None:
        """It is still stepped, so it has to be inert rather than merely unused."""
        world, shape = _world()
        one = world.add_body(model.Motion(type=model.DYNAMIC, mass=10.0),
                             collider=model.Collider(shape=shape),
                             position=(0.0, 50.0, 0.0))
        world.remove_body(one)
        before = world.position[one].copy()
        for _step in range(60):
            world.step(1.0 / 60.0)
        assert tuple(world.position[one]) == tuple(before)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
