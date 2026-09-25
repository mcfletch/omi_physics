"""A kinematic body sent to a place at a speed, and stopped on it.

A door rising to open, a lift going to a floor, a platform coming back: the
game says where and how fast, and the mover sets the velocity each frame and
puts the body exactly on the target at the end.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.kinematic import KinematicMover
from omi_physics.world import PhysicsWorld

STEP = 1.0 / 120.0
FRAME = 1.0 / 60.0


def world_with_door(position=(0.0, 1.0, 0.0)):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         fixed_dt=STEP, sleep_enabled=False)
    shape = world.add_shape(model.Shape.box((2.0, 2.0, 0.2)))
    door = world.add_body(model.Motion(type=model.KINEMATIC),
                          collider=model.Collider(shape=shape), position=position)
    return world, door


def run(world, mover, seconds):
    for _ in range(round(seconds / FRAME)):
        mover.update(FRAME)
        world.advance(FRAME)


def test_it_starts_where_the_body_is():
    world, door = world_with_door()
    mover = KinematicMover(world, door, speed=1.5)
    assert np.allclose(mover.target, (0.0, 1.0, 0.0))
    assert mover.arrived


def test_it_travels_at_its_speed():
    world, door = world_with_door()
    mover = KinematicMover(world, door, speed=1.5)
    mover.target = (0.0, 3.2, 0.0)
    assert not mover.arrived
    run(world, mover, 0.5)
    assert world.linear_velocity[door][1] == pytest.approx(1.5)
    assert world.position[door][1] == pytest.approx(1.0 + 0.75, abs=0.03)


def test_it_stops_exactly_on_the_target():
    world, door = world_with_door()
    mover = KinematicMover(world, door, speed=1.5)
    mover.target = (0.0, 3.2, 0.0)
    run(world, mover, 3.0)
    assert tuple(world.position[door]) == pytest.approx((0.0, 3.2, 0.0), abs=1e-9)
    assert not world.linear_velocity[door].any()
    assert mover.arrived


def test_it_goes_back_when_the_target_does():
    world, door = world_with_door()
    mover = KinematicMover(world, door, speed=1.5)
    mover.target = (0.0, 3.2, 0.0)
    run(world, mover, 3.0)
    mover.target = (0.0, 1.0, 0.0)
    run(world, mover, 3.0)
    assert tuple(world.position[door]) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_a_speed_of_nothing_is_refused():
    world, door = world_with_door()
    with pytest.raises(ValueError, match='speed'):
        KinematicMover(world, door, speed=0.0)
