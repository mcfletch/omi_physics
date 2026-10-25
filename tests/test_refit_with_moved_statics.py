"""Boxes of the bodies that move are refitted on a step in which a static body was moved too."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0


def world_with_a_ball_over_a_floor():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    floor = world.add_shape(model.Shape.box((400, 1, 400)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0, -0.5, 0))
    pad = world.add_shape(model.Shape.box((2, 1, 2)))
    mover = world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=pad),
                           position=(20, 0, 0))
    ball = world.add_shape(model.Shape.sphere(0.3))
    dropped = world.add_body(model.Motion(type=model.DYNAMIC, mass=2.0),
                             collider=model.Collider(shape=ball), position=(0, 3, 0))
    return world, mover, dropped


@pytest.mark.parametrize('how', ['place_body', 'write'])
def test_a_ball_rests_on_the_floor_while_a_static_body_is_moved_every_step(how):
    world, mover, ball = world_with_a_ball_over_a_floor()
    lowest = 99.0
    for number in range(240):
        at = (20 + 0.01 * number, 0, 0)
        if how == 'place_body':
            world.place_body(mover, position=at)
        else:
            world.position[mover] = at
        world.step(DT)
        lowest = min(lowest, float(world.position[ball][1]))
    assert lowest == pytest.approx(0.3, abs=0.05)
    assert float(world.position[ball][1]) == pytest.approx(0.3, abs=0.05)


def test_the_moved_static_body_has_its_box_where_it_is():
    world, mover, _ball = world_with_a_ball_over_a_floor()
    world.position[mover] = (30.0, 0, 0)
    world.step(DT)
    assert np.allclose((world.aabb_min[mover] + world.aabb_max[mover]) / 2.0, (30.0, 0.0, 0.0),
                       atol=1e-6)
