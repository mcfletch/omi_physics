"""The world velocity a character controller moved at over a frame.

Read by anything that has to follow the motion rather than cause it -- an
animation graph choosing walk over run, a footstep cadence, a camera that
leans into a fall.  It is the *measured* displacement per second, so a body
walking into a wall reports the nothing it actually travelled, not the speed it
was trying to.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld

DT = 1.0 / 60.0


def add_box(world, size, center):
    shape = world.add_shape(model.Shape.box(size))
    return world.add_body(model.Motion(type=model.STATIC),
                          collider=model.Collider(shape=shape), position=center)


def floor_world():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    add_box(world, (60, 1, 60), (0, -0.5, 0))
    return world


def standing_character(world, at=(0, 0, 0)):
    ch = CharacterController(world, CharacterCapabilities(), gravity=9.81)
    ch.safe_bind((at[0], 3.0, at[2]))
    return ch


def test_a_fresh_controller_is_at_rest():
    ch = CharacterController(floor_world(), CharacterCapabilities())
    assert np.allclose(ch.velocity, 0.0)


def test_walking_reports_its_ground_speed():
    ch = standing_character(floor_world())
    ch.set_move((1, 0, 0), mode='walk')
    ch.update(DT)
    assert ch.velocity[0] == pytest.approx(ch.caps.walkSpeed, rel=0.25)
    assert abs(ch.velocity[2]) < 0.1


def test_running_beats_walking():
    walk = standing_character(floor_world())
    walk.set_move((1, 0, 0), mode='walk')
    walk.update(DT)
    run = standing_character(floor_world())
    run.set_move((1, 0, 0), mode='run')
    run.update(DT)
    assert run.velocity[0] > walk.velocity[0]


def test_standing_still_reports_no_speed():
    ch = standing_character(floor_world())
    ch.update(DT)
    assert np.linalg.norm(ch.velocity) < 0.1


def test_velocity_is_the_measured_displacement():
    ch = standing_character(floor_world())
    ch.set_move((1, 0, 0), mode='run')
    before = ch.position.copy()
    ch.update(DT)
    assert np.allclose(ch.velocity, (ch.position - before) / DT)


def test_a_falling_body_reports_downward():
    world = floor_world()
    ch = CharacterController(world, CharacterCapabilities(), gravity=9.81)
    ch.safe_bind((0, 60.0, 0))          # above any floor within the snap search
    assert not ch.grounded
    ch.update(DT)
    assert ch.velocity[1] < 0.0


def test_a_body_walking_into_a_wall_reports_the_nothing_it_travelled():
    world = floor_world()
    add_box(world, (1, 4, 20), (1.5, 2, 0))     # a wall just ahead of +x
    ch = standing_character(world)
    ch.set_move((1, 0, 0), mode='run')
    for _ in range(30):                          # settle against the wall
        ch.update(DT)
    assert np.linalg.norm(ch.velocity[[0, 2]]) < 0.2


def test_a_fresh_bind_is_at_rest():
    ch = standing_character(floor_world())
    ch.set_move((1, 0, 0), mode='run')
    ch.update(DT)
    assert np.linalg.norm(ch.velocity) > 0.1
    ch.safe_bind((0, 3.0, 0))
    assert np.allclose(ch.velocity, 0.0)
