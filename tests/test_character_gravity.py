"""A jump is a parabola whatever the step: gravity is integrated exactly.

``position += v dt - g dt^2 / 2`` followed by ``v -= g dt`` puts every sample
of a flight on the true parabola, so the apex and the time to land are those of
``jumpHeight`` and ``gravity`` at any frame rate.
"""
import math

import numpy as np
import pytest

from omi_physics import model
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld

JUMP_HEIGHT = 1.2


def floor_world(gravity):
    world = PhysicsWorld(gravity=model.Gravity(gravity=gravity, direction=(0, -1, 0)))
    shape = world.add_shape(model.Shape.box((60, 1, 60)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape),
                   position=(0, -0.5, 0))
    return world


def flight(gravity, dt):
    """Jump from a standing start; return the feet's height at each step of the flight.

    The first entry is the height at the jump, the rest are sampled every ``dt``
    until the controller is back on the ground.
    """
    caps = CharacterCapabilities(jumpHeight=JUMP_HEIGHT, coyoteTime=0.0, jumpBuffer=0.0)
    character = CharacterController(floor_world(gravity), caps, gravity=gravity)
    character.safe_bind((0.0, 2.0, 0.0))
    assert character.grounded
    heights = [float(character.base()[1])]
    assert character.jump()
    for _ in range(int(3.0 / dt)):
        character.update(dt)
        if character.grounded:
            break
        heights.append(float(character.base()[1]))
    return np.array(heights)


@pytest.mark.parametrize('gravity', [9.81, 42.0])
@pytest.mark.parametrize('rate', [30, 60, 120, 240])
def test_the_apex_and_the_time_to_land_are_those_of_the_jump_height(gravity, rate):
    dt = 1.0 / rate
    heights = flight(gravity, dt)
    assert len(heights) > 3
    # Three samples fix the parabola: the second difference is -g dt^2, and the
    # first difference gives the launch speed.
    pulled = -(heights[2] - 2.0 * heights[1] + heights[0]) / (dt * dt)
    launch = (heights[1] - heights[0]) / dt + pulled * dt / 2.0
    assert pulled == pytest.approx(gravity, abs=1e-6)
    assert launch * launch / (2.0 * pulled) == pytest.approx(JUMP_HEIGHT, abs=1e-9)
    assert 2.0 * launch / pulled == pytest.approx(
        2.0 * math.sqrt(2.0 * JUMP_HEIGHT / gravity), abs=1e-9)


@pytest.mark.parametrize('gravity', [9.81, 42.0])
@pytest.mark.parametrize('rate', [30, 60, 120, 240])
def test_every_sample_of_the_flight_is_on_the_parabola(gravity, rate):
    dt = 1.0 / rate
    heights = flight(gravity, dt)
    launch = math.sqrt(2.0 * gravity * JUMP_HEIGHT)
    times = np.arange(len(heights)) * dt
    expected = heights[0] + launch * times - 0.5 * gravity * times * times
    assert np.max(np.abs(heights - expected)) < 1e-9


def test_a_fall_covers_the_distance_gravity_gives_it():
    """From rest, ``g t^2 / 2`` metres in ``t`` seconds, to the step."""
    gravity, dt = 9.81, 1.0 / 60.0
    character = CharacterController(floor_world(gravity), CharacterCapabilities(),
                                    position=(0.0, 20.0, 0.0), gravity=gravity)
    start = float(character.position[1])
    for step in range(1, 31):
        character.update(dt)
        elapsed = step * dt
        assert start - float(character.position[1]) == pytest.approx(
            0.5 * gravity * elapsed * elapsed, abs=1e-9)


def test_a_fall_stops_gaining_speed_at_terminal_velocity():
    gravity, dt = 9.81, 1.0 / 60.0
    caps = CharacterCapabilities(terminalVelocity=10.0)
    character = CharacterController(floor_world(gravity), caps,
                                    position=(0.0, 500.0, 0.0), gravity=gravity)
    for _ in range(120):
        character.update(dt)
    assert character.vy == -10.0
    before = float(character.position[1])
    character.update(dt)
    assert before - float(character.position[1]) == pytest.approx(10.0 * dt, abs=1e-9)


def test_the_step_that_reaches_terminal_velocity_covers_both_stretches():
    gravity, dt = 9.81, 0.1
    caps = CharacterCapabilities(terminalVelocity=5.0)
    character = CharacterController(floor_world(gravity), caps,
                                     position=(0.0, 500.0, 0.0), gravity=gravity)
    character.vy = -4.5
    before = float(character.position[1])
    character.update(dt)
    reach = 0.5 / gravity
    expected = 4.5 * reach + 0.5 * gravity * reach * reach + 5.0 * (dt - reach)
    assert before - float(character.position[1]) == pytest.approx(expected, abs=1e-9)
    assert character.vy == -5.0
