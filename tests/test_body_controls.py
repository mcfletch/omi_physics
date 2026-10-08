"""Steady forces and torques, exact rotation and each body's own sleep (no GL)."""
import math

import numpy as np
import pytest

from omi_physics import mathutil, model
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0
WEIGHTLESS = model.Gravity(gravity=0.0)


def _cube(world: PhysicsWorld, mass: float = 1.0, **motion) -> int:
    shape = world.add_shape(model.Shape.box((1, 1, 1)))
    return world.add_body(model.Motion(mass=mass, **motion),
                          collider=model.Collider(shape=shape))


def _run(world: PhysicsWorld, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        world.step(DT)


class TestSteadyForces:
    def test_a_force_pushes_every_step(self):
        world = PhysicsWorld(gravity=WEIGHTLESS)
        cube = _cube(world, mass=2.0)
        world.set_force(cube, (4.0, 0.0, 0.0))
        _run(world, 1.0)
        assert world.linear_velocity[cube][0] == pytest.approx(2.0)

    def test_a_torque_turns_every_step(self):
        world = PhysicsWorld(gravity=WEIGHTLESS)
        cube = _cube(world)
        world.set_torque(cube, (0.0, 1.0, 0.0))
        _run(world, 1.0)
        # A unit cube of 1 kg has 1/6 kg m^2 about each axis.
        assert world.angular_velocity[cube][1] == pytest.approx(6.0)

    def test_a_force_wakes_a_sleeping_body(self):
        world = PhysicsWorld(gravity=WEIGHTLESS)
        cube = _cube(world)
        _run(world, 1.0)
        assert not world.awake[cube]
        world.set_force(cube, (1.0, 0.0, 0.0))
        _run(world, 0.5)
        assert world.linear_velocity[cube][0] > 0.4

    def test_a_static_body_is_not_pushed(self):
        world = PhysicsWorld(gravity=WEIGHTLESS)
        cube = _cube(world, type=model.STATIC)
        world.set_force(cube, (1.0, 0.0, 0.0))
        _run(world, 0.5)
        assert np.allclose(world.position[cube], 0.0)

    def test_the_force_is_taken_away(self):
        world = PhysicsWorld(gravity=WEIGHTLESS)
        cube = _cube(world)
        world.set_force(cube, (1.0, 0.0, 0.0))
        _run(world, 0.5)
        world.set_force(cube, (0.0, 0.0, 0.0))
        speed = world.linear_velocity[cube][0]
        _run(world, 0.5)
        assert world.linear_velocity[cube][0] == pytest.approx(speed)


class TestExactRotation:
    def _spun(self, exact: bool) -> np.ndarray:
        world = PhysicsWorld(gravity=WEIGHTLESS, sleep_enabled=False)
        wheel = _cube(world, angularVelocity=(50.0, 0.0, 0.0))
        world.set_exact_rotation(wheel, exact)
        _run(world, 1.0)
        return mathutil.quat_to_axis_angle(world.orientation[wheel])

    def test_a_fast_spin_turns_as_far_as_it_should(self):
        axis_angle = self._spun(True)
        expected = math.remainder(50.0, 2 * math.pi)
        turned = axis_angle[3] if axis_angle[0] > 0 else -axis_angle[3]
        assert math.remainder(turned - expected, 2 * math.pi) == pytest.approx(0.0, abs=1e-6)

    def test_the_first_order_step_falls_behind(self):
        axis_angle = self._spun(False)
        expected = math.remainder(50.0, 2 * math.pi)
        turned = axis_angle[3] if axis_angle[0] > 0 else -axis_angle[3]
        assert abs(math.remainder(turned - expected, 2 * math.pi)) > 0.1


class TestOwnSleep:
    def test_a_body_that_may_not_sleep_stays_awake(self):
        world = PhysicsWorld(gravity=WEIGHTLESS)
        cube = _cube(world)
        world.set_sleep(cube, after=math.inf)
        _run(world, 2.0)
        assert world.awake[cube]

    def test_a_body_sleeps_at_its_own_speeds(self):
        world = PhysicsWorld(gravity=WEIGHTLESS)
        cube = _cube(world, linearVelocity=(0.5, 0.0, 0.0))
        world.set_sleep(cube, linear=1.0, angular=1.0, after=0.1)
        _run(world, 0.2)
        assert not world.awake[cube]

    def test_the_world_defaults_hold_where_a_body_says_nothing(self):
        world = PhysicsWorld(gravity=WEIGHTLESS)
        world.sleep_linear_speed = 1.0
        world.sleep_time = 0.1
        cube = _cube(world, linearVelocity=(0.5, 0.0, 0.0))
        _run(world, 0.2)
        assert not world.awake[cube]
