"""What a wheel is on, and what a tyre feels there.

Tarmac, gravel, wet grass and mud are the same tyre on different ground, and
the difference is two numbers: how much friction there is to steer and drive
with, and how hard the ground is to roll over. A car that leaves the road onto
soft going should lose the corner and then bog down, which is exactly those two.

The surface is set by whatever knows where the road is -- the vehicle has no
opinion about it -- and it is per wheel, because two wheels on the verge is the
usual way of finding out about the verge.
"""
import math

import numpy as np
import pytest

from omi_physics import model
from omi_physics.vehicle import (
    RaycastVehicle,
    Surface,
    VehicleTuning,
    car_wheels,
)
from omi_physics.world import PhysicsWorld

STEP = 1.0 / 120.0
EAST = -math.pi / 2.0


def _ground(world, size=4000.0):
    half = size / 2.0
    points = np.array([(-half, 0.0, -half), (half, 0.0, -half),
                       (-half, 0.0, half), (half, 0.0, half)], dtype='d')
    shape = world.add_shape(model.Shape.trimesh(
        points, np.array([(0, 1, 2), (1, 3, 2)], dtype='i')))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape))
    world.refit_aabbs()


def _car(surface=None):
    world = PhysicsWorld()
    _ground(world)
    body = world.add_body(model.Motion(type=model.DYNAMIC, mass=1200.0),
                          position=(0.0, 1.0, 0.0))
    car = RaycastVehicle(world, body, car_wheels(), VehicleTuning())
    car.place((0.0, 1.0, 0.0), EAST)
    if surface is not None:
        car.surface = surface
    return world, body, car


def _driven(surface=None, seconds=8.0, steer=0.0, throttle=1.0):
    world, body, car = _car(surface)
    for _ in range(int(seconds / STEP)):
        car.control(throttle=throttle, steer=steer)
        car.update(STEP)
        world.step(STEP)
    return car.speed(), np.asarray(world.position[body], dtype='d')


class TestWhatASurfaceIs:
    def test_a_car_is_on_tarmac_unless_told_otherwise(self) -> None:
        _world, _body, car = _car()
        assert car.surface.grip == 1.0
        assert car.surface.rolling == 0.0

    def test_a_wheel_takes_the_car_s_surface(self) -> None:
        _world, _body, car = _car(Surface(grip=0.4))
        assert all(wheel.surface().grip == 0.4 for wheel in car.wheels)

    def test_a_wheel_can_be_given_its_own(self) -> None:
        """Two wheels on the verge is the usual way of finding out about it."""
        _world, _body, car = _car()
        car.wheels[0].on = Surface(grip=0.3)
        assert car.wheels[0].surface().grip == 0.3
        assert car.wheels[1].surface().grip == 1.0

    def test_a_surface_reads_as_what_it_is(self) -> None:
        assert 'grip' in repr(Surface(grip=0.4))


class TestSoftGoing:
    def test_less_grip_means_less_acceleration(self) -> None:
        firm, _at = _driven(seconds=5.0)
        soft, _at = _driven(Surface(grip=0.3), seconds=5.0)
        assert soft < firm * 0.8

    def test_rolling_resistance_slows_a_rolling_car(self) -> None:
        rolling, _at = _driven(Surface(rolling=0.35), seconds=8.0)
        free, _at = _driven(seconds=8.0)
        assert rolling < free * 0.6

    def test_a_car_on_soft_going_bogs_down(self) -> None:
        """What leaving the road for the grass is meant to feel like."""
        firm, _at = _driven(seconds=10.0)
        speed, _at = _driven(Surface(grip=0.45, rolling=0.25), seconds=10.0)
        assert speed < firm * 0.5

    def test_it_still_moves_at_all(self) -> None:
        _speed, at = _driven(Surface(grip=0.45, rolling=0.25), seconds=10.0)
        assert float(at[0]) > 20.0

    def test_going_soft_enough_stops_it(self) -> None:
        """Mud, and the end of a run."""
        speed, _at = _driven(Surface(grip=0.35, rolling=0.5), seconds=10.0)
        assert speed < 2.0

    def test_a_surface_of_nothing_is_the_surface_it_had(self) -> None:
        firm, _at = _driven(seconds=5.0)
        same, _at = _driven(Surface(), seconds=5.0)
        assert same == pytest.approx(firm, rel=1e-6)

    def test_less_grip_means_a_wider_corner(self) -> None:
        """How far round a car gets on a held wheel: the one with grip turns."""
        _speed, firm = _driven(seconds=10.0, steer=0.25)
        _speed, soft = _driven(Surface(grip=0.25), seconds=10.0, steer=0.25)
        assert abs(float(soft[2])) < abs(float(firm[2]))


class TestItIsPerWheel:
    def _two_wheels_off(self, surface):
        """A car with its near-side wheels on ``surface``, and how far it got.

        The lateral grip of the wheels still on the road holds the car straight
        -- a yaw would be slip, and slip is what a tyre resists -- so what one
        side dragging costs is speed rather than direction.
        """
        world, body, car = _car()
        for wheel in car.wheels:
            if wheel.spec.position[0] < 0.0:
                wheel.on = surface
        for _ in range(int(10.0 / STEP)):
            car.control(throttle=1.0)
            car.update(STEP)
            world.step(STEP)
        return float(world.position[body][0])

    def test_one_side_on_soft_going_holds_the_car_back(self) -> None:
        soft = Surface(grip=0.3, rolling=0.6)
        assert self._two_wheels_off(soft) < self._two_wheels_off(Surface())

    def test_it_costs_less_than_all_four_wheels_off(self) -> None:
        soft = Surface(grip=0.3, rolling=0.6)
        _speed, both = _driven(soft, seconds=10.0)
        assert self._two_wheels_off(soft) > float(both[0])

    def test_clearing_a_wheel_puts_it_back_on_the_car_s_surface(self) -> None:
        _world, _body, car = _car(Surface(grip=0.5))
        car.wheels[0].on = Surface(grip=0.1)
        car.wheels[0].on = None
        assert car.wheels[0].surface().grip == 0.5


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
