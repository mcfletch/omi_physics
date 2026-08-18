"""How hard a car pulls at speed, and what stops it going faster.

A constant force at every speed is not what any drivetrain gives. A motor gives
constant *torque* up to a base speed and constant *power* above it, so the pull
falls away as the car gets going -- which is most of the difference between a
car that feels like it is accelerating and one that feels like it is on rails.

And nothing was pushing back. With no aerodynamic drag a car's top speed is set
by rolling resistance, which for anything with wheels means a top speed far
higher than the thing could ever reach. Drag rises with the square of speed, so
it is what actually decides how fast a car will go.
"""
import pytest

from omi_physics import model
from omi_physics.vehicle import RaycastVehicle, VehicleTuning, car_wheels
from omi_physics.world import PhysicsWorld

STEP = 1.0 / 120.0


def _world():
    world = PhysicsWorld()
    ground = world.add_shape(model.Shape.box((9000.0, 4.0, 9000.0)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=ground),
                   position=(0.0, -2.0, 0.0))
    return world


def _car(world, tuning=None, mass=1180.0):
    chassis = world.add_shape(model.Shape.box((1.85, 0.62, 4.2)))
    body = world.add_body(model.Motion(type=model.DYNAMIC, mass=mass),
                          collider=model.Collider(shape=chassis),
                          position=(0.0, 0.6, 0.0))
    return RaycastVehicle(world, body, car_wheels(
        wheelbase=2.55, track=1.58, height=-0.17, drive='rear', radius=0.33,
        suspension_travel=0.22, suspension_stiffness=26.0,
        suspension_damping=0.55, grip=1.9),
        tuning=tuning or VehicleTuning())


def _drive(world, car, seconds, throttle=1.0):
    for _ in range(int(seconds / STEP)):
        car.control(throttle=throttle)
        car.update(STEP)
        world.step(STEP)
    return car.speed()


class TestThePowerCurve:
    def test_below_the_base_speed_it_is_all_the_force_there_is(self) -> None:
        tuning = VehicleTuning(engine_force=9000.0, base_speed=15.0)
        assert tuning.drive_force(0.0) == pytest.approx(9000.0)
        assert tuning.drive_force(14.0) == pytest.approx(9000.0)

    def test_above_it_the_power_is_what_is_constant(self) -> None:
        tuning = VehicleTuning(engine_force=9000.0, base_speed=15.0)
        assert tuning.drive_force(30.0) == pytest.approx(4500.0)
        assert tuning.drive_force(45.0) == pytest.approx(3000.0)

    def test_no_base_speed_is_the_flat_force_it_always_was(self) -> None:
        tuning = VehicleTuning(engine_force=9000.0, base_speed=0.0)
        assert tuning.drive_force(80.0) == pytest.approx(9000.0)

    def test_the_pull_falls_away_as_the_car_gets_going(self) -> None:
        world = _world()
        car = _car(world, VehicleTuning(engine_force=9000.0, base_speed=8.0,
                                        drag=0.9))
        early = _drive(world, car, 3.0)
        later = _drive(world, car, 3.0) - early
        assert later < early * 0.6

    def test_and_a_flat_force_does_not(self) -> None:
        world = _world()
        car = _car(world, VehicleTuning(engine_force=9000.0, base_speed=0.0,
                                        drag=0.0))
        early = _drive(world, car, 3.0)
        later = _drive(world, car, 3.0) - early
        assert later > early * 0.75


class TestWhatStopsItGoingFaster:
    def test_drag_rises_with_the_square_of_speed(self) -> None:
        tuning = VehicleTuning(drag=0.4)
        assert tuning.drag_force(10.0) == pytest.approx(40.0)
        assert tuning.drag_force(20.0) == pytest.approx(160.0)

    def test_it_pushes_back_whichever_way_the_car_goes(self) -> None:
        tuning = VehicleTuning(drag=0.4)
        assert tuning.drag_force(-20.0) == pytest.approx(160.0)

    def test_a_car_with_drag_tops_out_lower(self) -> None:
        fast = _world()
        loose = _car(fast, VehicleTuning(engine_force=9000.0, drag=0.0))
        slippery = _drive(fast, loose, 40.0)
        slow = _world()
        blunt = _car(slow, VehicleTuning(engine_force=9000.0, drag=0.45))
        boxy = _drive(slow, blunt, 40.0)
        assert boxy < slippery * 0.8

    def test_and_stops_accelerating_rather_than_creeping_up(self) -> None:
        world = _world()
        car = _car(world, VehicleTuning(engine_force=9000.0, base_speed=15.0,
                                        drag=0.9))
        _drive(world, car, 40.0)
        settled = car.speed()
        _drive(world, car, 10.0)
        assert abs(car.speed() - settled) < 1.0

    def test_a_coasting_car_slows_down(self) -> None:
        world = _world()
        car = _car(world, VehicleTuning(engine_force=9000.0, drag=0.45))
        _drive(world, car, 12.0)
        rolling = car.speed()
        _drive(world, car, 6.0, throttle=0.0)
        assert car.speed() < rolling - 2.0


class TestItIsStillDrivable:
    def test_a_default_car_still_gets_going(self) -> None:
        world = _world()
        car = _car(world)
        assert _drive(world, car, 6.0) > 8.0

    def test_and_still_reaches_a_road_speed(self) -> None:
        world = _world()
        car = _car(world, VehicleTuning(engine_force=9000.0, base_speed=15.0,
                                        drag=0.9))
        assert 30.0 < _drive(world, car, 45.0) < 90.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
