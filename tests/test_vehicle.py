"""A raycast vehicle: does it hold itself up, go, stop, and turn?

The car is a rigid body with no wheels in it. Each wheel is a ray cast down
from where the wheel would be, a spring pushing the body up off what it finds,
and a patch of friction driving and steering at the contact. That is the model
every driving game uses, and these are the questions it has to answer.

Everything runs on a flat static floor with no rendering: a vehicle is numbers.
"""
import math

import numpy as np
import pytest

from omi_physics import model
from omi_physics.vehicle import RaycastVehicle, VehicleTuning, car_wheels
from omi_physics.world import PhysicsWorld

#: A floor big enough that a car at speed does not drive off it in the two or
#: three seconds a test runs for.
FLOOR = 4000.0
STEP = 1.0 / 120.0


def _world_with_floor(height=0.0, tilt=0.0):
    world = PhysicsWorld()
    shape = world.add_shape(model.Shape.box((FLOOR, 2.0, FLOOR)))
    orientation = (math.sin(tilt / 2.0), 0.0, 0.0, math.cos(tilt / 2.0))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape),
                   position=(0.0, height - 1.0, 0.0), orientation=orientation)
    return world


def _car(world, position=(0.0, 1.0, 0.0), tuning=None, wheels=None):
    chassis = world.add_shape(model.Shape.box((1.8, 0.6, 4.0)))
    body = world.add_body(
        model.Motion(type=model.DYNAMIC, mass=1200.0),
        collider=model.Collider(shape=chassis), position=position)
    return RaycastVehicle(world, body, wheels or car_wheels(),
                          tuning or VehicleTuning())


def _world_with_mesh_floor(height=0.0):
    """A floor that is a *surface*, as terrain is: no underside to stand on.

    A box has a bottom face, and a car that has fallen through one finds it and
    settles there -- which is not what a landscape does to a car that has gone
    under it.
    """
    world = PhysicsWorld()
    half = FLOOR / 2.0
    points = np.array([(-half, 0.0, -half), (half, 0.0, -half),
                       (half, 0.0, half), (-half, 0.0, half)], dtype='d')
    faces = np.array([(0, 1, 2), (0, 2, 3)], dtype='i')
    shape = world.add_shape(model.Shape.trimesh(points, faces))
    floor = world.add_body(model.Motion(type=model.STATIC),
                           collider=model.Collider(shape=shape),
                           position=(0.0, height, 0.0))
    return world, floor


def _hovering_car(world, position=(0.0, 1.0, 0.0), tuning=None, wheels=None):
    """A car with no chassis collider: the wheels are all that hold it up."""
    body = world.add_body(model.Motion(type=model.DYNAMIC, mass=1200.0),
                          position=position)
    return RaycastVehicle(world, body, wheels or car_wheels(),
                          tuning or VehicleTuning())


def _settle(world, vehicle, seconds=2.0, **controls):
    for _ in range(int(seconds / STEP)):
        vehicle.control(**controls)
        vehicle.update(STEP)
        world.step(STEP)
    return world.position[vehicle.body].copy()


class TestItHoldsItselfUp:
    def test_a_car_dropped_on_the_floor_comes_to_rest(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 2.0, 0))
        _settle(world, vehicle, seconds=3.0)
        assert abs(float(world.linear_velocity[vehicle.body][1])) < 0.1

    def test_it_rests_at_its_ride_height(self) -> None:
        """The springs carry the weight, so the body sits above the ground by
        rather less than the suspension's free length."""
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 2.0, 0))
        height = _settle(world, vehicle, seconds=3.0)[1]
        assert 0.2 < height < 1.2

    def test_it_does_not_sink_through_the_floor(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 2.0, 0))
        assert _settle(world, vehicle, seconds=5.0)[1] > 0.1

    def test_every_wheel_finds_the_ground(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 2.0, 0))
        _settle(world, vehicle, seconds=3.0)
        assert all(wheel.grounded for wheel in vehicle.wheels)
        assert vehicle.grounded

    def test_off_the_edge_of_the_world_no_wheel_is_grounded(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(FLOOR, 30.0, 0))
        vehicle.update(STEP)
        assert not vehicle.grounded
        assert not any(wheel.grounded for wheel in vehicle.wheels)

    def test_a_wheel_reports_how_far_it_has_compressed(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 2.0, 0))
        _settle(world, vehicle, seconds=3.0)
        for wheel in vehicle.wheels:
            assert 0.0 < wheel.compression < wheel.spec.suspension_travel

    def test_it_stays_upright(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 2.0, 0))
        _settle(world, vehicle, seconds=4.0)
        up = vehicle.up()
        assert float(up[1]) > 0.95


class TestItGoes:
    def test_throttle_moves_it_forward(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=1.5)
        start = world.position[vehicle.body].copy()
        _settle(world, vehicle, seconds=3.0, throttle=1.0)
        moved = world.position[vehicle.body] - start
        assert moved[2] < -3.0, "a car under power should go where it is pointing"

    def test_it_reports_its_speed(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=1.5)
        _settle(world, vehicle, seconds=3.0, throttle=1.0)
        assert vehicle.speed() > 3.0
        assert vehicle.forward_speed() > 3.0

    def test_reverse_goes_backwards(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=1.5)
        start = world.position[vehicle.body].copy()
        _settle(world, vehicle, seconds=3.0, throttle=-1.0)
        assert (world.position[vehicle.body] - start)[2] > 1.0

    def test_the_brake_stops_it(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=1.5)
        _settle(world, vehicle, seconds=3.0, throttle=1.0)
        rolling = vehicle.speed()
        _settle(world, vehicle, seconds=3.0, brake=1.0)
        assert vehicle.speed() < rolling * 0.2

    def test_it_does_not_accelerate_in_mid_air(self) -> None:
        """No wheel on the ground is no traction, whatever the pedal says."""
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 40.0, 0))
        for _ in range(30):
            vehicle.control(throttle=1.0)
            vehicle.update(STEP)
            world.step(STEP)
        assert abs(float(world.linear_velocity[vehicle.body][2])) < 0.05

    def test_it_slows_down_when_the_pedal_comes_off(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=1.5)
        _settle(world, vehicle, seconds=3.0, throttle=1.0)
        rolling = vehicle.speed()
        _settle(world, vehicle, seconds=4.0)
        assert vehicle.speed() < rolling


class TestItTurns:
    def _driving(self, seconds=2.5):
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=1.5)
        _settle(world, vehicle, seconds=seconds, throttle=1.0)
        return world, vehicle

    def test_steering_changes_its_heading(self) -> None:
        world, vehicle = self._driving()
        before = vehicle.forward().copy()
        _settle(world, vehicle, seconds=2.5, throttle=0.6, steer=1.0)
        assert float(np.dot(before, vehicle.forward())) < 0.98

    def test_left_and_right_turn_opposite_ways(self) -> None:
        def turned(steer):
            world, vehicle = self._driving()
            before = vehicle.forward().copy()
            _settle(world, vehicle, seconds=2.0, throttle=0.6, steer=steer)
            after = vehicle.forward()
            # The sign of the turn: which way the heading swung about up.
            return float(np.cross(before, after)[1])

        assert turned(1.0) * turned(-1.0) < 0.0

    def test_it_does_not_turn_standing_still(self) -> None:
        """A stationary car turns its wheels, not itself."""
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=2.0)
        before = vehicle.forward().copy()
        _settle(world, vehicle, seconds=2.0, steer=1.0)
        assert float(np.dot(before, vehicle.forward())) > 0.999

    def test_only_the_steered_wheels_turn(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        vehicle.control(steer=1.0)
        vehicle.update(STEP)
        turned = [wheel for wheel in vehicle.wheels if abs(wheel.steer_angle) > 1e-6]
        assert len(turned) == 2
        assert all(wheel.spec.steering for wheel in turned)

    def test_the_steering_is_capped(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0),
                       tuning=VehicleTuning(maximum_steer=0.5))
        vehicle.control(steer=10.0)
        vehicle.update(STEP)
        assert max(abs(wheel.steer_angle) for wheel in vehicle.wheels) <= 0.5


class TestItHandlesSlopes:
    def test_it_sits_square_on_a_slope(self) -> None:
        world = _world_with_floor(tilt=math.radians(10.0))
        vehicle = _car(world, position=(0, 3.0, 0))
        _settle(world, vehicle, seconds=4.0, brake=1.0)
        # The body leans with the ground rather than staying level or tipping.
        lean = math.degrees(math.acos(min(1.0, abs(float(vehicle.up()[1])))))
        assert 3.0 < lean < 20.0

    def test_the_handbrake_holds_it_on_the_slope(self) -> None:
        world = _world_with_floor(tilt=math.radians(8.0))
        vehicle = _car(world, position=(0, 3.0, 0))
        _settle(world, vehicle, seconds=2.0, brake=1.0)
        held = world.position[vehicle.body].copy()
        _settle(world, vehicle, seconds=3.0, brake=1.0)
        assert float(np.linalg.norm(world.position[vehicle.body] - held)) < 1.5


class TestTheWheelLayout:
    def test_the_default_car_has_four_wheels(self) -> None:
        assert len(car_wheels()) == 4

    def test_the_front_pair_steers_and_the_back_pair_drives(self) -> None:
        wheels = car_wheels()
        steering = [w for w in wheels if w.steering]
        driven = [w for w in wheels if w.driven]
        assert len(steering) == 2 and len(driven) == 2
        assert all(w.position[2] < 0 for w in steering), "front wheels lead"
        assert all(w.position[2] > 0 for w in driven)

    def test_the_track_and_wheelbase_can_be_set(self) -> None:
        wheels = car_wheels(wheelbase=3.0, track=2.0)
        assert max(w.position[2] for w in wheels) == pytest.approx(1.5)
        assert max(w.position[0] for w in wheels) == pytest.approx(1.0)

    def test_front_wheel_drive_is_expressible(self) -> None:
        wheels = car_wheels(drive='front')
        assert all(w.driven == w.steering for w in wheels)

    def test_all_wheel_drive_is_expressible(self) -> None:
        assert all(w.driven for w in car_wheels(drive='all'))

    def test_an_unknown_drive_layout_is_refused(self) -> None:
        with pytest.raises(ValueError, match='drive'):
            car_wheels(drive='diagonal')


class TestTheVehicleReportsItself:
    def test_it_names_its_axes(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world)
        assert np.allclose(vehicle.forward(), (0, 0, -1), atol=1e-9)
        assert np.allclose(vehicle.up(), (0, 1, 0), atol=1e-9)
        assert np.allclose(vehicle.right(), (1, 0, 0), atol=1e-9)

    def test_a_wheel_knows_where_it_is_in_the_world(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(10.0, 1.0, -5.0))
        vehicle.update(STEP)
        for wheel, spec in zip(vehicle.wheels, car_wheels(), strict=True):
            assert np.allclose(wheel.hub[[0, 2]],
                               np.array([10.0, -5.0]) + np.array(spec.position)[[0, 2]],
                               atol=1e-6)

    def test_controls_are_clamped_to_their_range(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world)
        vehicle.control(throttle=5.0, brake=-2.0, steer=-9.0)
        assert vehicle.throttle == 1.0
        assert vehicle.brake == 0.0
        assert vehicle.steer == -1.0

    def test_a_vehicle_with_no_wheels_is_refused(self) -> None:
        world = _world_with_floor()
        with pytest.raises(ValueError, match='wheel'):
            RaycastVehicle(world, 0, [], VehicleTuning())

    def test_it_can_be_placed(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world)
        vehicle.place((5.0, 3.0, -7.0), heading=math.pi / 2)
        assert np.allclose(world.position[vehicle.body], (5.0, 3.0, -7.0))
        assert np.allclose(vehicle.forward(), (-1, 0, 0), atol=1e-6)
        assert not world.linear_velocity[vehicle.body].any()


class TestASingleWheel:
    def test_a_stiffer_spring_rides_higher(self) -> None:
        world = _world_with_floor()
        soft = _car(world, position=(0, 2.0, 0),
                    wheels=car_wheels(suspension_stiffness=12.0))
        soft_height = _settle(world, soft, seconds=3.0)[1]
        world = _world_with_floor()
        stiff = _car(world, position=(0, 2.0, 0),
                     wheels=car_wheels(suspension_stiffness=40.0))
        assert _settle(world, stiff, seconds=3.0)[1] > soft_height

    def test_damping_settles_the_bounce(self) -> None:
        world = _world_with_floor()
        bouncy = _car(world, position=(0, 3.0, 0),
                      wheels=car_wheels(suspension_damping=0.05))
        heights = []
        for _ in range(int(3.0 / STEP)):
            bouncy.update(STEP)
            world.step(STEP)
            heights.append(float(world.position[bouncy.body][1]))
        world = _world_with_floor()
        damped = _car(world, position=(0, 3.0, 0),
                      wheels=car_wheels(suspension_damping=0.8))
        settled = []
        for _ in range(int(3.0 / STEP)):
            damped.update(STEP)
            world.step(STEP)
            settled.append(float(world.position[damped.body][1]))
        assert np.std(settled[-120:]) < np.std(heights[-120:])


class TestGroundThatRisesUnderIt:
    """A wheel looks below itself for the road, so a road that arrives *above*
    it is a road it cannot see -- and a car that cannot see the ground has no
    grip, no drive and nothing holding it up. It coasts, buried, until
    something else notices.

    Ground rises under a car for ordinary reasons: a lift, a moving platform, a
    landscape paging in at a finer level of detail than the one the car was
    driving on. Each wheel therefore looks a little above itself as well, and a
    wheel that finds the ground there is pushed back out on to it.
    """

    def _buried(self, above_hub=0.2, seconds=2.0, tuning=None):
        """A car with no chassis collider, so only the wheels can save it.

        A body that overlaps the ground is pushed out by the narrowphase, which
        would hide whether the wheels found anything -- and a raycast vehicle
        is meant to work without a chassis collider at all.

        The floor is raised until its surface is above every wheel's hub, which
        is exactly the case a downward ray cannot see: what a finer tile of a
        streamed landscape does when it replaces the coarse one the car was on.
        """
        world, floor = _world_with_mesh_floor()
        vehicle = _hovering_car(world, position=(0.0, 1.0, 0.0), tuning=tuning)
        _settle(world, vehicle, seconds=1.0)
        surface = max(float(wheel.hub[1]) for wheel in vehicle.wheels) + above_hub
        world.position[floor] = (0.0, surface, 0.0)
        world.wake(vehicle.body)
        _settle(world, vehicle, seconds=seconds)
        return world, vehicle, surface

    def test_it_ends_up_on_top_of_the_new_ground(self) -> None:
        world, vehicle, surface = self._buried()
        assert float(world.position[vehicle.body][1]) > surface

    def test_its_wheels_find_the_ground_again(self) -> None:
        _world, vehicle, _surface = self._buried()
        assert all(wheel.grounded for wheel in vehicle.wheels)

    def test_it_can_still_be_driven_afterwards(self) -> None:
        """The point of getting out: a car with no contact has no grip."""
        world, vehicle, _surface = self._buried()
        before = world.position[vehicle.body].copy()
        _settle(world, vehicle, seconds=2.0, throttle=1.0)
        assert abs(float(world.position[vehicle.body][2] - before[2])) > 5.0

    def test_ground_further_above_than_it_looks_is_not_found(self) -> None:
        """A car under a bridge is under a bridge, not hanging from it."""
        _world, vehicle, _surface = self._buried(above_hub=40.0, seconds=0.5)
        assert not any(wheel.grounded for wheel in vehicle.wheels)

    def test_the_ordinary_ride_height_is_unchanged(self) -> None:
        """Looking upwards must not lift a car that was sitting correctly."""
        world = _world_with_floor()
        vehicle = _car(world)
        resting = float(_settle(world, vehicle, seconds=2.0)[1])
        wheel = vehicle.wheels[0].spec
        expected = wheel.radius + wheel.suspension_travel
        assert resting == pytest.approx(expected, abs=0.25)

    def test_how_far_it_looks_is_a_tuning(self) -> None:
        """Nothing above the wheel is looked at when the recovery is nil."""
        _world, vehicle, _surface = self._buried(
            seconds=0.5, tuning=VehicleTuning(ground_recovery=0.0))
        assert not any(wheel.grounded for wheel in vehicle.wheels)
