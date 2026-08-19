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


#: A road's cut across, as one is actually built: two lanes falling away from a
#: crown so that it drains, a shoulder, and a verge dropping to the ground.
#: Offsets from the middle in metres, and how far each is below the crown.
CARRIAGEWAY = ((-5.3, -0.472), (-4.3, -0.122), (-3.6, -0.072), (0.0, 0.0),
               (3.6, -0.072), (4.3, -0.122), (5.3, -0.472))

#: How far apart the rings of a swept road are written, in metres.
RING = 5.0


def _world_with_crown(section=CARRIAGEWAY, ring=RING, length=1600.0):
    """A road, as a road is built: a ridge down the middle falling either side.

    This is the surface that asks the hardest question of four wheels at once.
    The two sides of the car stand on ground tilted opposite ways, so the
    sideways correction one wheel needs is not the one the other needs; and it
    is swept in rings a few metres apart, as any generated road is, so the
    ground under a wheel is a new triangle several times a second.
    """
    world = PhysicsWorld()
    rings = np.arange(-length / 2.0, length / 2.0 + ring, ring)
    points = np.array([(x, y, z) for z in rings for x, y in section], dtype='d')
    across = len(section)
    faces = []
    for row in range(len(rings) - 1):
        for column in range(across - 1):
            a = row * across + column
            faces += [(a, a + across, a + 1), (a + 1, a + across, a + across + 1)]
    shape = world.add_shape(model.Shape.trimesh(points,
                                                np.asarray(faces, dtype='i')))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape))
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


#: A fast road car: enough power to matter, wings that press it down and air
#: that holds it back. The combination is what a game tunes for, and what asks
#: the hardest questions of the wheels.
_FAST = VehicleTuning(engine_force=9000.0, brake_force=16000.0, base_speed=15.0,
                      downforce=6.0, drag=0.9)


class TestItGoesStraight:
    """A symmetric car, pointed straight and given the throttle, goes straight.

    What makes that a question at all is that four wheels push on one body in
    the same step: work each one out against a body the wheel before it has
    already moved and the four stop being symmetric, so the car wanders to
    whichever side happens to be worked out first. Five metres in eight seconds
    is a road's width, and a player correcting for it is correcting for the
    order of a loop.
    """

    def _driven(self, world, vehicle, seconds=8.0):
        _settle(world, vehicle, seconds=1.5)
        start = world.position[vehicle.body].copy()
        _settle(world, vehicle, seconds=seconds, throttle=1.0)
        return float((world.position[vehicle.body] - start)[0])

    def test_it_does_not_wander_off_to_one_side(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        assert abs(self._driven(world, vehicle)) < 0.5

    def test_and_which_side_is_not_decided_by_the_order_of_the_wheels(self) -> None:
        world = _world_with_floor()
        forwards = self._driven(world, _car(world, position=(0, 1.0, 0)))
        other = _world_with_floor()
        backwards = self._driven(
            other, _car(other, position=(0, 1.0, 0),
                        wheels=list(reversed(car_wheels()))))
        assert forwards == pytest.approx(backwards, abs=0.2)

    def test_it_stays_pointed_where_it_started(self) -> None:
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        self._driven(world, vehicle)
        forward = vehicle.forward()
        assert abs(math.degrees(math.atan2(float(forward[0]),
                                           -float(forward[2])))) < 0.5

    def test_it_does_not_shake_its_head_while_it_does(self) -> None:
        """Straight is a heading held steady, not one averaging out.

        Four wheels correct the same body's sideways scrub in the same step. Let
        each take all of what it sees and together they take more yaw out than
        there was, put some back the other way, and the car shimmies at the
        rate of the physics loop -- which averages to straight and looks like a
        car with a wheel out of balance.
        """
        world = _world_with_crown()
        vehicle = _car(world, position=(0, 1.0, 0), tuning=_FAST,
                       wheels=car_wheels(drive='rear', suspension_travel=0.22,
                                         suspension_stiffness=26.0,
                                         suspension_damping=0.55, grip=1.9))
        _settle(world, vehicle, seconds=1.5)
        worst = 0.0
        for _ in range(int(6.0 / STEP)):
            vehicle.control(throttle=1.0)
            vehicle.update(STEP)
            world.step(STEP)
            worst = max(worst, abs(float(
                world.angular_velocity[vehicle.body][1])))
        assert worst < 0.05, "shimmied at %.3f rad/s" % worst

    def test_it_goes_straight_along_a_crowned_road_too(self) -> None:
        """Which is every road: the two sides of the car stand on ground tilted
        opposite ways, and a car that cannot hold that line wanders off a road
        nobody has steered it away from."""
        world = _world_with_crown()
        vehicle = _car(world, position=(0, 1.0, 0), tuning=_FAST,
                       wheels=car_wheels(drive='rear', suspension_travel=0.22,
                                         suspension_stiffness=26.0,
                                         suspension_damping=0.55, grip=1.9))
        assert abs(self._driven(world, vehicle, seconds=12.0)) < 0.5

    def test_a_fast_car_with_wings_and_air_goes_straight_too(self) -> None:
        """The case that matters, because it is the one a game tunes for.

        Air resistance is a force on the *body*, handed to the wheels so that a
        tyre with nothing to push against cannot hold the car back with it.
        Handed out in equal quarters instead, the wheel the weight has come off
        under acceleration is asked for a quarter of the drag it has no grip to
        carry -- and what it gives up to find it is the sideways hold that was
        keeping the car straight.
        """
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0), tuning=_FAST)
        assert abs(self._driven(world, vehicle, seconds=12.0)) < 0.5


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


class TestACarLeftAlone:
    """It stays where it is.

    A car parked on a hill does not roll away: it is in gear, or it is on its
    handbrake, and either way letting go of the controls is not a decision to
    coast down the slope. A vehicle that does roll makes stopping anywhere but
    the flat a mistake, and turns every gentle grade in a world into something
    the player has to hold a key against.
    """

    #: About five and a half degrees, which is a steep road and a gentle hill.
    SLOPE = 0.1

    def _parked(self, world, vehicle, seconds=10.0):
        """Bring it to rest the way a driver does, let go, and see if it stays.

        On the brake while it settles, because a car put down on a hill is a car
        somebody stopped there: what is being asked is whether letting go of the
        controls is the same as deciding to coast away.
        """
        _settle(world, vehicle, seconds=1.5, brake=1.0)
        start = world.position[vehicle.body].copy()
        _settle(world, vehicle, seconds=seconds)
        return float(np.linalg.norm(world.position[vehicle.body] - start))

    def test_it_does_not_roll_down_a_slope(self) -> None:
        world = _world_with_floor(tilt=self.SLOPE)
        assert self._parked(world, _car(world, position=(0, 1.0, 0))) < 0.5

    def test_nor_creep_away_on_the_flat(self) -> None:
        world = _world_with_floor()
        assert self._parked(world, _car(world, position=(0, 1.0, 0))) < 0.2

    def test_a_car_told_not_to_hold_itself_rolls(self) -> None:
        """The holding is a number, and zero is a vehicle out of gear."""
        world = _world_with_floor(tilt=self.SLOPE)
        vehicle = _car(world, position=(0, 1.0, 0),
                       tuning=VehicleTuning(holding=0.0))
        assert self._parked(world, vehicle) > 5.0

    def test_the_throttle_still_gets_it_going(self) -> None:
        world = _world_with_floor(tilt=self.SLOPE)
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=1.5)
        _settle(world, vehicle, seconds=3.0, throttle=1.0)
        assert vehicle.speed() > 3.0

    def test_and_it_still_coasts_when_it_is_moving(self) -> None:
        """Holding is what a stopped car does, not a brake that is always on."""
        world = _world_with_floor()
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=1.5)
        _settle(world, vehicle, seconds=3.0, throttle=1.0)
        rolling = vehicle.speed()
        _settle(world, vehicle, seconds=1.0)
        assert vehicle.speed() > rolling * 0.7

    def test_and_then_stays_stopped_once_it_has_stopped(self) -> None:
        world = _world_with_floor(tilt=self.SLOPE)
        vehicle = _car(world, position=(0, 1.0, 0))
        _settle(world, vehicle, seconds=1.5)
        _settle(world, vehicle, seconds=3.0, throttle=1.0)
        _settle(world, vehicle, seconds=6.0, brake=1.0)
        assert self._parked(world, vehicle, seconds=8.0) < 0.5


class TestTheSteeringEasesOffWithSpeed:
    """The same input has to mean a gentler turn the faster the car is going.

    A steering lock is chosen for a car park: full lock at walking pace is a
    three-point turn, and full lock at forty metres a second is a request for
    ten g that ends with the car pointing at the trees. So the lock falls away
    as the speed rises, and it has to fall *fast enough that what it asks of the
    tyres stops growing* -- otherwise every extra ten miles an hour is another
    way to spin, and the driver's real control is how briefly they can touch a
    key.
    """

    #: The tuning the numbers below are read against: full lock is a third of a
    #: radian and it has halved by twenty metres a second.
    TUNING = VehicleTuning(maximum_steer=0.33, steer_falloff_speed=20.0)

    def _demand(self, speed):
        """What the lock asks of the tyres at this speed, bar the wheelbase.

        Cornering is ``v**2 / radius`` and the radius a steered wheel asks for
        is the wheelbase over its angle, so this is the acceleration a full-lock
        input is asking for, in metres a second squared per metre of wheelbase.
        """
        return speed ** 2 * self.TUNING.steer_lock(speed)

    def test_a_standing_car_has_all_of_its_lock(self) -> None:
        assert self.TUNING.steer_lock(0.0) == pytest.approx(0.33)

    def test_at_the_falloff_speed_it_has_half(self) -> None:
        assert self.TUNING.steer_lock(20.0) == pytest.approx(0.165)

    def test_and_well_past_it_far_less_than_half_again(self) -> None:
        """Halving once more for every doubling is not enough: the speed is
        squaring while the lock is only halving."""
        assert self.TUNING.steer_lock(40.0) < 0.33 / 4.0

    def test_what_it_asks_of_the_tyres_stops_growing(self) -> None:
        assert self._demand(80.0) < self._demand(40.0) * 1.3

    def test_though_it_still_grows_at_the_speeds_a_car_park_is_driven_at(self) -> None:
        """The fall-off is for the top end; at walking pace the lock is the
        lock, and a car that could not turn in its own length would be no use."""
        assert self._demand(5.0) > self._demand(1.0) * 4.0

    def test_a_lock_that_never_falls_off_is_still_allowed(self) -> None:
        assert VehicleTuning(maximum_steer=0.4, steer_falloff_speed=0.0
                             ).steer_lock(50.0) == pytest.approx(0.4)

    def test_a_tap_turns_the_car_more_at_low_speed_than_at_high(self) -> None:
        """The whole point of it, measured the way a road measures it.

        Not degrees a second -- a fast car covers more ground while it turns, so
        it swings further by that count however gently it is cornering. Degrees
        *per metre travelled* is what decides whether a touch of the key is a
        lane change or a lap of the car park, and it has to be the smaller
        number at speed.
        """
        def curved(seconds, throttle):
            world = _world_with_floor()
            vehicle = _car(world, position=(0, 1.0, 0), tuning=self.TUNING)
            _settle(world, vehicle, seconds=1.5)
            _settle(world, vehicle, seconds=seconds, throttle=throttle)
            before = vehicle.forward().copy()
            was = np.asarray(world.position[vehicle.body], dtype='d').copy()
            _settle(world, vehicle, seconds=0.5, throttle=throttle, steer=1.0)
            after = vehicle.forward()
            turned = abs(math.atan2(float(np.cross(before, after)[1]),
                                    float(np.dot(before, after))))
            travelled = float(np.linalg.norm(
                np.asarray(world.position[vehicle.body], dtype='d') - was))
            return math.degrees(turned) / max(travelled, 1e-6)
        assert curved(seconds=1.0, throttle=0.3) > \
            curved(seconds=8.0, throttle=1.0) * 3.0


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
