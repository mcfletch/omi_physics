"""A raycast vehicle: a rigid body held up by springs that end in rays.

There are no wheels in the simulation. Each wheel is a ray cast straight down
from where the wheel's suspension is mounted; what it finds is the ground, how
far away tells the spring how hard to push, and a patch of friction at that
point drives, brakes and steers the car. This is how driving games have modelled
cars for thirty years, because a stack of real rigid bodies joined by real
constraints is both slower and far harder to make behave.

Three forces act at each contact, and each answers a different question:

**the spring** -- how does the car stay up, and lean into a corner? A force
along the contact normal, proportional to how far the suspension is compressed
and damped by how fast that is changing.

**the drive** -- how does the car go and stop? A force along the wheel's
heading, from the throttle, the brake, and the rolling resistance that slows a
car with its foot off the pedal.

**the grip** -- why does the car go where it is pointed instead of sliding? A
force across the wheel's heading, cancelling the sideways speed at the contact.
That force and the drive force together are limited by the friction available
at that wheel, which is what makes a car under too much power or too much
steering let go.

The model is deliberately a *primitive*: it takes forces, wheels and a body, and
leaves gearing, aerodynamics, tyre wear and the feel of a particular car to the
game that mounts it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from . import mathutil
from .mathutil import Vec
from .raycast import raycast_many

__all__ = ['RaycastVehicle', 'Surface', 'TARMAC', 'VehicleTuning', 'Wheel',
           'WheelSpec', 'car_wheels']

UP = np.array([0.0, 1.0, 0.0])
FORWARD = np.array([0.0, 0.0, -1.0])
RIGHT = np.array([1.0, 0.0, 0.0])

#: Below this speed a car is standing still: steering it turns the wheels and
#: nothing else, and the brake holds it rather than fighting it.
CREEPING = 0.15


@dataclass
class WheelSpec:
    """Where a wheel is mounted and what it is for.

    ``position`` is the top of the suspension in the body's own frame, so a
    wheel hangs ``suspension_travel`` below it and its contact patch is
    ``radius`` below that again. Negative Z is forward, as it is for everything
    else in a Y-up right-handed world.

    ``suspension_stiffness`` is in multiples of the car's own weight per metre
    of compression: a value of 20 means a wheel compressed by a twentieth of a
    metre carries the whole car. Expressing it that way keeps a tuning that
    works when the car's mass changes. ``suspension_damping`` is a fraction of
    critical damping -- around 0.4 for a road car, higher for something stiff.

    ``grip`` scales the friction available at this wheel, which is how a
    handbrake on the rear axle, or a car with wider back tyres, is expressed.
    """

    position: tuple[float, float, float]
    radius: float = 0.34
    suspension_travel: float = 0.30
    suspension_stiffness: float = 22.0
    suspension_damping: float = 0.45
    steering: bool = False
    driven: bool = True
    braked: bool = True
    grip: float = 1.6


@dataclass
class VehicleTuning:
    """What the pedals and the wheel are worth.

    ``engine_force`` and ``brake_force`` are in newtons at full pedal, shared
    between the wheels that take them. ``maximum_steer`` is the front wheels'
    lock, in radians, and ``steer_speed`` is how fast the steering reaches it --
    a car whose wheels snap to full lock in one frame is undrivable.

    ``rolling_resistance`` is the fraction of the car's weight that opposes it
    rolling, and ``downforce`` presses it into the road in proportion to the
    square of its speed, which is what stops a fast car floating over crests.
    """

    engine_force: float = 6000.0
    brake_force: float = 14000.0
    reverse_fraction: float = 0.4
    maximum_steer: float = 0.55
    steer_speed: float = 4.0
    rolling_resistance: float = 0.015
    downforce: float = 0.0
    #: Steering lock falls off with speed, or a car twitches out of control at
    #: the top end. This is the speed, in m/s, at which the lock has halved.
    steer_falloff_speed: float = 30.0
    #: How far *above* each wheel the ground is still looked for, in metres.
    #:
    #: A wheel looks below itself for the road, so a road that arrives above it
    #: is a road it cannot see -- and a car that cannot see the ground has no
    #: grip, no drive and nothing holding it up. It coasts, buried, until
    #: something else notices. Ground rises under a car for ordinary reasons: a
    #: lift, a moving platform, a landscape paging in at a finer level of detail
    #: than the one the car was driving on. A wheel that finds the ground within
    #: this distance above itself is pushed back out on to it. Set it to 0 for a
    #: vehicle that should fall through anything it ends up under.
    ground_recovery: float = 1.0
    #: How fast the suspension may push a buried wheel back out, in m/s.
    #:
    #: A spring at full compression for as long as the wheel is under the
    #: ground fires a swallowed car into the air rather than setting it back on
    #: the road. While a wheel is buried the load is capped to what lifts the
    #: car at this speed and no faster, so it climbs out.
    recovery_speed: float = 2.0


@dataclass(frozen=True)
class Surface:
    """What a wheel is on, as the two numbers a tyre feels.

    Tarmac, gravel, wet grass and mud are the same tyre on different ground.
    ``grip`` scales the friction there is to steer and drive with, and
    ``rolling`` is the resistance to rolling over it, as a fraction of the load
    the wheel is carrying -- so soft going loses the corner *and then* bogs the
    car down, which is what leaving a road for a field feels like.

    The vehicle has no opinion about what it is on: whatever knows where the
    road is sets it, on the car or on a wheel.
    """

    grip: float = 1.0
    rolling: float = 0.0


#: Firm, dry, and what a car is on unless something says otherwise.
TARMAC = Surface()


@dataclass
class Wheel:
    """One wheel's state, refreshed every update -- the vehicle's read-out.

    A renderer draws the wheel at :attr:`contact` plus its radius along the
    contact normal when it is grounded, and hanging at full droop when it is
    not; a game reads :attr:`slip` to decide when to make tyre noise.
    """

    spec: WheelSpec
    grounded: bool = False
    #: World position of the top of the suspension.
    hub: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: Where the tyre meets the ground, when it does.
    contact: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: UP.copy())
    #: How far the suspension is compressed, in metres.
    compression: float = 0.0
    #: The load the spring is carrying, in newtons.
    load: float = 0.0
    #: This wheel's steer angle, in radians, after the rate limit.
    steer_angle: float = 0.0
    #: Sideways speed at the contact patch, in m/s: how much the tyre is
    #: scrubbing rather than rolling.
    slip: float = 0.0
    #: How far the suspension is pushed *past* the end of its travel, in
    #: metres: how deep the car is in the ground. Zero for a wheel that is
    #: where a wheel can be.
    bottomed: float = 0.0
    #: What *this* wheel is on, when it is not what the car is on. Two wheels
    #: on the verge is the usual way of finding out about the verge.
    on: "Optional[Surface]" = None
    #: What the car is on, shared; set by the vehicle when the wheel is made.
    _car_surface: "Optional[Callable[[], Surface]]" = field(
        default=None, repr=False)

    def surface(self) -> Surface:
        """The ground this wheel is on: its own, or the car's."""
        if self.on is not None:
            return self.on
        return self._car_surface() if self._car_surface is not None else TARMAC

    def centre(self) -> np.ndarray:
        """Where the wheel itself is, for something that wants to draw it."""
        if self.grounded:
            return self.contact + self.normal * self.spec.radius
        return self.hub - UP * (self.spec.suspension_travel + self.spec.radius)


def car_wheels(wheelbase: float = 2.6, track: float = 1.6,
               height: float = -0.1, drive: str = 'rear',
               **wheel: Any) -> list[WheelSpec]:
    """The four wheels of an ordinary car.

    ``wheelbase`` is front axle to rear axle and ``track`` is left wheel to
    right; ``height`` is where the suspension is mounted relative to the body's
    origin. ``drive`` is ``'rear'``, ``'front'`` or ``'all'``. Anything else in
    ``wheel`` goes to every :class:`WheelSpec` -- radius, stiffness, grip.
    """
    if drive not in ('rear', 'front', 'all'):
        raise ValueError(f"drive must be 'rear', 'front' or 'all', not {drive!r}")
    half_base, half_track = wheelbase / 2.0, track / 2.0
    out = []
    for front in (True, False):
        for side in (-1.0, 1.0):
            out.append(WheelSpec(
                position=(side * half_track, height,
                          -half_base if front else half_base),
                steering=front,
                driven=(drive == 'all' or (drive == 'front') == front),
                **wheel))
    return out


class RaycastVehicle:
    """A car: a rigid body, a set of wheels, and three pedals.

    Call :meth:`control` with the driver's inputs and :meth:`update` once per
    physics step, *before* stepping the world -- the vehicle applies impulses to
    its body and the world then integrates them along with everything else.

    The vehicle never moves its body directly. Everything it does is an impulse
    at a contact point, so a car that hits a wall, is hit by something, or drives
    off a cliff behaves like the rigid body it is.
    """

    def __init__(self, world: Any, body: int, wheels: Sequence[WheelSpec],
                 tuning: VehicleTuning | None = None) -> None:
        if not wheels:
            raise ValueError("a vehicle needs at least one wheel")
        self.world = world
        self.body = int(body)
        self.tuning = tuning or VehicleTuning()
        #: What the car as a whole is on. A wheel with no surface of its own
        #: takes this, so a game that knows only "the car is on grass" says it
        #: once.
        self.surface = TARMAC
        self.wheels = [Wheel(spec=spec, _car_surface=lambda: self.surface)
                       for spec in wheels]
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self._steer_angle = 0.0

    # -- the driver ------------------------------------------------------------

    def control(self, throttle: float = 0.0, brake: float = 0.0,
                steer: float = 0.0) -> None:
        """The driver's inputs: throttle -1..1 (negative reverses), brake 0..1,
        steer -1..1 (positive turns left, the way a positive yaw does)."""
        self.throttle = _clamp(throttle, -1.0, 1.0)
        self.brake = _clamp(brake, 0.0, 1.0)
        self.steer = _clamp(steer, -1.0, 1.0)

    # -- where it is -----------------------------------------------------------

    def rotation(self) -> np.ndarray:
        return np.asarray(mathutil.quat_to_matrix(self.world.orientation[self.body]))

    def forward(self) -> np.ndarray:
        return self.rotation() @ FORWARD

    def up(self) -> np.ndarray:
        return self.rotation() @ UP

    def right(self) -> np.ndarray:
        return self.rotation() @ RIGHT

    def position(self) -> np.ndarray:
        return np.asarray(self.world.position[self.body], dtype='d')

    def velocity(self) -> np.ndarray:
        return np.asarray(self.world.linear_velocity[self.body], dtype='d')

    def speed(self) -> float:
        """How fast it is going, in m/s, whichever way that is."""
        return float(np.linalg.norm(self.velocity()))

    def forward_speed(self) -> float:
        """How fast it is going *forwards*: negative when reversing."""
        return float(np.dot(self.velocity(), self.forward()))

    @property
    def grounded(self) -> bool:
        """Whether any wheel is on the ground."""
        return any(wheel.grounded for wheel in self.wheels)

    def place(self, position: Vec, heading: float = 0.0) -> None:
        """Put the car somewhere, facing ``heading`` radians round from -Z, still.

        For a start line, a respawn, or a reset after the driver ends up in a
        lake -- the car arrives level and stopped rather than carrying whatever
        it was doing into its new position.
        """
        self.world.position[self.body] = np.asarray(position, dtype='d')
        self.world.prev_position[self.body] = self.world.position[self.body]
        half = heading / 2.0
        self.world.orientation[self.body] = (0.0, math.sin(half), 0.0,
                                             math.cos(half))
        self.world.prev_orientation[self.body] = self.world.orientation[self.body]
        self.world.linear_velocity[self.body] = 0.0
        self.world.angular_velocity[self.body] = 0.0
        for wheel in self.wheels:
            wheel.grounded = False
            wheel.compression = 0.0
            wheel.load = 0.0

    # -- the step --------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Cast the wheels and apply a step's worth of impulses to the body."""
        if dt <= 0.0:
            return
        self._steer_angle = self._steered(dt)
        rotation = self.rotation()
        centre = self.position()
        mass = float(self.world.mass[self.body])
        gravity = float(np.linalg.norm(self.world.resolve_gravity()[self.body])) or 9.81
        driven = sum(1 for wheel in self.wheels if wheel.spec.driven) or 1
        braked = sum(1 for wheel in self.wheels if wheel.spec.braked) or 1

        self._cast_all(rotation, centre)
        for wheel in self.wheels:
            wheel.steer_angle = self._steer_angle if wheel.spec.steering else 0.0
            if not wheel.grounded:
                wheel.load = 0.0
                wheel.slip = 0.0
                continue
            self._suspend(wheel, dt, mass, gravity)
            self._drive(wheel, dt, rotation, mass, gravity, driven, braked)
        self._press_down(dt, mass)

    # -- the three forces ------------------------------------------------------

    def _cast_all(self, rotation: np.ndarray, centre: np.ndarray) -> None:
        """Find the ground under every wheel, in one cast.

        Four wheels of one car look at very nearly the same piece of the world,
        so they are cast together: which bodies are worth testing is decided
        once, and a landscape's mesh is asked once for the triangles near all
        four rather than once per wheel. On a streamed world that is most of a
        physics step.

        Each ray starts ``ground_recovery`` metres *above* its hub rather than
        at it, so ground that has come up under the car -- see
        :attr:`VehicleTuning.ground_recovery` -- is still found and the
        suspension pushes the wheel back out on to it. Compression is measured
        from the hub as it always was, and capped at full travel, so the
        recovery is a firm shove rather than an unbounded one.
        """
        up = rotation @ UP
        down = -up
        overhead = max(0.0, self.tuning.ground_recovery)
        origins = []
        reaches = []
        for wheel in self.wheels:
            wheel.hub = centre + rotation @ np.asarray(wheel.spec.position,
                                                       dtype='d')
            origins.append(wheel.hub + up * overhead)
            reaches.append(wheel.spec.suspension_travel + wheel.spec.radius)
        # One distance for the bundle, since a cast has one reach: the longest
        # any wheel wants, with each wheel's own limit applied to its answer.
        furthest = max(reaches) + overhead
        hits = raycast_many(self.world, origins, [down] * len(self.wheels),
                            max_distance=furthest, skip=(self.body,))
        for wheel, reach, hit in zip(self.wheels, reaches, hits, strict=True):
            self._place(wheel, reach, overhead, hit)

    @staticmethod
    def _place(wheel: Wheel, reach: float, overhead: float,
               hit: Any) -> None:
        """What one wheel makes of what its ray found."""
        if hit is None or float(hit.distance) > reach + overhead:
            wheel.grounded = False
            wheel.compression = 0.0
            wheel.bottomed = 0.0
            return
        wheel.grounded = True
        wheel.contact = np.asarray(hit.point, dtype='d')
        wheel.normal = np.asarray(hit.normal, dtype='d')
        squash = max(0.0, reach - (float(hit.distance) - overhead))
        wheel.compression = min(wheel.spec.suspension_travel, squash)
        wheel.bottomed = squash - wheel.compression

    def _suspend(self, wheel: Wheel, dt: float, mass: float,
                 gravity: float) -> None:
        """Hold the body up, and damp it settling."""
        spec = wheel.spec
        normal = wheel.normal
        # Stiffness is in car-weights per metre, so a tuning survives a change
        # of mass; damping is a fraction of this spring's critical damping.
        stiffness = spec.suspension_stiffness * mass * gravity
        share = mass / max(1, len(self.wheels))
        critical = 2.0 * math.sqrt(max(stiffness * share, 1e-9))
        speed_along_normal = float(np.dot(self._point_velocity(wheel.contact), normal))
        force = (stiffness * wheel.compression
                 - spec.suspension_damping * critical * speed_along_normal)
        # A spring pushes, never pulls: a wheel at full droop lets the car fall.
        wheel.load = max(0.0, force)
        if wheel.bottomed > 0.0:
            # The suspension has run out of travel and the car is in the
            # ground. Climbing out of it, not bouncing off it: the load is
            # whatever lifts the car at the recovery speed, and no more.
            allowed = self.tuning.recovery_speed - speed_along_normal
            wheel.load = min(wheel.load, max(
                0.0, share * allowed / max(dt, 1e-9)))
        self._impulse(wheel.contact, normal * (wheel.load * dt))

    def _drive(self, wheel: Wheel, dt: float, rotation: np.ndarray, mass: float,
               gravity: float, driven: int, braked: int) -> None:
        """Push it along, hold it back, and stop it sliding sideways."""
        spec = wheel.spec
        heading = _project(self._heading(wheel, rotation), wheel.normal)
        sideways = np.cross(wheel.normal, heading)
        contact_velocity = self._point_velocity(wheel.contact)
        along = float(np.dot(contact_velocity, heading))
        across = float(np.dot(contact_velocity, sideways))
        wheel.slip = across

        drive = 0.0
        if spec.driven and self.throttle:
            share = self.tuning.engine_force / driven
            drive = share * self.throttle * (
                1.0 if self.throttle > 0 else self.tuning.reverse_fraction)
        if spec.braked and self.brake:
            stopping = self.tuning.brake_force / braked * self.brake
            # Brake to a stop, not backwards through it.
            drive -= _clamp(stopping * math.copysign(1.0, along), -abs(along) *
                            mass / max(dt, 1e-6), abs(along) * mass / max(dt, 1e-6))
        # Rolling resistance: the road's own, plus whatever the ground under
        # this wheel adds. Soft going is mostly this.
        on = wheel.surface()
        rolling = self.tuning.rolling_resistance + on.rolling
        drive -= rolling * wheel.load * math.copysign(
            1.0, along) if abs(along) > CREEPING else 0.0

        # The sideways force needed to stop the tyre scrubbing this step, and
        # the longitudinal force asked of it, share one friction budget -- which
        # is what the ground under the wheel has to offer.
        grip_force = -across * mass / max(1, len(self.wheels)) / max(dt, 1e-6)
        budget = spec.grip * on.grip * max(wheel.load, 0.0)
        combined = math.hypot(drive, grip_force)
        if combined > budget > 0.0:
            scale = budget / combined
            drive *= scale
            grip_force *= scale
        self._impulse(wheel.contact, (heading * drive + sideways * grip_force) * dt)

    def _press_down(self, dt: float, mass: float) -> None:
        """Aerodynamic downforce: what keeps a fast car on the road over a crest."""
        if not self.tuning.downforce or not self.grounded:
            return
        speed = self.speed()
        force = self.tuning.downforce * speed * speed
        self._impulse(self.position(), -UP * (force * dt))

    # -- the machinery ---------------------------------------------------------

    def _steered(self, dt: float) -> float:
        """The steer angle, rate-limited and eased off at speed."""
        lock = self.tuning.maximum_steer
        falloff = self.tuning.steer_falloff_speed
        if falloff > 0:
            lock = lock / (1.0 + abs(self.forward_speed()) / falloff)
        wanted = self.steer * lock
        step = self.tuning.steer_speed * self.tuning.maximum_steer * dt
        return _clamp(wanted, self._steer_angle - step, self._steer_angle + step)

    def _heading(self, wheel: Wheel, rotation: np.ndarray) -> np.ndarray:
        """Which way this wheel is pointing, in world space."""
        angle = wheel.steer_angle
        if not angle:
            return rotation @ FORWARD
        local = np.array([-math.sin(angle), 0.0, -math.cos(angle)])
        return rotation @ local

    def _point_velocity(self, point: np.ndarray) -> np.ndarray:
        """How fast a point fixed to the body is moving through the world."""
        offset = point - self.position()
        return (np.asarray(self.world.linear_velocity[self.body], dtype='d')
                + np.cross(np.asarray(self.world.angular_velocity[self.body],
                                      dtype='d'), offset))

    def _impulse(self, point: np.ndarray, impulse: np.ndarray) -> None:
        """Apply an impulse at a world point: a push and the spin it imparts."""
        if not np.any(impulse):
            return
        offset = point - self.position()
        self.world.apply_impulse(self.body, impulse)
        self.world.apply_angular_impulse(self.body, np.cross(offset, impulse))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _project(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """``vector`` flattened into the plane of ``normal``, unit length.

    A wheel's heading has to lie along the ground it is standing on, or driving
    up a slope would push the car into the hill.
    """
    flattened = vector - normal * float(np.dot(vector, normal))
    length = float(np.linalg.norm(flattened))
    if length < 1e-9:                            # pragma: no cover - a vertical wall
        return vector
    return flattened / length
