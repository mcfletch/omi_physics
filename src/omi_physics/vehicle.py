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
from typing import Any

import numpy as np

from . import mathutil
from .mathutil import Vec
from .raycast import raycast

__all__ = ['RaycastVehicle', 'VehicleTuning', 'Wheel', 'WheelSpec', 'car_wheels']

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
        self.wheels = [Wheel(spec=spec) for spec in wheels]
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

        for wheel in self.wheels:
            wheel.steer_angle = self._steer_angle if wheel.spec.steering else 0.0
            self._cast(wheel, rotation, centre)
            if not wheel.grounded:
                wheel.load = 0.0
                wheel.slip = 0.0
                continue
            self._suspend(wheel, dt, mass, gravity)
            self._drive(wheel, dt, rotation, mass, gravity, driven, braked)
        self._press_down(dt, mass)

    # -- the three forces ------------------------------------------------------

    def _cast(self, wheel: Wheel, rotation: np.ndarray,
              centre: np.ndarray) -> None:
        """Find the ground under one wheel, or report it hanging."""
        spec = wheel.spec
        wheel.hub = centre + rotation @ np.asarray(spec.position, dtype='d')
        down = -(rotation @ UP)
        reach = spec.suspension_travel + spec.radius
        hit = raycast(self.world, wheel.hub, down, max_distance=reach,
                      skip=(self.body,))
        if hit is None:
            wheel.grounded = False
            wheel.compression = 0.0
            return
        wheel.grounded = True
        wheel.contact = np.asarray(hit.point, dtype='d')
        wheel.normal = np.asarray(hit.normal, dtype='d')
        wheel.compression = max(0.0, reach - float(hit.distance))

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
        drive -= self.tuning.rolling_resistance * wheel.load * math.copysign(
            1.0, along) if abs(along) > CREEPING else 0.0

        # The sideways force needed to stop the tyre scrubbing this step, and
        # the longitudinal force asked of it, share one friction budget.
        grip_force = -across * mass / max(1, len(self.wheels)) / max(dt, 1e-6)
        budget = spec.grip * max(wheel.load, 0.0)
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
