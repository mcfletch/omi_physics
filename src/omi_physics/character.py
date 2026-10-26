"""Character controller — a kinematic capsule with move-and-slide.

A game-usable first-person controller: tiered ground speed
(walk/run/sprint/crouch), jump when grounded, fly/noclip, step-up over small
ledges, and sliding on steep slopes.  It also implements **safe viewpoint
binding** (depenetration + ground snap) so a camera placed low or inside geometry
never leaves the user stuck in the ground.

The capsule is *kinematic* (move-and-slide against the world's static colliders)
rather than a dynamic rigid body — the standard approach for responsive avatars.
It reads the world through a :class:`~omi_physics.collisionquery.CollisionQuery`,
so it runs against a live :class:`~omi_physics.world.PhysicsWorld` or against a
published copy of one. Dynamic bodies are obstacles and floors for it. A body no
heavier than ``pushMass`` that blocks a step is named in ``pushing`` for the
caller to move with a kinematic stand-in, and a body it stands on carries it.
Pure CPU; the GL-facing :class:`PhysicsViewPlatform` wraps this.
"""
import enum
import math
import operator
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from . import collide, model
from .body import (
    BoxProxy,
    CapsuleProxy,
    HeightFieldProxy,
    Proxy,
    TriangleMeshProxy,
    make_proxy,
    pose_key,
)
from .collisionquery import CollisionQuery, DynamicBody, WorldQuery
from .contactevents import ContactEvent
from .mathutil import Vec, flat_length, length
from .world import PhysicsWorld

UP = np.array([0.0, 1.0, 0.0])

#: The depth of a ``(push, depth)`` contact. Named rather than a lambda, so the
#: sort in :meth:`CharacterController._push_out` calls C rather than Python once
#: per comparison.
_depth_of = operator.itemgetter(1)
IDENT = np.array([0.0, 0.0, 0.0, 1.0])
IDENTITY = np.eye(3)
FLAT = np.array([1.0, 0.0, 1.0])


def _depth_of_contact(contact: Any) -> float:
    """A contact's depth, for picking the deepest."""
    return float(contact.depth)



#: The gathers of static bodies an update keeps, the newest first.
_GATHERS_KEPT = 4


class _StaticBoxes:
    """The oriented boxes among some static bodies, as arrays a capsule is tested against at once."""

    def __init__(self, bodies: list[int], proxy_of: Any) -> None:
        boxes = [(j, proxy) for j in bodies
                 if isinstance(proxy := proxy_of(j), BoxProxy)]
        #: The body of each box, in the order given.
        self.rows = [j for j, _ in boxes]
        self._centers = np.array([proxy.center for _, proxy in boxes]).reshape(-1, 3)
        self._halves = np.array([proxy.half for _, proxy in boxes]).reshape(-1, 3)
        self._rotations = np.array([proxy.R for _, proxy in boxes]).reshape(-1, 3, 3)
        #: Each box as numbers, for :func:`~omi_physics.collide.capsule_box_push`.
        self._numbers = [(j, tuple(proxy.center.tolist()), tuple(proxy.half.tolist()),
                          tuple(tuple(row) for row in proxy.R.tolist())) for j, proxy in boxes]
        #: Each box's bounds, as ``(x0, y0, z0, x1, y1, z1)``.
        self._bounds = [(*low.tolist(), *high.tolist())
                        for low, high in (proxy.aabb() for _, proxy in boxes)]

    #: The most boxes a capsule is tested against one at a time with numbers.
    FEW = 6

    def pushes(self, capsule: CapsuleProxy) -> dict[int, tuple[Any, float]]:
        """The push out of each box ``capsule`` overlaps, and its depth, by body."""
        if not self.rows:
            return {}
        if len(self.rows) <= self.FEW:
            return self._few_pushes(capsule)
        hit, pushes, depths = collide.capsule_box_pushes(capsule, self._centers, self._halves,
                                                         self._rotations)
        return {self.rows[k]: (pushes[k], float(depths[k])) for k in np.flatnonzero(hit).tolist()}

    def _few_pushes(self, capsule: CapsuleProxy) -> dict[int, tuple[Any, float]]:
        """:meth:`pushes` box by box with numbers, leaving out boxes the capsule's box misses."""
        p0 = tuple(capsule.p0.tolist())
        p1 = tuple(capsule.p1.tolist())
        radius = capsule.radius
        lo = [min(p0[i], p1[i]) - radius for i in range(3)]
        hi = [max(p0[i], p1[i]) + radius for i in range(3)]
        found: dict[int, tuple[Any, float]] = {}
        for (j, center, half, rotation), (x0, y0, z0, x1, y1, z1) in zip(
                self._numbers, self._bounds, strict=True):
            if (lo[0] > x1 or lo[1] > y1 or lo[2] > z1 or hi[0] < x0 or hi[1] < y0
                    or hi[2] < z0):
                continue
            pushed = collide.capsule_box_push(p0, p1, radius, center, half, rotation)
            if pushed is not None:
                found[j] = pushed
        return found


class _Gather:
    """The static bodies near a box, gathered once for the depenetrations that fit in it.

    ``grounds`` is the height field of each body when every one of them is a
    height field, else None; ``meshes`` is each body and its triangle mesh
    when every one of them is a triangle mesh, else None.
    """

    __slots__ = ('_bounds', 'bodies', 'boxes', 'grounds', 'high', 'low', 'meshes')

    def __init__(self, low: np.ndarray, high: np.ndarray, bodies: list[int], proxy_of: Any) -> None:
        self.low, self.high = low, high
        self._bounds = (*low.tolist(), *high.tolist())
        self.bodies = bodies
        self.boxes = _StaticBoxes(bodies, proxy_of)
        proxies = [proxy_of(j) for j in bodies]
        self.grounds: list[HeightFieldProxy] | None = [
            proxy for proxy in proxies if isinstance(proxy, HeightFieldProxy)]
        if len(self.grounds) != len(bodies):
            self.grounds = None
        self.meshes: list[tuple[int, TriangleMeshProxy]] | None = [
            (j, proxy) for j, proxy in zip(bodies, proxies, strict=True)
            if isinstance(proxy, TriangleMeshProxy)]
        if not bodies or len(self.meshes) != len(bodies):
            self.meshes = None

    def holds(self, lo: tuple[float, float, float], hi: tuple[float, float, float]) -> bool:
        """Whether the box ``lo`` to ``hi`` lies inside the box the bodies were gathered for."""
        x0, y0, z0, x1, y1, z1 = self._bounds
        return (lo[0] >= x0 and lo[1] >= y0 and lo[2] >= z0
                and hi[0] <= x1 and hi[1] <= y1 and hi[2] <= z1)

class CharacterState(enum.Enum):
    """What a character is doing, as :attr:`CharacterController.state` reports it.

    The first match in this order: ``FLYING``, ``SWIMMING``, ``DOWN``,
    ``AIRBORNE`` (off the ground), ``SLIDING``, ``CROUCHING``, ``STANDING``.
    """

    STANDING = 'standing'
    CROUCHING = 'crouching'
    SLIDING = 'sliding'
    AIRBORNE = 'airborne'
    DOWN = 'down'
    SWIMMING = 'swimming'
    FLYING = 'flying'


@dataclass(slots=True)
class _Slide:
    """A slide under way: how long it lasts, how much of that is the rise, and how far in."""

    duration: float
    rise: float
    elapsed: float = 0.0
    #: Whether the input still steers the slide, at the speed of its mode.
    steer: bool = False
    #: How long the slide was asked to last, which :meth:`CharacterController.extend_slide` does not change.
    nominal: float = 0.0


@dataclass(slots=True)
class _Obstacle:
    """A dynamic body as the capsule collides with it this update."""

    body: int
    proxy: Proxy
    velocity: np.ndarray
    light: bool


@dataclass
class CharacterCapabilities:
    """Non-OMI gameplay config driving the controller (see plan §engine-config)."""
    walkSpeed: float = 3.0
    runSpeed: float = 6.0
    sprintSpeed: float = 9.0
    crouchSpeed: float = 1.5
    canSprint: bool = True
    canCrouch: bool = True
    canJump: bool = True
    canFly: bool = True
    flySpeed: float = 8.0
    #: Metres per second under water.  Slower than walking, because water is
    #: what you push against and should feel like it.
    swimSpeed: float = 2.0
    #: How fast water bleeds off a swimmer's vertical speed, per second.  This
    #: is what makes sinking read as sinking rather than as falling: without
    #: it a swimmer with no buoyancy accelerates to terminal velocity, and a
    #: pool becomes a hole in the floor.  It also settles the steady sink
    #: rate, which is ``(1 - buoyancy) * gravity / swimDrag``.
    swimDrag: float = 4.0
    jumpHeight: float = 1.2
    #: A jump from the ground keeps the vertical speed the run along the ground had, so a jump
    #: off a falling slope rises ``jumpHeight`` above the ground and not above the point it left.
    jumpFollowsSlope: bool = False
    stepHeight: float = 0.35
    maxSlope: float = 50.0            # degrees
    eyeHeight: float = 1.6
    standHeight: float = 1.8
    crouchHeight: float = 1.0
    radius: float = 0.3
    airControl: float = 0.4
    pushFriction: float = 6.0         # per second; bleeds off impulse speed on the ground
    #: Fastest the capsule may **fall**, m/s.  A real quantity -- a human
    #: reaches about 55 m/s spread-eagled -- and the one that decides how much
    #: work a frame can be asked for: collision is stepped finely enough to
    #: keep up with the capsule, so the fastest it may go is what sets the
    #: ceiling on the substep count (:meth:`CharacterController.max_substeps`).
    #: Raise it and falls stay fast for longer and cost proportionally more;
    #: 0 lets the fall accelerate without limit, which also removes the
    #: guarantee that a step cannot outrun collision.
    terminalVelocity: float = 55.0
    suppressOverVoid: bool = False    # hover instead of dropping into a bottomless gap
    #: Seconds after *falling* off something during which a jump is still
    #: allowed.  Running over a step, a ramp lip or a seam between two
    #: colliders leaves the capsule airborne for a frame or two at a time, and
    #: without this every press landing in one of those frames is swallowed
    #: with no feedback -- the commonest complaint about a first-person
    #: controller, and worst at speed, where it happens most.  0 disables it.
    coyoteTime: float = 0.12
    #: Seconds a refused jump is remembered for, so one asked for just before
    #: landing fires on landing rather than being dropped.  0 disables it.
    jumpBuffer: float = 0.12
    #: Kilograms. A blow's impulse divided by this is the change in velocity
    #: (:meth:`CharacterController.knockback`).
    mass: float = 80.0
    #: The heaviest dynamic body, in kilograms, that the character moves by
    #: walking into it. A heavier one blocks the step instead.
    pushMass: float = 200.0
    #: Fastest, in metres a second, that a body is pushed at. None is
    #: ``walkSpeed``.
    pushSpeed: float | None = None


class CharacterController:
    """Kinematic capsule avatar: move-and-slide against the world's colliders."""

    def __init__(self, world: PhysicsWorld | CollisionQuery,
                 capabilities: CharacterCapabilities | None = None,
                 position: Vec = (0, 0, 0), gravity: float = 9.81,
                 body: bool = False, ignore: Iterable[int] = ()) -> None:
        """Place a capsule of ``capabilities`` at ``position`` (its centre) in ``world``.

        ``world`` is a :class:`~omi_physics.world.PhysicsWorld` or any
        :class:`~omi_physics.collisionquery.CollisionQuery`.

        ``body`` gives the character a body in the world as well: a kinematic
        sensor capsule, standing height, that follows it on every
        :meth:`update` and :meth:`safe_bind` (:attr:`body`). It needs a
        ``PhysicsWorld``.

        ``ignore`` names bodies the capsule does not collide with, such as the
        kinematic stand-in that carries this character's velocity into the
        world (:attr:`pushing`).
        """
        #: The world the character walks in, or None when it was given a
        #: collision query in place of one.
        self.world: PhysicsWorld | None
        #: What the capsule collides with and casts rays at.
        self.query: CollisionQuery
        if isinstance(world, PhysicsWorld):
            self.world, self.query = world, WorldQuery(world)
        else:
            self.world, self.query = None, world
            if body:
                raise ValueError('a character body needs a PhysicsWorld, '
                                 'not a collision query')
        self.ignore = set(ignore)
        self.caps = capabilities or CharacterCapabilities()
        self.gravity_mag = gravity
        self.position = np.asarray(position, dtype='d')     # capsule centre
        self.vy = 0.0
        #: ``maxSlope`` as its cosine, and the limit it was taken from. See
        #: :meth:`_walkable`, which is asked about every contact of every
        #: depenetration iteration and so is not the place for an arccos.
        self._slope_limit: float | None = None
        self._slope_cosine = 1.0
        #: World velocity the capsule actually moved at over the last
        #: :meth:`update`, in metres a second.  Measured from where it ended up
        #: rather than from what it was asked for, so a body stopped by a wall
        #: or turned by a slope reports where it went, which is what anything
        #: following the motion -- an animation, a footstep, a leaning camera --
        #: needs rather than the intent behind it.
        self.velocity = np.zeros(3)
        self.grounded = False
        #: Surface normal underfoot while grounded, else None.  What the move
        #: direction is projected onto, so speed is spent along the ground
        #: rather than along the horizon.
        self.ground_normal: np.ndarray | None = None
        self.crouching = False
        self.flying = False
        #: Under water: collides like walking, moves like flying, and is
        #: pulled by whatever fraction of gravity :attr:`buoyancy` leaves.
        self.swimming = False
        #: Fraction of gravity that pushes back up while swimming.  1.0 hangs
        #: where it is, 0.0 sinks at full weight, above 1.0 rises.
        self.buoyancy = 0.0
        #: Swimming velocity, which is state rather than input: water is
        #: pushed against, so a stroke builds speed and letting go coasts.
        self._swim_velocity = np.zeros(3)
        #: Which way the surface last met under water lies: +1 above (a
        #: ceiling), -1 below (a bottom), 0 for open water.  See _update_swim.
        self._swim_contact = 0.0
        self.mode = 'walk'
        self.move_dir = np.zeros(3)                         # world horizontal unit
        self.fly_dir = np.zeros(3)
        self.push = np.zeros(3)                             # impulse-carried horizontal m/s
        self.stuck = False
        #: Seconds since the capsule last left the ground by *falling*, or None
        #: when it is grounded or left by jumping.  See ``coyoteTime``.
        self._airborne_for: float | None = None
        #: Seconds since a jump was asked for and refused, or None.  See
        #: ``jumpBuffer``.
        self._jump_wanted_for: float | None = None
        #: Metres a step-up advanced beyond what its frame was due, taken back
        #: out of the frames that follow.  See :meth:`_try_step_up`.
        self._step_debt: float = 0.0
        # Per *avatar*, unlike the proxy cache the query keeps: it is keyed on
        # where this capsule is, and two avatars are in different places.  See
        # `_near_triangles`.
        self._near_cache: dict[int, tuple[np.ndarray, np.ndarray,
                                          np.ndarray]] = {}
        #: The static bodies gathered during the update under way; None outside an
        #: update. See :meth:`_keep_gathers`.
        self._gathers: list[_Gather] | None = None
        #: The last gather, held across updates while the query's static layout
        #: (``static_epoch``) and :attr:`ignore` are those it was made under.
        self._held: tuple[int, frozenset[int], _Gather] | None = None
        #: Whether a capsule among height fields alone is collided with them with
        #: numbers (:meth:`_push_out`). False collides it through
        #: :func:`~omi_physics.collide.collide`, with the same answer, for comparison.
        self.ground_fast_path = True
        #: The character's body in the world, or None. A sensor
        #: (:meth:`~omi_physics.world.PhysicsWorld.is_sensor`): it enters
        #: trigger volumes and its touches with moving bodies are reported,
        #: and nothing is solved against it, so the character still collides
        #: only with what the controller collides it with.
        self.body: int | None = None
        #: Static colliders the capsule touches, each with the unit normal from
        #: the surface into the capsule and the point it touches at.
        self._touching: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        #: Velocity the capsule arrived with at the start of the last update.
        self._arrived = np.zeros(3)
        #: The dynamic body to move, with the velocity to move it at, when the
        #: last :meth:`update` was blocked by one the character can push, else
        #: None. A caller that keeps a kinematic stand-in for the character in
        #: its physics world gives the stand-in that velocity, so the solver
        #: pushes the body and the capsule follows on the next frames.
        self.pushing: tuple[int, np.ndarray] | None = None
        #: Whether the character is knocked down (:meth:`knock_down`).
        self.down = False
        #: The way the character was last told to move, as a flat unit vector.
        self._heading = np.zeros(3)
        self._slide: _Slide | None = None
        #: A height that replaces the standing and crouching ones while a slide
        #: or a knock-down holds the capsule at another.
        self._pose_height: float | None = None
        #: Speed the character landed at in the last :meth:`update`.
        self._landing: float | None = None
        #: Downward speed at the end of a free-fall step, before the ground has
        #: its say. It becomes the landing speed if the step ends on the ground.
        self._impact = 0.0
        #: The dynamic bodies the capsule collides with this update.
        self._obstacles: list[_Obstacle] = []
        self._obstacle_of: dict[int, _Obstacle] = {}
        #: Proxies of the dynamic bodies, kept while a body keeps its pose.
        self._obstacle_proxies: dict[int, tuple[model.Shape, tuple, Proxy]] = {}
        #: The dynamic body the last depenetration found the ground in, or None
        #: for a static one.
        self._ground_source: int | None = None
        self._standing_on: int | None = None
        if body:
            assert self.world is not None
            shape = self.world.add_shape(self._shape(self.caps.standHeight))
            self.body = self.world.add_body(model.Motion(type=model.KINEMATIC),
                                            collider=model.Collider(shape=shape),
                                            position=tuple(self.position), sensor=True)

    # -- geometry --------------------------------------------------------
    @property
    def height(self) -> float:
        """Current capsule height (crouch or stand, per pose, or a slide's or fall's own)."""
        if self._pose_height is not None:
            return self._pose_height
        return self.caps.crouchHeight if self.crouching else self.caps.standHeight

    @property
    def state(self) -> CharacterState:
        """What the character is doing, for choosing an animation."""
        if self.flying:
            return CharacterState.FLYING
        if self.swimming:
            return CharacterState.SWIMMING
        if self.down:
            return CharacterState.DOWN
        if not self.grounded:
            return CharacterState.AIRBORNE
        if self._slide is not None:
            return CharacterState.SLIDING
        if self.crouching:
            return CharacterState.CROUCHING
        return CharacterState.STANDING

    @property
    def slide_progress(self) -> float:
        """How far through the slide's own length the character is, from 0 to 1; 0 when it is not sliding.

        A slide that was not extended is as far through as its time. One that was extended stays
        where it is while the capsule is low, however long that goes on, and goes through the rise
        over the end, as the slide asked for does.
        """
        slide = self._slide
        if slide is None or slide.nominal <= 0.0:
            return 0.0
        remaining = slide.duration - slide.elapsed
        if remaining > slide.rise:
            held = min(slide.elapsed, slide.nominal - slide.rise)
        else:
            held = slide.nominal - remaining
        return min(max(held / slide.nominal, 0.0), 1.0)

    @property
    def landing(self) -> float | None:
        """Speed in metres a second the character landed at in the last :meth:`update`, else None.

        The downward speed at the end of the step that put it on the ground
        after being off it.
        """
        return self._landing

    @property
    def standing_on(self) -> int | None:
        """The id of the dynamic body underfoot, or None on the static world or in the air."""
        return self._standing_on if self.grounded else None

    def _resize(self, height: float) -> None:
        """Set the capsule's height, keeping its feet where they are."""
        before = self.height
        self._pose_height = height
        if height != before:
            self.position = self.position + UP * (0.5 * (height - before))

    #: How far the capsule that is tested for standing up is raised, in metres.
    #: A capsule resting on a slope touches it, and a touch reads as an overlap
    #: to the collision test; the raise leaves the contact alone and asks only
    #: whether something is in the way.
    STAND_CLEARANCE = 0.02

    def _blocked_at(self, height: float) -> bool:
        """Whether a capsule of ``height`` on the same feet would overlap something.

        The test capsule is raised by :data:`STAND_CLEARANCE`, so the ground
        the character stands on does not count as in the way.
        """
        lift = 0.5 * (height - self.height) + self.STAND_CLEARANCE
        return self._overlaps(make_proxy(self._shape(height), self.position + UP * lift, IDENT))

    def _shape(self, height: float | None = None) -> model.Shape:
        """Capsule shape at ``height`` (defaults to the current pose height)."""
        h = self.height if height is None else height
        mid = max(h - 2 * self.caps.radius, 1e-3)
        return model.Shape.capsule(height=mid, radius=self.caps.radius)

    def _proxy(self, position: Vec | None = None,
               height: float | None = None) -> Proxy:
        """World-space collision proxy for the capsule at ``position`` and ``height``."""
        pos = self.position if position is None else position
        h = self.height if height is None else height
        # The proxy make_proxy builds for an upright capsule, without the shape it is built from.
        mid = max(h - 2 * self.caps.radius, 1e-3)
        return CapsuleProxy(np.asarray(pos, dtype='d'), 0.5 * float(mid),
                            float(self.caps.radius), IDENTITY)

    def base(self) -> np.ndarray:
        """World position of the capsule's feet (the point below the centre)."""
        return self.position - UP * (self.height * 0.5)

    def eye(self) -> np.ndarray:
        """World position of the eye/camera (eye height above the feet)."""
        return self.base() + UP * self.caps.eyeHeight

    # -- input -----------------------------------------------------------
    def set_move(self, direction: Vec, mode: str = 'walk') -> None:
        """Set the desired horizontal move (flattened to the ground plane) and speed tier."""
        d = np.asarray(direction, dtype='d')
        d = d * FLAT                                        # flatten to ground plane
        n = np.linalg.norm(d)
        self.move_dir = d / n if n > 1e-9 else np.zeros(3)
        if n > 1e-9:
            self._heading = self.move_dir
        self.mode = mode

    def set_fly_move(self, direction: Vec) -> None:
        """Set the desired fly move (full 3D unit direction, no ground flattening)."""
        d = np.asarray(direction, dtype='d')
        n = np.linalg.norm(d)
        self.fly_dir = d / n if n > 1e-9 else np.zeros(3)

    def speed(self) -> float:
        """Ground speed for the current pose and mode (walk/run/sprint/crouch).

        A steered slide goes at the speed of the mode, not the crouch's."""
        slide = self._slide
        if self.crouching and not (slide is not None and slide.steer):
            return self.caps.crouchSpeed
        if self.mode == 'sprint' and self.caps.canSprint:
            return self.caps.sprintSpeed
        if self.mode == 'run':
            return self.caps.runSpeed
        return self.caps.walkSpeed

    def _intent(self) -> np.ndarray:
        """Ground velocity the input asks for. Nothing while knocked down or in a slide that is not steered."""
        slide = self._slide
        if self.down or (slide is not None and not slide.steer):
            return np.zeros(3)
        return self.move_dir * self.speed()

    def jump(self) -> bool:
        """Launch upward if able; return whether the jump fired *now*.

        Able means grounded, or airborne only because the capsule walked off
        something within the last ``coyoteTime`` seconds.  A jump that is
        refused for timing alone is remembered for ``jumpBuffer`` seconds and
        taken on landing, so False here does not always mean the press was
        thrown away.

        The two windows forgive *falling*, never jumping: a capsule that left
        the ground under its own power has no coyote time, so there is no free
        double jump.
        """
        if not self._may_jump():
            return False
        if self._can_launch():
            self._launch()
            return True
        # Refused for timing, not for good: hold it for the landing.
        if self.caps.jumpBuffer > 0.0:
            self._jump_wanted_for = 0.0
        return False

    def _may_jump(self) -> bool:
        """Whether a jump is permitted at all, timing aside."""
        return bool(self.caps.canJump and not self.crouching and not self.flying
                    and not self.down)

    def _can_launch(self) -> bool:
        """Whether the capsule is on the ground, or close enough in time to it."""
        if self.grounded:
            return True
        return (self.caps.coyoteTime > 0.0 and self._airborne_for is not None
                and self._airborne_for <= self.caps.coyoteTime)

    def _launch(self) -> None:
        self.vy = np.sqrt(2.0 * self.gravity_mag * self.caps.jumpHeight)
        if self.caps.jumpFollowsSlope and self.grounded:
            self.vy += float(self._along_ground(self._intent() + self.push)[1])
        self.grounded = False
        # Neither window survives a launch: coyote time forgives falling only,
        # and a buffered press has now been spent.
        self._airborne_for = None
        self._jump_wanted_for = None

    def _tick_jump_windows(self, dt: float, was_grounded: bool) -> None:
        """Age the coyote and buffer windows, and take a buffered jump on landing.

        Run at the end of a step, once the ground state for the frame is
        settled, so "landed this frame" is a fact rather than a prediction.
        """
        if self.grounded:
            self._airborne_for = None
        elif was_grounded:
            # Left the ground this frame.  A launch has already cleared this,
            # so reaching here means it fell.
            self._airborne_for = 0.0
        elif self._airborne_for is not None:
            self._airborne_for += dt

        if self._jump_wanted_for is None:
            return
        if self.grounded and self._may_jump():
            self._launch()
            return
        self._jump_wanted_for += dt
        if self._jump_wanted_for > self.caps.jumpBuffer:
            self._jump_wanted_for = None

    def apply_impulse(self, velocity: Vec) -> None:
        """Launch the capsule at ``velocity`` (m/s), discarding its current motion.

        This is the jump-pad / spring / blast primitive: the vertical part drives
        the usual ballistic arc, and the horizontal part is *carried* -- it moves
        the capsule on its own, unscaled by ``airControl``, until ground friction
        bleeds it away.  Ungrounding is part of the launch: a grounded capsule is
        snapped back onto its surface each step, which would eat the impulse.
        """
        v = np.asarray(velocity, dtype='d')
        self.push = v * FLAT
        self.vy = float(v[1])
        self.grounded = False
        # Launched, not fallen: no coyote time off a jump pad, and a press held
        # from before the launch must not fire the moment it lands.
        self._airborne_for = None
        self._jump_wanted_for = None

    def knockback(self, impulse: Vec) -> None:
        """Change the velocity by ``impulse`` (kilogram metres a second) over ``caps.mass``.

        The blow adds to the motion the character has. Its horizontal part is
        carried like an :meth:`apply_impulse` and bled off by ground friction;
        an upward part leaves the ground.
        """
        change = np.asarray(impulse, dtype='d') / self.caps.mass
        self.push = self.push + change * FLAT
        self.vy += float(change[1])
        if self.vy > 0.0:
            self.grounded = False
            self._airborne_for = None
            self._jump_wanted_for = None

    def slide(self, duration: float, rise: float = 0.3, steer: bool = False) -> bool:
        """Slide for ``duration`` seconds; return whether it began.

        A crouch that keeps the momentum the character has and ignores input.
        Ground friction (``pushFriction``) bleeds the speed off. With ``steer``
        the slide takes no momentum, friction does not act, and the input goes
        on moving the character at the speed of its mode. The capsule is
        ``crouchHeight`` tall until the last ``rise`` seconds and then grows to
        ``standHeight``, waiting under a ceiling that leaves no room, and the
        slide ends crouched if it never gets any. It begins only from the
        ground, walking, and not already sliding or knocked down.
        """
        if (duration <= 0.0 or not self.grounded or self.flying or self.swimming
                or self.down or self._slide is not None):
            return False
        momentum = self.velocity * FLAT
        if flat_length(momentum) < 1e-6:
            momentum = self._intent()
        self.push = np.zeros(3) if steer else momentum
        self._slide = _Slide(duration, min(max(rise, 0.0), duration), steer=steer, nominal=duration)
        self._resize(self.caps.crouchHeight)
        self.crouching = True
        return True

    def extend_slide(self, duration: float) -> bool:
        """Have the slide last ``duration`` seconds more, from now; return whether there is one.

        The capsule goes low again, the rise comes at the end of the new length, and
        :attr:`slide_progress` is the time slid over the new length.
        """
        slide = self._slide
        if slide is None:
            return False
        slide.duration = slide.elapsed + max(duration, 0.0)
        slide.rise = min(slide.rise, max(duration, 0.0))
        return True

    def end_slide(self) -> bool:
        """End a slide at once and stand, or crouch where there is no room; return whether there was one."""
        if self._slide is None:
            return False
        self._slide = None
        if self._blocked_at(self.caps.standHeight):
            self.crouching = True
            self._resize(self.caps.crouchHeight)
        else:
            self.crouching = False
            self._resize(self.caps.standHeight)
        self._pose_height = None
        return True

    def knock_down(self) -> None:
        """Lie on the ground: input is ignored, the speed drops to zero and the capsule is low.

        Gravity still acts. :meth:`get_up` ends it.
        """
        self._slide = None
        self.down = True
        self.push = np.zeros(3)
        self._jump_wanted_for = None
        self._resize(self.caps.crouchHeight)

    def get_up(self, start_speed: float | None = None) -> None:
        """End a knock-down and stand, or crouch where there is no room to stand.

        ``start_speed`` (metres a second) is carried along the last direction
        the character was told to move in and bled off by ground friction.
        """
        if not self.down:
            return
        self.down = False
        height = self.caps.crouchHeight if self.crouching else self.caps.standHeight
        if height > self.height and self._blocked_at(height):
            self.crouching = True
            height = self.caps.crouchHeight
        self._resize(height)
        self._pose_height = None
        if start_speed:
            self.push = self._heading * start_speed

    def set_crouch(self, crouch: bool) -> bool:
        """Crouch or stand; standing fails (returns False) if the head is blocked.

        A slide holds its own pose, so this changes nothing during one.
        """
        if self._slide is not None:
            return crouch
        if crouch and self.caps.canCrouch:
            self.crouching = True
        elif not crouch and self.crouching:
            if self._can_stand():
                self.crouching = False
                return True
            return False
        return True

    def set_fly(self, flying: bool) -> bool:
        """Enter or leave fly/noclip; return False if flight is disallowed."""
        if flying and not self.caps.canFly:
            return False
        self.flying = flying
        if flying:
            self.grounded = False
            self._airborne_for = None
            self._jump_wanted_for = None
        return True

    def set_swim(self, swimming: bool, buoyancy: float = 0.9) -> None:
        """Enter or leave the water.

        A separate state from :meth:`set_fly` rather than a use of it, and the
        difference is the point: flying is noclip and free of gravity, while a
        swimmer still collides with the world and is still pulled by whatever
        fraction of gravity ``buoyancy`` leaves.  Swimming as a fly gives a
        player who can leave a pool through its wall.

        Vertical speed is dropped on the way in *and* out.  Coming in, a
        falling body should enter the water and not carry its fall through it;
        going out, a swimmer rising fast who breaks the surface with that speed
        intact is launched into the air.
        """
        was = self.swimming
        self.swimming = bool(swimming)
        self.buoyancy = float(buoyancy)
        if self.swimming != was:
            self.vy = 0.0
            self._swim_velocity = np.zeros(3)
            self._swim_contact = 0.0
        if self.swimming:
            self.grounded = False
            self.ground_normal = None
            self._airborne_for = None
            self._jump_wanted_for = None

    def _can_stand(self) -> bool:
        """No static geometry where the standing capsule's head would be."""
        return not self._blocked_at(self.caps.standHeight)

    # -- world queries ---------------------------------------------------
    def _static_bodies(self, lo: np.ndarray | None = None,
                       hi: np.ndarray | None = None) -> Iterator[int]:
        """Yield the id of each solid body that does not move under the solver.

        With a box, ``lo`` to ``hi``, only the bodies whose bounding boxes overlap it:
        a capsule is tested against what is near it and not against a world.
        """
        ignore = self.ignore
        if lo is None or hi is None:
            found = self.query.static_bodies()
        else:
            found = self.query.static_near(lo, hi)
        for j in found:
            if j not in ignore:
                yield j

    def _static_proxy(self, j: int) -> Proxy:
        """The query's world-space proxy for a static body.

        The query keeps these, and for a live world shares them between every
        avatar walking in it: a trimesh proxy is a transformed copy of the
        geometry with a uniform grid over it, so one per controller would be
        one copy of the level per person in it. Sharing is correct because
        these bodies never move.
        """
        return self.query.static_proxy(j)

    def _gather_obstacles(self, lo: np.ndarray, hi: np.ndarray) -> None:
        """Read the dynamic bodies in the box ``lo`` to ``hi`` as this update's obstacles."""
        kept: dict[int, tuple[model.Shape, tuple, Proxy]] = {}
        obstacles: list[_Obstacle] = []
        for found in self.query.dynamic_near(lo, hi):
            if found.body in self.ignore:
                continue
            obstacles.append(_Obstacle(found.body, self._obstacle_proxy(found, kept),
                                       found.velocity, found.mass <= self.caps.pushMass))
        self._obstacle_proxies = kept
        self._obstacles = obstacles
        self._obstacle_of = {obstacle.body: obstacle for obstacle in obstacles}

    def _obstacle_proxy(self, found: DynamicBody,
                        kept: dict[int, tuple[model.Shape, tuple, Proxy]]) -> Proxy:
        """A proxy of ``found`` at its pose, reused while the body keeps its shape and pose."""
        key = pose_key(found.position, found.orientation)
        cached = self._obstacle_proxies.get(found.body)
        if cached is not None and cached[0] is found.shape and cached[1] == key:
            proxy = cached[2]
        else:
            proxy = make_proxy(found.shape, found.position, found.orientation)
        kept[found.body] = (found.shape, key, proxy)
        return proxy

    def _overlaps(self, proxy: Proxy) -> bool:
        """True if ``proxy`` intersects any static collider or dynamic obstacle."""
        lo, hi = proxy.aabb()
        near = list(self._static_bodies(lo, hi))
        if isinstance(proxy, CapsuleProxy):
            boxes = _StaticBoxes(near, self._static_proxy)
            if boxes.pushes(proxy):
                return True
            near = [j for j in near if j not in boxes.rows]
        for j in near:
            other = self._static_proxy(j)
            if (self.ground_fast_path and isinstance(other, HeightFieldProxy)
                    and isinstance(proxy, CapsuleProxy) and np.array_equal(proxy.R, IDENTITY)):
                x, y, z = proxy.center.tolist()
                if collide.upright_capsule_heightfield_pushes(other, x, y, z, proxy.half_height,
                                                              proxy.radius):
                    return True
            elif collide.collide(0, 1, proxy, other):
                return True
        return any(collide.collide(0, 1, proxy, obstacle.proxy)
                   for obstacle in self._obstacles)

    #: How far past the capsule a cached candidate gather reaches, in metres.
    #: Wide enough that the depenetration iterations, the ground probe and the
    #: step tests all fall inside one gather -- their positions differ by
    #: centimetres -- and narrow enough that what it hands back is still mostly
    #: near the capsule.
    NEAR_MARGIN = 0.5

    def _near_triangles(self, body: int, mesh: TriangleMeshProxy,
                        capsule: CapsuleProxy) -> np.ndarray:
        """Triangles near the capsule, gathered once and reused while it fits.

        A frame asks for depenetration nine to twelve times -- three iterations
        inside each of the ground probe, the move, the step-up and the
        step-down -- at positions centimetres apart. The broad phase does not
        change its answer over that, so it is asked once for a box grown by
        :attr:`NEAR_MARGIN` and the gather is reused until the capsule leaves
        it. A superset is safe: the exact test rejects the rest.
        """
        lo, hi = capsule.aabb()
        return self._near_triangles_in(body, mesh, lo, hi)

    def _near_triangles_in(self, body: int, mesh: TriangleMeshProxy, lo: Any, hi: Any
                           ) -> np.ndarray:
        """:meth:`_near_triangles` for a capsule whose box runs from ``lo`` to ``hi``."""
        cached = self._near_cache.get(body)
        if cached is not None:
            low, high, verts = cached
            # Spelled out over the three axes rather than through two np.all
            # calls: each dispatches into the reduction machinery, and this
            # question is asked of every body on every depenetration iteration
            # -- which is the whole point of the cache being cheap.
            if (lo[0] >= low[0] and lo[1] >= low[1] and lo[2] >= low[2]
                    and hi[0] <= high[0] and hi[1] <= high[1]
                    and hi[2] <= high[2]):
                return verts
        low = np.asarray(lo, dtype='d') - self.NEAR_MARGIN
        high = np.asarray(hi, dtype='d') + self.NEAR_MARGIN
        verts = mesh.candidate_vertices(low, high)
        self._near_cache[body] = (low, high, verts)
        return verts

    def _keep_gathers(self) -> None:
        """Keep the static bodies gathered by depenetration for the rest of the update.

        The world does not change during an update, so a gather made for a box
        that holds the capsule's box answers as a new one would: it may hold
        more bodies, and those touch nothing the capsule's box does not reach.
        """
        self._gathers = []
        held = self._held
        if held is not None:
            if held[0] == self._static_epoch() and held[1] == self.ignore:
                self._gathers.append(held[2])
            else:
                self._held = None

    def _static_epoch(self) -> int | None:
        """The query's static layout epoch, or None where it cannot say when its statics change."""
        epoch = getattr(self.query, 'static_epoch', None)
        return None if epoch is None else epoch()

    #: How far past the capsule a gather that is held across updates reaches, in
    #: metres: a walker crosses it in a few updates, and what it holds besides
    #: grounds is tested exactly and rejected at once.
    HOLD_MARGIN = 1.0

    def _gather_statics(self, lo: np.ndarray, hi: np.ndarray) -> '_Gather':
        """The static bodies near the box ``lo`` to ``hi``, from a gather this update kept if one holds it."""
        kept = self._gathers
        if kept is not None:
            for gathered in kept:
                low, high = gathered.low, gathered.high
                if (lo[0] >= low[0] and lo[1] >= low[1] and lo[2] >= low[2]
                        and hi[0] <= high[0] and hi[1] <= high[1] and hi[2] <= high[2]):
                    return gathered
        epoch = self._static_epoch()
        margin = self.NEAR_MARGIN if epoch is None else self.HOLD_MARGIN
        low, high = lo - margin, hi + margin
        gathered = _Gather(low, high, list(self._static_bodies(low, high)), self._static_proxy)
        if kept is not None:
            kept.insert(0, gathered)
            del kept[_GATHERS_KEPT:]
        if epoch is not None:
            self._held = (epoch, frozenset(self.ignore), gathered)
        return gathered

    def _push_out(self, position: Vec,
                  iterations: int = 3) -> tuple[np.ndarray, np.ndarray | None]:
        """Depenetrate the capsule from static colliders and obstacles; return (pos, ground_n).

        Contacts are resolved by sequential projection (deepest first): each
        contact only contributes the penetration *not already* resolved by the
        corrections applied so far.  This resolves distinct directions together
        (floor-up + wall-sideways, so the capsule can't sink or slip through)
        without the over-push that summing many near-parallel contacts on a curved
        surface would cause — which otherwise springs the capsule back out and
        makes it bounce off columns instead of sliding/stopping.

        Where the static bodies near the capsule are all height fields and no
        dynamic body is near, the contacts are those of the spheres down the
        capsule's axis against each ground, found with numbers
        (:func:`~omi_physics.collide.upright_capsule_heightfield_pushes`): no
        capsule proxy, box test or contact object is made. Where they are all
        triangle meshes and no dynamic body is near, the triangles near the
        capsule are collided with its axis as numbers
        (:func:`~omi_physics.collide.upright_capsule_mesh_pushes`).
        """
        px, py, pz = (float(v) for v in np.asarray(position, dtype='d'))
        radius = self.caps.radius
        half = 0.5 * max(self.height - 2 * radius, 1e-3)
        fast = self.ground_fast_path
        slope_cosine = self._slope_cos()
        ground_n: tuple[float, float, float] | np.ndarray | None = None
        ground_source: int | None = None
        gathered: _Gather | None = None
        for _ in range(iterations):
            # The capsule's box, as CapsuleProxy.aabb finds it.
            lo = (px - radius, (py - half) - radius, pz - radius)
            hi = (px + radius, (py + half) + radius, pz + radius)
            # The static bodies near the capsule, asked for once and kept while
            # the corrections leave it inside the box they were gathered for.
            if gathered is None or not gathered.holds(lo, hi):
                gathered = self._gather_statics(np.array(lo), np.array(hi))
            contacts: list[tuple[Any, float]] = []
            owners: list[int] = []
            grounds = gathered.grounds if fast and not self._obstacles else None
            meshes = gathered.meshes if grounds is None and not self._obstacles else None
            if grounds is not None:
                for field in grounds:
                    contacts.extend(collide.upright_capsule_heightfield_pushes(
                        field, px, py, pz, half, radius))
                first_dynamic = len(contacts)
            elif meshes is not None:
                for j, mesh in meshes:
                    contacts.extend(collide.upright_capsule_mesh_pushes(
                        self._near_triangles_in(j, mesh, lo, hi), px, py, pz, half, radius))
                first_dynamic = len(contacts)
            else:
                first_dynamic = self._contacts_with_everything(
                    gathered, np.array([px, py, pz]), contacts, owners)
            # Ground is a surface this capsule could stand on, which is what
            # ``maxSlope`` says.  A fixed threshold here would disagree with it
            # -- and a face counted as ground but too steep to walk is one the
            # player can climb, since every part of the controller that seats
            # or steps trusts this.
            for index, (push, _depth) in enumerate(contacts):
                if float(push[1]) >= slope_cosine:
                    ground_n = push
                    ground_source = (owners[index - first_dynamic]
                                     if index >= first_dynamic else None)
            if not contacts:
                break
            contacts.sort(key=_depth_of, reverse=True)       # deepest first
            # The running correction is carried as three floats rather than as
            # an array: this is the innermost loop of the whole controller --
            # every contact of every iteration of every depenetration -- and a
            # three-element dot and add through numpy costs several times the
            # arithmetic in it. The order of operations is the same, so the
            # answer is the same to the bit.
            cx = cy = cz = 0.0
            for push, depth in contacts:
                nx, ny, nz = float(push[0]), float(push[1]), float(push[2])
                remaining = depth + 1e-4 - (cx * nx + cy * ny + cz * nz)
                if remaining > 0:
                    cx += nx * remaining
                    cy += ny * remaining
                    cz += nz * remaining
            px, py, pz = px + cx, py + cy, pz + cz
        self._ground_source = ground_source
        normal = None if ground_n is None else np.array(ground_n, dtype='d')
        return np.array([px, py, pz]), normal

    def _contacts_with_everything(self, gathered: '_Gather', pos: np.ndarray,
                                  contacts: list[tuple[Any, float]], owners: list[int]) -> int:
        """Add the capsule's contacts at ``pos`` with the bodies of ``gathered`` and the obstacles.

        Each is ``(push, depth)``, ``push`` the unit normal from the surface into
        the capsule; the owner of each obstacle contact goes to ``owners``.
        Returns how many contacts come before the obstacles' own.
        """
        capsule = cast(CapsuleProxy, self._proxy(pos))
        boxed = gathered.boxes.pushes(capsule)
        fast = self.ground_fast_path
        for j in gathered.bodies:
            found = boxed.get(j)
            if found is not None:
                contacts.append(found)
                continue
            Pj = self._static_proxy(j)
            if isinstance(Pj, BoxProxy):
                continue
            if isinstance(Pj, TriangleMeshProxy):
                # The level, which is nearly always every contact there is.
                # Straight to the two numbers this loop reads, so a frame
                # does not build twenty thousand contact objects to look at
                # a normal and a depth and drop them.
                pushes, depths = collide.capsule_mesh_pushes(
                    capsule, Pj, self._near_triangles(j, Pj, capsule))
                contacts.extend(zip(pushes, depths, strict=True))
                continue
            if fast and isinstance(Pj, HeightFieldProxy):
                contacts.extend(collide.upright_capsule_heightfield_pushes(
                    Pj, float(pos[0]), float(pos[1]), float(pos[2]),
                    capsule.half_height, capsule.radius))
                continue
            for c in collide.collide(0, 1, capsule, Pj):
                contacts.append((-c.normal, c.depth))   # from world into char
        # Dynamic bodies come after the static world, so a contact tie goes
        # to the body that can carry the capsule.
        first_dynamic = len(contacts)
        for obstacle in self._obstacles:
            for c in collide.collide(0, 1, capsule, obstacle.proxy):
                contacts.append((-c.normal, c.depth))
                owners.append(obstacle.body)
        return first_dynamic

    # -- movement --------------------------------------------------------
    #: How much of the capsule's own extent it may cross in one collision step.
    #: Collision is *discrete* -- each step places the capsule and then resolves
    #: whatever it overlaps -- so a step that carries it clean past a surface
    #: leaves nothing to overlap and nothing to be stopped by.  Half leaves the
    #: contact plenty of depth to be found at.
    SUBSTEP_REACH = 0.5

    def update(self, dt: float) -> None:
        """Advance the avatar: move-and-slide, gravity, ground/slope handling.

        Split into substeps short enough that collision cannot be outrun.  A
        fall from a third-storey window reaches ~15 m/s, which at a 20 fps
        frame is three quarters of a metre in one step: enough to place the
        capsule beyond the floor it should have landed on, where the floor is
        no longer something it overlaps and so no longer something it can be
        stopped by.  Stepping it in shorter pieces is what makes the landing
        depend on the geometry rather than on the frame rate.

        The dynamic bodies near the capsule are read once, at the start, and
        stay where they were found for the whole update. A body the character
        stands on carries it by the velocity it was found with.
        """
        self._gather_for(dt)
        self._keep_gathers()
        try:
            self._advance_pose(dt)
            start = self.position.copy()
            self._arrived = self.velocity.copy()
            self.pushing = None
            self._landing = None
            for piece in self._substeps(dt):
                was_grounded = self.grounded
                self._step(piece)
                self._carry(piece)
                # After the step, so "left the ground" and "landed" are settled
                # facts rather than predictions, and so every early return above is
                # covered.
                if self.grounded and not was_grounded and self._impact > 0.0:
                    self._landing = max(self._landing or 0.0, self._impact)
                if not self.flying:
                    self._tick_jump_windows(piece, was_grounded)
        finally:
            self._gathers = None
        self.velocity = (self.position - start) / dt if dt > 0 else np.zeros(3)
        self._carry_body()

    def _gather_for(self, dt: float) -> None:
        """Read the dynamic bodies this update can reach.

        The box is the capsule grown by :attr:`NEAR_MARGIN`, the step height and
        the furthest it can travel in ``dt`` at the speeds it has.
        """
        if self.flying:
            self._obstacles, self._obstacle_of = [], {}
            return
        caps = self.caps
        speed = max(self.speed(), caps.swimSpeed) + length(self.push)
        fall = abs(self.vy) + self.gravity_mag * dt
        reach = self.NEAR_MARGIN + caps.stepHeight + (speed + fall) * dt
        half = np.array([caps.radius + reach, 0.5 * max(self.height, caps.standHeight) + reach,
                         caps.radius + reach])
        self._gather_obstacles(self.position - half, self.position + half)

    def _advance_pose(self, dt: float) -> None:
        """Move a slide along: the capsule stays low, then rises, then the slide ends.

        The rise waits while the taller capsule would overlap something. A
        slide that never gets the room ends crouched.
        """
        slide = self._slide
        if slide is None:
            return
        slide.elapsed += dt
        done = slide.elapsed >= slide.duration - 1e-9
        into = slide.elapsed - (slide.duration - slide.rise)
        fraction = 1.0 if done else min(max(into / slide.rise, 0.0), 1.0) if slide.rise else 0.0
        low, high = self.caps.crouchHeight, self.caps.standHeight
        height = low + (high - low) * fraction
        if height > self.height and self._blocked_at(height):
            height = self.height
        self._resize(height)
        if not done:
            return
        self._slide = None
        if height >= high - 1e-9:
            self.crouching = False
            self._pose_height = None
        else:
            self.crouching = True
            self._resize(low)
            self._pose_height = None

    def _carry(self, dt: float) -> None:
        """Move the capsule with the dynamic body it stands on."""
        if not self.grounded or self._standing_on is None:
            return
        obstacle = self._obstacle_of.get(self._standing_on)
        if obstacle is not None:
            self.position = self.position + obstacle.velocity * dt

    #: How far past the capsule's surface a static collider still counts as
    #: touched, in metres. A capsule resting on a floor is pushed just clear
    #: of it, so touching is measured with this much to spare.
    TOUCH_SKIN = 0.02

    def _carry_body(self) -> None:
        """Put :attr:`body` where the capsule is, and report what it began or stopped touching.

        The body moves at the capsule's velocity, so a touch it has with a
        moving body has a closing speed. The controller's own collisions, with
        the static world, are reported here while the world records contacts.
        """
        world = self.world
        if self.body is None or world is None:
            return
        world.place_body(self.body, position=self.position)
        world.linear_velocity[self.body] = self.velocity
        if world.contact_reporting == 'off':
            self._touching = {}
            return
        now = self._touches_static()
        events = [self._event(j, 'begin', normal, point)
                  for j, (normal, point) in now.items() if j not in self._touching]
        events += [self._event(j, 'end', normal, point, reason='separated')
                   for j, (normal, point) in self._touching.items() if j not in now]
        self._touching = now
        world.report_contact_events([event for event in events if event is not None])

    def _touches_static(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """The static colliders within :data:`TOUCH_SKIN` of the capsule."""
        middle = max(self.height - 2 * self.caps.radius, 1e-3)
        skin = model.Shape.capsule(height=middle, radius=self.caps.radius + self.TOUCH_SKIN)
        capsule = make_proxy(skin, self.position, IDENT)
        found: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        lo, hi = capsule.aabb()
        near = list(self._static_bodies(lo, hi))
        boxed = _StaticBoxes(near, self._static_proxy).pushes(cast(CapsuleProxy, capsule))
        for j in near:
            other = self._static_proxy(j)
            if isinstance(other, BoxProxy):
                if j in boxed:
                    normal = np.asarray(boxed[j][0], dtype='d')
                    found[j] = (normal, self._surface_point(normal))
                continue
            if isinstance(other, TriangleMeshProxy):
                pushes, depths = collide.capsule_mesh_pushes(
                    cast(CapsuleProxy, capsule), other,
                    self._near_triangles(j, other, cast(CapsuleProxy, capsule)))
                if not len(depths):
                    continue
                normal = np.asarray(pushes[int(np.argmax(depths))], dtype='d')
            else:
                contacts = collide.collide(0, 1, capsule, other)
                if not contacts:
                    continue
                normal = -max(contacts, key=_depth_of_contact).normal
            found[j] = (normal, self._surface_point(normal))
        return found

    def _surface_point(self, normal: np.ndarray) -> np.ndarray:
        """Where the capsule's surface meets a collider whose normal is ``normal``."""
        half = max(self.height * 0.5 - self.caps.radius, 0.0)
        end = self.position - UP * half * float(np.sign(normal[1]))
        return end - normal * self.caps.radius

    def _event(self, j: int, phase: str, normal: np.ndarray, point: np.ndarray,
               reason: str | None = None) -> Any:
        """A contact event between static body ``j`` and :attr:`body`; None if unreported."""
        world, mine = self.world, cast(int, self.body)
        if world is None or not world.reports_pair(j, mine):
            return None
        approach = (max(0.0, -float(np.dot(self._arrived, normal)))
                    if phase == 'begin' else 0.0)
        # The normal runs from the lower-indexed body to the other; ``normal``
        # runs from the surface into the capsule.
        a, b = (j, mine) if j < mine else (mine, j)
        return ContactEvent(phase, world.ref(a), world.ref(b), point,
                            normal if j < mine else -normal, approach, 0.0, 0.0, 0.0,
                            0.0, 1 if phase == 'begin' else 0, world.step_count,
                            world.time, reason=reason)

    def remove(self) -> None:
        """Take :attr:`body` out of the world; the controller keeps working without it.

        What the capsule was touching ends with ``reason='removed'``.
        """
        world = self.world
        if self.body is None or world is None:
            return
        if world.contact_reporting != 'off':
            world.report_contact_events([
                event for event in (self._event(j, 'end', normal, point, reason='removed')
                                    for j, (normal, point) in self._touching.items())
                if event is not None])
        self._touching = {}
        world.remove_body(self.body)
        self.body = None

    def _reach(self) -> tuple[float, float]:
        """How far the capsule may travel before a surface stops overlapping it.

        Two numbers because the capsule is not round: it is **as wide as its
        radius and as tall as its height**, so it can cross far more ground
        vertically than horizontally before a floor it has passed is no longer
        something it touches.  Budgeting the vertical by the narrow measure
        would substep an ordinary walk for nothing.
        """
        return (self.caps.radius * self.SUBSTEP_REACH,
                self.height * 0.5 * self.SUBSTEP_REACH)

    def terminal_velocity(self) -> float:
        """Fastest the capsule may fall, or ``inf`` where none was set."""
        limit = float(self.caps.terminalVelocity)
        return limit if limit > 0 else float('inf')

    def max_substeps(self, dt: float) -> int:
        """The most substeps a frame of ``dt`` can ever need.

        **Calculated, not chosen.**  A fixed ceiling is a number nobody can
        check: raise the fall speed past what it allows and a step outruns
        collision again, silently and only at speed.  Derived from terminal
        velocity the two cannot disagree -- whatever the capsule is allowed to
        reach is what the stepping is sized for.

        Unbounded (``sys.maxsize``) when ``terminalVelocity`` is 0, which is
        the honest answer rather than a safe-looking one: with no top speed
        there is no bound on the work a frame might need and no guarantee a
        step cannot outrun collision.  The stepping still follows the actual
        speed, so the cost stays proportional rather than jumping to the
        sentinel.
        """
        limit = self.terminal_velocity()
        _across, along = self._reach()
        if along <= 0 or dt <= 0:
            return 1
        if limit == float('inf'):
            return sys.maxsize
        return max(1, math.ceil(limit * dt / along))

    def _substeps(self, dt: float) -> list[float]:
        """``dt`` cut into pieces no one of which can outrun collision.

        Sized from the speed the capsule will actually reach this frame,
        gravity included, so a fall is stepped finely exactly while it is fast
        and costs nothing at all once it is standing still.  The count cannot
        exceed :meth:`max_substeps`, because the speed feeding it cannot exceed
        terminal velocity.
        """
        if dt <= 0:
            return [dt]
        vertical = min(abs(self.vy) + self.gravity_mag * dt,
                       self.terminal_velocity()) * dt
        horizontal = length(self._intent() + self.push) * dt
        across, along = self._reach()
        if across <= 0 or along <= 0:
            return [dt]
        count = max(math.ceil(horizontal / across),
                    math.ceil(vertical / along), 1)
        count = min(count, self.max_substeps(dt))
        return [dt / count] * count

    def _step(self, dt: float) -> None:
        """One movement step, before the jump windows are aged."""
        self._impact = 0.0
        self._standing_on = None
        if self.flying:
            self._update_fly(dt)
            return
        if self.swimming:
            self._update_swim(dt)
            return
        was_grounded = self.grounded
        if self.grounded:
            # Friction is a contact effect. Bleeding impulse speed off in flight
            # instead would shorten every launch by however long it hangs there.
            self.push = self.push * max(0.0, 1.0 - self.caps.pushFriction * dt)
        control = 1.0 if self.grounded else self.caps.airControl
        horiz = self._intent() * control + self.push
        horiz = self._settle_step_debt(horiz, dt)
        wanted = horiz * FLAT
        if self.grounded:
            horiz = self._along_ground(horiz)

        target = self.position + horiz * dt
        resolved, _ = self._push_out(target)
        if not self._on_walkable_ground() and resolved[1] > target[1]:
            # Pushing out of a surface travels along its normal, so walking into
            # anything tilted lifts the capsule -- and a player holding forward
            # against a cliff climbs it.  Off walkable ground the move asked for
            # no height, so any it gained is that lift and is dropped.  On a
            # walkable slope the climb is already in ``target`` (the move was
            # turned along the ground) and the seating that follows is real.
            resolved = np.array([resolved[0], target[1], resolved[2]])
        moved = flat_length(resolved - self.position)
        want = flat_length(horiz) * dt
        blocker = self._pushable_blocker(resolved, wanted) if want > 1e-6 else None
        if want > 1e-6 and moved < 0.7 * want:
            stepped = self._try_step_up(horiz, dt)
            if stepped is not None:
                resolved, blocker = stepped, None
        if blocker is not None:
            self.pushing = (blocker, self._push_velocity(wanted))
        self.position = resolved

        # Step-down: while grounded and not rising, stick to the surface over
        # small down-steps instead of launching off the edge into a fall (which
        # otherwise reads as a slow sink followed by a jump).
        if was_grounded and self.vy <= 1e-6:
            seated, gn = self._snap_down(self.caps.stepHeight)
            if gn is not None:
                # Seat vertically and keep the horizontal position the move
                # already resolved.  The seat is found by pushing out of the
                # surface, which travels along its normal -- back down the hill
                # on any slope -- and letting that move the capsule sideways
                # takes a slice off every step of a climb.
                self.position = np.array(
                    [self.position[0], seated[1], self.position[2]])
                self.grounded = True
                self.ground_normal = gn
                self._standing_on = self._ground_source
                self.vy = 0.0
                return

        # Over a bottomless void (no geometry below the footprint), hover instead
        # of dropping.  Gated on a non-rising velocity so a jump *across* a void
        # still arcs up normally and only floats once it stops rising.
        if self.caps.suppressOverVoid and self.vy <= 0.0 and self._over_void():
            self.vy = 0.0
            self.grounded = False
            return

        vtarget = self.position + UP * self._fall(dt)
        self._impact = max(0.0, -self.vy)
        vresolved, ground_n = self._push_out(vtarget)
        self.position = vresolved
        self._update_grounded(ground_n)
        if self.grounded:
            self._apply_slope_slide(ground_n, dt)

    def _fall(self, dt: float) -> float:
        """Advance :attr:`vy` by one step of gravity; return the height gained.

        Constant acceleration is integrated exactly, ``v dt - g dt^2 / 2``, so a
        jump or a fall covers the same distance in the same time at every frame
        rate. Past :meth:`terminal_velocity` the capsule falls at that speed for
        the rest of the step.
        """
        g = self.gravity_mag
        v0 = self.vy
        limit = self.terminal_velocity()
        v1 = v0 - g * dt
        if v1 >= -limit:
            self.vy = v1
            return v0 * dt - 0.5 * g * dt * dt
        # The limit is reached part-way through the step.
        reach = min(max((v0 + limit) / g, 0.0), dt) if g > 0.0 else 0.0
        self.vy = -limit
        return v0 * reach - 0.5 * g * reach * reach - limit * (dt - reach)

    def _pushable_blocker(self, at: np.ndarray, wanted: np.ndarray) -> int | None:
        """The pushable body in the way of a step taken to ``at``, or None.

        A body is in the way when it is within :attr:`PUSH_REACH` of the capsule
        and presses on it against the direction it is going. The deepest light
        one is the answer, unless a heavy one is in the way too: that one blocks
        the step.
        """
        if not self._obstacles:
            return None
        speed = flat_length(wanted)
        if speed <= 1e-9:
            return None
        mid = max(self.height - 2 * self.caps.radius, 1e-3)
        reach = make_proxy(model.Shape.capsule(height=mid, radius=self.caps.radius
                                               + self.PUSH_REACH), at, IDENT)
        found, deepest = None, 0.0
        for obstacle in self._obstacles:
            for c in collide.collide(0, 1, reach, obstacle.proxy):
                push = -c.normal                          # from the body into the capsule
                across = flat_length(push)
                if across < 0.3 or float(np.dot(push * FLAT, wanted)) / (across * speed) > -0.3:
                    continue
                if not obstacle.light:
                    return None
                if c.depth > deepest:
                    found, deepest = obstacle.body, c.depth
        return found

    #: How close to a body, in metres, the capsule is taken to be pushing it. A
    #: kinematic stand-in for the character whose capsule is wider than the
    #: controller's by up to this much overlaps a body the character touches,
    #: which is what makes the solver move it.
    PUSH_REACH = 0.05

    def _push_velocity(self, wanted: np.ndarray) -> np.ndarray:
        """``wanted`` limited to ``pushSpeed``, or to ``walkSpeed`` where that is None."""
        limit = self.caps.walkSpeed if self.caps.pushSpeed is None else self.caps.pushSpeed
        speed = length(wanted)
        return wanted * (limit / speed) if speed > limit else wanted.copy()

    def _snap_down(self, max_drop: float,
                   step: float = 0.04) -> tuple[np.ndarray, np.ndarray | None]:
        """Find ground within ``max_drop`` below the capsule and seat on it."""
        drop = step
        while drop <= max_drop + 1e-9:
            seated, gn = self._push_out(self.position - UP * drop, iterations=2)
            if gn is not None:
                return seated, gn
            drop += step
        return self.position, None

    def _over_void(self) -> bool:
        """True when no static geometry lies anywhere below the capsule footprint.

        Casts straight-down infinite rays from the capsule base — its centre plus
        four points on the radius ring — and tests each against static-collider
        AABBs.  A void is where *every* ray misses: as long as one footprint point
        is over solid ground the character is at an edge, not over a void, and
        normal gravity applies so it falls or slides off naturally.

        AABB-level detection catches genuine voids (the edge of the world, a gap
        with no geometry in the column); it does not see holes *inside* a single
        collider's bounding box, and errs toward "not a void" (fall) when unsure.
        """
        r = self.caps.radius
        base = self.base()
        oy = base[1]
        origins = ((base[0], base[2]), (base[0] + r, base[2]),
                   (base[0] - r, base[2]), (base[0], base[2] + r),
                   (base[0], base[2] - r))
        # Below the feet, over the whole column: the box has no bottom.
        under = self._static_bodies(np.array([base[0] - r, -np.inf, base[2] - r]),
                                    np.array([base[0] + r, oy, base[2] + r]))
        aabbs = [self._static_proxy(j).aabb() for j in under]
        for ox, oz in origins:
            for lo, hi in aabbs:
                if lo[0] <= ox <= hi[0] and lo[2] <= oz <= hi[2] and lo[1] <= oy:
                    return False                       # ground under this point
        return True

    def _update_fly(self, dt: float) -> None:
        """Advance the avatar one step in fly/noclip mode (free movement, no collision)."""
        # Fly is noclip: move freely, no depenetration. Depenetrating every frame
        # while overlapping geometry (e.g. after a stuck safe-bind fell back to
        # fly) oscillates the position; noclip lets the user simply fly out.
        move = (self.move_dir + self.fly_dir) * self.caps.flySpeed
        if self.down:
            move = np.zeros(3)
        self.position = self.position + move * dt
        self.vy = 0.0
        self.push = np.zeros(3)                 # noclip answers to input alone

    def _update_swim(self, dt: float) -> None:
        """Advance the avatar one step under water.

        Halfway between the other two, and deliberately so.  The *movement* is
        flying's -- a direction in three dimensions, at one speed, with no
        ground to walk along and no slope to slide down.  The *collision* is
        walking's: the capsule is depenetrated from the world every step, so a
        pool has a bottom, a wall and a ceiling.

        The vertical is what is neither.  Gravity is scaled by what buoyancy
        does not cancel, and drag bleeds the result away, so a sink settles at
        a steady speed instead of accelerating; the two together are most of
        what tells a player they are in water rather than in air.
        """
        self.grounded = False
        self.ground_normal = None
        # Water is something you push against, so a stroke builds speed and
        # letting go coasts.  Movement that switched on and off is what makes
        # a swim read as walking with the floor removed.
        desired = (self.move_dir + self.fly_dir) * self.caps.swimSpeed
        if self.down:
            desired = np.zeros(3)
        self._swim_velocity += ((desired - self._swim_velocity)
                                * min(1.0, self.caps.swimDrag * dt))
        self.vy += (self.buoyancy - 1.0) * self.gravity_mag * dt
        # A surface already being rested against opposes what pushes into it,
        # so that push must not accumulate.  Depenetration leaves the capsule a
        # hair clear of what it met, and a hair is room enough for a frame of
        # buoyancy to build the speed back and cross it again: without this a
        # swimmer held under a ceiling bounces against it for ever, a couple of
        # millimetres at a time, and reads as still rising half the frames.
        if self._swim_contact * self.vy > 0.0:
            self.vy = 0.0
        self.vy *= max(0.0, 1.0 - self.caps.swimDrag * dt)
        # An impulse -- a jump pad firing into a pool -- carries through the
        # water and is bled off by the same drag, so it slows rather than
        # stopping dead at the surface.
        self.push = self.push * max(0.0, 1.0 - self.caps.swimDrag * dt)
        velocity = self._swim_velocity + UP * self.vy + self.push
        target = self.position + velocity * dt
        resolved, _ = self._push_out(target)
        # Meeting the bottom or the ceiling has to stop the drift, not merely
        # be undone by it: speed accumulating against the floor would rocket
        # the swimmer the moment they turned upward.
        if abs(resolved[1] - target[1]) > 1e-6:
            self._swim_contact = 1.0 if velocity[1] > 0.0 else -1.0
            self.vy = 0.0
        elif self._swim_contact * velocity[1] < 0.0:
            # Swimming away from it: whatever was met is behind us now.
            self._swim_contact = 0.0
        self.position = resolved

    def _settle_step_debt(self, horiz: np.ndarray, dt: float) -> np.ndarray:
        """Take back what a step-up advanced beyond its frame's due.

        Paid off over the frames that follow rather than all at once, so the
        capsule keeps moving -- a stall would read as the stutter the whole
        one-motion step exists to avoid.  It makes the *average* speed right;
        it does not make the step-up frame itself right, which is the bug
        recorded on :meth:`_try_step_up`.
        """
        if self._step_debt <= 0.0:
            return horiz
        due = flat_length(horiz) * dt
        if due <= 1e-9:
            return horiz
        paid = min(self._step_debt, due)
        self._step_debt -= paid
        return horiz * (1.0 - paid / due)

    def _try_step_up(self, horiz: np.ndarray, dt: float) -> np.ndarray | None:
        """Try to climb a small ledge ahead; return the seated position or None if not a step."""
        speed = flat_length(horiz)
        if speed < 1e-6:
            return None
        hdir = horiz / speed
        lifted = self.position + UP * self.caps.stepHeight
        if self._overlaps(self._proxy(lifted)):
            return None                                     # too tall / ceiling above
        # Move forward far enough to clear the step edge and land *on* the step
        # in one motion (a small per-frame velocity alone catches the edge and
        # never gets over it, which an earlier version hid with an airborne hop
        # -> the jitter).
        forward = max(speed * dt, self.caps.radius + 0.05)
        ahead, _ = self._push_out(lifted + hdir * forward)
        if flat_length(ahead - self.position) < 0.5 * forward:
            return None                                     # blocked -> not a step
        dropped, ground_n = self._push_out(ahead - UP * self.caps.stepHeight)
        # Only accept a step onto solid ground that is actually higher than here
        # (never leave the avatar hanging, and never "step" down onto the floor)
        # and that could be stood on.  Without the slope test a face steeper
        # than ``maxSlope`` is climbable one step-height at a time, so anything
        # rising less than ``stepHeight`` per frame -- which at running speed is
        # most cliffs -- can simply be walked up.
        if (ground_n is None or not self._walkable(ground_n)
                or dropped[1] <= self.position[1] + 1e-3):
            return None
        # There *is* a step.  Now cross it at this frame's own pace: the probe
        # above went as far as it had to in order to answer the question, and
        # moving that far would carry the capsule several frames' distance in
        # one -- which the eye reads as a sharp jump forward however carefully
        # the excess is paid back over the frames that follow.  An average that
        # is right and an instant that is wrong is still wrong.
        # The step is mounted in one motion, which is further than this frame
        # was due -- so the excess is owed.  Without it a flight of stairs is
        # climbed one step per *frame* rather than at running speed, and the
        # better the frame rate the faster the stairs.
        #
        # **This is the cause of a known bug**: the whole mount lands in a
        # single frame, and the eye reads that as a sharp jump forward however
        # carefully the distance is paid back afterwards.  See
        # ``tests/test_character_step_pace.py``, which measures it, and the
        # bug's entry in twitch's PROJECT-PLAN -- three fixes were tried and
        # each traded the lurch for a stall.  An average that is right and an
        # instant that is wrong is still wrong, and this is the wrong one that
        # at least always works.
        advanced = flat_length(dropped - self.position)
        self._step_debt += max(0.0, advanced - speed * dt)
        return dropped

    def _along_ground(self, velocity: np.ndarray) -> np.ndarray:
        """A horizontal velocity turned to run along the surface, same speed.

        Moved horizontally into a slope, the capsule penetrates it and is pushed
        back out along the surface normal, whose horizontal component opposes
        the motion -- so the distance covered falls off with the slope and a
        walkable ramp feels like wading.  Redirecting first means the whole
        speed is spent going somewhere: a run up a ramp covers the same metres
        per second as a run along the flat, and the climb is the vertical part
        of that rather than something taken out of it.

        Only for slopes that can be walked.  Steeper than ``maxSlope`` the
        surface is a wall, and projecting onto it would let the player run up
        it.
        """
        normal = self.ground_normal
        if normal is None or not self._walkable(normal):
            return velocity
        speed = length(velocity)
        if speed <= 1e-9 or abs(float(normal[1])) <= 1e-6:
            return velocity
        # The *heading* is the player's and is never turned: only the height
        # needed to follow the surface is added, and the result is rescaled so
        # the speed along the surface is the speed that was asked for.  A plain
        # projection onto the plane would also rotate the heading, which on a
        # floor whose contact normal is a hair off vertical steers the player
        # sideways for as long as they walk.
        climb = -(float(normal[0]) * float(velocity[0])
                  + float(normal[2]) * float(velocity[2])) / float(normal[1])
        along = np.array([velocity[0], climb, velocity[2]])
        reach = length(along)
        if reach <= 1e-9:
            return velocity
        return along * (speed / reach)

    def _on_walkable_ground(self) -> bool:
        """Whether the capsule is standing on something it could walk along."""
        return bool(self.grounded and self.ground_normal is not None
                    and self._walkable(self.ground_normal))

    def _walkable(self, normal: np.ndarray) -> bool:
        """Whether a surface with this normal is shallow enough to stand on.

        Asked as a cosine rather than an angle. ``arccos`` decreases over
        [-1, 1], so a slope within ``maxSlope`` is exactly a normal whose
        component along up is at or above the cosine of it -- the same question
        without the two transcendentals, which matters because depenetration
        asks it of every contact it resolves.
        """
        return float(normal[1]) >= self._slope_cos()

    def _slope_cos(self) -> float:
        """The cosine of ``maxSlope``, kept against the limit it was taken from,
        so a capability changed mid-run is picked up."""
        limit = self.caps.maxSlope
        if limit != self._slope_limit:
            self._slope_limit = float(limit)
            self._slope_cosine = math.cos(math.radians(self._slope_limit))
        return self._slope_cosine

    #: How far below itself the capsule looks for floor when nothing touched it
    #: this step.  Enough to bridge the gap a step opens under a walker, small
    #: enough not to hunt for ground that is not there.
    GROUND_PROBE = 0.05

    def _update_grounded(self, ground_n: np.ndarray | None) -> None:
        """Set the grounded flag from the vertical contact, probing just below if needed.

        **A rising capsule is never grounded**, whatever is under it.  One frame
        after a jump the capsule has climbed only ``vy * dt``, which on a fast
        machine is less than the probe reaches -- so without this the launch is
        snapped straight back onto the floor and its velocity zeroed in the same
        frame it started, and the faster the machine the more jumps vanish.  It
        also still *touches* the floor it is leaving, so the contact above says
        "ground" too; a capsule on its way up has left deliberately and neither
        answer applies to it.
        """
        if self.vy > 0.0:
            self.grounded = False
            self.ground_normal = None
            return
        if ground_n is not None:
            self.grounded = True
            self.ground_normal = ground_n
            self._standing_on = self._ground_source
            self.vy = max(self.vy, 0.0)
            return
        probe, gn = self._push_out(self.position - UP * self.GROUND_PROBE)
        if gn is not None:
            self.position = probe
            self.grounded = True
            self.ground_normal = gn
            self._standing_on = self._ground_source
            self.vy = 0.0
        else:
            self.grounded = False
            self.ground_normal = None

    def _apply_slope_slide(self, ground_n: np.ndarray | None, dt: float) -> None:
        """Slide the avatar downhill and unground it when the slope exceeds ``maxSlope``."""
        if ground_n is None:
            return
        slope = np.degrees(np.arccos(np.clip(np.dot(ground_n, UP), -1, 1)))
        if slope > self.caps.maxSlope:
            downhill = ground_n - UP * np.dot(ground_n, UP)
            n = np.linalg.norm(downhill)
            if n > 1e-6:
                slide = -downhill / n
                self.position, _ = self._push_out(
                    self.position + slide * self.gravity_mag * dt * dt * 4.0)
                self.grounded = False
                self.ground_normal = None

    # -- safe viewpoint binding -----------------------------------------
    def safe_bind(self, position: Vec, search: float = 4.0) -> bool:
        """Place the capsule at ``position`` without ending up stuck in geometry.

        Depenetrate from any overlapping static collider, then snap the base onto
        the floor below (or pop up if bound beneath it).  If no free space is
        found, keep the pose but flag ``stuck`` (and enter fly).
        """
        bound = self._bind(position, search)
        self._carry_body()
        return bound

    def _bind(self, position: Vec, search: float) -> bool:
        """:meth:`safe_bind` without carrying the body along."""
        self.position = np.asarray(position, dtype='d').copy()
        self.stuck = False
        self.flying = False                         # a fresh bind starts on foot
        self.push = np.zeros(3)                     # ...and at rest
        self.vy = 0.0
        self.velocity = np.zeros(3)
        reach = np.array([self.caps.radius, 0.5 * self.caps.standHeight + search,
                          self.caps.radius]) + self.NEAR_MARGIN
        self._gather_obstacles(self.position - reach, self.position + reach)
        resolved, _ground_n = self._push_out(self.position, iterations=12)
        self.position = resolved
        if self._overlaps(self._proxy()):
            self.stuck = True
            self.set_fly(True)
            return False
        self._ground_snap(search)
        return True

    def _ground_snap(self, search: float) -> None:
        """Drop the capsule up to ``search`` metres onto the first floor below and seat it.

        Nothing within reach leaves it airborne, and it must say so: the pose
        before the bind is no evidence about the one after it, and a caller
        asking whether there is ground underfoot -- to decide between walking
        and flying there -- gets the wrong answer from a stale yes.
        """
        step = 0.05
        drop = 0.0
        self.grounded = False
        while drop < search:
            probe = self.position - UP * (drop + step)
            _, gn = self._push_out(probe, iterations=2)
            if gn is not None:
                seated, _ = self._push_out(probe)
                self.position = seated
                self.grounded = True
                self._standing_on = self._ground_source
                self.vy = 0.0
                return
            drop += step
