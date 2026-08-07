"""Character controller — a kinematic capsule with move-and-slide.

A game-usable first-person controller: tiered ground speed
(walk/run/sprint/crouch), jump when grounded, fly/noclip, step-up over small
ledges, and sliding on steep slopes.  It also implements **safe viewpoint
binding** (depenetration + ground snap) so a camera placed low or inside geometry
never leaves the user stuck in the ground.

The capsule is *kinematic* (move-and-slide against the world's static colliders)
rather than a dynamic rigid body — the standard approach for responsive avatars.
Pure CPU; the GL-facing :class:`PhysicsViewPlatform` wraps this.
"""
import math
import sys
import weakref
from typing import (Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING,
                    cast)
import numpy as np

from dataclasses import dataclass

from . import model
from .body import CapsuleProxy, TriangleMeshProxy, make_proxy, Proxy
from . import collide
from .mathutil import Vec

if TYPE_CHECKING:
    from .world import PhysicsWorld

UP = np.array([0.0, 1.0, 0.0])
IDENT = np.array([0.0, 0.0, 0.0, 1.0])

#: World-space proxies for each world's static colliders, keyed by world and
#: then by body index.  Weak, so an unloaded level takes its collision copy
#: with it rather than keeping it alive for the rest of the process.  See
#: :meth:`CharacterController._static_proxy` for why it is shared.
_STATIC_PROXIES: "weakref.WeakKeyDictionary[Any, Dict[int, Proxy]]" = \
    weakref.WeakKeyDictionary()


def _static_proxies(world: "PhysicsWorld") -> Dict[int, Proxy]:
    """The proxy cache belonging to ``world``, made on first use."""
    found = _STATIC_PROXIES.get(world)
    if found is None:
        found = _STATIC_PROXIES[world] = {}
    return found


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


class CharacterController:
    """Kinematic capsule avatar: move-and-slide against the world's static colliders."""

    def __init__(self, world: "PhysicsWorld",
                 capabilities: Optional[CharacterCapabilities] = None,
                 position: Vec = (0, 0, 0), gravity: float = 9.81) -> None:
        """Place a capsule of ``capabilities`` at ``position`` (its centre) in ``world``."""
        self.world = world
        self.caps = capabilities or CharacterCapabilities()
        self.gravity_mag = gravity
        self.position = np.asarray(position, dtype='d')     # capsule centre
        self.vy = 0.0
        self.grounded = False
        #: Surface normal underfoot while grounded, else None.  What the move
        #: direction is projected onto, so speed is spent along the ground
        #: rather than along the horizon.
        self.ground_normal: Optional[np.ndarray] = None
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
        self.mode = 'walk'
        self.move_dir = np.zeros(3)                         # world horizontal unit
        self.fly_dir = np.zeros(3)
        self.push = np.zeros(3)                             # impulse-carried horizontal m/s
        self.stuck = False
        #: Seconds since the capsule last left the ground by *falling*, or None
        #: when it is grounded or left by jumping.  See ``coyoteTime``.
        self._airborne_for: Optional[float] = None
        #: Seconds since a jump was asked for and refused, or None.  See
        #: ``jumpBuffer``.
        self._jump_wanted_for: Optional[float] = None
        #: Metres a step-up advanced beyond what its frame was due, taken back
        #: out of the frames that follow.  See :meth:`_try_step_up`.
        self._step_debt: float = 0.0
        # Static body -> world-space proxy, shared with every other avatar in
        # this world.  See `_static_proxy`.
        self._proxy_cache: Dict[int, Proxy] = _static_proxies(world)
        # Per *avatar*, unlike the proxy cache above: it is keyed on where this
        # capsule is, and two avatars are in different places.  See
        # `_near_triangles`.
        self._near_cache: Dict[int, Tuple[np.ndarray, np.ndarray,
                                          np.ndarray]] = {}

    # -- geometry --------------------------------------------------------
    @property
    def height(self) -> float:
        """Current capsule height (crouch or stand, per pose)."""
        return self.caps.crouchHeight if self.crouching else self.caps.standHeight

    def _shape(self, height: Optional[float] = None) -> model.Shape:
        """Capsule shape at ``height`` (defaults to the current pose height)."""
        h = self.height if height is None else height
        mid = max(h - 2 * self.caps.radius, 1e-3)
        return model.Shape.capsule(height=mid, radius=self.caps.radius)

    def _proxy(self, position: Optional[Vec] = None,
               height: Optional[float] = None) -> Proxy:
        """World-space collision proxy for the capsule at ``position`` and ``height``."""
        pos = self.position if position is None else position
        return make_proxy(self._shape(height), pos, IDENT)

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
        d = d * np.array([1, 0, 1])                         # flatten to ground plane
        n = np.linalg.norm(d)
        self.move_dir = d / n if n > 1e-9 else np.zeros(3)
        self.mode = mode

    def set_fly_move(self, direction: Vec) -> None:
        """Set the desired fly move (full 3D unit direction, no ground flattening)."""
        d = np.asarray(direction, dtype='d')
        n = np.linalg.norm(d)
        self.fly_dir = d / n if n > 1e-9 else np.zeros(3)

    def speed(self) -> float:
        """Ground speed for the current pose and mode (walk/run/sprint/crouch)."""
        if self.crouching:
            return self.caps.crouchSpeed
        if self.mode == 'sprint' and self.caps.canSprint:
            return self.caps.sprintSpeed
        if self.mode == 'run':
            return self.caps.runSpeed
        return self.caps.walkSpeed

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
        return bool(self.caps.canJump and not self.crouching and not self.flying)

    def _can_launch(self) -> bool:
        """Whether the capsule is on the ground, or close enough in time to it."""
        if self.grounded:
            return True
        return (self.caps.coyoteTime > 0.0 and self._airborne_for is not None
                and self._airborne_for <= self.caps.coyoteTime)

    def _launch(self) -> None:
        self.vy = np.sqrt(2.0 * self.gravity_mag * self.caps.jumpHeight)
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
        self.push = v * np.array([1, 0, 1])
        self.vy = float(v[1])
        self.grounded = False
        # Launched, not fallen: no coyote time off a jump pad, and a press held
        # from before the launch must not fire the moment it lands.
        self._airborne_for = None
        self._jump_wanted_for = None

    def set_crouch(self, crouch: bool) -> bool:
        """Crouch or stand; standing fails (returns False) if the head is blocked."""
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
        if self.swimming:
            self.grounded = False
            self.ground_normal = None
            self._airborne_for = None
            self._jump_wanted_for = None

    def _can_stand(self) -> bool:
        """No static geometry where the standing capsule's head would be."""
        stand = self._proxy(height=self.caps.standHeight)
        # raise so the feet stay put while the head extends upward
        lift = 0.5 * (self.caps.standHeight - self.caps.crouchHeight)
        stand = make_proxy(self._shape(self.caps.standHeight),
                           self.position + UP * lift, IDENT)
        return not self._overlaps(stand)

    # -- world queries ---------------------------------------------------
    def _static_bodies(self) -> Iterator[int]:
        """Yield the index of each non-kinematic body that has a collider."""
        w = self.world
        for j in range(w.body_count):
            if w.collider_shape[j] >= 0 and w.motion_type[j] != 2:
                yield j

    def _static_proxy(self, j: int) -> Proxy:
        """Cached world-space proxy for a static body (they never move, so a
        trimesh's transformed vertices are computed once, not per query).

        The cache is the **world's**, shared by every avatar walking in it: a
        trimesh proxy is not a handle but a transformed copy of the geometry
        with a uniform grid over it, so one per controller is one copy of the
        level per person in the level.  With opponents in the world that is
        paid again the first time each of them takes a step, which reads as a
        hitch at the start of a fight.  Sharing is correct precisely because
        these bodies never move.
        """
        proxy = self._proxy_cache.get(j)
        if proxy is None:
            w = self.world
            proxy = make_proxy(w.shapes[w.collider_shape[j]],
                               w.position[j], w.orientation[j])
            self._proxy_cache[j] = proxy
        return proxy

    def _overlaps(self, proxy: Proxy) -> bool:
        """True if ``proxy`` intersects any static collider."""
        for j in self._static_bodies():
            if collide.collide(0, 1, proxy, self._static_proxy(j)):
                return True
        return False

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
        cached = self._near_cache.get(body)
        if cached is not None:
            low, high, verts = cached
            if np.all(lo >= low) and np.all(hi <= high):
                return verts
        low, high = lo - self.NEAR_MARGIN, hi + self.NEAR_MARGIN
        verts = mesh.candidate_vertices(low, high)
        self._near_cache[body] = (low, high, verts)
        return verts

    def _push_out(self, position: Vec,
                  iterations: int = 3) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Depenetrate the capsule from static colliders; return (pos, ground_n).

        Contacts are resolved by sequential projection (deepest first): each
        contact only contributes the penetration *not already* resolved by the
        corrections applied so far.  This resolves distinct directions together
        (floor-up + wall-sideways, so the capsule can't sink or slip through)
        without the over-push that summing many near-parallel contacts on a curved
        surface would cause — which otherwise springs the capsule back out and
        makes it bounce off columns instead of sliding/stopping.
        """
        pos = np.asarray(position, dtype='d').copy()
        ground_n: Optional[np.ndarray] = None
        for _ in range(iterations):
            contacts: List[Tuple[np.ndarray, float]] = []
            for j in self._static_bodies():
                Pj = self._static_proxy(j)
                if isinstance(Pj, TriangleMeshProxy):
                    # The level, which is nearly always every contact there is.
                    # Straight to the two numbers this loop reads, so a frame
                    # does not build twenty thousand contact objects to look at
                    # a normal and a depth and drop them.
                    capsule = cast(CapsuleProxy, self._proxy(pos))
                    pushes, depths = collide.capsule_mesh_pushes(
                        capsule, Pj, self._near_triangles(j, Pj, capsule))
                    contacts.extend(zip(pushes, depths, strict=True))
                    continue
                for c in collide.collide(0, 1, self._proxy(pos), Pj):
                    contacts.append((-c.normal, c.depth))   # from world into char
            # Ground is a surface this capsule could stand on, which is what
            # ``maxSlope`` says.  A fixed threshold here would disagree with it
            # -- and a face counted as ground but too steep to walk is one the
            # player can climb, since every part of the controller that seats
            # or steps trusts this.
            for push, _depth in contacts:
                if self._walkable(push):
                    ground_n = push
            if not contacts:
                break
            contacts.sort(key=lambda pc: -pc[1])            # deepest first
            correction = np.zeros(3)
            for push, depth in contacts:
                remaining = depth + 1e-4 - float(np.dot(correction, push))
                if remaining > 0:
                    correction = correction + push * remaining
            pos = pos + correction
        return pos, ground_n

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
        """
        for piece in self._substeps(dt):
            was_grounded = self.grounded
            self._step(piece)
            # After the step, so "left the ground" and "landed" are settled
            # facts rather than predictions, and so every early return above is
            # covered.
            if not self.flying:
                self._tick_jump_windows(piece, was_grounded)

    def _reach(self) -> Tuple[float, float]:
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

    def _substeps(self, dt: float) -> List[float]:
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
        horizontal = float(np.linalg.norm(
            self.move_dir * self.speed() + self.push)) * dt
        across, along = self._reach()
        if across <= 0 or along <= 0:
            return [dt]
        count = max(math.ceil(horizontal / across),
                    math.ceil(vertical / along), 1)
        count = min(count, self.max_substeps(dt))
        return [dt / count] * count

    def _step(self, dt: float) -> None:
        """One movement step, before the jump windows are aged."""
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
        horiz = self.move_dir * self.speed() * control + self.push
        horiz = self._settle_step_debt(horiz, dt)
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
        moved = np.linalg.norm((resolved - self.position)[[0, 2]])
        want = np.linalg.norm(horiz[[0, 2]]) * dt
        if want > 1e-6 and moved < 0.7 * want:
            stepped = self._try_step_up(horiz, dt)
            if stepped is not None:
                resolved = stepped
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
                self.vy = 0.0
                return

        # Over a bottomless void (no geometry below the footprint), hover instead
        # of dropping.  Gated on a non-rising velocity so a jump *across* a void
        # still arcs up normally and only floats once it stops rising.
        if self.caps.suppressOverVoid and self.vy <= 0.0 and self._over_void():
            self.vy = 0.0
            self.grounded = False
            return

        self.vy = max(self.vy - self.gravity_mag * dt, -self.terminal_velocity())
        vtarget = self.position + UP * (self.vy * dt)
        vresolved, ground_n = self._push_out(vtarget)
        self.position = vresolved
        self._update_grounded(ground_n)
        if self.grounded:
            self._apply_slope_slide(ground_n, dt)

    def _snap_down(self, max_drop: float,
                   step: float = 0.04) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
        aabbs = [self._static_proxy(j).aabb() for j in self._static_bodies()]
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
        self._swim_velocity += ((desired - self._swim_velocity)
                                * min(1.0, self.caps.swimDrag * dt))
        self.vy += (self.buoyancy - 1.0) * self.gravity_mag * dt
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
            self.vy = 0.0
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
        due = float(np.linalg.norm(horiz[[0, 2]])) * dt
        if due <= 1e-9:
            return horiz
        paid = min(self._step_debt, due)
        self._step_debt -= paid
        return horiz * (1.0 - paid / due)

    def _try_step_up(self, horiz: np.ndarray, dt: float) -> Optional[np.ndarray]:
        """Try to climb a small ledge ahead; return the seated position or None if not a step."""
        speed = np.linalg.norm(horiz[[0, 2]])
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
        if np.linalg.norm((ahead - self.position)[[0, 2]]) < 0.5 * forward:
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
        advanced = float(np.linalg.norm((dropped - self.position)[[0, 2]]))
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
        speed = float(np.linalg.norm(velocity))
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
        length = float(np.linalg.norm(along))
        if length <= 1e-9:
            return velocity
        return along * (speed / length)

    def _on_walkable_ground(self) -> bool:
        """Whether the capsule is standing on something it could walk along."""
        return bool(self.grounded and self.ground_normal is not None
                    and self._walkable(self.ground_normal))

    def _walkable(self, normal: np.ndarray) -> bool:
        """Whether a surface with this normal is shallow enough to stand on."""
        slope = np.degrees(np.arccos(np.clip(float(np.dot(normal, UP)), -1, 1)))
        return bool(slope <= self.caps.maxSlope)

    #: How far below itself the capsule looks for floor when nothing touched it
    #: this step.  Enough to bridge the gap a step opens under a walker, small
    #: enough not to hunt for ground that is not there.
    GROUND_PROBE = 0.05

    def _update_grounded(self, ground_n: Optional[np.ndarray]) -> None:
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
            self.vy = max(self.vy, 0.0)
            return
        probe, gn = self._push_out(self.position - UP * self.GROUND_PROBE)
        if gn is not None:
            self.position = probe
            self.grounded = True
            self.ground_normal = gn
            self.vy = 0.0
        else:
            self.grounded = False
            self.ground_normal = None

    def _apply_slope_slide(self, ground_n: Optional[np.ndarray], dt: float) -> None:
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
        self.position = np.asarray(position, dtype='d').copy()
        self.stuck = False
        self.flying = False                         # a fresh bind starts on foot
        self.push = np.zeros(3)                     # ...and at rest
        self.vy = 0.0
        resolved, ground_n = self._push_out(self.position, iterations=12)
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
                self.vy = 0.0
                return
            drop += step
