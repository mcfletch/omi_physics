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
from typing import Dict, Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np

from dataclasses import dataclass

from . import model
from .body import make_proxy, CapsuleProxy, Proxy
from . import collide
from .mathutil import Vec

if TYPE_CHECKING:
    from .world import PhysicsWorld

UP = np.array([0.0, 1.0, 0.0])
IDENT = np.array([0.0, 0.0, 0.0, 1.0])


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
    jumpHeight: float = 1.2
    stepHeight: float = 0.35
    maxSlope: float = 50.0            # degrees
    eyeHeight: float = 1.6
    standHeight: float = 1.8
    crouchHeight: float = 1.0
    radius: float = 0.3
    airControl: float = 0.4
    suppressOverVoid: bool = False    # hover instead of dropping into a bottomless gap


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
        self.crouching = False
        self.flying = False
        self.mode = 'walk'
        self.move_dir = np.zeros(3)                         # world horizontal unit
        self.fly_dir = np.zeros(3)
        self.stuck = False
        self._proxy_cache: Dict[int, Proxy] = {}            # static body -> proxy

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
        """Launch upward if grounded and able; return whether the jump fired."""
        if self.caps.canJump and self.grounded and not self.crouching:
            self.vy = np.sqrt(2.0 * self.gravity_mag * self.caps.jumpHeight)
            self.grounded = False
            return True
        return False

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
        return True

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
        trimesh's transformed vertices are computed once, not per query)."""
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
            contacts = []
            for j in self._static_bodies():
                Pj = self._static_proxy(j)
                for c in collide.collide(0, 1, self._proxy(pos), Pj):
                    push = -c.normal                        # from world into char
                    contacts.append((push, c.depth))
                    if push[1] > 0.5:
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
    def update(self, dt: float) -> None:
        """Advance the avatar one step: move-and-slide, gravity, ground/slope handling."""
        if self.flying:
            self._update_fly(dt)
            return
        was_grounded = self.grounded
        control = 1.0 if self.grounded else self.caps.airControl
        horiz = self.move_dir * self.speed() * control

        target = self.position + horiz * dt
        resolved, _ = self._push_out(target)
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
                self.position = seated
                self.grounded = True
                self.vy = 0.0
                return

        # Over a bottomless void (no geometry below the footprint), hover instead
        # of dropping.  Gated on a non-rising velocity so a jump *across* a void
        # still arcs up normally and only floats once it stops rising.
        if self.caps.suppressOverVoid and self.vy <= 0.0 and self._over_void():
            self.vy = 0.0
            self.grounded = False
            return

        self.vy -= self.gravity_mag * dt
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

    def _try_step_up(self, horiz: np.ndarray, dt: float) -> Optional[np.ndarray]:
        """Try to climb a small ledge ahead; return the seated position or None if not a step."""
        speed = np.linalg.norm(horiz[[0, 2]])
        if speed < 1e-6:
            return None
        hdir = horiz / speed
        lifted = self.position + UP * self.caps.stepHeight
        if self._overlaps(self._proxy(lifted)):
            return None                                     # too tall / ceiling above
        # Move forward far enough to clear the step edge and land *on* the step in
        # one motion (a small per-frame velocity alone catches the edge and never
        # gets over it, which the old code hid with an airborne hop -> the jitter).
        forward = max(speed * dt, self.caps.radius + 0.05)
        ahead, _ = self._push_out(lifted + hdir * forward)
        if np.linalg.norm((ahead - self.position)[[0, 2]]) < 0.5 * forward:
            return None                                     # blocked -> not a step
        dropped, ground_n = self._push_out(ahead - UP * self.caps.stepHeight)
        # Only accept a step onto solid ground that is actually higher than here
        # (never leave the avatar hanging, and never "step" down onto the floor).
        if ground_n is None or dropped[1] <= self.position[1] + 1e-3:
            return None
        return dropped

    def _update_grounded(self, ground_n: Optional[np.ndarray]) -> None:
        """Set the grounded flag from the vertical contact, probing just below if needed."""
        if ground_n is not None:
            self.grounded = True
            self.vy = max(self.vy, 0.0)
            return
        probe, gn = self._push_out(self.position - UP * 0.05)
        if gn is not None:
            self.position = probe
            self.grounded = True
            self.vy = 0.0
        else:
            self.grounded = False

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
        resolved, ground_n = self._push_out(self.position, iterations=12)
        self.position = resolved
        if self._overlaps(self._proxy()):
            self.stuck = True
            self.set_fly(True)
            return False
        self._ground_snap(search)
        return True

    def _ground_snap(self, search: float) -> None:
        """Drop the capsule up to ``search`` metres onto the first floor below and seat it."""
        step = 0.05
        drop = 0.0
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
