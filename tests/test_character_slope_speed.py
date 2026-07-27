"""Speed is a speed along the ground, not a speed along the horizon.

Walking into a slope, a capsule moved horizontally penetrates it and is pushed
back out along the surface normal, and the normal has a horizontal component
opposing the motion.  The distance actually covered therefore falls off with the
slope -- roughly with its cosine squared, and worse once the push-out iterates --
so a ramp that is perfectly walkable makes the player feel like they are wading.

The fix is to spend the speed *along the surface*: the move direction is
projected onto the ground plane before it is used, so a run up a ramp covers the
same metres per second as a run along the flat, and the climb is the vertical
part of that rather than something taken out of it.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld
from omi_physics.character import CharacterController, CharacterCapabilities

DT = 1.0 / 60.0
RUN = CharacterCapabilities().runSpeed


def tilted_world(degrees):
    """One large slab tilted about z, so +x is uphill and -x is downhill."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    shape = world.add_shape(model.Shape.box((400, 2, 60)))
    half = np.radians(degrees) / 2.0
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape), position=(0, -1.0, 0),
                   orientation=(0, 0, float(np.sin(half)), float(np.cos(half))))
    return world


def surface_speed(degrees, direction=(1, 0, 0), mode='run', caps=None):
    """Metres per second covered along the surface, after a second of settling."""
    world = tilted_world(degrees)
    ch = CharacterController(world, caps or CharacterCapabilities(), gravity=9.81)
    ch.safe_bind((0, 4.0, 0))
    ch.set_move(direction, mode=mode)
    for _ in range(60):
        ch.update(DT)
    start = ch.position.copy()
    for _ in range(60):
        ch.update(DT)
    return float(np.linalg.norm(ch.position - start))


class TestRunningUpASlope:
    @pytest.mark.parametrize('degrees', [0, 5, 10, 15, 20, 25, 30])
    def test_the_pace_is_the_same_at_every_walkable_slope(self, degrees):
        assert surface_speed(degrees) == pytest.approx(RUN, rel=0.06)

    def test_downhill_is_the_same_pace_as_uphill(self):
        assert surface_speed(20, direction=(-1, 0, 0)) == pytest.approx(
            surface_speed(20), rel=0.06)

    def test_across_the_slope_is_the_same_pace(self):
        """Traversing is neither climbing nor descending and must not be slower."""
        assert surface_speed(20, direction=(0, 0, 1)) == pytest.approx(
            RUN, rel=0.06)

    def test_a_slope_still_costs_the_climb_in_height(self):
        """The pace is along the surface, so the climb is part of it, not free."""
        world = tilted_world(20)
        ch = CharacterController(world, CharacterCapabilities(), gravity=9.81)
        ch.safe_bind((0, 4.0, 0))
        ch.set_move((1, 0, 0), mode='run')
        for _ in range(60):
            ch.update(DT)
        start = ch.position.copy()
        for _ in range(60):
            ch.update(DT)
        climbed = ch.position[1] - start[1]
        assert climbed == pytest.approx(RUN * np.sin(np.radians(20)), rel=0.1)

    def test_walking_is_still_slower_than_running_on_a_slope(self):
        assert surface_speed(20, mode='walk') < surface_speed(20, mode='run')


class TestNothingElseChanged:
    def test_flat_ground_is_untouched(self):
        assert surface_speed(0) == pytest.approx(RUN, rel=0.01)

    def test_a_slope_past_the_limit_is_still_not_climbable(self):
        """Projecting the move must not turn a wall into a ramp."""
        caps = CharacterCapabilities(maxSlope=40.0)
        world = tilted_world(60)
        ch = CharacterController(world, caps, gravity=9.81)
        ch.safe_bind((0, 6.0, 0))
        ch.set_move((1, 0, 0), mode='run')
        start = ch.position[1]
        for _ in range(120):
            ch.update(DT)
        assert ch.position[1] <= start + 0.5

    def test_an_airborne_capsule_is_not_projected_onto_a_stale_surface(self):
        """Air control is horizontal; the last ground seen is not underfoot."""
        world = tilted_world(25)
        ch = CharacterController(world, CharacterCapabilities(), gravity=9.81)
        ch.safe_bind((0, 4.0, 0))
        ch.set_move((1, 0, 0), mode='run')
        for _ in range(30):
            ch.update(DT)
        ch.apply_impulse(np.array([0.0, 8.0, 0.0]))
        rising = [ch.position[1]]
        for _ in range(10):
            ch.update(DT)
            rising.append(ch.position[1])
        assert rising[-1] > rising[0]


class TestRunningDownSteps:
    """Walking off a ledge no taller than ``stepHeight`` must not be a fall.

    At speed the capsule clears the edge in one frame and is then still
    overlapping the step it just left, so a seat found by pushing out of the
    world resolves against *that* wall rather than the floor below and reports
    no ground at all.  The capsule then falls the height of a stair it was
    supposed to walk down -- for a sixth of a second, every step, which is long
    enough to swallow a jump and short enough that nobody sees why.
    """

    def stairs(self, rise, tile=2.0, count=80):
        """A staircase descending in +x, one ``rise`` per tile.

        Descending monotonically, not a sawtooth: a repeating up-down pattern
        makes every second edge a drop of several ``rise``, which is a cliff
        and is *supposed* to be a fall.
        """
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        for i in range(count):
            shape = world.add_shape(model.Shape.box((tile, 20.0, 20)))
            world.add_body(model.Motion(type=model.STATIC),
                           collider=model.Collider(shape=shape),
                           position=(i * tile, -10.0 - rise * i, 0))
        return world

    def airborne_fraction(self, rise, speed, caps=None):
        caps = caps or CharacterCapabilities(runSpeed=speed)
        ch = CharacterController(self.stairs(rise), caps, gravity=9.81)
        ch.safe_bind((2.0, 3.0, 0))
        ch.set_move((1, 0, 0), mode='run')
        for _ in range(40):
            ch.update(DT)
        air = 0
        for _ in range(600):
            ch.update(DT)
            if not ch.grounded:
                air += 1
        return air / 600.0

    @pytest.mark.parametrize('speed', [3.0, 6.0, 12.0])
    def test_footing_is_kept_over_steps_within_the_step_height(self, speed):
        caps = CharacterCapabilities(runSpeed=speed, stepHeight=0.45)
        assert self.airborne_fraction(0.40, speed, caps) < 0.02

    def test_a_drop_taller_than_the_step_height_is_still_a_fall(self):
        """Stepping down is for stairs, not for cliffs."""
        caps = CharacterCapabilities(runSpeed=6.0, stepHeight=0.2)
        assert self.airborne_fraction(0.8, 6.0, caps) > 0.1

    def test_it_holds_at_the_speed_an_arena_runs_at(self):
        """12 m/s over 16-unit stairs is the case that started this."""
        caps = CharacterCapabilities(runSpeed=12.19, stepHeight=0.457,
                                     standHeight=1.42, radius=0.406,
                                     eyeHeight=1.4, crouchHeight=0.71)
        assert self.airborne_fraction(0.406, 12.19, caps) < 0.02


class TestRunningUpSteps:
    """A staircase is climbed at running pace, not at frame-rate pace.

    Mounting a step has to move the capsule's *centre* past the edge -- about a
    radius, in one motion, because a capsule stopped against the riser is a
    radius behind it and a smaller probe never reaches over.  That single motion
    is further than a frame's running travel, so a flight of stairs climbed one
    step per frame goes faster than the same distance on the flat, and faster
    still the better the frame rate.  The distance is owed rather than given:
    what a step advanced beyond the frame's due is taken back out of the frames
    that follow.
    """

    def rising_stairs(self, rise, tile=2.0, count=60):
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        for i in range(count):
            shape = world.add_shape(model.Shape.box((tile, 40.0, 20)))
            world.add_body(model.Motion(type=model.STATIC),
                           collider=model.Collider(shape=shape),
                           position=(i * tile, -20.0 + rise * i, 0))
        return world

    def pace(self, rise, fps, seconds=1.5, caps=None):
        caps = caps or CharacterCapabilities()
        dt = 1.0 / fps
        ch = CharacterController(self.rising_stairs(rise), caps, gravity=9.81)
        ch.safe_bind((2.0, 3.0, 0))
        ch.set_move((1, 0, 0), mode='run')
        for _ in range(int(0.5 / dt)):
            ch.update(dt)
        start = ch.position.copy()
        for _ in range(int(seconds / dt)):
            ch.update(dt)
        return float(np.linalg.norm(ch.position - start)) / seconds

    @pytest.mark.parametrize('fps', [60, 144, 300])
    def test_stairs_are_no_faster_than_the_flat(self, fps):
        assert self.pace(0.30, fps) <= self.pace(0.0, fps) * 1.05

    @pytest.mark.parametrize('fps', [60, 144, 300])
    def test_and_not_so_slow_that_a_stair_becomes_a_wall(self, fps):
        assert self.pace(0.30, fps) >= self.pace(0.0, fps) * 0.7

    def test_the_pace_no_longer_depends_on_the_frame_rate(self):
        slow, fast = self.pace(0.30, 60), self.pace(0.30, 300)
        assert fast == pytest.approx(slow, rel=0.1)

    def test_the_flat_is_untouched(self):
        assert self.pace(0.0, 144) == pytest.approx(RUN, rel=0.02)
