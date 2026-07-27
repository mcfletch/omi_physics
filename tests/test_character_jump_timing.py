"""Jump has to fire when the player meant it, not only when the capsule agrees.

A jump refused because ``grounded`` happened to be false on that one frame is
the single most-reported feel problem in a first-person controller, and it is
worst exactly where it is most noticed: running.  A capsule moving at speed over
a step, a ramp lip or a seam between two colliders leaves the ground for a frame
or two at a time, and every press landing in one of those frames is swallowed
with no feedback at all.

Two windows fix it, and both are measured in seconds rather than frames so they
hold at any frame rate:

* **Coyote time** -- a jump is still allowed for a moment after walking off
  something, provided the capsule left the ground by falling rather than by
  jumping.
* **Jump buffering** -- a jump asked for just before landing fires on landing
  rather than being dropped.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld
from omi_physics.character import CharacterController, CharacterCapabilities

DT = 1.0 / 60.0


def add_box(world, size, center):
    shape = world.add_shape(model.Shape.box(size))
    return world.add_body(model.Motion(type=model.STATIC),
                          collider=model.Collider(shape=shape), position=center)


def ledge_world():
    """A floor that stops at x = 0, so walking on gets you into open air."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    add_box(world, (20, 1, 20), (-10, -0.5, 0))
    return world


def floor_world():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    add_box(world, (60, 1, 60), (0, -0.5, 0))
    return world


def standing_character(world, caps=None, at=(0, 0, 0)):
    ch = CharacterController(world, caps or CharacterCapabilities(), gravity=9.81)
    ch.safe_bind((at[0], 3.0, at[2]))
    return ch


def run_until_airborne(ch, limit=240):
    """Run forward off the ledge, stopping on the first airborne frame."""
    ch.set_move((1, 0, 0), mode='run')
    for _ in range(limit):
        ch.update(DT)
        if not ch.grounded:
            return True
    return False


def fall_to(ch, height, limit=600):
    """Fall until the capsule's base is within ``height`` of the floor."""
    for _ in range(limit):
        ch.update(DT)
        if ch.grounded:
            return False
        if ch.vy < 0 and ch.base()[1] <= height:
            return True
    return False


def launches_without_being_asked(ch, seconds=1.5):
    """Whether the capsule *starts* rising with no further call to jump().

    A gain in upward speed, not merely having some: free flight only ever loses
    it, so a gain is a launch -- and requiring it to be positive rules out the
    rise from a negative velocity to zero, which is landing.
    """
    for _ in range(int(seconds / DT)):
        before = ch.vy
        ch.update(DT)
        if ch.vy > 0.0 and ch.vy > before:
            return True
    return False


class TestCoyoteTime:
    def test_a_jump_just_after_walking_off_still_fires(self):
        """The player was running on the ground a frame ago and pressed jump."""
        ch = standing_character(ledge_world(), at=(-2, 0, 0))
        assert run_until_airborne(ch)
        assert ch.jump()

    def test_it_launches_the_capsule_rather_than_only_reporting_true(self):
        ch = standing_character(ledge_world(), at=(-2, 0, 0))
        assert run_until_airborne(ch)
        falling = ch.vy
        ch.jump()
        assert ch.vy > falling

    def test_the_window_closes(self):
        ch = standing_character(ledge_world(), at=(-2, 0, 0))
        assert run_until_airborne(ch)
        for _ in range(int(0.5 / DT)):
            ch.update(DT)
        assert not ch.jump()

    def test_a_second_jump_in_the_air_is_still_refused(self):
        """Coyote time forgives falling, not jumping: no free double jump."""
        ch = standing_character(floor_world())
        assert ch.jump()
        ch.update(DT)
        assert not ch.jump()

    def test_it_can_be_switched_off(self):
        caps = CharacterCapabilities(coyoteTime=0.0)
        ch = standing_character(ledge_world(), caps=caps, at=(-2, 0, 0))
        assert run_until_airborne(ch)
        assert not ch.jump()


class TestJumpBuffering:
    def test_a_jump_asked_for_just_before_landing_fires_on_landing(self):
        ch = standing_character(floor_world())
        assert ch.jump()
        assert fall_to(ch, 0.15)                # nearly down
        assert not ch.jump()                    # refused now...
        assert launches_without_being_asked(ch)  # ...and taken on landing

    def test_a_jump_asked_for_long_before_landing_is_forgotten(self):
        ch = standing_character(floor_world())
        assert ch.jump()
        assert fall_to(ch, 1.0)                 # still a long way up
        assert not ch.jump()
        assert not launches_without_being_asked(ch)
        assert ch.grounded

    def test_the_buffer_is_spent_once(self):
        ch = standing_character(floor_world())
        assert ch.jump()
        assert fall_to(ch, 0.15)
        assert not ch.jump()
        assert launches_without_being_asked(ch)  # the buffered jump
        assert not launches_without_being_asked(ch, seconds=2.0)
        assert ch.grounded

    def test_it_can_be_switched_off(self):
        ch = standing_character(floor_world(),
                                caps=CharacterCapabilities(jumpBuffer=0.0))
        assert ch.jump()
        assert fall_to(ch, 0.15)
        assert not ch.jump()
        assert not launches_without_being_asked(ch)
        assert ch.grounded

    def test_a_crouching_capsule_does_not_buffer_a_jump_it_cannot_take(self):
        """A refusal on grounds other than timing is a refusal, not a delay."""
        ch = standing_character(floor_world())
        ch.set_crouch(True)
        assert not ch.jump()
        assert not launches_without_being_asked(ch, seconds=0.5)


class TestNothingElseChanged:
    def test_a_grounded_jump_still_fires_at_once(self):
        ch = standing_character(floor_world())
        assert ch.jump()
        assert ch.vy > 0.0
        assert not ch.grounded

    def test_a_capsule_that_cannot_jump_still_cannot(self):
        caps = CharacterCapabilities(canJump=False)
        ch = standing_character(floor_world(), caps=caps)
        assert not ch.jump()
        for _ in range(int(0.5 / DT)):
            ch.update(DT)
        assert ch.vy == pytest.approx(0.0, abs=1e-6)

    def test_a_crouching_capsule_still_cannot_jump(self):
        ch = standing_character(floor_world())
        ch.set_crouch(True)
        assert not ch.jump()
        assert ch.vy == pytest.approx(0.0, abs=1e-6)

    def test_an_impulse_still_ungrounds_and_launches(self):
        ch = standing_character(floor_world())
        ch.apply_impulse(np.array([0.0, 5.0, 0.0]))
        assert not ch.grounded
        assert ch.vy == pytest.approx(5.0)


class TestAJumpSurvivesTheFrameRate:
    """A launch must leave the ground however short the frame was.

    The ground probe looks a few centimetres below the capsule and re-seats it
    when it finds floor, which is what keeps a walker attached to the ground.
    One frame after a jump the capsule has risen only ``vy * dt``, and at a high
    frame rate that is *less* than the probe reaches -- so the jump is snapped
    back onto the floor and its velocity zeroed, in the same frame it started.
    The faster the machine, the more jumps vanish, and since the frame time
    varies it takes some presses and not others.
    """

    @pytest.mark.parametrize('fps', [30, 60, 144, 240, 500, 1000])
    def test_the_capsule_leaves_the_ground_at_any_frame_rate(self, fps):
        dt = 1.0 / fps
        ch = standing_character(floor_world())
        base = ch.base()[1]
        assert ch.jump()
        for _ in range(int(0.15 * fps)):
            ch.update(dt)
        assert ch.base()[1] > base + 0.1

    @pytest.mark.parametrize('fps', [144, 500])
    def test_the_launch_velocity_is_not_zeroed_on_the_first_frame(self, fps):
        ch = standing_character(floor_world())
        ch.jump()
        launched = ch.vy
        ch.update(1.0 / fps)
        assert ch.vy > launched * 0.5

    @pytest.mark.parametrize('fps', [144, 500])
    def test_a_rising_capsule_is_not_reported_as_grounded(self, fps):
        """Grounded while rising re-arms every path that re-seats the capsule."""
        ch = standing_character(floor_world())
        ch.jump()
        ch.update(1.0 / fps)
        assert not ch.grounded

    def test_a_walker_is_still_held_against_the_floor(self):
        """The probe earns its keep: level movement must not drift off."""
        ch = standing_character(floor_world())
        ch.set_move((1, 0, 0), mode='run')
        for _ in range(120):
            ch.update(1.0 / 240)
        assert ch.grounded
