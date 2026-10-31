"""Launching the character with an impulse (jump pads, explosions, springs)."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld

DT = 1.0 / 60.0
GRAVITY = 9.81


def floor_world():
    world = PhysicsWorld(gravity=model.Gravity(gravity=GRAVITY, direction=(0, -1, 0)))
    shape = world.add_shape(model.Shape.box((120, 1, 120)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape), position=(0, -0.5, 0))
    return world


def standing_character(caps=None):
    world = floor_world()
    character = CharacterController(world, caps or CharacterCapabilities(),
                                    gravity=GRAVITY)
    character.safe_bind((0.0, 3.0, 0.0))
    assert character.grounded
    return character


def fly_for(character, seconds):
    """Step ``seconds`` of simulation, returning the highest base height seen."""
    peak = character.base()[1]
    for _ in range(round(seconds / DT)):
        character.update(DT)
        peak = max(peak, character.base()[1])
    return peak


class TestLaunching:
    def test_an_upward_impulse_lifts_the_capsule_off_the_ground(self):
        character = standing_character()
        character.apply_impulse((0.0, 8.0, 0.0))
        character.update(DT)
        assert not character.grounded
        assert character.base()[1] > 0.05

    def test_the_capsule_reaches_the_apex_the_impulse_asks_for(self):
        # v = sqrt(2*g*h) is the launch speed for an apex of h, which is how a
        # jump pad is aimed: pick the height, solve for the speed.
        height = 4.0
        character = standing_character()
        character.apply_impulse((0.0, np.sqrt(2 * GRAVITY * height), 0.0))
        assert fly_for(character, 1.5) == pytest.approx(height, rel=0.1)

    def test_a_horizontal_impulse_carries_the_capsule_with_no_input(self):
        character = standing_character()
        character.apply_impulse((6.0, 6.0, 0.0))
        for _ in range(30):
            character.update(DT)
        # Nothing is pressed, so only the impulse can be moving it.
        assert np.linalg.norm(character.move_dir) == 0.0
        assert character.position[0] > 2.0

    def test_air_control_does_not_throttle_the_impulse(self):
        # airControl scales *walking* while airborne; scaling the launch too
        # would silently shorten every jump pad by that factor.
        caps = CharacterCapabilities(airControl=0.1)
        character = standing_character(caps)
        character.apply_impulse((6.0, 6.0, 0.0))
        for _ in range(30):
            character.update(DT)
        assert character.position[0] > 2.0

    def test_an_impulse_replaces_the_one_before_it(self):
        character = standing_character()
        character.apply_impulse((6.0, 6.0, 0.0))
        character.update(DT)
        character.apply_impulse((-6.0, 6.0, 0.0))
        start = character.position[0]
        for _ in range(30):
            character.update(DT)
        assert character.position[0] < start

    def test_a_launch_is_not_swallowed_by_the_step_down_snap(self):
        # Standing on the pad the capsule is grounded, and the grounded branch
        # snaps back to the surface unless the launch ungrounds it first.
        character = standing_character()
        character.apply_impulse((0.0, 8.0, 0.0))
        assert character.grounded is False
        assert character.vy == pytest.approx(8.0)


class TestSettling:
    def test_the_capsule_lands_and_comes_to_rest(self):
        character = standing_character()
        character.apply_impulse((6.0, 6.0, 0.0))
        for _ in range(300):
            character.update(DT)
        assert character.grounded
        settled = character.position[0]
        for _ in range(60):
            character.update(DT)
        assert character.position[0] == pytest.approx(settled, abs=0.05)

    def test_the_carried_speed_survives_the_whole_flight(self):
        # Friction is a *ground* effect; bleeding it off in the air would turn
        # every long pad jump into a short one.
        character = standing_character()
        character.apply_impulse((6.0, 6.0, 0.0))
        character.update(DT)
        first = character.push.copy()
        for _ in range(30):
            character.update(DT)
        assert not character.grounded
        assert character.push == pytest.approx(first)

    def test_friction_is_what_stops_it(self):
        slippery = standing_character(CharacterCapabilities(pushFriction=0.0))
        sticky = standing_character(CharacterCapabilities(pushFriction=12.0))
        for character in (slippery, sticky):
            character.apply_impulse((6.0, 6.0, 0.0))
            for _ in range(300):
                character.update(DT)
        assert slippery.position[0] > sticky.position[0]


class TestModeChanges:
    def test_flying_ignores_a_pending_impulse(self):
        character = standing_character()
        character.apply_impulse((6.0, 6.0, 0.0))
        character.set_fly(True)
        start = character.position.copy()
        for _ in range(30):
            character.update(DT)
        assert character.position == pytest.approx(start)

    def test_a_fresh_bind_forgets_an_impulse(self):
        character = standing_character()
        character.apply_impulse((6.0, 6.0, 0.0))
        character.safe_bind((0.0, 3.0, 0.0))
        assert character.push == pytest.approx(np.zeros(3))
        assert character.vy == 0.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


NOT_FINITE = [float('nan'), float('inf'), -float('inf')]


class TestWhatIsRefused:
    """A value no step can use is refused where it is given, and the character is untouched."""

    @pytest.mark.parametrize('mass', [0.0, -80.0, *NOT_FINITE])
    def test_a_knockback_needs_a_positive_finite_mass(self, mass):
        character = standing_character(CharacterCapabilities(mass=mass))
        with pytest.raises(ValueError, match='mass'):
            character.knockback((10.0, 0.0, 0.0))
        assert character.push == pytest.approx([0.0, 0.0, 0.0])

    @pytest.mark.parametrize('bad', NOT_FINITE)
    def test_a_knockback_needs_a_finite_impulse(self, bad):
        character = standing_character()
        with pytest.raises(ValueError, match='impulse'):
            character.knockback((bad, 0.0, 0.0))
        assert character.grounded

    @pytest.mark.parametrize('bad', NOT_FINITE)
    def test_a_launch_needs_a_finite_velocity(self, bad):
        character = standing_character()
        with pytest.raises(ValueError, match='velocity'):
            character.apply_impulse((0.0, bad, 0.0))
        assert character.grounded

    @pytest.mark.parametrize('bad', NOT_FINITE)
    def test_a_move_needs_a_finite_direction(self, bad):
        character = standing_character()
        with pytest.raises(ValueError, match='direction'):
            character.set_move((bad, 0.0, 1.0))
        with pytest.raises(ValueError, match='direction'):
            character.set_fly_move((bad, 0.0, 1.0))

    @pytest.mark.parametrize('speed', [-1.0, *NOT_FINITE])
    def test_a_move_needs_a_finite_speed_of_zero_or_more(self, speed):
        character = standing_character()
        with pytest.raises(ValueError, match='speed'):
            character.set_move((1.0, 0.0, 0.0), speed=speed)
        assert character.move_speed is None

    @pytest.mark.parametrize('dt', [-DT, *NOT_FINITE])
    def test_an_update_needs_a_finite_time_of_zero_or_more(self, dt):
        character = standing_character()
        before = character.position.copy()
        with pytest.raises(ValueError, match='dt'):
            character.update(dt)
        assert character.position == pytest.approx(before)

    def test_an_update_of_no_time_moves_nothing(self):
        character = standing_character()
        before = character.position.copy()
        character.update(0.0)
        assert character.position == pytest.approx(before)
