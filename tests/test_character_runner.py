"""A character that runs without stopping: a slide that steers and goes on, and a jump that follows the slope."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.character import (
    CharacterCapabilities,
    CharacterController,
    CharacterState,
)
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0
GRAVITY = 42.0


def floor_world() -> PhysicsWorld:
    world = PhysicsWorld(gravity=model.Gravity(gravity=GRAVITY, direction=(0, -1, 0)))
    shape = world.add_shape(model.Shape.box((400, 1, 400)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape),
                   position=(0, -0.5, 0))
    return world


def tilted_world(degrees: float) -> PhysicsWorld:
    """One slab tilted about z, so +x is uphill and -x is downhill."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=GRAVITY, direction=(0, -1, 0)))
    shape = world.add_shape(model.Shape.box((800, 2, 60)))
    half = np.radians(degrees) / 2.0
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape),
                   position=(0, -1.0, 0),
                   orientation=(0, 0, float(np.sin(half)), float(np.cos(half))))
    return world


def runner(world: PhysicsWorld | None = None, at: tuple[float, float, float] = (0.0, 2.0, 0.0),
           **caps: float) -> CharacterController:
    caps.setdefault('runSpeed', 10.0)
    caps.setdefault('crouchHeight', 0.6)
    caps.setdefault('standHeight', 1.7)
    caps.setdefault('radius', 0.15)
    caps.setdefault('jumpHeight', 1.5)
    caps.setdefault('airControl', 1.0)
    caps.setdefault('coyoteTime', 0.0)
    caps.setdefault('jumpBuffer', 0.0)
    character = CharacterController(world or floor_world(), CharacterCapabilities(**caps),
                                    gravity=GRAVITY)
    character.safe_bind(at)
    assert character.grounded
    return character


def step(character: CharacterController, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        character.update(DT)


class TestSteeredSlide:
    def test_it_goes_where_the_input_says_at_the_speed_of_the_mode(self) -> None:
        character = runner()
        character.set_move((1, 0, 0), 'run')
        step(character, 0.5)
        assert character.slide(0.8, 0.3, steer=True)
        character.set_move((0, 0, 1), 'run')
        step(character, 0.3)
        assert character.state is CharacterState.SLIDING
        assert character.velocity[2] == pytest.approx(10.0, abs=0.05)
        assert character.velocity[0] == pytest.approx(0.0, abs=0.05)

    def test_the_speed_is_the_modes_each_moment_and_not_the_crouch_speed(self) -> None:
        character = runner(crouchSpeed=1.0)
        character.set_move((1, 0, 0), 'run')
        step(character, 0.3)
        character.slide(0.8, 0.3, steer=True)
        character.mode = 'run'
        character.caps.runSpeed = 16.0
        step(character, 0.2)
        assert character.velocity[0] == pytest.approx(16.0, abs=0.1)

    def test_it_is_as_low_as_any_slide_and_rises_over_the_same_last_moments(self) -> None:
        character = runner()
        character.set_move((1, 0, 0), 'run')
        character.slide(0.8, 0.3, steer=True)
        step(character, 0.3)
        assert character.height == pytest.approx(0.6, abs=1e-9)
        step(character, 0.6)
        assert character.height == pytest.approx(1.7, abs=1e-9)
        assert character.state is CharacterState.STANDING

    def test_a_slide_that_is_not_steered_still_ignores_the_input(self) -> None:
        character = runner()
        character.set_move((1, 0, 0), 'run')
        step(character, 0.5)
        character.slide(0.8, 0.3)
        character.set_move((-1, 0, 0), 'run')
        before = float(character.position[0])
        step(character, 0.3)
        assert float(character.position[0]) > before


class TestExtendingASlide:
    def test_it_gives_the_slide_that_long_again_from_now_and_the_body_goes_low_again(self) -> None:
        character = runner()
        character.slide(0.8, 0.3)
        step(character, 0.65)
        assert character.height > 0.6 + 0.1
        assert character.extend_slide(0.8)
        step(character, 0.01)
        assert character.height == pytest.approx(0.6, abs=1e-9)
        step(character, 0.45)
        assert character.height == pytest.approx(0.6, abs=1e-9)
        assert character.state is CharacterState.SLIDING
        step(character, 0.4)
        assert character.height == pytest.approx(1.7, abs=1e-9)
        assert character.state is CharacterState.STANDING

    def test_the_progress_stays_in_the_low_part_of_the_slide_for_as_long_as_it_is_gone_on_with(
            self) -> None:
        character = runner()
        character.slide(0.8, 0.3)
        for _ in range(8):
            step(character, 0.2)
            character.extend_slide(0.8)
            assert character.slide_progress <= 0.5 / 0.8 + 1e-9
        step(character, 0.5)
        assert character.slide_progress == pytest.approx(0.5 / 0.8, abs=0.01)

    def test_a_slide_let_go_goes_through_the_rise_over_its_end(self) -> None:
        character = runner()
        character.slide(0.8, 0.3)
        step(character, 0.4)
        character.extend_slide(0.8)
        step(character, 0.6)
        assert character.slide_progress == pytest.approx((0.8 - 0.2) / 0.8, abs=0.01)
        step(character, 0.19)
        assert character.slide_progress == pytest.approx(1.0, abs=0.02)

    def test_a_slide_that_was_not_extended_is_as_far_through_as_its_time(self) -> None:
        character = runner()
        character.slide(0.8, 0.3)
        step(character, 0.2)
        assert character.slide_progress == pytest.approx(0.25, abs=0.01)
        step(character, 0.4)
        assert character.slide_progress == pytest.approx(0.75, abs=0.01)

    def test_a_character_that_is_not_sliding_has_nothing_to_extend(self) -> None:
        character = runner()
        assert not character.extend_slide(0.8)
        step(character, 0.1)
        assert character.state is CharacterState.STANDING


class TestEndingASlide:
    def test_it_stands_the_character_up_and_a_jump_is_then_heard(self) -> None:
        character = runner()
        character.slide(0.8, 0.3)
        step(character, 0.2)
        assert not character.jump()
        assert character.end_slide()
        assert character.state is CharacterState.STANDING
        assert character.height == pytest.approx(1.7, abs=1e-9)
        assert character.jump()

    def test_a_character_that_is_not_sliding_is_left_alone(self) -> None:
        character = runner()
        assert not character.end_slide()
        assert character.state is CharacterState.STANDING

    def test_it_crouches_where_there_is_no_room_to_stand(self) -> None:
        character = runner()
        character.slide(0.8, 0.3)
        step(character, 0.2)
        roof = character.world.add_shape(model.Shape.box((400, 1, 400)))
        character.world.add_body(model.Motion(type=model.STATIC),
                                 collider=model.Collider(shape=roof), position=(0, 1.4 + 0.5, 0))
        assert character.end_slide()
        assert character.state is CharacterState.CROUCHING


def relative_jump(world: PhysicsWorld, degrees: float, follows: bool) -> tuple[float, float]:
    """The highest a jump rises above the slope, and how long it is in the air, running downhill."""
    slope = np.tan(np.radians(degrees))
    character = runner(world, at=(0.0, 3.0, 0.0), runSpeed=20.0, jumpFollowsSlope=follows)
    character.set_move((-1, 0, 0), 'run')
    step(character, 1.0)
    assert character.grounded
    assert character.jump()
    peak, steps = 0.0, 0
    while True:
        character.update(DT)
        steps += 1
        x, y = float(character.base()[0]), float(character.base()[1])
        peak = max(peak, y - slope * x)
        if character.grounded or steps > 400:
            return peak, steps * DT


class TestAJumpOnASlope:
    def test_off_a_downhill_it_rises_above_the_ground_by_the_jump_height_when_it_follows_the_slope(
            self) -> None:
        peak, air = relative_jump(tilted_world(10.0), 10.0, True)
        assert peak == pytest.approx(1.5, rel=0.03)
        assert air == pytest.approx(2.0 * np.sqrt(2.0 * 1.5 / GRAVITY), abs=0.03)

    def test_it_rises_higher_above_a_falling_slope_when_it_does_not(self) -> None:
        peak, air = relative_jump(tilted_world(10.0), 10.0, False)
        assert peak > 1.5 * 1.25
        assert air > 2.0 * np.sqrt(2.0 * 1.5 / GRAVITY) + 0.05

    def test_on_the_flat_the_two_are_the_same(self) -> None:
        followed = relative_jump(floor_world(), 0.0, True)
        plain = relative_jump(floor_world(), 0.0, False)
        assert followed == pytest.approx(plain, abs=1e-9)
        assert followed[0] == pytest.approx(1.5, rel=0.01)
