"""What a character is doing: its state, a slide, a knock-down and a landing."""
import itertools
import math

import numpy as np
import pytest

from omi_physics import model
from omi_physics.character import (
    CharacterCapabilities,
    CharacterController,
    CharacterState,
)
from omi_physics.world import PhysicsWorld

DT = 1.0 / 60.0
GRAVITY = 9.81


def floor_world() -> PhysicsWorld:
    world = PhysicsWorld(gravity=model.Gravity(gravity=GRAVITY, direction=(0, -1, 0)))
    shape = world.add_shape(model.Shape.box((200, 1, 200)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape),
                   position=(0, -0.5, 0))
    return world


def add_ceiling(character: CharacterController, height: float) -> None:
    """A roof whose underside is ``height`` above the floor, over the whole level."""
    world = character.world
    roof = world.add_shape(model.Shape.box((200, 1, 200)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=roof),
                   position=(0, height + 0.5, 0))


def runner(world: PhysicsWorld | None = None, **caps: float) -> CharacterController:
    character = CharacterController(world or floor_world(), CharacterCapabilities(**caps),
                                    gravity=GRAVITY)
    character.safe_bind((0.0, 2.0, 0.0))
    assert character.grounded
    return character


def step(character: CharacterController, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        character.update(DT)


def ground_speed(character: CharacterController) -> float:
    return math.hypot(float(character.velocity[0]), float(character.velocity[2]))


class TestState:
    def test_standing_on_the_ground(self) -> None:
        character = runner()
        step(character, 0.1)
        assert character.state is CharacterState.STANDING

    def test_crouching(self) -> None:
        character = runner()
        character.set_crouch(True)
        step(character, 0.5)
        assert character.state is CharacterState.CROUCHING

    def test_airborne_after_a_jump(self) -> None:
        character = runner()
        assert character.jump()
        step(character, 0.1)
        assert character.state is CharacterState.AIRBORNE

    def test_flying(self) -> None:
        character = runner()
        character.set_fly(True)
        assert character.state is CharacterState.FLYING

    def test_swimming(self) -> None:
        character = runner()
        character.set_swim(True)
        step(character, 0.1)
        assert character.state is CharacterState.SWIMMING

    def test_sliding(self) -> None:
        character = runner()
        assert character.slide(0.8, 0.3)
        step(character, 0.1)
        assert character.state is CharacterState.SLIDING

    def test_down(self) -> None:
        character = runner()
        character.knock_down()
        step(character, 0.1)
        assert character.state is CharacterState.DOWN


class TestSlide:
    def test_it_keeps_the_crouch_height_for_half_a_second_then_rises_over_three_tenths(
            self) -> None:
        caps = CharacterCapabilities()
        character = runner()
        character.set_move((1, 0, 0), 'run')
        step(character, 0.5)
        assert character.slide(0.8, 0.3)
        heights = []
        for _ in range(48):
            character.update(DT)
            heights.append(character.height)
        low, high = caps.crouchHeight, caps.standHeight
        assert heights[0] == pytest.approx(low, abs=1e-9)
        assert heights[29] == pytest.approx(low, abs=1e-9)             # t = 0.5 s
        assert heights[38] == pytest.approx(low + 0.5 * (high - low), abs=1e-9)   # 0.65 s
        assert heights[-1] == pytest.approx(high, abs=1e-9)             # 0.8 s
        assert heights == sorted(heights)
        assert character.state is CharacterState.STANDING
        assert not character.crouching

    def test_the_feet_stay_on_the_floor_through_it(self) -> None:
        character = runner()
        character.set_move((1, 0, 0), 'run')
        step(character, 0.3)
        character.slide(0.8, 0.3)
        for _ in range(48):
            character.update(DT)
            assert character.grounded
            assert abs(float(character.base()[1])) < 0.02

    def test_it_carries_the_speed_it_began_with_and_friction_bleeds_it_off(self) -> None:
        character = runner()
        character.set_move((1, 0, 0), 'run')
        step(character, 0.5)
        running = ground_speed(character)
        assert running == pytest.approx(6.0, abs=0.05)
        character.slide(0.8, 0.3)
        speeds = []
        for _ in range(30):
            character.update(DT)
            speeds.append(ground_speed(character))
        assert speeds[0] > 0.8 * running
        assert all(later < earlier for earlier, later in itertools.pairwise(speeds))
        assert speeds[-1] < 0.2 * running

    def test_input_is_ignored_while_it_lasts(self) -> None:
        character = runner()
        character.set_move((1, 0, 0), 'run')
        step(character, 0.5)
        character.slide(0.8, 0.3)
        character.set_move((-1, 0, 0), 'run')
        before = float(character.position[0])
        step(character, 0.3)
        assert float(character.position[0]) > before

    def test_it_ends_and_the_input_is_heard_again(self) -> None:
        character = runner()
        character.slide(0.8, 0.3)
        step(character, 0.9)
        assert character.state is CharacterState.STANDING
        character.set_move((0, 0, 1), 'walk')
        step(character, 0.5)
        assert character.velocity[2] == pytest.approx(3.0, abs=0.05)

    def test_it_cannot_start_in_the_air(self) -> None:
        character = runner()
        character.jump()
        step(character, 0.1)
        assert not character.slide(0.8, 0.3)
        assert character.state is CharacterState.AIRBORNE

    def test_a_jump_is_refused_while_it_lasts(self) -> None:
        character = runner()
        character.slide(0.8, 0.3)
        step(character, 0.2)
        assert not character.jump()
        assert character.grounded

    def test_the_rise_waits_for_a_low_ceiling(self) -> None:
        caps = CharacterCapabilities()
        character = runner()
        character.slide(0.8, 0.3)
        add_ceiling(character, 1.4)
        step(character, 1.0)
        assert character.height == pytest.approx(caps.crouchHeight, abs=1e-9)
        assert character.state is CharacterState.CROUCHING
        assert not character.set_crouch(False)


class TestKnockedDown:
    def test_input_moves_it_no_further(self) -> None:
        character = runner()
        character.set_move((1, 0, 0), 'run')
        step(character, 0.5)
        character.knock_down()
        step(character, 0.1)
        at = character.position.copy()
        character.set_move((0, 0, 1), 'sprint')
        step(character, 1.0)
        assert float(np.linalg.norm(character.position - at)) < 1e-6
        assert ground_speed(character) == pytest.approx(0.0, abs=1e-6)

    def test_it_lies_low(self) -> None:
        caps = CharacterCapabilities()
        character = runner()
        character.knock_down()
        step(character, 0.1)
        assert character.height == pytest.approx(caps.crouchHeight)
        assert abs(float(character.base()[1])) < 0.02

    def test_it_cannot_jump(self) -> None:
        character = runner()
        character.knock_down()
        assert not character.jump()

    def test_it_still_falls(self) -> None:
        character = CharacterController(floor_world(), CharacterCapabilities(),
                                        position=(0.0, 10.0, 0.0), gravity=GRAVITY)
        character.knock_down()
        step(character, 0.5)
        assert character.position[1] < 10.0 - 1.0

    def test_get_up_hears_the_input_again(self) -> None:
        caps = CharacterCapabilities()
        character = runner()
        character.knock_down()
        step(character, 0.2)
        character.get_up()
        assert character.state is CharacterState.STANDING
        assert character.height == pytest.approx(caps.standHeight)
        character.set_move((0, 0, 1), 'walk')
        step(character, 0.5)
        assert character.velocity[2] == pytest.approx(3.0, abs=0.05)

    def test_get_up_restarts_at_a_start_speed_along_the_last_heading(self) -> None:
        character = runner()
        character.set_move((1, 0, 0), 'run')
        step(character, 0.3)
        character.knock_down()
        step(character, 0.2)
        character.set_move((0, 0, 0))
        character.get_up(start_speed=4.0)
        character.update(DT)
        assert float(character.velocity[0]) == pytest.approx(4.0 * (1 - 6 * DT), abs=0.1)
        assert float(character.velocity[2]) == pytest.approx(0.0, abs=1e-6)

    def test_get_up_under_a_low_ceiling_stays_crouched(self) -> None:
        character = runner()
        character.knock_down()
        add_ceiling(character, 1.4)
        step(character, 0.2)
        character.get_up()
        assert character.state is CharacterState.CROUCHING


class TestLanding:
    def test_it_is_none_until_the_character_lands(self) -> None:
        character = runner()
        assert character.landing is None
        step(character, 0.5)
        assert character.landing is None

    def land(self, height: float) -> float:
        character = CharacterController(floor_world(), CharacterCapabilities(),
                                        position=(0.0, height + 0.9, 0.0), gravity=GRAVITY)
        for _ in range(600):
            character.update(DT)
            if character.landing is not None:
                assert character.grounded
                return character.landing
        raise AssertionError('never landed')

    @pytest.mark.parametrize('height', [2.0, 6.0])
    def test_a_fall_lands_at_the_speed_gravity_gave_it(self, height: float) -> None:
        assert self.land(height) == pytest.approx(math.sqrt(2.0 * GRAVITY * height), rel=0.05)

    def test_a_longer_fall_lands_harder(self) -> None:
        assert self.land(6.0) > 1.5 * self.land(2.0)

    def test_it_is_reported_for_the_one_update_only(self) -> None:
        character = CharacterController(floor_world(), CharacterCapabilities(),
                                        position=(0.0, 3.0, 0.0), gravity=GRAVITY)
        reports = []
        for _ in range(240):
            character.update(DT)
            reports.append(character.landing)
        assert sum(report is not None for report in reports) == 1

    def test_a_jump_lands_at_the_speed_it_left(self) -> None:
        character = runner(coyoteTime=0.0, jumpBuffer=0.0)
        character.jump()
        landed = None
        for _ in range(240):
            character.update(DT)
            if character.landing is not None:
                landed = character.landing
        assert landed == pytest.approx(math.sqrt(2.0 * GRAVITY * 1.2), rel=0.05)

    def test_walking_is_not_landing(self) -> None:
        character = runner()
        character.set_move((1, 0, 0), 'run')
        for _ in range(120):
            character.update(DT)
            assert character.landing is None
