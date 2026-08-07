"""Swimming: a capsule in water, which is neither walking nor flying.

Flying is noclip and free of gravity; a swimmer is neither. Water still lets
you through a doorway and not through a wall, and it still has a bottom you
sink to and a surface you rise to -- which is the whole difference between a
pool you swim in and a pool you fly around inside.

``buoyancy`` is the fraction of gravity that pushes back up, so 1.0 hangs where
it is, 0.0 sinks at full weight and anything above 1.0 rises.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld


def slab(world, y, extent=50.0, thickness=1.0):
    """A wide horizontal box, as a floor or a ceiling."""
    shape = world.add_shape(model.Shape.box(
        size=(extent * 2, thickness, extent * 2)))
    return world.add_body(model.Motion(type=model.STATIC),
                          collider=model.Collider(shape=shape),
                          position=(0.0, y, 0.0))


def wall(world, x, extent=50.0, thickness=1.0):
    """A tall vertical box across the x axis."""
    shape = world.add_shape(model.Shape.box(
        size=(thickness, extent, extent * 2)))
    return world.add_body(model.Motion(type=model.STATIC),
                          collider=model.Collider(shape=shape),
                          position=(float(x), 0.0, 0.0))


def swimmer(world=None, buoyancy=0.9, height=5.0, **caps):
    world = world if world is not None else PhysicsWorld(
        gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    character = CharacterController(world, CharacterCapabilities(**caps),
                                    gravity=9.81)
    character.position = np.array([0.0, float(height), 0.0])
    character.set_swim(True, buoyancy=buoyancy)
    return character


def run(character, seconds=2.0, dt=1 / 60.0):
    for _ in range(int(seconds / dt)):
        character.update(dt)
    return character


class TestEnteringAndLeaving:

    def test_a_swimmer_is_swimming(self):
        assert swimmer().swimming

    def test_swimming_is_not_flying(self):
        """They are different states and must not be conflated.

        Flying is noclip; swimming collides.  A swim implemented as a fly is a
        player who can leave a pool through its wall.
        """
        assert not swimmer().flying

    def test_leaving_the_water_stops_the_swimming(self):
        character = swimmer()
        character.set_swim(False)
        assert not character.swimming

    def test_a_swimmer_is_never_grounded(self):
        """Standing on the bottom of a pool is still swimming, not walking."""
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        slab(world, y=0.0)
        assert not run(swimmer(world, buoyancy=0.0, height=3.0), 3.0).grounded

    def test_leaving_the_water_leaves_the_vertical_speed_behind(self):
        """Surfacing must not fling the player, nor drop them like a stone.

        A swimmer rising at speed who breaks the surface with that speed
        intact is launched into the air; one who kept a sink speed would be
        slammed down.  Neither is what leaving water looks like.
        """
        character = swimmer(buoyancy=2.0)
        run(character, 1.0)
        assert character.vy > 0.0
        character.set_swim(False)
        assert character.vy == 0.0


class TestBuoyancy:

    def test_neutral_buoyancy_hangs_where_it_is(self):
        """1.0 is exactly cancelled gravity: no rise and no sink."""
        character = run(swimmer(buoyancy=1.0), 2.0)
        assert character.position[1] == pytest.approx(5.0, abs=0.05)

    def test_no_buoyancy_sinks(self):
        assert run(swimmer(buoyancy=0.0), 2.0).position[1] < 5.0 - 0.5

    def test_more_than_neutral_buoyancy_rises(self):
        assert run(swimmer(buoyancy=1.5), 2.0).position[1] > 5.0 + 0.2

    def test_a_swimmer_sinks_far_more_slowly_than_a_faller(self):
        """The difference in speed is most of what says "this is water".

        Sinking at the same rate as falling is what makes a pool read as a
        hole in the floor.
        """
        sinking = run(swimmer(buoyancy=0.0), 2.0)
        falling = swimmer(buoyancy=0.0)
        falling.set_swim(False)
        run(falling, 2.0)
        assert abs(sinking.vy) < abs(falling.vy) * 0.5

    def test_the_sink_reaches_a_steady_speed_rather_than_accelerating(self):
        """Drag, not terminal velocity: water resists from the first metre."""
        character = swimmer(buoyancy=0.0)
        run(character, 3.0)
        early = character.vy
        run(character, 3.0)
        assert character.vy == pytest.approx(early, rel=0.05)


class TestCollision:
    """The half that a noclip swim gets wrong."""

    def test_a_swimmer_cannot_pass_through_a_wall(self):
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        wall(world, x=2.0)
        character = swimmer(world, buoyancy=1.0)
        character.set_fly_move((1.0, 0.0, 0.0))
        run(character, 3.0)
        assert character.position[0] < 2.0

    def test_a_swimmer_settles_on_the_bottom_rather_than_through_it(self):
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        slab(world, y=0.0)
        character = run(swimmer(world, buoyancy=0.0, height=4.0), 6.0)
        assert character.base()[1] > -0.2

    def test_hitting_the_bottom_stops_the_sink(self):
        """Otherwise the speed builds against the floor and surfacing rockets."""
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        slab(world, y=0.0)
        character = run(swimmer(world, buoyancy=0.0, height=4.0), 6.0)
        assert character.vy == pytest.approx(0.0, abs=0.05)

    def test_hitting_the_ceiling_stops_the_rise(self):
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        slab(world, y=6.0)
        character = run(swimmer(world, buoyancy=2.0, height=2.0), 6.0)
        assert character.vy == pytest.approx(0.0, abs=0.05)


class TestSwimmingUnderControl:

    def test_the_up_command_beats_the_sink_at_the_usual_buoyancy(self):
        """Holding "up" in water has to actually get you out of it."""
        character = swimmer(buoyancy=0.9)
        character.set_fly_move((0.0, 1.0, 0.0))
        assert run(character, 2.0).position[1] > 5.0

    def test_a_body_with_no_buoyancy_sinks_faster_than_it_can_swim(self):
        """The extreme end of the knob, pinned because it is a design choice.

        The stroke is *added* to the drift rather than overriding it, so with
        buoyancy at 0 -- "sinks like a stone" -- the steady sink of
        ``gravity / swimDrag`` outruns ``swimSpeed`` and holding up merely
        slows the descent.  A game that wants lava to be escapable raises the
        buoyancy of lava; it does not need different physics.
        """
        character = swimmer(buoyancy=0.0)
        character.set_fly_move((0.0, 1.0, 0.0))
        sinking = run(character, 2.0).position[1]
        assert sinking < 5.0
        assert sinking > run(swimmer(buoyancy=0.0), 2.0).position[1]

    def test_a_settled_stroke_carries_at_the_swim_speed(self):
        """Measured after the stroke has built, not across the build-up."""
        character = swimmer(buoyancy=1.0, swimSpeed=2.0)
        character.set_fly_move((0.0, 0.0, -1.0))
        run(character, 2.0)
        settled = character.position[2]
        run(character, 1.0)
        assert settled - character.position[2] == pytest.approx(2.0, rel=0.05)

    def test_a_stroke_builds_rather_than_switching_on(self):
        """The first moment of a stroke is slower than the settled one."""
        character = swimmer(buoyancy=1.0, swimSpeed=2.0)
        character.set_fly_move((0.0, 0.0, -1.0))
        run(character, 0.1)
        first = -character.position[2]
        run(character, 2.0)
        before = character.position[2]
        run(character, 0.1)
        assert first < (before - character.position[2]) * 0.9

    def test_swimming_is_slower_than_walking(self):
        """Water is what you push against, and it should feel like it."""
        assert CharacterCapabilities().swimSpeed < CharacterCapabilities().walkSpeed

    def test_letting_go_coasts_to_a_stop_rather_than_stopping_dead(self):
        character = swimmer(buoyancy=1.0)
        character.set_fly_move((0.0, 0.0, -1.0))
        run(character, 1.0)
        moving = character.position[2]
        character.set_fly_move((0.0, 0.0, 0.0))
        character.update(1 / 60.0)
        assert character.position[2] < moving

    def test_an_impulse_still_carries_a_swimmer(self):
        """A jump pad firing into water must not simply stop at the surface."""
        character = swimmer(buoyancy=1.0)
        character.apply_impulse((4.0, 0.0, 0.0))
        run(character, 0.5)
        assert character.position[0] > 0.2
