"""Climbing a step is a stride, not a lurch.

Mounting a ledge used to happen in **one motion**: the capsule was lifted, moved
far enough to clear the edge in a single frame, and the excess was booked as a
debt paid back over the frames that followed.  That made the *average* speed
right and the *instant* wrong — and the instant is what a player sees, as a
sharp forward jump of the better part of half a metre every time they walked
onto a step.

So what is asserted here is the **per-frame** displacement, not the arrival.  A
test that only checked where the capsule ended up would have passed throughout.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld

#: twitch's own proportions, in metres: an 18-unit step, a 300-unit walk.
STEP_HEIGHT = 18.0 * 0.0254
WALK_SPEED = 300.0 * 0.0254
RADIUS = 16.0 * 0.0254
STAND = 56.0 * 0.0254


def _quad(world, points):
    indices = np.array([(0, 1, 2), (0, 2, 3)], dtype='i')
    shape = world.add_shape(model.Shape.trimesh(np.array(points, dtype='d'),
                                                indices))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape), position=(0, 0, 0))


def stepped_world(height=STEP_HEIGHT, at=2.0, extent=20.0):
    """A floor, a riser at ``at``, and a higher floor beyond it."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    e = extent
    _quad(world, [(-e, 0.0, -e), (at, 0.0, -e), (at, 0.0, e), (-e, 0.0, e)])
    _quad(world, [(at, 0.0, -e), (at, height, -e), (at, height, e), (at, 0.0, e)])
    _quad(world, [(at, height, -e), (e, height, -e), (e, height, e), (at, height, e)])
    return world


def walker(world, speed=WALK_SPEED):
    caps = CharacterCapabilities(stepHeight=STEP_HEIGHT, walkSpeed=speed,
                                 radius=RADIUS, standHeight=STAND,
                                 crouchHeight=STAND * 0.5, eyeHeight=STAND * 0.82)
    character = CharacterController(world, caps, gravity=9.81)
    character.safe_bind((0.0, STAND * 0.5, 0.0))
    character.set_move((1.0, 0.0, 0.0), mode='walk')
    return character


def strides(character, dt=1 / 60.0, frames=120):
    """The horizontal distance covered in each of ``frames`` frames."""
    covered = []
    previous = character.position.copy()
    for _frame in range(frames):
        character.update(dt)
        moved = character.position - previous
        covered.append(float(np.linalg.norm(moved[[0, 2]])))
        previous = character.position.copy()
    return covered


#: **This bug is open.**  Three fixes were tried and each traded the lurch for
#: a stall — crossing at the frame's own pace leaves the capsule inside the
#: riser and the step-down snap drags it back; rising in place leaves it
#: airborne over the lower floor with nothing to seat on; taking the shortest
#: advance that stands never finds one, because a capsule resting exactly at
#: the step's height still grazes the riser and the deepest contact is a
#: vertical face.  The measurement below is kept and marked expected-to-fail so
#: the bug stays visible and the next attempt has a number to aim at.
LURCH_IS_OPEN = pytest.mark.xfail(
    reason='mounting a step advances the whole climb in one frame; see '
           'PROJECT-PLAN §3b B1', strict=True)


class TestThePaceOfAStep:

    @LURCH_IS_OPEN
    def test_no_frame_covers_more_than_it_was_due(self):
        """The bug, stated as a rule: a stride is a stride, every frame.

        A tolerance rather than an equality because collision resolution moves
        the capsule a little as it seats — but a *lurch* is several times the
        frame's due, not a few per cent over it.
        """
        dt = 1 / 60.0
        character = walker(stepped_world())
        due = character.caps.walkSpeed * dt
        worst = max(strides(character, dt=dt))
        assert worst <= due * 1.35, (
            f'one frame covered {worst:.3f} m where {due:.3f} m was due '
            f'({worst / due:.1f} times)')

    def test_the_step_is_actually_climbed(self):
        """The pace must not be bought by refusing to climb at all."""
        character = walker(stepped_world())
        strides(character)
        assert character.position[0] > 2.5
        assert character.base()[1] > STEP_HEIGHT * 0.5

    @LURCH_IS_OPEN
    def test_a_flight_of_steps_is_climbed_at_walking_pace(self):
        """Not one step per frame: the frame rate must not set the pace."""
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        e = 20.0
        for index in range(6):
            near, far = 1.0 + index * 0.6, 1.6 + index * 0.6
            height = index * STEP_HEIGHT
            _quad(world, [(near - 0.6, height, -e), (near, height, -e),
                          (near, height, e), (near - 0.6, height, e)])
            _quad(world, [(near, height, -e), (near, height + STEP_HEIGHT, -e),
                          (near, height + STEP_HEIGHT, e), (near, height, e)])
            _quad(world, [(near, height + STEP_HEIGHT, -e), (far, height + STEP_HEIGHT, -e),
                          (far, height + STEP_HEIGHT, e), (near, height + STEP_HEIGHT, e)])
        _quad(world, [(-e, 0.0, -e), (1.0, 0.0, -e), (1.0, 0.0, e), (-e, 0.0, e)])
        character = walker(world)
        dt = 1 / 60.0
        due = character.caps.walkSpeed * dt
        assert max(strides(character, dt=dt, frames=180)) <= due * 1.35

    @LURCH_IS_OPEN
    def test_a_faster_walker_is_still_within_its_own_due(self):
        dt = 1 / 60.0
        character = walker(stepped_world(), speed=WALK_SPEED * 1.6)
        due = character.caps.walkSpeed * dt
        assert max(strides(character, dt=dt)) <= due * 1.35

    def test_walking_on_the_flat_is_unaffected(self):
        """The change must not cost anything where there is no step."""
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        _quad(world, [(-20.0, 0.0, -20.0), (20.0, 0.0, -20.0),
                      (20.0, 0.0, 20.0), (-20.0, 0.0, 20.0)])
        character = walker(world)
        dt = 1 / 60.0
        covered = strides(character, dt=dt, frames=60)
        due = character.caps.walkSpeed * dt
        assert sum(covered[10:]) == pytest.approx(due * 50, rel=0.05)

    def test_a_step_too_tall_is_still_refused(self):
        character = walker(stepped_world(height=STEP_HEIGHT * 4))
        strides(character)
        assert character.position[0] < 2.5
