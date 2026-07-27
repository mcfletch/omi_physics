"""Falling onto mesh geometry: the ground catches you, however fast you arrive.

A trimesh is what a loaded scene becomes (OpenGLContext's
``physics.gltf_world.collision_world_from_scene`` turns a whole glTF group into
one), so this is the collision every walkable floor in a real scene goes
through -- and a floor that only stops a slow walker is not a floor.
"""
import sys

import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld
from omi_physics.character import CharacterController, CharacterCapabilities

STOREY = 3.0


def trimesh_floor(world, y=0.0, extent=200.0):
    e = extent
    pts = np.array([(-e, y, -e), (e, y, -e), (e, y, e), (-e, y, e)], dtype='d')
    idx = np.array([(0, 1, 2), (0, 2, 3)], dtype='i')
    shape = world.add_shape(model.Shape.trimesh(pts, idx))
    return world.add_body(model.Motion(type=model.STATIC),
                          collider=model.Collider(shape=shape), position=(0, 0, 0))


def floor_world():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    trimesh_floor(world)
    return world


def dropped(height, dt, world=None):
    """Release a character at ``height`` and run until it settles."""
    world = world or floor_world()
    ch = CharacterController(world, CharacterCapabilities(), gravity=9.81)
    ch.position = np.array([0.0, float(height), 0.0])
    ch.grounded = False
    ch.vy = 0.0
    deepest = ch.base()[1]
    for _ in range(3000):
        ch.update(dt)
        deepest = min(deepest, ch.base()[1])
        if ch.grounded:
            break
    return ch, deepest


class TestDepenetrationGoesTheRightWay:
    """A capsule inside the floor is pushed **out of** it, not through it.

    The contact normal for a capsule against a triangle used to be taken from
    whichever end of the capsule was nearest the surface.  Once the lower end
    passed below a floor that end is still the nearest, so the push pointed
    *down* and the resolve drove the character through the very surface it had
    hit.
    """

    @pytest.fixture
    def character(self):
        return CharacterController(floor_world(), CharacterCapabilities(),
                                   gravity=9.81)

    @pytest.mark.parametrize('centre', [0.9, 0.7, 0.5, 0.3, 0.1, 0.01])
    def test_a_capsule_over_the_floor_ends_up_above_it(self, character, centre):
        """Above the surface it is resolved above it, however deep it has sunk.

        A zero-thickness floor has no inside, so "which side" is the side the
        capsule's centre is on; the failure this pins is being resolved to the
        *other* one.
        """
        resolved, _ = character._push_out(np.array([0.0, centre, 0.0]))
        assert resolved[1] >= 0.9 - 1e-3, (
            'placed at y=%s, resolved to y=%.4f -- pushed through the floor'
            % (centre, resolved[1]))

    def test_a_capsule_under_the_floor_is_left_under_it(self, character):
        """A ceiling pushes down, so the same rule has to work upside down."""
        resolved, _ = character._push_out(np.array([0.0, -0.1, 0.0]))
        assert resolved[1] <= -0.9 + 1e-3, resolved[1]

    @pytest.mark.parametrize('centre', [0.9, 0.5, 0.1])
    def test_the_floor_is_reported_as_ground(self, character, centre):
        _resolved, ground = character._push_out(np.array([0.0, centre, 0.0]))
        assert ground is not None, 'no ground found at y=%s' % (centre,)
        assert ground[1] > 0.5, ground

    def test_a_capsule_clear_of_the_floor_is_left_alone(self, character):
        resolved, ground = character._push_out(np.array([0.0, 4.0, 0.0]))
        assert resolved[1] == pytest.approx(4.0)
        assert ground is None


class TestAFallIsCaughtHoweverFastItArrives:
    """The reported bug: from the third or fourth storey you land inside the floor."""

    @pytest.mark.parametrize('storey', range(1, 13))
    @pytest.mark.parametrize('dt', [1 / 60.0, 1 / 30.0, 0.05])
    def test_landing_from_any_storey_rests_on_the_floor(self, storey, dt):
        ch, _deepest = dropped(storey * STOREY, dt)
        assert ch.grounded, 'storey %d at dt=%.4f never landed' % (storey, dt)
        assert ch.base()[1] == pytest.approx(0.0, abs=0.05)

    @pytest.mark.parametrize('storey', [3, 4, 5])
    def test_the_landing_never_dips_into_the_floor(self, storey):
        """Not merely 'ends up right' -- it is never seen inside the floor."""
        _ch, deepest = dropped(storey * STOREY, 0.05)
        assert deepest > -0.05, 'sank %.3f m into the floor' % (-deepest,)

    def test_a_very_long_fall_still_lands(self):
        ch, _deepest = dropped(120.0, 0.05)
        assert ch.grounded
        assert ch.base()[1] == pytest.approx(0.0, abs=0.05)

    def test_it_lands_on_the_floor_it_hit_not_one_below(self):
        """Two storeys: a fall from the upper one stops on the lower slab."""
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81,
                                                   direction=(0, -1, 0)))
        trimesh_floor(world, y=0.0)
        trimesh_floor(world, y=6.0, extent=20.0)
        ch, _ = dropped(18.0, 0.05, world=world)
        assert ch.grounded
        assert ch.base()[1] == pytest.approx(6.0, abs=0.05)


class TestTerminalVelocityBoundsTheFall:
    """How fast the capsule may fall is a stated speed, not an emergent one.

    It is also what makes the substep count finite: collision is stepped finely
    enough to keep up with the capsule, so the fastest the capsule may go is
    what decides how much work a frame can ever be asked for.
    """

    def test_a_long_fall_settles_at_terminal_velocity(self):
        caps = CharacterCapabilities(terminalVelocity=20.0)
        world = floor_world()
        ch = CharacterController(world, caps, gravity=9.81)
        ch.position = np.array([0.0, 4000.0, 0.0])
        ch.grounded = False
        ch.vy = 0.0
        fastest = 0.0
        for _ in range(400):
            ch.update(0.05)
            fastest = max(fastest, -ch.vy)
        assert fastest == pytest.approx(20.0, abs=0.5)

    def test_it_never_exceeds_terminal_velocity(self):
        caps = CharacterCapabilities(terminalVelocity=30.0)
        ch = CharacterController(floor_world(), caps, gravity=9.81)
        ch.position = np.array([0.0, 4000.0, 0.0])
        ch.grounded = False
        ch.vy = 0.0
        for _ in range(400):
            ch.update(0.05)
            assert -ch.vy <= 30.0 + 1e-6, -ch.vy

    def test_a_terminal_fall_still_lands_on_the_floor(self):
        caps = CharacterCapabilities(terminalVelocity=55.0)
        world = floor_world()
        ch = CharacterController(world, caps, gravity=9.81)
        ch.position = np.array([0.0, 500.0, 0.0])
        ch.grounded = False
        ch.vy = 0.0
        for _ in range(4000):
            ch.update(0.05)
            if ch.grounded:
                break
        assert ch.grounded
        assert ch.base()[1] == pytest.approx(0.0, abs=0.05)

    def test_jumping_is_not_clamped_by_it(self):
        """Terminal velocity is about falling; a launch is not a fall."""
        caps = CharacterCapabilities(terminalVelocity=1.0)
        ch = CharacterController(floor_world(), caps, gravity=9.81)
        ch.safe_bind((0, 1.0, 0))
        ch.jump()
        assert ch.vy > 1.0


class TestTheSubstepCountFollowsFromTerminalVelocity:
    """The ceiling on per-frame work is calculated, not chosen.

    A fixed cap is a number nobody can check: raise the fall speed past what it
    allows and tunnelling returns, silently.  Deriving it from the fastest the
    capsule may travel means the two can never disagree.
    """

    def _controller(self, terminal):
        return CharacterController(
            floor_world(), CharacterCapabilities(terminalVelocity=terminal),
            gravity=9.81)

    def test_the_ceiling_covers_a_frame_at_terminal_velocity(self):
        ch = self._controller(55.0)
        dt = 0.05
        ch.vy = -55.0
        assert len(ch._substeps(dt)) <= ch.max_substeps(dt)

    def test_a_faster_terminal_velocity_buys_more_substeps(self):
        slow = self._controller(20.0).max_substeps(0.05)
        fast = self._controller(60.0).max_substeps(0.05)
        assert fast > slow

    def test_standing_still_costs_one_substep(self):
        ch = self._controller(55.0)
        ch.safe_bind((0, 1.0, 0))
        assert len(ch._substeps(1 / 60.0)) == 1

    def test_a_terminal_fall_is_stepped_finely_enough(self):
        """Every substep stays inside what collision can see."""
        ch = self._controller(55.0)
        ch.vy = -55.0
        dt = 0.05
        pieces = ch._substeps(dt)
        travel = 55.0 * (dt / len(pieces))
        assert travel <= ch.height * 0.5, travel

    def test_a_shorter_frame_needs_fewer_substeps(self):
        ch = self._controller(55.0)
        assert ch.max_substeps(1 / 60.0) < ch.max_substeps(0.05)

    def test_an_unlimited_fall_says_so_rather_than_pretending(self):
        """No top speed means no bound -- and the method admits it."""
        ch = CharacterController(floor_world(),
                                 CharacterCapabilities(terminalVelocity=0.0),
                                 gravity=9.81)
        assert ch.terminal_velocity() == float('inf')
        assert ch.max_substeps(0.05) == sys.maxsize

    def test_an_unlimited_fall_still_only_pays_for_its_speed(self):
        ch = CharacterController(floor_world(),
                                 CharacterCapabilities(terminalVelocity=0.0),
                                 gravity=9.81)
        ch.vy = -30.0
        assert len(ch._substeps(0.05)) < 20
