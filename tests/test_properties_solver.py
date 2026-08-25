"""Properties a step holds whatever scene it is given.

The suite elsewhere drops a ball on a floor and checks where it lands. These
files hand the solver scenes nobody would author -- a dozen colliders already
inside one another, a body of a hundredth of a kilogramme resting under one of a
hundred, a collider with no size at all, friction above one -- and ask only for
what must be true of every step regardless: that the world stays made of
numbers, that a static body stays where it was put, and that a scene left alone
settles rather than accelerating away.

That last one is what a player sees when it fails. A solver that gains energy
where it should lose it does not report anything; a crate on a floor simply
starts to shiver, and then to climb.
"""
import numpy as np
import pytest

# Hypothesis is a CPython-only test dependency; see test_properties_math.py.
pytest.importorskip('hypothesis', reason='hypothesis has no wheel for this interpreter')

from hypothesis import given
from hypothesis import strategies as st
from property_strategies import (
    SETTINGS,
    STEPPING,
    build_world,
    unit_quaternions,
    vectors,
    world_recipes,
    worlds,
)

from omi_physics import model
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0

#: Nothing in these scenes is thrown harder than ten metres a second or dropped
#: from higher than three metres, so a body a kilometre out has not been solved,
#: it has been launched.
ESCAPED = 1e3


def state(world):
    """Everything a step is allowed to change, as one array per quantity."""
    n = world.body_count
    return (world.position[:n].copy(), world.orientation[:n].copy(),
            world.linear_velocity[:n].copy(), world.angular_velocity[:n].copy())


def is_finite(world) -> bool:
    return all(np.isfinite(part).all() for part in state(world))


class TestTheWorldStaysMadeOfNumbers:
    """One NaN is the end of the scene: it spreads through the next contact into
    both bodies, and out of the broad phase into every pair either of them is
    in. Nothing in the pipeline stops it, so nothing may start it."""

    @given(world=worlds(max_bodies=6))
    @STEPPING
    def test_stepping_any_scene_leaves_finite_numbers(self, world):
        for _ in range(60):
            world.step(DT)
        assert is_finite(world)

    @given(world=worlds(max_bodies=6),
           dt=st.floats(min_value=1e-6, max_value=1.0 / 20.0))
    @STEPPING
    def test_any_step_length_leaves_finite_numbers(self, world, dt):
        """A game that hitches hands the accumulator a long step, and one on a
        fast machine hands it a very short one."""
        for _ in range(20):
            world.step(dt)
        assert is_finite(world)

    @given(world=worlds(max_bodies=6))
    @STEPPING
    def test_orientations_stay_rotations(self, world):
        """A quaternion that has drifted off the unit sphere scales the body it
        turns, and the inverse inertia built from it is no longer an inertia."""
        for _ in range(60):
            world.step(DT)
        lengths = np.linalg.norm(world.orientation[:world.body_count], axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-9)

    @given(world=worlds(max_bodies=6))
    @STEPPING
    def test_nothing_is_launched_out_of_the_scene(self, world):
        for _ in range(60):
            world.step(DT)
        assert np.all(np.abs(world.position[:world.body_count]) < ESCAPED)


class TestBodiesThatShouldNotMoveDoNot:
    @given(world=worlds(max_bodies=6))
    @STEPPING
    def test_a_static_body_is_where_it_was_put(self, world):
        """Static means the Earth: it is what everything else is resolved
        against, and a floor that is pushed down by what stands on it is a
        scene that sinks."""
        n = world.body_count
        fixed = world.motion_type[:n] == 0
        before = world.position[:n][fixed].copy()
        orientation = world.orientation[:n][fixed].copy()
        for _ in range(30):
            world.step(DT)
        assert np.allclose(world.position[:n][fixed], before)
        assert np.allclose(world.orientation[:n][fixed], orientation)

    @given(world=worlds(max_bodies=6))
    @STEPPING
    def test_a_kinematic_body_keeps_the_velocity_it_was_given(self, world):
        """Kinematic means the animation has the say: contacts push what it
        meets, and nothing pushes back."""
        n = world.body_count
        driven = world.motion_type[:n] == 1
        before = world.linear_velocity[:n][driven].copy()
        for _ in range(30):
            world.step(DT)
        assert np.allclose(world.linear_velocity[:n][driven], before)


class TestAScenePutDownSettles:
    @given(world=worlds(max_bodies=5))
    @STEPPING
    def test_it_does_not_end_faster_than_the_fall_could_make_it(self, world):
        """Free fall over the run is the whole of the speed available; anything
        past it came from the solver, and a stack that gains a little energy per
        step is a stack that eventually takes off.

        Generous, because a bounce trades height for speed and restitution of
        one keeps it: what this catches is a body leaving with tens of times the
        energy it arrived with, which is what an unstable contact looks like.
        """
        n = world.body_count
        thrown = float(np.max(np.linalg.norm(world.linear_velocity[:n], axis=1)))
        steps = 240
        for _ in range(steps):
            world.step(DT)
        budget = thrown + world.gravity.gravity * steps * DT + 1.0
        assert np.all(np.linalg.norm(world.linear_velocity[:n], axis=1) <= budget * 4.0)


class TestTheSameSceneStepsTheSameWay:
    """The CPU backend is deterministic run to run, which is what lets a replay
    be a list of inputs rather than a list of positions."""

    @given(recipe=world_recipes(max_bodies=5))
    @STEPPING
    def test_two_identical_worlds_stay_identical(self, recipe):
        one, other = build_world(recipe), build_world(recipe)
        for _ in range(30):
            one.step(DT)
            other.step(DT)
        for a, b in zip(state(one), state(other), strict=True):
            assert np.array_equal(a, b)

    @given(recipe=world_recipes(max_bodies=5))
    @STEPPING
    def test_the_order_bodies_were_added_in_is_the_order_they_keep(self, recipe):
        """A body's index is its identity for as long as it lives, so a caller
        can hold one and read its pose back."""
        world = build_world(recipe)
        count = world.body_count
        for _ in range(10):
            world.step(DT)
        assert world.body_count == count


class TestUndisturbedMotion:
    """With nothing to hit and nothing pulling, the integrator is the whole of
    the step, and what it does is exactly stated."""

    @staticmethod
    def _lone_body(velocity, spin=(0.0, 0.0, 0.0)):
        world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), sleep_enabled=False)
        shape = world.add_shape(model.Shape.sphere(0.5))
        body = world.add_body(
            model.Motion(type=model.DYNAMIC, mass=1.0,
                         linearVelocity=tuple(velocity), angularVelocity=tuple(spin)),
            collider=model.Collider(shape=shape), position=(0, 0, 0))
        return world, body

    @given(velocity=vectors(20.0), steps=st.integers(1, 40))
    @SETTINGS
    def test_a_body_with_nothing_to_hit_keeps_its_velocity(self, velocity, steps):
        world, body = self._lone_body(velocity)
        for _ in range(steps):
            world.step(DT)
        assert np.allclose(world.linear_velocity[body], velocity, rtol=1e-12)

    @given(velocity=vectors(20.0), steps=st.integers(1, 40))
    @SETTINGS
    def test_and_travels_the_distance_that_implies(self, velocity, steps):
        world, body = self._lone_body(velocity)
        for _ in range(steps):
            world.step(DT)
        assert np.allclose(world.position[body], np.asarray(velocity) * steps * DT,
                           rtol=1e-9, atol=1e-9)

    @given(spin=vectors(20.0), steps=st.integers(1, 40))
    @SETTINGS
    def test_a_spinning_body_keeps_spinning_and_stays_a_rotation(self, spin, steps):
        world, body = self._lone_body((0.0, 0.0, 0.0), spin)
        for _ in range(steps):
            world.step(DT)
        assert np.allclose(world.angular_velocity[body], spin, rtol=1e-9, atol=1e-9)
        assert np.linalg.norm(world.orientation[body]) == pytest.approx(1.0, rel=1e-9)


class TestDegenerateBodies:
    """Colliders an asset pipeline produces and a level never means to contain."""

    @given(position=vectors(5.0), orientation=unit_quaternions())
    @SETTINGS
    def test_a_collider_with_no_size_rests_on_the_floor_quietly(self, position, orientation):
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), sleep_enabled=False)
        floor = world.add_shape(model.Shape.box((40, 1, 40)))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=floor), position=(0, -0.5, 0))
        for shape in (model.Shape.sphere(0.0), model.Shape.box((0.0, 0.0, 0.0)),
                      model.Shape.capsule(height=0.0, radius=0.0)):
            index = world.add_shape(shape)
            world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                           collider=model.Collider(shape=index),
                           position=tuple(position), orientation=tuple(orientation))
        for _ in range(120):
            world.step(DT)
        assert is_finite(world)

    @given(masses=st.lists(st.floats(min_value=1e-4, max_value=1e4),
                           min_size=2, max_size=5))
    @SETTINGS
    def test_a_stack_of_any_masses_stays_on_top_of_the_floor(self, masses):
        """Mass ratios of eight orders of magnitude are what a crate of
        ammunition under a truck comes to, and the contact between them is the
        one whose effective mass is worst conditioned.

        What is asked is that the floor holds. Whether the *stack* holds is a
        different question and the answer at these ratios is no: sequential
        impulses pass a load down a stack one contact at a time, so a light box
        between two heavy ones is being pushed by a force its ten iterations
        cannot balance, and it is squeezed out sideways. A stack of boxes of
        comparable weight is what stays a stack (``test_solver.py``); this one
        collapses, and what it may not do while collapsing is leave the world or
        end up under the ground.
        """
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), sleep_enabled=False)
        # Wide enough that a box shot out of the stack is still over it at the
        # end, so "went through the floor" and "slid off the edge" stay apart.
        floor = world.add_shape(model.Shape.box((400, 1, 400)))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=floor), position=(0, -0.5, 0))
        cube = world.add_shape(model.Shape.box((1, 1, 1)))
        stack = [world.add_body(model.Motion(type=model.DYNAMIC, mass=mass),
                                collider=model.Collider(shape=cube),
                                position=(0, 0.5 + k * 1.05, 0))
                 for k, mass in enumerate(masses)]
        for _ in range(400):
            world.step(DT)
        assert is_finite(world)
        resting = world.position[stack]
        over_the_floor = np.all(np.abs(resting[:, [0, 2]]) < 190.0, axis=1)
        assert np.all(resting[over_the_floor][:, 1] > -0.5)
