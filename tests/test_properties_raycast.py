"""Properties a ray cast holds against any world.

A cast is what a game asks the physics when it needs an answer now: what is the
gun pointing at, is there ground under this wheel, can this guard see that
player. The answer is acted on immediately and rarely checked, so what it must
never be is *plausible* -- a hit behind the shooter, a distance that does not
match the point, a normal facing away from the ray.

The bundled form is asked about too, since a vehicle takes its wheel rays that
way: it shares work between rays, and what it may not do is give a different
answer for it.
"""
import numpy as np
import pytest

# Hypothesis is a CPython-only test dependency; see test_properties_math.py.
pytest.importorskip('hypothesis', reason='hypothesis has no wheel for this interpreter')

from hypothesis import given
from hypothesis import strategies as st
from property_strategies import (
    SETTINGS,
    directions,
    shapes,
    unit_quaternions,
    vectors,
    worlds,
)

from omi_physics import model, raycast
from omi_physics.world import PhysicsWorld

REACH = 500.0


def unit(v):
    length = np.linalg.norm(v)
    return None if length < 1e-9 else np.asarray(v, dtype='d') / length


class TestAHitDescribesItself:
    """The four numbers in a hit have to agree with one another: a caller uses
    the distance to decide and the point and normal to draw."""

    @given(world=worlds(max_bodies=6), origin=vectors(20.0), direction=directions())
    @SETTINGS
    def test_the_point_is_where_the_distance_says(self, world, origin, direction):
        heading = unit(direction)
        if heading is None:
            return
        hit = raycast.raycast(world, origin, direction, max_distance=REACH)
        if hit is None:
            return
        assert 0.0 <= hit.distance <= REACH
        expected = np.asarray(origin, dtype='d') + heading * hit.distance
        assert np.allclose(hit.point, expected, atol=1e-6 * max(hit.distance, 1.0))

    @given(world=worlds(max_bodies=6), origin=vectors(20.0), direction=directions())
    @SETTINGS
    def test_the_normal_faces_back_along_the_ray(self, world, origin, direction):
        """Documented of :class:`RayHit`, and what lets an impact effect be
        oriented without knowing which way the surface was wound."""
        heading = unit(direction)
        if heading is None:
            return
        hit = raycast.raycast(world, origin, direction, max_distance=REACH)
        if hit is None:
            return
        assert np.linalg.norm(hit.normal) == pytest.approx(1.0, rel=1e-6)
        assert float(np.dot(hit.normal, heading)) <= 1e-9

    @given(world=worlds(max_bodies=6), origin=vectors(20.0), direction=directions())
    @SETTINGS
    def test_it_names_a_body_that_is_in_the_world(self, world, origin, direction):
        hit = raycast.raycast(world, origin, direction, max_distance=REACH)
        if hit is None:
            return
        assert 0 <= hit.body < world.body_count


class TestWhatIsNotACast:
    @given(world=worlds(max_bodies=4), origin=vectors(20.0))
    @SETTINGS
    def test_a_ray_with_no_direction_meets_nothing(self, world, origin):
        """Documented as a miss rather than an error: a caller with a velocity
        should not have to check whether it is standing still."""
        assert raycast.raycast(world, origin, (0.0, 0.0, 0.0)) is None

    @given(world=worlds(max_bodies=4), origin=vectors(20.0), direction=directions(),
           reach=st.floats(min_value=-10.0, max_value=0.0))
    @SETTINGS
    def test_a_ray_with_no_reach_meets_nothing(self, world, origin, direction, reach):
        assert raycast.raycast(world, origin, direction, max_distance=reach) is None

    @given(world=worlds(max_bodies=4), origin=vectors(20.0), direction=directions())
    @SETTINGS
    def test_skipping_every_body_meets_nothing(self, world, origin, direction):
        assert raycast.raycast(world, origin, direction, max_distance=REACH,
                               skip=range(world.body_count)) is None


class TestHowTheRayIsGivenDoesNotMatter:
    @given(world=worlds(max_bodies=6), origin=vectors(20.0), direction=directions(),
           scale=st.floats(min_value=1e-3, max_value=1e3))
    @SETTINGS
    def test_the_length_of_the_direction_changes_nothing(self, world, origin,
                                                         direction, scale):
        """`direction` need not be normalised, so a caller can pass a velocity."""
        if unit(direction) is None:
            return
        one = raycast.raycast(world, origin, direction, max_distance=REACH)
        other = raycast.raycast(world, origin, np.asarray(direction) * scale,
                                max_distance=REACH)
        assert (one is None) == (other is None)
        if one is not None:
            assert one.body == other.body
            assert one.distance == pytest.approx(other.distance, rel=1e-6, abs=1e-9)


class TestTheBundleAnswersWhatTheRaysWould:
    """A vehicle casts one ray per wheel every step and they share their work.
    What comes back has to be, ray for ray, what each would have answered alone."""

    @given(world=worlds(max_bodies=6),
           rays=st.lists(st.tuples(vectors(20.0), directions()),
                         min_size=1, max_size=6))
    @SETTINGS
    def test_a_bundle_matches_the_rays_cast_one_at_a_time(self, world, rays):
        origins = [o for o, _ in rays]
        directions = [d for _, d in rays]
        together = raycast.raycast_many(world, origins, directions, max_distance=REACH)
        assert len(together) == len(rays)
        for bundled, (origin, direction) in zip(together, rays, strict=True):
            alone = raycast.raycast(world, origin, direction, max_distance=REACH)
            assert (bundled is None) == (alone is None)
            if alone is not None:
                assert bundled.body == alone.body
                assert bundled.distance == pytest.approx(alone.distance, rel=1e-9)

    @given(world=worlds(max_bodies=4))
    @SETTINGS
    def test_an_empty_bundle_is_an_empty_answer(self, world):
        assert raycast.raycast_many(world, np.zeros((0, 3)), np.zeros((0, 3))) == []


class TestASingleBodyIsWhereItSaysItIs:
    """The properties above hold when nothing is hit at all, so one case has to
    fix that a cast finds what is in front of it."""

    @given(shape=shapes(degenerate=False), centre=vectors(10.0),
           orientation=unit_quaternions(), direction=directions())
    @SETTINGS
    def test_a_ray_aimed_at_a_body_finds_it(self, shape, centre, orientation, direction):
        heading = unit(direction)
        if heading is None:
            return
        world = PhysicsWorld()
        index = world.add_shape(shape)
        body = world.add_body(model.Motion(type=model.STATIC),
                              collider=model.Collider(shape=index),
                              position=tuple(centre), orientation=tuple(orientation))
        start = np.asarray(centre, dtype='d') - heading * 50.0
        hit = raycast.raycast(world, start, heading, max_distance=REACH)
        assert hit is not None and hit.body == body
        # Aimed at the centre from outside, so the surface is nearer than it is.
        assert hit.distance <= 50.0 + 1e-6

    @given(shape=shapes(degenerate=False), centre=vectors(10.0),
           orientation=unit_quaternions(), direction=directions())
    @SETTINGS
    def test_a_ray_pointed_away_from_it_does_not(self, shape, centre, orientation,
                                                 direction):
        heading = unit(direction)
        if heading is None:
            return
        world = PhysicsWorld()
        index = world.add_shape(shape)
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=index),
                       position=tuple(centre), orientation=tuple(orientation))
        start = np.asarray(centre, dtype='d') - heading * 50.0
        assert raycast.raycast(world, start, -heading, max_distance=REACH) is None


class TestLineOfSight:
    @given(world=worlds(max_bodies=6), start=vectors(20.0), end=vectors(20.0))
    @SETTINGS
    def test_it_agrees_with_a_cast_between_the_two_points(self, world, start, end):
        gap = np.asarray(end, dtype='d') - np.asarray(start, dtype='d')
        span = float(np.linalg.norm(gap))
        if span < 1e-6:
            return
        clear = raycast.line_of_sight(world, start, end)
        blocker = raycast.raycast(world, start, gap, max_distance=span)
        assert clear == (blocker is None)
