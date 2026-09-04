"""Properties the quaternion and vector helpers hold for any input.

:mod:`omi_physics.mathutil` is beneath everything else here -- the integrator
turns orientations with it, the solver builds each contact's inverse inertia
from it, the narrow phase places every proxy through it -- so an input it
mishandles surfaces somewhere far away as a body that has quietly become NaN.

What is asserted is the algebra: rotations compose, lengths survive them, the
matrix form and the quaternion form agree, and the routines that promise to
broadcast do.
"""
import numpy as np
import pytest

# Hypothesis is a CPython-only test dependency: the wheels it publishes do not
# cover the PyPy in the matrix, and it is a compiled package now, so a PyPy row
# would try to build it from source. What these files cover is arithmetic, which
# the rest of the suite covers by example on every interpreter.
pytest.importorskip('hypothesis', reason='hypothesis has no wheel for this interpreter')

from hypothesis import given
from hypothesis import strategies as st
from property_strategies import (
    SETTINGS,
    any_quaternions,
    rough_quaternions,
    unit_quaternions,
    vectors,
)

from omi_physics import mathutil

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


class TestNormalisationSurvivesAnything:
    @given(v=vectors())
    @SETTINGS
    def test_a_normalised_vector_is_finite(self, v):
        """Including the zero vector, which has no direction to return."""
        assert np.isfinite(mathutil.normalize(v)).all()

    @given(v=vectors())
    @SETTINGS
    def test_a_vector_worth_normalising_comes_back_unit(self, v):
        if np.linalg.norm(v) < 1e-6:
            return
        assert np.linalg.norm(mathutil.normalize(v)) == pytest.approx(1.0, rel=1e-9)

    @given(q=any_quaternions())
    @SETTINGS
    def test_normalising_a_quaternion_is_finite(self, q):
        assert np.isfinite(mathutil.quat_normalize(q)).all()

    @given(q=rough_quaternions())
    @SETTINGS
    def test_a_quaternion_worth_normalising_comes_back_unit(self, q):
        assert np.linalg.norm(mathutil.quat_normalize(q)) == pytest.approx(1.0, rel=1e-9)


class TestRotationsCompose:
    @given(a=unit_quaternions(), b=unit_quaternions())
    @SETTINGS
    def test_the_product_of_two_rotations_is_a_rotation(self, a, b):
        assert np.linalg.norm(mathutil.quat_mul(a, b)) == pytest.approx(1.0, rel=1e-9)

    @given(a=unit_quaternions(), b=unit_quaternions(), c=unit_quaternions())
    @SETTINGS
    def test_composition_is_associative(self, a, b, c):
        left = mathutil.quat_mul(mathutil.quat_mul(a, b), c)
        right = mathutil.quat_mul(a, mathutil.quat_mul(b, c))
        assert np.allclose(left, right, atol=1e-12)

    @given(q=unit_quaternions())
    @SETTINGS
    def test_identity_leaves_a_rotation_alone(self, q):
        assert np.allclose(mathutil.quat_mul(q, IDENTITY), q, atol=1e-12)
        assert np.allclose(mathutil.quat_mul(IDENTITY, q), q, atol=1e-12)

    @given(q=unit_quaternions())
    @SETTINGS
    def test_a_rotation_times_its_conjugate_is_the_identity(self, q):
        product = mathutil.quat_mul(q, mathutil.quat_conjugate(q))
        assert np.allclose(np.abs(product), IDENTITY, atol=1e-12)

    @given(q=unit_quaternions())
    @SETTINGS
    def test_conjugating_leaves_the_original_alone(self, q):
        """The caller's quaternion is theirs; the inverse is a fresh array."""
        before = q.copy()
        mathutil.quat_conjugate(q)
        assert np.array_equal(q, before)


class TestRotatingAVector:
    @given(q=unit_quaternions(), v=vectors())
    @SETTINGS
    def test_rotation_preserves_length(self, q, v):
        rotated = mathutil.quat_rotate(q, v)
        assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(v), rel=1e-9, abs=1e-9)

    @given(q=unit_quaternions(), v=vectors())
    @SETTINGS
    def test_rotating_back_returns_the_vector(self, q, v):
        there = mathutil.quat_rotate(q, v)
        back = mathutil.quat_rotate(mathutil.quat_conjugate(q), there)
        assert np.allclose(back, v, rtol=1e-9, atol=1e-9)

    @given(q=unit_quaternions(), v=vectors())
    @SETTINGS
    def test_the_matrix_form_agrees_with_the_quaternion_form(self, q, v):
        assert np.allclose(mathutil.quat_to_matrix(q) @ v,
                           mathutil.quat_rotate(q, v), rtol=1e-9, atol=1e-9)

    @given(q=unit_quaternions(), a=unit_quaternions(), v=vectors())
    @SETTINGS
    def test_rotating_twice_is_rotating_by_the_product(self, q, a, v):
        both = mathutil.quat_rotate(mathutil.quat_mul(q, a), v)
        stepwise = mathutil.quat_rotate(q, mathutil.quat_rotate(a, v))
        assert np.allclose(both, stepwise, rtol=1e-9, atol=1e-9)


class TestTheMatrixIsARotationMatrix:
    @given(q=rough_quaternions())
    @SETTINGS
    def test_it_is_orthonormal_even_from_an_unnormalised_quaternion(self, q):
        """`quat_to_matrix` normalises what it is given, so a caller who stored
        a quaternion through a float32 round trip still gets a rotation."""
        m = mathutil.quat_to_matrix(q)
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)

    @given(q=unit_quaternions())
    @SETTINGS
    def test_it_does_not_reflect(self, q):
        assert np.linalg.det(mathutil.quat_to_matrix(q)) == pytest.approx(1.0, rel=1e-9)


class TestAxisAngle:
    @given(q=unit_quaternions())
    @SETTINGS
    def test_the_axis_is_a_direction_and_the_angle_is_in_range(self, q):
        axis_angle = mathutil.quat_to_axis_angle(q)
        assert np.linalg.norm(axis_angle[:3]) == pytest.approx(1.0, rel=1e-9)
        assert 0.0 <= axis_angle[3] <= 2 * np.pi

    @pytest.mark.parametrize('angle', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    def test_a_barely_turned_body_still_gives_a_unit_axis(self, angle):
        """The angles a body spinning slowly produces every frame.

        Recovering the axis from ``sqrt(1 - w*w)`` loses most of its digits as
        ``w`` approaches one -- the subtraction cancels -- and the axis comes
        back a little short. ``threaded.py`` publishes this to whatever is
        drawing the world, which then has a rotation that scales what it turns.
        """
        axis_angle = mathutil.quat_to_axis_angle(
            mathutil.quat_from_axis_angle((0.0, 0.0, 1.0), angle))
        assert np.linalg.norm(axis_angle[:3]) == pytest.approx(1.0, rel=1e-12)
        assert axis_angle[3] == pytest.approx(angle, rel=1e-9)

    @given(q=unit_quaternions())
    @SETTINGS
    def test_the_round_trip_is_the_same_rotation(self, q):
        """A quaternion and its negation are the same rotation, so the round
        trip is checked by what it does to space rather than by its four numbers."""
        axis_angle = mathutil.quat_to_axis_angle(q)
        again = mathutil.quat_from_axis_angle(axis_angle[:3], axis_angle[3])
        assert np.allclose(mathutil.quat_to_matrix(again),
                           mathutil.quat_to_matrix(q), atol=1e-7)

    @given(axis=vectors(1.0),
           angle=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False))
    @SETTINGS
    def test_building_from_any_axis_gives_a_rotation(self, axis, angle):
        """A zero axis has no rotation to describe; what it must not do is
        produce a quaternion the integrator will spread through the world."""
        q = mathutil.quat_from_axis_angle(axis, angle)
        assert np.isfinite(q).all()


class TestIntegratingOrientation:
    @given(q=unit_quaternions(), omega=vectors(50.0),
           dt=st.floats(min_value=1e-4, max_value=0.1))
    @SETTINGS
    def test_the_orientation_stays_a_rotation(self, q, omega, dt):
        """However fast the body is spinning and however long the step."""
        assert np.linalg.norm(mathutil.quat_integrate(q, omega, dt)) == pytest.approx(1.0, rel=1e-9)

    @given(q=unit_quaternions(), dt=st.floats(min_value=1e-4, max_value=0.1))
    @SETTINGS
    def test_a_body_that_is_not_spinning_does_not_turn(self, q, dt):
        assert np.allclose(mathutil.quat_integrate(q, np.zeros(3), dt), q, atol=1e-12)

    @given(q=unit_quaternions(), axis=unit_quaternions(),
           speed=st.floats(min_value=0.0, max_value=20.0))
    @SETTINGS
    def test_small_steps_agree_with_one_larger_one(self, q, axis, speed):
        """The integrator is first order, so the two answers differ by a
        rounding-scale amount over a step a game would take, not by a rotation."""
        omega = mathutil.normalize(axis[:3]) * speed
        big = mathutil.quat_integrate(q, omega, 1.0 / 60.0)
        small = q
        for _ in range(10):
            small = mathutil.quat_integrate(small, omega, 1.0 / 600.0)
        assert np.allclose(mathutil.quat_to_matrix(big),
                           mathutil.quat_to_matrix(small), atol=5e-2)


class TestCrossProduct:
    @given(a=vectors(), b=vectors())
    @SETTINGS
    def test_it_matches_numpy(self, a, b):
        assert np.allclose(mathutil.cross3(a, b), np.cross(a, b), rtol=1e-12, atol=0.0)

    @given(a=vectors(), b=vectors())
    @SETTINGS
    def test_it_is_perpendicular_to_both(self, a, b):
        c = mathutil.cross3(a, b)
        scale = max(np.linalg.norm(a) * np.linalg.norm(b), 1.0)
        assert abs(np.dot(c, a)) <= 1e-9 * scale * max(np.linalg.norm(a), 1.0)
        assert abs(np.dot(c, b)) <= 1e-9 * scale * max(np.linalg.norm(b), 1.0)

    @given(a=vectors())
    @SETTINGS
    def test_a_vector_crossed_with_itself_vanishes(self, a):
        assert np.allclose(mathutil.cross3(a, a), 0.0, atol=1e-9)


class TestLengthOfOneVector:
    """``length`` and ``flat_length`` exist to be cheap, so what has to be held
    is that being cheap did not make them different: they stand in for
    ``numpy.linalg.norm`` in the controller's innermost loops, and a
    disagreement there is a body that steps somewhere else."""

    @given(v=vectors())
    @SETTINGS
    def test_it_matches_numpy(self, v):
        assert mathutil.length(v) == float(np.linalg.norm(v))

    @given(v=vectors())
    @SETTINGS
    def test_the_flat_one_measures_x_and_z(self, v):
        assert mathutil.flat_length(v) == float(np.linalg.norm(v[[0, 2]]))

    @given(v=vectors())
    @SETTINGS
    def test_the_flat_one_ignores_height(self, v):
        lifted = np.array([v[0], v[1] + 17.0, v[2]])
        assert mathutil.flat_length(lifted) == mathutil.flat_length(v)


class TestTheIdentityRotation:
    """``quat_to_matrix`` answers the identity without building it. Most of
    what it is asked for is a proxy being rebuilt at a new position and an
    unchanged orientation, so that path is the common one and has to give
    exactly what the general one would."""

    def test_it_is_the_identity_matrix(self):
        found = mathutil.quat_to_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
        assert np.array_equal(found, np.eye(3))

    def test_each_caller_gets_its_own(self):
        """A shared matrix would let one caller's write reach every other."""
        first = mathutil.quat_to_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
        first[0, 0] = 99.0
        second = mathutil.quat_to_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
        assert second[0, 0] == 1.0

    @given(q=unit_quaternions())
    @SETTINGS
    def test_every_other_rotation_still_goes_the_long_way(self, q):
        """The shortcut must not answer for a quaternion that is not identity."""
        found = mathutil.quat_to_matrix(np.asarray(q, dtype='d'))
        turned = mathutil.quat_rotate(np.asarray(q, dtype='d'),
                                      np.array([0.0, 1.0, 0.0]))
        assert np.allclose(found @ np.array([0.0, 1.0, 0.0]), turned, atol=1e-12)


class TestTheHelpersBroadcast:
    """The module's contract is that one call handles a whole world's worth of
    bodies, which is what lets the integrator turn every orientation in one op.
    A batch must therefore answer exactly what the same inputs answer one at a
    time."""

    @given(qs=st.lists(unit_quaternions(), min_size=1, max_size=8),
           v=vectors())
    @SETTINGS
    def test_rotating_a_batch_matches_rotating_each(self, qs, v):
        batch = np.array(qs)
        together = mathutil.quat_rotate(batch, np.broadcast_to(v, (len(qs), 3)))
        for i, q in enumerate(qs):
            assert np.allclose(together[i], mathutil.quat_rotate(q, v), atol=1e-12)

    @given(qs=st.lists(unit_quaternions(), min_size=1, max_size=8))
    @SETTINGS
    def test_a_batch_of_matrices_matches_one_at_a_time(self, qs):
        batch = mathutil.quat_to_matrix(np.array(qs))
        assert batch.shape == (len(qs), 3, 3)
        for i, q in enumerate(qs):
            assert np.allclose(batch[i], mathutil.quat_to_matrix(q), atol=1e-12)

    @given(qs=st.lists(unit_quaternions(), min_size=1, max_size=8),
           omega=vectors(20.0), dt=st.floats(min_value=1e-4, max_value=0.1))
    @SETTINGS
    def test_integrating_a_batch_matches_integrating_each(self, qs, omega, dt):
        batch = mathutil.quat_integrate(np.array(qs),
                                        np.broadcast_to(omega, (len(qs), 3)), dt)
        for i, q in enumerate(qs):
            assert np.allclose(batch[i], mathutil.quat_integrate(q, omega, dt), atol=1e-12)

    @given(qs=st.lists(unit_quaternions(), min_size=1, max_size=8))
    @SETTINGS
    def test_a_batch_of_axis_angles_matches_one_at_a_time(self, qs):
        batch = mathutil.quat_to_axis_angle(np.array(qs))
        for i, q in enumerate(qs):
            assert np.allclose(batch[i], mathutil.quat_to_axis_angle(q), atol=1e-12)
