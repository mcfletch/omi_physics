"""Unit tests for :class:`KinematicAnimator`.

A kinematic body is moved by the integrator from its *velocity* (see
``backend.integrate_positions`` / ``world.moving_mask``), never by gravity or
contact.  The animator turns a pose function ``t -> (position, orientation)`` into
the per-frame linear+angular velocity that makes the body track that pose — and,
because it is a real velocity, the contact solver **carries riders** (a marble on
an elevator rises with it).  This backs the demo's always-running elevators and
rotating arms.
"""
import numpy as np

from omi_physics import model, mathutil
from omi_physics.world import PhysicsWorld
from omi_physics.kinematic import KinematicAnimator


def _kinematic_box(world, size=(4, 0.5, 4), position=(0, 0, 0)):
    shape = world.add_shape(model.Shape.box(size))
    return world.add_body(model.Motion(type=model.KINEMATIC),
                          collider=model.Collider(shape=shape),
                          position=position)


def test_vertical_pose_sets_upward_velocity():
    world = PhysicsWorld(sleep_enabled=False)
    i = _kinematic_box(world, position=(0, 0, 0))
    # Rise at a constant 2 m/s.
    anim = KinematicAnimator(world, i, lambda t: ((0.0, 2.0 * t, 0.0),
                                                  (0.0, 0.0, 0.0, 1.0)))
    anim.update(0.1)
    assert np.allclose(world.linear_velocity[i], (0.0, 2.0, 0.0), atol=1e-6)


def test_position_tracks_the_pose_after_stepping():
    world = PhysicsWorld(sleep_enabled=False)
    i = _kinematic_box(world, position=(0, 0, 0))
    anim = KinematicAnimator(world, i, lambda t: ((0.0, 2.0 * t, 0.0),
                                                  (0.0, 0.0, 0.0, 1.0)))
    dt = 1 / 60.0
    for _ in range(30):
        anim.update(dt)
        world.step(dt)
    # After 0.5 s at 2 m/s it should be ~1.0 m up.
    assert abs(world.position[i][1] - 1.0) < 0.05


def test_rotating_pose_sets_angular_velocity_about_axis():
    world = PhysicsWorld(sleep_enabled=False)
    i = _kinematic_box(world, size=(6, 0.4, 0.4), position=(0, 1, 0))
    rate = 1.5  # rad/s about +Y

    def pose(t):
        return (0.0, 1.0, 0.0), tuple(mathutil.quat_from_axis_angle((0, 1, 0), rate * t))

    anim = KinematicAnimator(world, i, pose)
    anim.update(0.05)
    w = world.angular_velocity[i]
    assert np.allclose(w, (0.0, rate, 0.0), atol=1e-3)


def test_elevator_carries_a_resting_marble_upward():
    """A dynamic sphere sitting on a rising kinematic platform rises with it."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    plat = _kinematic_box(world, size=(6, 0.5, 6), position=(0, 0, 0))
    ball_shape = world.add_shape(model.Shape.sphere(0.5))
    ball = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=ball_shape),
                          position=(0, 0.75, 0))       # resting on the platform top

    anim = KinematicAnimator(world, plat, lambda t: ((0.0, 1.0 * t, 0.0),
                                                     (0.0, 0.0, 0.0, 1.0)))
    dt = 1 / 60.0
    for _ in range(120):
        anim.update(dt)
        world.step(dt)
    # Platform rose ~2 m; the ball should have been carried most of the way up,
    # not left behind on the ground.
    assert world.position[plat][1] > 1.5
    assert world.position[ball][1] > 1.2


def test_pose_function_may_return_position_only():
    """A 3-vector pose (no orientation) is treated as translation with no spin."""
    world = PhysicsWorld(sleep_enabled=False)
    i = _kinematic_box(world)
    anim = KinematicAnimator(world, i, lambda t: (1.0 * t, 0.0, 0.0))
    anim.update(0.1)
    assert np.allclose(world.linear_velocity[i], (1.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(world.angular_velocity[i], (0.0, 0.0, 0.0), atol=1e-6)
