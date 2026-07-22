"""Phase 6 joints: distance/point limits and motor drives (no GL)."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld
from omi_physics.joints import PointConstraint, DistanceConstraint, AngularMotor

DT = 1.0 / 120.0


def test_distance_constraint_keeps_pendulum_arm_length():
    """A point-mass bob on a rigid arm swings and preserves its length."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    pivot = (0.0, 5.0, 0.0)
    bob = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                         position=(2.0, 5.0, 0.0))     # 2 m arm, horizontal
    world.add_joint_constraint(DistanceConstraint(
        -1, bob, anchor_a=pivot, anchor_b=(0, 0, 0), length=2.0))
    lengths = []
    lowest = 5.0
    for _ in range(600):
        world.step(DT)
        lengths.append(np.linalg.norm(world.position[bob] - pivot))
        lowest = min(lowest, world.position[bob][1])
    assert max(lengths) == pytest.approx(2.0, abs=0.05)
    assert min(lengths) == pytest.approx(2.0, abs=0.05)
    assert lowest < 3.3                                 # swung down to near the bottom (y≈3)


def test_point_constraint_pins_extended_body_at_offset():
    """An extended (inertial) box pinned at a corner swings so its centre drops."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    shape = world.add_shape(model.Shape.box((2, 0.4, 0.4)))
    bar = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                         collider=model.Collider(shape=shape), position=(1.0, 5.0, 0.0))
    pivot = (0.0, 5.0, 0.0)                            # pin the bar's left end
    world.add_joint_constraint(PointConstraint(-1, bar, anchor=pivot))
    for _ in range(400):
        world.step(DT)
    pinned = world.position[bar] + _rotate(world, bar, (-1.0, 0.0, 0.0))
    assert np.allclose(pinned, pivot, atol=0.05)       # corner stays on the pivot
    assert world.position[bar][1] < 4.9                # the bar swung down


def _rotate(world, i, local):
    from omi_physics import mathutil
    R = mathutil.quat_to_matrix(world.orientation[i])
    return R @ np.asarray(local, dtype='d')


def test_distance_constraint_holds_between_two_bodies():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    anchor = world.add_body(model.Motion(type=model.STATIC), position=(0, 5, 0))
    hanging = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                             position=(0, 2, 0))       # 3 m below
    world.add_joint_constraint(DistanceConstraint(
        anchor, hanging, anchor_a=(0, 0, 0), anchor_b=(0, 0, 0), length=3.0))
    for _ in range(600):
        world.step(DT)
    dist = np.linalg.norm(world.position[hanging] - world.position[anchor])
    assert dist == pytest.approx(3.0, abs=0.05)


def test_angular_motor_reaches_velocity_target():
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), sleep_enabled=False)
    shape = world.add_shape(model.Shape.box((1, 1, 1)))
    body = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=shape), position=(0, 0, 0))
    world.add_joint_constraint(AngularMotor(body, axis=(0, 1, 0), target=3.0,
                                            max_force=50.0))
    for _ in range(240):
        world.step(DT)
    assert world.angular_velocity[body][1] == pytest.approx(3.0, abs=0.1)


def test_motor_force_limit_caps_spin_up():
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), sleep_enabled=False)
    shape = world.add_shape(model.Shape.box((1, 1, 1)))
    body = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=shape), position=(0, 0, 0))
    world.add_joint_constraint(AngularMotor(body, axis=(0, 1, 0), target=100.0,
                                            max_force=0.5))
    world.step(DT)
    assert world.angular_velocity[body][1] < 5.0       # force budget throttles it


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
