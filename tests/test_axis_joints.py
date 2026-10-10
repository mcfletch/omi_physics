"""Hinges, sliders, universal and two-axis hinges, ball and motor joints (no GL)."""
import numpy as np
import pytest

from omi_physics import jointrows, mathutil, model
from omi_physics.joints import (
    BallJoint,
    HingeJoint,
    MotorJoint,
    SliderJoint,
    TwoAxisHingeJoint,
    UniversalJoint,
)
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0
DOWN = model.Gravity(gravity=9.81, direction=(0, -1, 0))
WEIGHTLESS = model.Gravity(gravity=0.0)


def _world(gravity: model.Gravity = DOWN) -> PhysicsWorld:
    return PhysicsWorld(gravity=gravity, sleep_enabled=False)


def _box(world: PhysicsWorld, position, size=(1.0, 0.2, 0.2), mass=1.0, **motion) -> int:
    shape = world.add_shape(model.Shape.box(size))
    return world.add_body(model.Motion(mass=mass, **motion),
                          collider=model.Collider(shape=shape), position=position)


def _run(world: PhysicsWorld, seconds: float, each=None) -> None:
    for _ in range(round(seconds / DT)):
        world.step(DT)
        if each is not None:
            each()


def _turned(world: PhysicsWorld, body: int, local) -> np.ndarray:
    return mathutil.quat_to_matrix(world.orientation[body]) @ np.asarray(local, dtype='d')


@pytest.fixture(params=['native', 'python'])
def kernel(request, monkeypatch):
    """Each test runs with the compiled row solver and with its Python twin."""
    if request.param == 'python':
        monkeypatch.setattr(jointrows, '_native', None)
    elif jointrows._native is None:
        pytest.skip('the compiled row solver is not built')
    return request.param


class TestBallJoint:
    def test_the_anchor_is_held_while_the_body_swings(self, kernel):
        world = _world()
        bar = _box(world, (0.5, 0.0, 0.0))
        joint = world.add_joint_constraint(BallJoint(-1, bar, anchor=(0.0, 0.0, 0.0)))
        _run(world, 0.4)
        assert np.allclose(joint.anchor_b(world), (0, 0, 0), atol=0.01)
        assert world.position[bar][1] < -0.3

    def test_two_bodies_share_the_anchor(self, kernel):
        world = _world()
        top = _box(world, (0.5, 0.0, 0.0), mass=1000.0, gravityFactor=0.0)
        hung = _box(world, (1.5, 0.0, 0.0))
        joint = world.add_joint_constraint(BallJoint(top, hung, anchor=(1.0, 0.0, 0.0)))
        _run(world, 0.5)
        assert np.allclose(joint.anchor_a(world), joint.anchor_b(world), atol=0.01)


class TestHingeJoint:
    def test_it_swings_only_about_its_axis(self, kernel):
        world = _world()
        bar = _box(world, (0.5, 0.0, 0.0))
        world.angular_velocity[bar] = (3.0, 0.0, 0.0)       # a twist the hinge must remove
        joint = world.add_joint_constraint(
            HingeJoint(-1, bar, anchor=(0, 0, 0), axis=(0, 0, 1)))
        _run(world, 0.4)
        assert abs(world.position[bar][2]) < 0.01
        assert abs(_turned(world, bar, (0, 0, 1))[2] - 1.0) < 0.01
        assert np.allclose(joint.anchor_b(world), (0, 0, 0), atol=0.01)
        assert joint.angle(world) < -0.5

    def test_the_angle_and_its_rate(self, kernel):
        world = _world(WEIGHTLESS)
        bar = _box(world, (0.5, 0.0, 0.0))
        joint = world.add_joint_constraint(
            HingeJoint(-1, bar, anchor=(0, 0, 0), axis=(0, 0, 1)))
        world.angular_velocity[bar] = (0.0, 0.0, 1.0)
        world.linear_velocity[bar] = (0.0, 0.5, 0.0)
        _run(world, 0.5)
        assert joint.angle(world) == pytest.approx(0.5, abs=0.02)
        assert joint.angle_rate(world) == pytest.approx(1.0, abs=0.02)

    def test_a_stop_holds_the_swing(self, kernel):
        world = _world()
        bar = _box(world, (0.5, 0.0, 0.0))
        joint = world.add_joint_constraint(
            HingeJoint(-1, bar, anchor=(0, 0, 0), axis=(0, 0, 1), lower=-0.3, upper=0.3))
        _run(world, 2.0)
        assert joint.angle(world) == pytest.approx(-0.3, abs=0.03)

    def test_a_bouncing_stop_throws_the_swing_back(self, kernel):
        world = _world()
        bar = _box(world, (0.5, 0.0, 0.0))
        joint = world.add_joint_constraint(
            HingeJoint(-1, bar, anchor=(0, 0, 0), axis=(0, 0, 1), lower=-0.3, upper=0.3,
                       stop_bounce=1.0))
        rates = []
        _run(world, 0.6, each=lambda: rates.append(joint.angle_rate(world)))
        assert max(rates[len(rates) // 3:]) > 1.0

    def test_a_motor_drives_the_hinge(self, kernel):
        world = _world(WEIGHTLESS)
        wheel = _box(world, (0.0, 0.0, 0.0), size=(1, 1, 1))
        joint = world.add_joint_constraint(
            HingeJoint(-1, wheel, anchor=(0, 0, 0), axis=(1, 0, 0),
                       motor_speed=2.0, motor_torque=50.0))
        _run(world, 1.0)
        assert joint.angle_rate(world) == pytest.approx(2.0, abs=0.02)

    def test_a_weak_motor_takes_its_time(self, kernel):
        world = _world(WEIGHTLESS)
        wheel = _box(world, (0.0, 0.0, 0.0), size=(1, 1, 1))
        joint = world.add_joint_constraint(
            HingeJoint(-1, wheel, anchor=(0, 0, 0), axis=(1, 0, 0),
                       motor_speed=100.0, motor_torque=0.1))
        _run(world, 0.5)
        # 0.1 N m on 1/6 kg m^2 for half a second.
        assert joint.angle_rate(world) == pytest.approx(0.3, abs=0.02)


class TestSliderJoint:
    def test_it_slides_along_its_axis_and_does_not_turn(self, kernel):
        world = _world()
        down = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
        block = _box(world, (0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2))
        world.angular_velocity[block] = (1.0, 2.0, 3.0)
        joint = world.add_joint_constraint(SliderJoint(-1, block, axis=down))
        _run(world, 0.5)
        along = float(np.dot(world.position[block], down))
        across = world.position[block] - along * down
        assert along > 0.5
        assert np.linalg.norm(across) < 0.01
        assert np.linalg.norm(world.angular_velocity[block]) < 0.05
        assert joint.separation(world) == pytest.approx(along, abs=0.01)

    def test_stops_hold_the_separation(self, kernel):
        world = _world()
        block = _box(world, (0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2))
        joint = world.add_joint_constraint(
            SliderJoint(-1, block, axis=(0, 1, 0), lower=-0.25, upper=1.0))
        _run(world, 1.0)
        assert joint.separation(world) == pytest.approx(-0.25, abs=0.02)

    def test_the_force_pushes_the_bodies_apart(self, kernel):
        world = _world(WEIGHTLESS)
        left = _box(world, (-0.5, 0.0, 0.0), size=(0.2, 0.2, 0.2))
        right = _box(world, (0.5, 0.0, 0.0), size=(0.2, 0.2, 0.2))
        joint = world.add_joint_constraint(SliderJoint(left, right, axis=(1, 0, 0),
                                                       force=1.0))
        _run(world, 1.0)
        # 1 N on each 1 kg body for a second: each moves half a metre.
        assert joint.separation(world) == pytest.approx(1.0, abs=0.02)
        assert world.position[left][0] == pytest.approx(-1.0, abs=0.02)
        assert joint.separation_rate(world) == pytest.approx(2.0, abs=0.05)


class TestUniversalJoint:
    def test_a_spin_is_carried_through_the_cross(self, kernel):
        world = _world(WEIGHTLESS)
        driving = _box(world, (-0.5, 0.0, 0.0))
        driven = _box(world, (0.5, 0.0, 0.0))
        world.angular_velocity[driving] = (4.0, 0.0, 0.0)
        joint = world.add_joint_constraint(UniversalJoint(
            driving, driven, anchor=(0, 0, 0), axis1=(0, 1, 0), axis2=(0, 0, 1)))
        _run(world, 0.5)
        assert world.angular_velocity[driven][0] == pytest.approx(2.0, abs=0.1)
        assert abs(float(np.dot(joint.axis1(world), joint.axis2(world)))) < 0.01

    def test_it_bends_about_either_axis(self, kernel):
        world = _world(WEIGHTLESS)
        held = _box(world, (-0.5, 0.0, 0.0), gravityFactor=0.0, mass=1e6)
        hung = _box(world, (0.5, 0.0, 0.0))
        world.angular_velocity[hung] = (0.0, 1.0, 1.0)
        world.linear_velocity[hung] = (0.0, 0.5, -0.5)
        world.add_joint_constraint(UniversalJoint(
            held, hung, anchor=(0, 0, 0), axis1=(0, 1, 0), axis2=(0, 0, 1)))
        _run(world, 0.5)
        end = world.position[hung]
        assert end[1] > 0.1 and end[2] < -0.1


class TestTwoAxisHingeJoint:
    def _wheel(self, world, **settings):
        car = _box(world, (0.0, 1.0, 0.0), size=(2, 0.5, 1), mass=1e6, gravityFactor=0.0)
        wheel = world.add_body(model.Motion(mass=1.0), collider=model.Collider(
            shape=world.add_shape(model.Shape.sphere(0.3))), position=(1.0, 1.0, 0.0))
        joint = world.add_joint_constraint(TwoAxisHingeJoint(
            car, wheel, anchor=(1.0, 1.0, 0.0), axis1=(0, 1, 0), axis2=(0, 0, 1),
            **settings))
        return car, wheel, joint

    def test_the_wheel_spins_on_its_axle(self, kernel):
        world = _world(WEIGHTLESS)
        _, wheel, joint = self._wheel(world, motor2_speed=5.0, motor2_torque=10.0,
                                      lower1=0.0, upper1=0.0)
        _run(world, 1.0)
        assert joint.angle2_rate(world) == pytest.approx(5.0, abs=0.05)
        assert np.allclose(world.position[wheel], (1.0, 1.0, 0.0), atol=0.01)

    def test_steering_turns_the_axle(self, kernel):
        world = _world(WEIGHTLESS)
        _, _, joint = self._wheel(world, motor1_speed=0.5, motor1_torque=10.0,
                                  lower1=-0.4, upper1=0.4)
        _run(world, 2.0)
        assert joint.angle1(world) == pytest.approx(0.4, abs=0.03)
        turned = joint.axis2(world)
        assert turned[0] == pytest.approx(np.sin(0.4), abs=0.03)

    def test_the_suspension_gives_under_load(self, kernel):
        world = _world()
        _, wheel, _ = self._wheel(world, lower1=0.0, upper1=0.0,
                                  suspension_stiffness=200.0, suspension_damping=20.0)
        _run(world, 3.0)
        # 1 kg on 200 N/m sags 9.81 / 200 metres.
        assert world.position[wheel][1] == pytest.approx(1.0 - 9.81 / 200.0, abs=0.005)


class TestMotorJoint:
    def test_torques_turn_the_bodies_against_each_other(self, kernel):
        world = _world(WEIGHTLESS)
        one = _box(world, (-1.0, 0.0, 0.0), size=(1, 1, 1))
        two = _box(world, (1.0, 0.0, 0.0), size=(1, 1, 1))
        world.add_joint_constraint(MotorJoint(one, two, axes=[(0, 1, 0)], torques=[0.5]))
        _run(world, 1.0)
        # 0.5 N m on 1/6 kg m^2 for a second, equal and opposite.
        assert world.angular_velocity[two][1] == pytest.approx(3.0, abs=0.02)
        assert world.angular_velocity[one][1] == pytest.approx(-3.0, abs=0.02)

    def test_the_angle_about_each_axis(self, kernel):
        world = _world(WEIGHTLESS)
        one = _box(world, (-1.0, 0.0, 0.0), size=(1, 1, 1))
        two = _box(world, (1.0, 0.0, 0.0), size=(1, 1, 1))
        joint = world.add_joint_constraint(
            MotorJoint(one, two, axes=[(0, 1, 0), (1, 0, 0)], torques=[0.0, 0.0]))
        world.angular_velocity[two] = (0.0, 0.6, 0.0)
        _run(world, 0.5)
        assert joint.angle(world, 0) == pytest.approx(0.3, abs=0.01)
        assert joint.angle_rate(world, 0) == pytest.approx(0.6, abs=0.01)
        assert joint.angle(world, 1) == pytest.approx(0.0, abs=0.01)


def test_the_compiled_and_python_solvers_agree(monkeypatch):
    if jointrows._native is None:
        pytest.skip('the compiled row solver is not built')

    def swing():
        world = _world()
        bar = _box(world, (0.5, 0.0, 0.0))
        world.add_joint_constraint(
            HingeJoint(-1, bar, anchor=(0, 0, 0), axis=(0, 0, 1), lower=-0.5, upper=0.5,
                       stop_bounce=0.5))
        _run(world, 1.0)
        return world.position[bar].copy(), world.orientation[bar].copy()

    native = swing()
    monkeypatch.setattr(jointrows, '_native', None)
    python = swing()
    assert np.allclose(native[0], python[0], atol=1e-9)
    assert np.allclose(native[1], python[1], atol=1e-9)


def test_a_stiff_chain_does_not_sag_apart(kernel):
    """Three links hung end to end stay joined: warm starting carries the load."""
    world = _world()
    links = [_box(world, (0.5 + i, 0.0, 0.0)) for i in range(3)]
    joints = [world.add_joint_constraint(BallJoint(-1, links[0], anchor=(0, 0, 0)))]
    joints += [world.add_joint_constraint(BallJoint(links[i], links[i + 1],
                                                    anchor=(1.0 + i, 0, 0)))
               for i in range(2)]
    _run(world, 2.0)
    for joint in joints:
        assert np.linalg.norm(joint.anchor_a(world) - joint.anchor_b(world)) < 0.02


def test_jointed_bodies_pass_through_each_other_unless_told_to_collide():
    def overlap(collide_connected: bool) -> int:
        world = _world(WEIGHTLESS)
        one = _box(world, (0.0, 0.0, 0.0), size=(1, 1, 1))
        two = _box(world, (0.5, 0.0, 0.0), size=(1, 1, 1))
        joint = BallJoint(one, two, anchor=(0.25, 0.0, 0.0))
        joint.collide_connected = collide_connected
        world.add_joint_constraint(joint)
        world.step(DT)
        return len(world.contacts)

    assert overlap(False) == 0
    assert overlap(True) > 0


def test_a_removed_joint_holds_nothing():
    world = _world()
    bar = _box(world, (0.5, 0.0, 0.0))
    joint = world.add_joint_constraint(BallJoint(-1, bar, anchor=(0.0, 0.0, 0.0)))
    world.remove_joint_constraint(joint)
    _run(world, 0.5)
    assert world.position[bar][1] == pytest.approx(-0.5 * 9.81 * 0.25, abs=0.03)
    assert world.row_joints == []
