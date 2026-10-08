"""Which pairs collide, and with what bounce, friction and give (no GL)."""
import numpy as np
import pytest

from omi_physics import model, solver
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0
DOWN = model.Gravity(gravity=9.81, direction=(0, -1, 0))


@pytest.fixture(params=['native', 'python'])
def kernel(request, monkeypatch):
    """Each test runs with the compiled contact solver and with its Python twin."""
    if request.param == 'python':
        monkeypatch.setattr(solver, '_native', None)
    elif solver._native is None:
        pytest.skip('the compiled contact solver is not built')
    return request.param


def _floor_world(rule=None) -> tuple[PhysicsWorld, int]:
    world = PhysicsWorld(gravity=DOWN, sleep_enabled=False)
    floor = world.add_shape(model.Shape.box((40, 1, 40)))
    ground = world.add_body(model.Motion(type=model.STATIC),
                            collider=model.Collider(shape=floor), position=(0, -0.5, 0))
    if rule is not None:
        world.set_pair_rule(rule)
    return world, ground


def _ball(world: PhysicsWorld, height: float = 1.0) -> int:
    shape = world.add_shape(model.Shape.sphere(0.25))
    return world.add_body(model.Motion(mass=1.0), collider=model.Collider(shape=shape),
                          position=(0, height, 0))


def _crate(world: PhysicsWorld, velocity=(0, 0, 0)) -> int:
    shape = world.add_shape(model.Shape.box((0.5, 0.5, 0.5)))
    return world.add_body(model.Motion(mass=1.0, linearVelocity=velocity),
                          collider=model.Collider(shape=shape), position=(0, 0.25, 0))


def _run(world: PhysicsWorld, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        world.step(DT)


def test_a_rule_saying_no_lets_the_pair_pass(kernel):
    world, _ = _floor_world(lambda a, b: False)
    ball = _ball(world)
    _run(world, 1.0)
    assert world.position[ball][1] < -2.0


def test_a_rule_saying_nothing_leaves_the_materials_in_charge(kernel):
    world, _ = _floor_world(lambda a, b: None)
    ball = _ball(world)
    _run(world, 1.0)
    assert world.position[ball][1] == pytest.approx(0.25, abs=0.02)


def test_the_rule_is_asked_with_the_pair(kernel):
    asked = []

    def rule(a, b):
        asked.append((a, b))

    world, ground = _floor_world(rule)
    ball = _ball(world, height=0.2)
    world.step(DT)
    assert (min(ground, ball), max(ground, ball)) in asked


def test_full_bounce(kernel):
    world, _ = _floor_world(lambda a, b: model.ContactParameters(restitution=1.0,
                                                                 bounce_speed=0.1))
    ball = _ball(world, height=1.25)
    highest_after = []

    def peak():
        highest_after.append(world.position[ball][1])

    for step in range(round(1.5 / DT)):
        world.step(DT)
        if step > round(0.5 / DT):
            peak()
    assert max(highest_after) > 1.1


def test_no_bounce_below_the_bounce_speed(kernel):
    world, _ = _floor_world(lambda a, b: model.ContactParameters(restitution=1.0,
                                                                 bounce_speed=100.0))
    ball = _ball(world, height=1.25)
    _run(world, 1.5)
    assert world.position[ball][1] == pytest.approx(0.25, abs=0.02)


def test_friction_in_one_direction_only(kernel):
    rule = lambda a, b: model.ContactParameters(  # noqa: E731 a rule is a small function
        friction=(2.0, 0.0), friction_direction=(1.0, 0.0, 0.0))
    world, _ = _floor_world(rule)
    crate = _crate(world, velocity=(2.0, 0.0, 2.0))
    _run(world, 1.0)
    assert abs(world.linear_velocity[crate][0]) < 0.05
    assert world.linear_velocity[crate][2] == pytest.approx(2.0, abs=0.05)


def test_a_moving_surface_carries_what_rests_on_it(kernel):
    rule = lambda a, b: model.ContactParameters(  # noqa: E731 a rule is a small function
        friction=1.0, friction_direction=(1.0, 0.0, 0.0), surface_speed=(1.5, 0.0))
    world, _ = _floor_world(rule)
    crate = _crate(world)
    _run(world, 1.0)
    # The floor is body a or b of the pair; the crate slides one way or the other.
    assert abs(world.linear_velocity[crate][0]) == pytest.approx(1.5, abs=0.05)
    assert abs(world.linear_velocity[crate][2]) < 0.01


def test_a_soft_contact_gives_more_than_a_hard_one(kernel):
    def rest(softness):
        world, _ = _floor_world(lambda a, b: model.ContactParameters(softness=softness,
                                                                     error_correction=0.2))
        crate = _crate(world)
        _run(world, 1.0)
        return world.position[crate][1]

    assert rest(0.05) < rest(0.0) - 0.005


def test_without_error_correction_an_overlap_stays(kernel):
    world, _ = _floor_world(lambda a, b: model.ContactParameters(error_correction=0.0))
    world.gravity = model.Gravity(gravity=0.0)
    crate = _crate(world)
    world.place_body(crate, (0.0, 0.15, 0.0))
    _run(world, 0.5)
    assert world.position[crate][1] == pytest.approx(0.15, abs=0.005)


def test_the_compiled_and_python_solvers_agree_on_parameters(monkeypatch):
    if solver._native is None:
        pytest.skip('the compiled contact solver is not built')

    def slide():
        world, _ = _floor_world(lambda a, b: model.ContactParameters(
            restitution=0.5, bounce_speed=0.2, friction=(0.8, 0.3),
            friction_direction=(1.0, 0.0, 1.0), surface_speed=(0.2, -0.1),
            slip=(0.01, 0.02), softness=0.001, error_correction=0.5))
        crate = _crate(world, velocity=(1.0, -1.0, 0.5))
        world.place_body(crate, (0.0, 0.5, 0.0))
        _run(world, 1.0)
        return world.position[crate].copy(), world.orientation[crate].copy()

    native = slide()
    monkeypatch.setattr(solver, '_native', None)
    python = slide()
    assert np.allclose(native[0], python[0], atol=1e-9)
    assert np.allclose(native[1], python[1], atol=1e-9)
