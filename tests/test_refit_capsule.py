"""A capsule's box is derived from its pose, as a box's is, and is the box its proxy measures."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics import world as world_module
from omi_physics.body import world_aabb
from omi_physics.mathutil import quat_from_axis_angle
from omi_physics.world import PhysicsWorld


def capsule_world(turn):
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    shape = model.Shape.capsule(height=1.2, radius=0.3)
    body = world.add_body(model.Motion(type=model.KINEMATIC, linearVelocity=(1.0, 0.5, 0.0),
                                       angularVelocity=(0.3, 0.2, 0.1)),
                          collider=model.Collider(shape=world.add_shape(shape)),
                          position=(1.0, 2.0, 3.0), orientation=turn)
    return world, body, shape


@pytest.mark.parametrize('seed', range(6))
def test_a_capsule_box_is_the_one_its_proxy_measures(seed):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    world, body, shape = capsule_world(quat_from_axis_angle(axis / np.linalg.norm(axis),
                                                           float(rng.uniform(0, 6))))
    for _ in range(10):
        world.step(1 / 60)
        world.refit_aabbs()
        low, high = world_aabb(shape, world.position[body], world.orientation[body])
        assert world.aabb_min[body] == pytest.approx(low - 0.05, abs=1e-12)
        assert world.aabb_max[body] == pytest.approx(high + 0.05, abs=1e-12)


def test_a_moving_capsule_is_not_measured_through_its_proxy(monkeypatch):
    world, _body, _shape = capsule_world((0.0, 0.0, 0.0, 1.0))
    measured = []
    real = world_module.world_aabb
    monkeypatch.setattr(world_module, 'world_aabb',
                        lambda *a, **k: measured.append(1) or real(*a, **k))
    for _ in range(10):
        world.step(1 / 60)
    assert measured == []


@pytest.mark.parametrize('seed', range(4))
def test_a_few_bodies_fitted_with_numbers_have_the_boxes_many_are_given(seed):
    """Boxes, spheres and capsules, placed one at a time and refitted a few at a time."""
    rng = np.random.default_rng(seed)
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    shapes = [world.add_shape(model.Shape.box(tuple(rng.uniform(0.1, 2.0, 3)))),
              world.add_shape(model.Shape.sphere(0.7)),
              world.add_shape(model.Shape.capsule(height=1.5, radius=0.2))]
    bodies = []
    for _ in range(12):
        axis = rng.normal(size=3)
        turn = quat_from_axis_angle(axis / np.linalg.norm(axis), float(rng.uniform(0, 6)))
        bodies.append(world.add_body(model.Motion(type=model.KINEMATIC),
                                     collider=model.Collider(shape=int(rng.choice(shapes))),
                                     position=tuple(rng.uniform(-5, 5, 3)), orientation=turn))
    world.refit_aabbs()
    many = world.aabb_min.copy(), world.aabb_max.copy()
    for body in bodies:
        world.place_body(body, position=world.position[body])
    assert world.aabb_min == pytest.approx(many[0], abs=1e-12)
    assert world.aabb_max == pytest.approx(many[1], abs=1e-12)
    for start in range(0, 12, 3):
        world.refit_aabbs(rows=np.array(bodies[start:start + 3]))
    assert world.aabb_min == pytest.approx(many[0], abs=1e-12)
    assert world.aabb_max == pytest.approx(many[1], abs=1e-12)
