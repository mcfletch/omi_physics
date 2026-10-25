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
