"""Phase 4 gravity zones: point pull, priority/replace, stop (no GL)."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld
from omi_physics.gravity import GravityVolume, SphereRegion

DT = 1.0 / 120.0


def test_point_gravity_pulls_body_toward_center():
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))   # no global gravity
    world.add_gravity_volume(GravityVolume(
        model.Gravity(type=model.POINT, gravity=9.81, center=(0, 0, 0)),
        SphereRegion((0, 0, 0), 100.0)))
    body = world.add_body(model.Motion(type=model.DYNAMIC), position=(10, 0, 0))
    for _ in range(120):
        world.step(DT)
    assert world.position[body][0] < 10.0                 # moved inward (−x)
    assert world.linear_velocity[body][0] < 0


def test_higher_priority_replace_overrides_global():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    world.add_gravity_volume(GravityVolume(
        model.Gravity(type=model.DIRECTIONAL, gravity=9.81, direction=(0, 1, 0),
                      priority=5, replace=True),
        SphereRegion((0, 0, 0), 100.0)))
    body = world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 0, 0))
    for _ in range(120):
        world.step(DT)
    assert world.linear_velocity[body][1] > 0             # pushed up, not down


def test_stop_zone_halts_gravity_accumulation():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    world.add_gravity_volume(GravityVolume(
        model.Gravity(gravity=0.0, priority=5, stop=True),
        SphereRegion((0, 0, 0), 100.0)))
    body = world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 0, 0))
    for _ in range(120):
        world.step(DT)
    assert np.allclose(world.linear_velocity[body], 0.0)  # weightless


def test_region_bounds_limit_effect():
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    world.add_gravity_volume(GravityVolume(
        model.Gravity(type=model.DIRECTIONAL, gravity=9.81, direction=(0, -1, 0)),
        SphereRegion((0, 0, 0), 1.0)))
    inside = world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 0, 0))
    outside = world.add_body(model.Motion(type=model.DYNAMIC), position=(50, 0, 0))
    for _ in range(120):
        world.step(DT)
    assert world.linear_velocity[inside][1] < -1.0
    assert np.allclose(world.linear_velocity[outside], 0.0)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
