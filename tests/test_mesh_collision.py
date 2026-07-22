"""Phase 3 mesh collision: sphere/convex onto a trimesh floor (no GL)."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0


def trimesh_floor(world, y=0.0, extent=6.0):
    e = extent
    pts = np.array([(-e, y, -e), (e, y, -e), (e, y, e), (-e, y, e)], dtype='d')
    idx = np.array([(0, 1, 2), (0, 2, 3)], dtype='i')
    shape = world.add_shape(model.Shape.trimesh(pts, idx))
    return world.add_body(model.Motion(type=model.STATIC),
                          collider=model.Collider(shape=shape), position=(0, 0, 0))


def test_sphere_rests_on_trimesh_floor():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    trimesh_floor(world, y=0.0)
    s = world.add_shape(model.Shape.sphere(0.5))
    ball = world.add_body(model.Motion(type=model.DYNAMIC),
                          collider=model.Collider(shape=s), position=(0, 3.0, 0))
    for _ in range(600):
        world.step(DT)
    assert world.position[ball][1] == pytest.approx(0.5, abs=0.08)
    assert abs(world.linear_velocity[ball][1]) < 0.3


def test_convex_prop_rests_on_trimesh_floor():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    trimesh_floor(world, y=0.0)
    corners = np.array([(sx, sy, sz) for sx in (-0.5, 0.5) for sy in (-0.5, 0.5)
                        for sz in (-0.5, 0.5)], dtype='d')
    c = world.add_shape(model.Shape.convex(corners))
    prop = world.add_body(model.Motion(type=model.DYNAMIC),
                          collider=model.Collider(shape=c), position=(0, 2.5, 0))
    for _ in range(700):
        world.step(DT)
    assert world.position[prop][1] == pytest.approx(0.5, abs=0.12)
    assert abs(world.linear_velocity[prop][1]) < 0.4


def test_sphere_misses_hole_falls_through_gap():
    """A ball outside the floor extent keeps falling (sanity: no phantom floor)."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    trimesh_floor(world, y=0.0, extent=1.0)
    s = world.add_shape(model.Shape.sphere(0.3))
    ball = world.add_body(model.Motion(type=model.DYNAMIC),
                          collider=model.Collider(shape=s), position=(5.0, 1.0, 0))
    for _ in range(200):
        world.step(DT)
    assert world.position[ball][1] < -1.0


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
