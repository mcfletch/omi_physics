"""Cylinders and cones as flat-ended solids, not capsules (no GL)."""
import numpy as np
import pytest

from omi_physics import mathutil, model
from omi_physics.body import make_proxy, world_aabb
from omi_physics.collide import collide
from omi_physics.raycast import raycast
from omi_physics.world import PhysicsWorld, inertia_diagonal

GRAVITY = model.Gravity(gravity=9.81, direction=(0, -1, 0))
DT = 1.0 / 120.0
LEVEL = (0, 0, 0, 1)


def _cone(height: float = 2.0, radius: float = 1.0) -> model.Shape:
    return model.Shape(type='cylinder', height=height, radiusBottom=radius, radiusTop=0.0)


def test_a_cylinder_has_flat_ends():
    """A capsule's rounded end reaches past the height; a cylinder's does not."""
    lo, hi = world_aabb(model.Shape.cylinder(height=2.0, radius=0.5), (0, 0, 0), LEVEL)
    assert np.allclose(lo, (-0.5, -1.0, -0.5))
    assert np.allclose(hi, (0.5, 1.0, 0.5))


def test_a_cylinder_on_its_side_is_bounded_on_its_side():
    side = tuple(mathutil.quat_from_axis_angle((0, 0, 1), np.pi / 2))
    lo, hi = world_aabb(model.Shape.cylinder(height=2.0, radius=0.5), (0, 0, 0), side)
    assert np.allclose(hi - lo, (2.0, 1.0, 1.0))


def test_a_cone_narrows_to_its_tip():
    tip = make_proxy(_cone(), (0, 0, 0), LEVEL).support(np.array([0.0, 1.0, 0.0]))
    assert np.allclose(tip, (0.0, 1.0, 0.0))
    rim = make_proxy(_cone(), (0, 0, 0), LEVEL).support(np.array([1.0, -1.0, 0.0]))
    assert np.allclose(rim, (1.0, -1.0, 0.0))


def test_a_ball_just_past_the_corner_of_a_cylinder_misses_it():
    """Beyond the rim, where a capsule's round end would still reach."""
    cylinder = make_proxy(model.Shape.cylinder(height=2.0, radius=0.5), (0, 0, 0), LEVEL)
    ball = make_proxy(model.Shape.sphere(0.1), (0.0, 1.15, 0.0), LEVEL)
    assert collide(0, 1, cylinder, ball) == []
    touching = make_proxy(model.Shape.sphere(0.1), (0.0, 1.05, 0.0), LEVEL)
    contacts = collide(0, 1, cylinder, touching)
    assert contacts
    assert np.allclose(contacts[0].normal, (0, 1, 0), atol=1e-3)


def test_a_cylinder_stands_on_its_end():
    world = PhysicsWorld(gravity=GRAVITY, sleep_enabled=False)
    floor = world.add_shape(model.Shape.box((20, 1, 20)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0, -0.5, 0))
    can = world.add_body(model.Motion(mass=1.0), collider=model.Collider(
        shape=world.add_shape(model.Shape.cylinder(height=1.0, radius=0.4))),
        position=(0, 0.6, 0))
    for _ in range(240):
        world.step(DT)
    assert world.position[can][1] == pytest.approx(0.5, abs=0.03)


def test_a_ray_down_onto_a_cylinder_meets_its_flat_top():
    world = PhysicsWorld()
    shape = world.add_shape(model.Shape.cylinder(height=2.0, radius=0.5))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape))
    hit = raycast(world, (0.45, 5.0, 0.0), (0, -1, 0))
    assert hit is not None
    assert hit.distance == pytest.approx(4.0)
    assert np.allclose(hit.normal, (0, 1, 0))


def test_a_ray_across_a_cone_meets_its_slope():
    world = PhysicsWorld()
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=world.add_shape(_cone())))
    # At y = 0 the cone's radius is 0.5.
    hit = raycast(world, (5.0, 0.0, 0.0), (-1, 0, 0))
    assert hit is not None
    assert hit.distance == pytest.approx(4.5)
    assert hit.normal[0] > 0.8 and hit.normal[1] > 0.3
    assert raycast(world, (5.0, 1.2, 0.0), (-1, 0, 0)) is None


def test_the_inertia_of_a_cone():
    diag = inertia_diagonal(_cone(height=2.0, radius=1.0), 1.0)
    # About its own centre of volume: 3/10 m r^2 along the axis.
    assert diag[1] == pytest.approx(0.3)
