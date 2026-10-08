"""Compound shapes: several shapes held at offsets in one body's frame (no GL)."""
import numpy as np
import pytest

from omi_physics import mathutil, model, omi_gltf
from omi_physics.body import make_proxy, world_aabb
from omi_physics.collide import collide
from omi_physics.raycast import raycast
from omi_physics.world import PhysicsWorld, inertia_diagonal

GRAVITY = model.Gravity(gravity=9.81, direction=(0, -1, 0))
DT = 1.0 / 120.0
TURN_Z = tuple(mathutil.quat_from_axis_angle((0, 0, 1), np.pi / 2))


def _dumbbell() -> model.Shape:
    """Two unit spheres two metres either side of the origin along x."""
    return model.Shape.compound([
        model.ShapePart(model.Shape.sphere(0.5), position=(-2.0, 0.0, 0.0)),
        model.ShapePart(model.Shape.sphere(0.5), position=(2.0, 0.0, 0.0)),
    ])


def test_a_compound_is_built_from_parts():
    shape = _dumbbell()
    assert shape.type == 'compound'
    assert len(shape.parts) == 2
    assert shape.parts[1].position == (2.0, 0.0, 0.0)


def test_a_compound_accepts_shape_position_orientation_triples():
    shape = model.Shape.compound([(model.Shape.box((1, 1, 1)), (0, 1, 0), (0, 0, 0, 1))])
    assert isinstance(shape.parts[0], model.ShapePart)
    assert shape.parts[0].position == (0, 1, 0)


def test_the_bounds_of_a_compound_cover_its_parts_where_the_body_is():
    lo, hi = world_aabb(_dumbbell(), (10.0, 0.0, 0.0), (0, 0, 0, 1))
    assert np.allclose(lo, (7.5, -0.5, -0.5))
    assert np.allclose(hi, (12.5, 0.5, 0.5))


def test_the_parts_turn_with_the_body():
    lo, hi = world_aabb(_dumbbell(), (0.0, 0.0, 0.0), TURN_Z)
    assert np.allclose(lo, (-0.5, -2.5, -0.5))
    assert np.allclose(hi, (0.5, 2.5, 0.5))


def test_a_part_turned_in_the_body_is_turned_in_the_world():
    shape = model.Shape.compound([
        model.ShapePart(model.Shape.box((4.0, 1.0, 1.0)), orientation=TURN_Z)])
    lo, hi = world_aabb(shape, (0.0, 0.0, 0.0), (0, 0, 0, 1))
    assert np.allclose(hi - lo, (1.0, 4.0, 1.0))


def test_contacts_come_from_the_part_that_touches():
    ball = make_proxy(model.Shape.sphere(0.5), (2.0, 0.9, 0.0), (0, 0, 0, 1))
    contacts = collide(0, 1, make_proxy(_dumbbell(), (0, 0, 0), (0, 0, 0, 1)), ball)
    assert contacts
    assert all(c.point[0] > 1.0 for c in contacts)
    assert np.allclose(contacts[0].normal, (0, 1, 0), atol=1e-6)


def test_nothing_touches_the_space_between_the_parts():
    ball = make_proxy(model.Shape.sphere(0.5), (0.0, 0.0, 0.0), (0, 0, 0, 1))
    assert collide(0, 1, make_proxy(_dumbbell(), (0, 0, 0), (0, 0, 0, 1)), ball) == []


def test_compound_against_compound():
    near = make_proxy(_dumbbell(), (0, 0, 0), (0, 0, 0, 1))
    far = make_proxy(_dumbbell(), (4.0, 0.9, 0.0), (0, 0, 0, 1))
    contacts = collide(0, 1, near, far)
    assert contacts
    assert all(c.depth > 0 for c in contacts)


def test_a_compound_rests_on_the_ground_on_both_parts():
    world = PhysicsWorld(gravity=GRAVITY, sleep_enabled=False)
    floor = world.add_shape(model.Shape.box((20, 1, 20)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0, -0.5, 0))
    bell = world.add_body(model.Motion(mass=2.0),
                          collider=model.Collider(shape=world.add_shape(_dumbbell())),
                          position=(0, 2.0, 0))
    for _ in range(360):
        world.step(DT)
    assert world.position[bell][1] == pytest.approx(0.5, abs=0.03)
    # Level: neither end has sunk below the other.
    R = mathutil.quat_to_matrix(world.orientation[bell])
    assert abs((R @ np.array([1.0, 0.0, 0.0]))[1]) < 0.02


def test_a_ray_finds_a_part_and_misses_the_gap():
    world = PhysicsWorld()
    shape = world.add_shape(_dumbbell())
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape))
    hit = raycast(world, (2.0, 5.0, 0.0), (0, -1, 0))
    assert hit is not None
    assert hit.distance == pytest.approx(4.5)
    assert raycast(world, (0.0, 5.0, 0.0), (0, -1, 0)) is None


def test_the_inertia_of_a_compound_counts_where_its_mass_is():
    """Two balls far apart are much harder to turn end over end than about their axis."""
    diag = inertia_diagonal(_dumbbell(), 2.0)
    # Each ball: 1 kg at 2 m about y and z, plus its own 0.4 m r^2.
    own = 0.4 * 1.0 * 0.25
    assert diag[0] == pytest.approx(2 * own)
    assert diag[1] == pytest.approx(2 * (own + 4.0))
    assert diag[2] == pytest.approx(2 * (own + 4.0))


def test_the_mass_of_a_compound_is_shared_by_volume():
    big = model.Shape.box((2, 2, 2))
    small = model.Shape.box((1, 1, 1))
    shape = model.Shape.compound([model.ShapePart(big, position=(0, 0, 0)),
                                  model.ShapePart(small, position=(0, 10, 0))])
    diag = inertia_diagonal(shape, 9.0)
    # 8/9 of the mass in the big box, 1/9 at ten metres.
    expected_x = (8.0 / 12.0) * (4 + 4) + (1.0 / 12.0) * 2 + 1.0 * 100.0
    assert diag[0] == pytest.approx(expected_x)


def test_a_compound_is_not_written_as_one_omi_shape():
    document = omi_gltf.PhysicsDocument(shapes=[_dumbbell()])
    with pytest.raises(ValueError, match='compound'):
        omi_gltf.export_extensions(document)
