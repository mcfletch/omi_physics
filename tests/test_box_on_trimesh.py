"""A box resting on a static triangle mesh is held by its corners, and friction holds it."""
import math

import numpy as np
import pytest

from omi_physics import hull, model
from omi_physics.body import BoxProxy, make_proxy
from omi_physics.collide import box_face_contacts, collide, face_across_internal_edge
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0
IDENTITY = np.identity(3)


def slab_mesh():
    """A 6 by 0.4 by 4 slab as triangles, the way a scene's block is cooked."""
    points = np.array([[sx * 3, sy * 0.2, sz * 2] for sx in (-1, 1) for sy in (-1, 1)
                       for sz in (-1, 1)], dtype='d')
    vertices, faces = hull.convex_hull(points)
    return model.Shape.trimesh(vertices, faces)


def world_with_a_box_on_a_slab(degrees, friction=0.5, velocity=(0.0, 0.0, 0.0)):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    material = world.add_material(model.Material(staticFriction=friction, dynamicFriction=friction))
    angle = math.radians(degrees)
    turn = (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=world.add_shape(slab_mesh()),
                                           physicsMaterial=material),
                   position=(0, 0, 0), orientation=turn)
    up = np.array([-math.sin(angle), math.cos(angle), 0.0])
    box = world.add_body(
        model.Motion(type=model.DYNAMIC, mass=5.0, linearVelocity=velocity),
        collider=model.Collider(shape=world.add_shape(model.Shape.box((0.8, 0.8, 0.8))),
                                physicsMaterial=material),
        position=tuple(0.6 * up), orientation=turn)
    return world, box


def run(world, box, seconds=2.0):
    start = np.array(world.position[box])
    for _ in range(int(seconds / DT)):
        world.step(DT)
    return float(np.linalg.norm(world.position[box] - start))


@pytest.mark.parametrize('degrees', [0, 10, 20])
def test_a_box_holds_on_a_slope_gentler_than_its_friction(degrees):
    world, box = world_with_a_box_on_a_slab(degrees)
    assert run(world, box) < 0.2


def test_a_box_slides_on_a_slope_steeper_than_its_friction():
    world, box = world_with_a_box_on_a_slab(40)
    assert run(world, box, seconds=1.0) > 1.0


def test_friction_brings_a_sliding_box_to_rest_on_a_level_mesh():
    world, box = world_with_a_box_on_a_slab(0, velocity=(-2.0, 0.0, 0.0))
    run(world, box, seconds=2.0)
    assert float(np.linalg.norm(world.linear_velocity[box])) < 0.05


def test_the_four_corners_of_a_box_lying_on_a_triangle_are_its_contacts():
    triangle = np.array([(-5.0, 0.0, -5.0), (5.0, 0.0, -5.0), (0.0, 0.0, 5.0)])
    box = BoxProxy(np.array([0.0, 0.39, 0.0]), np.array([0.4, 0.4, 0.4]), IDENTITY)
    found = box_face_contacts(box, triangle, np.array([0.0, 1.0, 0.0]), True)
    assert len(found) == 4
    assert all(depth == pytest.approx(0.01) for _point, depth in found)
    assert all(point[1] == pytest.approx(0.0) for point, _depth in found)


def test_a_triangle_under_the_middle_of_a_face_has_no_corner_over_it():
    triangle = np.array([(-0.5, 0.0, -0.5), (0.5, 0.0, -0.5), (0.0, 0.0, 0.5)])
    box = BoxProxy(np.array([0.0, 1.9, 0.0]), np.array([2.0, 2.0, 2.0]), IDENTITY)
    assert box_face_contacts(box, triangle, np.array([0.0, 1.0, 0.0]), True) == []


def test_the_contacts_agree_with_the_mesh_being_either_body():
    mesh = make_proxy(slab_mesh(), (0, 0, 0), (0, 0, 0, 1))
    box = make_proxy(model.Shape.box((0.8, 0.8, 0.8)), (0.0, 0.39, 0.0), (0, 0, 0, 1))
    first = collide(0, 1, mesh, box)
    second = collide(0, 1, box, mesh)
    assert len(first) == len(second) >= 4
    assert np.allclose(first[0].normal, -second[0].normal)


def slope_mesh(angle, cells=4, size=10.0):
    """A plane tilted by ``angle`` about z, as a grid of coplanar triangles."""
    grid = np.linspace(-size, size, cells + 1)
    points = np.array([(x, x * math.tan(angle), z) for z in grid for x in grid], dtype='d')
    faces = []
    for j in range(cells):
        for i in range(cells):
            a = j * (cells + 1) + i
            b, c = a + 1, a + cells + 1
            faces += [(a, c, b), (b, c, c + 1)]
    return model.Shape.trimesh(points, np.array(faces, dtype='i'))


def world_with_a_box_across_a_mesh_edge(degrees):
    """A 1 m cube resting across the shared edge of two coplanar slope triangles."""
    angle = math.radians(degrees)
    world = PhysicsWorld()
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=world.add_shape(slope_mesh(angle))),
                   position=(0, 0, 0))
    turn = (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
    x0 = 0.3
    box = world.add_body(
        model.Motion(type=model.DYNAMIC),
        collider=model.Collider(shape=world.add_shape(model.Shape.box((1, 1, 1)))),
        position=(x0, x0 * math.tan(angle) + 0.5 / math.cos(angle) + 0.01, 0.2),
        orientation=turn)
    return world, box, np.array([-math.sin(angle), math.cos(angle), 0.0])


def test_a_box_across_a_shared_mesh_edge_takes_the_slopes_normal():
    world, _box, up = world_with_a_box_across_a_mesh_edge(20.0)
    normals = []
    for _ in range(4):
        world.step(1 / 60)
        normals.extend(contact.normal for contact in world.contacts)
    assert normals
    assert all(abs(float(np.dot(normal, up))) > 0.999 for normal in normals)


def test_a_box_across_a_shared_mesh_edge_holds_on_a_20_degree_slope():
    world, box, _up = world_with_a_box_across_a_mesh_edge(20.0)
    start = np.array(world.position[box])
    for _ in range(300):
        world.step(1 / 60)
    assert float(np.linalg.norm(world.position[box] - start)) < 0.02


def test_a_faces_diagonal_is_its_only_internal_edge():
    mesh = make_proxy(slab_mesh(), (0, 0, 0), (0, 0, 0, 1))
    assert mesh.internal_edges.shape == (12, 3)
    assert (mesh.internal_edges.sum(axis=1) == 1).all()


def test_triangles_repeating_their_vertices_find_the_same_internal_edges():
    shape = slope_mesh(math.radians(20.0), cells=1)
    unwelded = model.Shape.trimesh(shape.points[shape.indices].reshape(-1, 3),
                                   np.arange(6, dtype='i').reshape(2, 3))
    mesh = make_proxy(unwelded, (0, 0, 0), (0, 0, 0, 1))
    assert mesh.internal_edges.sum() == 2


def test_a_sideways_push_across_a_flat_edge_becomes_the_face_normal():
    triangle = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (5.0, 0.0, 0.0)])
    box = BoxProxy(np.array([-0.3, 0.49, 1.0]), np.array([0.5, 0.5, 0.5]), IDENTITY)
    sideways = np.array([-1.0, 0.0, 0.0])
    flat = face_across_internal_edge(box, triangle, sideways, np.array([True, False, False]))
    assert flat == pytest.approx([0.0, 1.0, 0.0])


def test_a_sideways_push_across_a_free_edge_or_a_crease_stands():
    triangle = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0), (5.0, 0.0, 0.0)])
    box = BoxProxy(np.array([-0.3, 0.49, 1.0]), np.array([0.5, 0.5, 0.5]), IDENTITY)
    sideways = np.array([-1.0, 0.0, 0.0])
    assert face_across_internal_edge(box, triangle, sideways,
                                     np.array([False, True, True])) is None
