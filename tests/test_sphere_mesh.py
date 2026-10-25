"""A sphere against a triangle mesh: every touching triangle tested at once, as a capsule's are.

The contacts are held against GJK and EPA one triangle at a time, and against
the nearest point of each triangle, which is exact where EPA's tolerance on the
sphere's curve is the larger error.
"""
import numpy as np
import pytest

from omi_physics import collide, gjk, model
from omi_physics.body import SphereProxy, TriangleMeshProxy
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0


def bumpy_mesh(seed: int, cells: int = 6, size: float = 6.0, bowl: float = 0.0) -> TriangleMeshProxy:
    """A grid of random heights, raised toward its rim by ``bowl`` times the squared distance."""
    rng = np.random.default_rng(seed)
    n = cells + 1
    xs = np.linspace(-size / 2, size / 2, n)
    grid_x, grid_z = np.meshgrid(xs, xs)
    heights = rng.uniform(-0.3, 0.3, grid_x.shape) + bowl * (grid_x ** 2 + grid_z ** 2)
    points = np.stack([grid_x.ravel(), heights.ravel(), grid_z.ravel()], axis=1)
    index = np.arange(n * n).reshape(n, n)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    triangles = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    return TriangleMeshProxy(points, triangles, np.zeros(3), np.eye(3))


@pytest.mark.parametrize('seed', range(8))
def test_a_sphere_on_a_mesh_touches_the_triangles_gjk_finds_as_deep(seed):
    rng = np.random.default_rng(100 + seed)
    mesh = bumpy_mesh(seed)
    for _ in range(60):
        centre = rng.uniform((-2.5, -0.2, -2.5), (2.5, 0.8, 2.5))
        sphere = SphereProxy(centre, float(rng.uniform(0.1, 0.6)))
        found = collide.collide(0, 1, sphere, mesh)
        lo, hi = sphere.aabb()
        expected = [contact for triangle in mesh.triangles_overlapping(lo, hi)
                    for contact in gjk.collide_convex(0, 1, sphere, triangle)]
        # Where the centre has passed below a triangle's plane the two part ways
        # by design (a capsule is pushed back to the side it is on), so the
        # comparison is made where the centre is above every triangle it touches.
        if any(np.dot(-c.normal, centre - c.point) < 0.0 for c in expected):
            continue
        # EPA stops within its tolerance of the sphere's curve, which off a
        # triangle's edge or corner is some 1e-4 m; the exact depths are held
        # in the test below.
        deep = sorted(c.depth for c in expected if c.depth > 1e-3)
        mine = sorted(c.depth for c in found if c.depth > 1e-3)
        assert mine == pytest.approx(deep, abs=5e-4)


def test_a_sphere_contact_is_pushed_from_the_nearest_point_of_the_triangle():
    rng = np.random.default_rng(7)
    mesh = bumpy_mesh(3)
    checked = 0
    for _ in range(200):
        centre = rng.uniform((-2.5, 0.0, -2.5), (2.5, 0.6, 2.5))
        sphere = SphereProxy(centre, 0.4)
        for contact in collide.collide(0, 1, sphere, mesh):
            # The point lies halfway between the sphere's deepest point and the
            # triangle, as EPA places it; the triangle's nearest point is
            # half the depth back along the normal.
            on_triangle = contact.point - contact.normal * (0.5 * contact.depth)
            gap = centre - on_triangle
            assert np.linalg.norm(gap) == pytest.approx(0.4 - contact.depth, abs=1e-9)
            assert np.allclose(-contact.normal, gap / np.linalg.norm(gap), atol=1e-9)
            checked += 1
    assert checked > 50


def test_a_sphere_on_a_mesh_asks_gjk_nothing(monkeypatch):
    asked = []
    original = gjk.collide_convex
    monkeypatch.setattr(gjk, 'collide_convex', lambda *args: asked.append(args) or original(*args))
    collide.collide(0, 1, SphereProxy((0.0, 0.2, 0.0), 0.5), bumpy_mesh(1))
    assert asked == []


def test_balls_dropped_on_a_mesh_come_to_rest_on_it():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    mesh = bumpy_mesh(2, cells=12, bowl=0.2)
    shape = world.add_shape(model.Shape.trimesh(mesh.world_pts, mesh.indices))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape))
    ball = world.add_shape(model.Shape.sphere(0.25))
    balls = [world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                            collider=model.Collider(shape=ball), position=(x, 2.2, z))
             for x in (-1.5, 0.0, 1.5) for z in (-1.5, 0.0, 1.5)]
    for _ in range(1200):
        world.step(DT)
    for body in balls:
        assert np.abs(world.position[body][[0, 2]]).max() < 3.0
        assert -0.3 < world.position[body][1] < 2.5
        assert np.abs(world.linear_velocity[body]).max() < 0.05
