"""A box against a triangle mesh, tested analytically rather than by GJK.

Anything box-shaped resting on a level -- a crate, a car's chassis, a piece of
debris -- is one dynamic box against a few dozen triangles of the ground, every
step. Running GJK/EPA once per candidate triangle costs more than everything
else in the step put together, and GJK is at its worst on exactly this shape:
big flat triangles, where the polytope expansion crawls.

The separating-axis test answers the same question directly, and answers it for
every candidate triangle in one pass.
"""
import math

import numpy as np
import pytest

from omi_physics.body import BoxProxy
from omi_physics.collide import box_triangle_batch

IDENTITY = np.identity(3)


def _box(centre=(0.0, 0.0, 0.0), half=(1.0, 1.0, 1.0), rotation=IDENTITY):
    return BoxProxy(np.asarray(centre, 'd'), np.asarray(half, 'd'),
                    np.asarray(rotation, 'd'))


def _floor(y=0.0, extent=10.0):
    """Two triangles making a square of ground."""
    return np.array([
        [(-extent, y, -extent), (extent, y, -extent), (-extent, y, extent)],
        [(extent, y, -extent), (extent, y, extent), (-extent, y, extent)],
    ], dtype='d')


def _turn(axis, angle):
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


class TestWhetherTheyTouch:
    def test_a_box_above_the_ground_does_not(self) -> None:
        hit, _, _, _ = box_triangle_batch(_box(centre=(0, 5, 0)), _floor())
        assert not hit.any()

    def test_a_box_resting_on_it_does(self) -> None:
        hit, _, _, _ = box_triangle_batch(_box(centre=(0, 0.9, 0)), _floor())
        assert hit.any()

    def test_a_box_beside_the_triangles_does_not(self) -> None:
        hit, _, _, _ = box_triangle_batch(_box(centre=(50, 0, 0)), _floor())
        assert not hit.any()

    def test_a_box_exactly_touching_is_not_a_penetration(self) -> None:
        hit, _, _, depths = box_triangle_batch(_box(centre=(0, 1.0, 0)), _floor())
        assert not hit.any() or float(depths.max()) < 1e-6

    def test_no_triangles_is_no_contact(self) -> None:
        hit, points, _normals, _depths = box_triangle_batch(
            _box(), np.zeros((0, 3, 3)))
        assert len(hit) == 0 and len(points) == 0

    def test_a_turned_box_reaches_further_down_than_a_square_one(self) -> None:
        """Standing on an edge, a two-unit cube reaches sqrt(2) below its
        centre rather than the one it reaches sitting flat."""
        rotation = _turn((1.0, 0.0, 0.0), math.pi / 4)
        assert box_triangle_batch(
            _box(centre=(0, 1.2, 0), rotation=rotation), _floor())[0].any()
        assert not box_triangle_batch(
            _box(centre=(0, 1.2, 0)), _floor())[0].any()


class TestWhichWayItIsPushed:
    def test_a_box_in_the_floor_is_pushed_up(self) -> None:
        hit, _, normals, _ = box_triangle_batch(_box(centre=(0, 0.5, 0)), _floor())
        pushed = normals[hit]
        assert len(pushed)
        assert all(float(normal[1]) > 0.9 for normal in pushed)

    def test_a_box_under_a_ceiling_is_pushed_down(self) -> None:
        """The side it is on decides, not the winding: a triangle soup does not
        promise one."""
        hit, _, normals, _ = box_triangle_batch(_box(centre=(0, -0.5, 0)), _floor())
        assert all(float(normal[1]) < -0.9 for normal in normals[hit])

    def test_a_box_against_a_wall_is_pushed_sideways(self) -> None:
        wall = np.array([
            [(0.0, -10, -10), (0.0, 10, -10), (0.0, -10, 10)],
            [(0.0, 10, -10), (0.0, 10, 10), (0.0, -10, 10)],
        ], dtype='d')
        hit, _, normals, _ = box_triangle_batch(_box(centre=(0.5, 0, 0)), wall)
        assert all(abs(float(normal[0])) > 0.9 for normal in normals[hit])

    def test_the_depth_is_how_far_in_it_is(self) -> None:
        hit, _, _, depths = box_triangle_batch(_box(centre=(0, 0.7, 0)), _floor())
        assert float(depths[hit].max()) == pytest.approx(0.3, abs=1e-6)

    def test_the_contact_is_on_the_surface_it_met(self) -> None:
        hit, points, _, _ = box_triangle_batch(_box(centre=(0, 0.6, 0)), _floor())
        for point in points[hit]:
            assert abs(float(point[1])) < 1e-6            # on the floor plane
            assert abs(float(point[0])) <= 1.001          # under the box


class TestItAgreesWithTheGeneralTest:
    """The analytic answer must be the answer GJK would have given."""

    @pytest.mark.parametrize('height', [0.4, 0.7, 0.95, 1.05, 1.4])
    def test_touching_or_not_matches_gjk(self, height) -> None:
        from omi_physics.body import TriangleProxy
        from omi_physics.gjk import collide_convex
        box = _box(centre=(0.3, height, -0.2))
        floor = _floor()
        analytic = box_triangle_batch(box, floor)[0].any()
        general = any(collide_convex(0, 1, box, TriangleProxy(*triangle))
                      for triangle in floor)
        assert bool(analytic) == bool(general)

    def test_a_resting_box_settles_at_the_same_height(self) -> None:
        """Through the world, which is what actually matters."""
        from omi_physics import model
        from omi_physics.world import PhysicsWorld

        world = PhysicsWorld()
        floor = _floor(extent=20.0)
        points = floor.reshape(-1, 3)
        faces = np.arange(len(points), dtype='i').reshape(-1, 3)
        shape = world.add_shape(model.Shape.trimesh(points, faces))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=shape))
        box = world.add_shape(model.Shape.box((2.0, 2.0, 2.0)))
        falling = world.add_body(model.Motion(type=model.DYNAMIC, mass=10.0),
                                 collider=model.Collider(shape=box),
                                 position=(0.0, 4.0, 0.0))
        for _ in range(360):
            world.step(1.0 / 120.0)
        assert float(world.position[falling][1]) == pytest.approx(1.0, abs=0.15)
