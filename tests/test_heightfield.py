"""Height-field shapes: a static ground given as a grid or a function.

``Shape.heightfield`` holds a surface ``y = h(x, z)`` in the body's frame.
Spheres, capsules, boxes, cylinders and convex hulls collide with it, and rays
are cast at it. The cases here drop real bodies onto slopes of up to 60 degrees
and compare where they stop with the plane the slope was built from.
"""
import math

import numpy as np
import pytest

from omi_physics import collide, mathutil, model
from omi_physics.body import make_proxy
from omi_physics.heightfield import GridSurface
from omi_physics.raycast import raycast, raycast_many
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0
RNG_SEED = 20261025


def tilted_plane(angle_deg: float, heading: float, size: float = 200.0
                 ) -> tuple[model.Shape, float, float]:
    """A grid for the plane ``y = a x + b z`` rising ``angle_deg`` towards ``heading``.

    Returns the shape and the slopes ``a`` and ``b``.
    """
    slope = math.tan(math.radians(angle_deg))
    a, b = slope * math.cos(heading), slope * math.sin(heading)
    axis = np.arange(-size / 2, size / 2 + 1.0)
    x, z = np.meshgrid(axis, axis)
    shape = model.Shape.heightfield(a * x + b * z, origin=(-size / 2, 0.0, -size / 2),
                                    spacing=1.0)
    return shape, a, b


def plane_distance(point: np.ndarray, a: float, b: float) -> float:
    """Signed distance from ``point`` to the plane ``y = a x + b z``, positive above."""
    return float((point[1] - a * point[0] - b * point[2]) / math.sqrt(1 + a * a + b * b))


def ground_world(shape: model.Shape, position=(0.0, 0.0, 0.0), orientation=(0, 0, 0, 1),
                 friction: float = 3.0) -> tuple[PhysicsWorld, int]:
    """A world holding ``shape`` as static ground made of a grippy material."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), fixed_dt=DT)
    material = world.add_material(model.Material(staticFriction=friction,
                                                 dynamicFriction=friction))
    index = world.add_shape(shape)
    ground = world.add_body(model.Motion(type=model.STATIC),
                            collider=model.Collider(shape=index, physicsMaterial=material),
                            position=position, orientation=orientation)
    return world, ground


def drop(world: PhysicsWorld, shape: model.Shape, position, orientation=(0, 0, 0, 1),
         friction: float = 3.0) -> int:
    material = world.add_material(model.Material(staticFriction=friction,
                                                 dynamicFriction=friction))
    index = world.add_shape(shape)
    return world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                          collider=model.Collider(shape=index, physicsMaterial=material),
                          position=position, orientation=orientation)


def corners(size) -> np.ndarray:
    half = 0.5 * np.asarray(size, dtype='d')
    return np.array([(sx * half[0], sy * half[1], sz * half[2])
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def world_points(world: PhysicsWorld, body: int, local: np.ndarray) -> np.ndarray:
    rotation = mathutil.quat_to_matrix(world.orientation[body])
    return local @ rotation.T + world.position[body]


class TestGridSurface:
    def test_height_at_a_node_is_the_node(self) -> None:
        shape = model.Shape.heightfield([[0.0, 1.0], [2.0, 3.0]], spacing=2.0)
        surface = shape.surface
        assert surface.heights(np.array([0.0, 2.0]), np.array([0.0, 2.0])) == pytest.approx(
            [0.0, 3.0])
        assert shape.type == 'heightfield'

    def test_height_between_nodes_is_bilinear(self) -> None:
        shape = model.Shape.heightfield([[0.0, 1.0], [2.0, 3.0]])
        surface = shape.surface
        assert surface.heights(np.array([0.5]), np.array([0.5]))[0] == pytest.approx(1.5)
        assert surface.heights(np.array([0.25]), np.array([0.75]))[0] == pytest.approx(
            0.25 * 1.0 + 0.75 * 2.0)

    def test_rows_run_along_z_and_columns_along_x(self) -> None:
        shape = model.Shape.heightfield([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
                                        origin=(10.0, 100.0, -3.0), spacing=(1.0, 4.0))
        surface = shape.surface
        # The second row starts 4 m along z from the origin, and the origin's y is added.
        assert surface.heights(np.array([11.0]), np.array([1.0]))[0] == pytest.approx(105.0)
        assert surface.heights(np.array([11.0]), np.array([-3.0]))[0] == pytest.approx(100.0)

    def test_normal_of_a_plane_is_exact(self) -> None:
        shape, a, b = tilted_plane(35.0, 1.1, size=10.0)
        x = np.array([-3.3, 0.2, 4.4])
        z = np.array([2.1, -1.7, 0.4])
        expected = np.array([-a, 1.0, -b]) / math.sqrt(1 + a * a + b * b)
        assert shape.surface.normals(x, z) == pytest.approx(np.tile(expected, (3, 1)))

    def test_normal_follows_the_cell(self) -> None:
        shape = model.Shape.heightfield([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        flat = shape.surface.normals(np.array([0.5]), np.array([0.5]))[0]
        steep = shape.surface.normals(np.array([1.9]), np.array([1.0]))[0]
        assert flat == pytest.approx([0.0, 1.0, 0.0])
        assert steep[0] < -0.5

    def test_outside_the_grid_there_is_no_surface(self) -> None:
        shape = model.Shape.heightfield(np.zeros((3, 3)))
        heights = shape.surface.heights(np.array([-0.1, 2.1, 1.0, 1.0]),
                                        np.array([1.0, 1.0, -0.1, 2.1]))
        assert np.all(np.isneginf(heights))

    def test_bounds_hold_the_grid(self) -> None:
        shape = model.Shape.heightfield([[1.0, 2.0], [3.0, 7.0]], origin=(1, 10, 2), spacing=2)
        low, high = shape.surface.aabb()
        assert low == pytest.approx([1, 11, 2])
        assert high == pytest.approx([3, 17, 4])

    @pytest.mark.parametrize('heights', [np.zeros(4), np.zeros((1, 4)), np.zeros((4, 1))])
    def test_a_grid_needs_two_rows_and_two_columns(self, heights) -> None:
        with pytest.raises(ValueError, match='2 rows'):
            model.Shape.heightfield(heights)

    def test_heights_must_be_finite_and_spacing_positive(self) -> None:
        with pytest.raises(ValueError, match='finite'):
            model.Shape.heightfield([[0.0, np.nan], [0.0, 0.0]])
        with pytest.raises(ValueError, match='spacing'):
            model.Shape.heightfield(np.zeros((2, 2)), spacing=0.0)


class TestFunctionSurface:
    def test_sample_is_called_with_floats_by_default(self) -> None:
        seen = []

        def sample(x, z):
            seen.append((type(x), type(z)))
            return 0.1 * x + 0.2 * z

        shape = model.Shape.heightfield(sample=sample)
        heights = shape.surface.heights(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        assert heights == pytest.approx([0.3, 0.4])
        assert set(seen) == {(float, float)}

    def test_a_vectorized_sample_is_called_once_with_arrays(self) -> None:
        calls = []

        def sample(x, z):
            calls.append(np.shape(x))
            return np.sin(x) + z

        shape = model.Shape.heightfield(sample=sample, vectorized=True)
        shape.surface.heights(np.linspace(0, 1, 50), np.zeros(50))
        assert calls == [(50,)]

    def test_the_default_normal_is_the_central_difference(self) -> None:
        shape = model.Shape.heightfield(sample=lambda x, z: 2.0 * math.sin(0.3 * x) * math.cos(
            0.2 * z))
        x, z = 1.3, -2.1
        dx = 2.0 * 0.3 * math.cos(0.3 * x) * math.cos(0.2 * z)
        dz = -2.0 * 0.2 * math.sin(0.3 * x) * math.sin(0.2 * z)
        expected = np.array([-dx, 1.0, -dz]) / math.sqrt(1 + dx * dx + dz * dz)
        got = shape.surface.normals(np.array([x]), np.array([z]))[0]
        assert got == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize('vectorized', [True, False])
    def test_a_sample_asks_for_its_heights_and_slopes_in_one_call(self, vectorized) -> None:
        calls = []

        def sample(x, z):
            calls.append(np.shape(x))
            return np.sin(0.7 * x) * np.cos(0.4 * z) + 0.1 * x if vectorized else (
                math.sin(0.7 * x) * math.cos(0.4 * z) + 0.1 * x)

        surface = model.Shape.heightfield(sample=sample, vectorized=vectorized,
                                          extent=((-5.0, -5.0), (5.0, 5.0))).surface
        x = np.array([0.3, -2.0, 4.9995, 7.0, -5.0])
        z = np.array([1.1, 0.4, 0.0, 0.0, 2.0])
        heights, normals = surface.heights(x, z), surface.normals(x, z)
        calls.clear()
        sampled = surface.sample(x, z)
        assert np.array_equal(sampled[0], heights) and np.array_equal(sampled[1], normals)
        if vectorized:
            # The five points of each stencil that lie inside the extent, together.
            assert calls == [(18,)]

    def test_a_given_normal_is_used(self) -> None:
        shape = model.Shape.heightfield(sample=lambda x, z: 0.0,
                                        normal=lambda x, z: (0.0, 0.0, 2.0))
        assert shape.surface.normals(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(
            [0.0, 0.0, 1.0])

    def test_an_extent_limits_the_surface(self) -> None:
        shape = model.Shape.heightfield(sample=lambda x, z: 1.0,
                                        extent=((0.0, 0.0), (4.0, 2.0)))
        heights = shape.surface.heights(np.array([1.0, 5.0, 1.0]), np.array([1.0, 1.0, 3.0]))
        assert heights[0] == 1.0
        assert np.isneginf(heights[1:]).all()
        low, high = shape.surface.aabb()
        assert low[0] == 0.0 and high[2] == 2.0
        assert np.isinf(low[1]) and np.isinf(high[1])

    def test_without_an_extent_the_field_is_unbounded(self) -> None:
        shape = model.Shape.heightfield(sample=lambda x, z: 0.0)
        low, high = shape.surface.aabb()
        assert np.isneginf(low).all() and np.isposinf(high).all()

    def test_a_height_range_bounds_the_box(self) -> None:
        shape = model.Shape.heightfield(sample=lambda x, z: 0.0, height_range=(-1.0, 3.0))
        low, high = shape.surface.aabb()
        assert (low[1], high[1]) == (-1.0, 3.0)

    def test_one_form_only(self) -> None:
        with pytest.raises(ValueError, match='heights or a sample'):
            model.Shape.heightfield()
        with pytest.raises(ValueError, match='heights or a sample'):
            model.Shape.heightfield(np.zeros((2, 2)), sample=lambda x, z: 0.0)


class TestSphere:
    def test_a_sphere_dropped_on_a_slope_stays_on_its_surface(self) -> None:
        """50 random slopes up to 60 degrees: the ball rolls and keeps its radius from the plane."""
        rng = np.random.default_rng(RNG_SEED)
        radius = 1.0
        for _ in range(50):
            angle = rng.uniform(0.0, 60.0)
            heading = rng.uniform(0.0, 2 * math.pi)
            shape, a, b = tilted_plane(angle, heading)
            world, _ground = ground_world(shape)
            x, z = rng.uniform(-10, 10, 2)
            ball = drop(world, model.Shape.sphere(radius),
                        (x, a * x + b * z + 1.5, z))
            gaps = []
            for step in range(240):
                world.step(DT)
                if step > 120:
                    gaps.append(plane_distance(world.position[ball], a, b) - radius)
            assert max(abs(gap) for gap in gaps) <= 0.01 * radius, (angle, heading)

    def test_a_ball_rolls_down_a_slope_and_stays_on_it(self) -> None:
        shape, a, b = tilted_plane(30.0, 0.0)
        world, _ground = ground_world(shape, friction=0.8)
        ball = drop(world, model.Shape.sphere(1.0), (0.0, 1.0 / math.cos(math.atan(a)), 0.0),
                    friction=0.8)
        start = world.position[ball].copy()
        for _ in range(360):
            world.step(DT)
            gap = plane_distance(world.position[ball], a, b) - 1.0
            assert abs(gap) <= 0.01
        travelled = start[0] - world.position[ball][0]
        assert travelled > 3.0

    def test_a_sphere_above_the_ground_makes_no_contact(self) -> None:
        shape, _a, _b = tilted_plane(20.0, 0.0, size=20.0)
        ground = make_proxy(shape, (0, 0, 0), (0, 0, 0, 1))
        ball = make_proxy(model.Shape.sphere(0.5), (0.0, 2.0, 0.0), (0, 0, 0, 1))
        assert collide.collide(0, 1, ground, ball) == []

    def test_contact_normal_runs_from_the_ground_to_the_mover(self) -> None:
        shape, a, b = tilted_plane(20.0, 0.0, size=20.0)
        ground = make_proxy(shape, (0, 0, 0), (0, 0, 0, 1))
        height = 0.4 / math.cos(math.atan(a)) - 0.1
        ball = make_proxy(model.Shape.sphere(0.5), (0.0, height, 0.0), (0, 0, 0, 1))
        (contact,) = collide.collide(0, 1, ground, ball)
        expected = np.array([-a, 1.0, -b]) / math.sqrt(1 + a * a + b * b)
        assert contact.normal == pytest.approx(expected)
        assert contact.a == 0 and contact.b == 1
        (flipped,) = collide.collide(1, 0, ball, ground)
        assert flipped.normal == pytest.approx(-expected)
        assert (flipped.a, flipped.b) == (1, 0)
        assert contact.depth == pytest.approx(flipped.depth)

    def test_a_sphere_beyond_the_extent_falls(self) -> None:
        shape = model.Shape.heightfield(np.zeros((5, 5)), spacing=1.0)
        world, _ground = ground_world(shape)
        ball = drop(world, model.Shape.sphere(0.3), (20.0, 1.0, 2.0))
        for _ in range(120):
            world.step(DT)
        assert world.position[ball][1] < -2.0


def slope_orientation(a: float, b: float, yaw: float) -> np.ndarray:
    """A pose with local +Y along the normal of ``y = a x + b z``, turned ``yaw`` about it."""
    normal = np.array([-a, 1.0, -b]) / math.sqrt(1 + a * a + b * b)
    axis = np.cross((0.0, 1.0, 0.0), normal)
    length = float(np.linalg.norm(axis))
    tilt = (mathutil.quat_from_axis_angle(axis / length, math.acos(normal[1]))
            if length > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0]))
    return mathutil.quat_mul(tilt, mathutil.quat_from_axis_angle((0, 1, 0), yaw))


class TestBoxAndHull:
    """Bodies wide enough not to tip over at 60 degrees, set down on the slope."""

    SLAB = (4.0, 2.0, 4.0)

    def test_a_box_set_on_a_slope_comes_to_rest_on_it(self) -> None:
        rng = np.random.default_rng(RNG_SEED + 1)
        for _ in range(50):
            angle = rng.uniform(0.0, 60.0)
            heading = rng.uniform(0.0, 2 * math.pi)
            shape, a, b = tilted_plane(angle, heading)
            world, _ground = ground_world(shape)
            x, z = rng.uniform(-10, 10, 2)
            pose = slope_orientation(a, b, rng.uniform(0, 2 * math.pi))
            up = mathutil.quat_to_matrix(pose)[:, 1]
            start = np.array([x, a * x + b * z, z]) + up * 1.4
            body = drop(world, model.Shape.box(self.SLAB), start, pose)
            for _ in range(480):
                world.step(DT)
            lowest = min(plane_distance(p, a, b)
                         for p in world_points(world, body, corners(self.SLAB)))
            assert abs(lowest) <= 0.01 * 1.0, (angle, heading, lowest)
            assert np.linalg.norm(world.linear_velocity[body]) < 0.2

    def test_a_box_dropped_flat_on_a_gentle_slope_settles_onto_it(self) -> None:
        rng = np.random.default_rng(RNG_SEED + 5)
        for _ in range(20):
            angle = rng.uniform(5.0, 30.0)
            heading = rng.uniform(0.0, 2 * math.pi)
            shape, a, b = tilted_plane(angle, heading)
            world, _ground = ground_world(shape)
            x, z = rng.uniform(-10, 10, 2)
            yaw = mathutil.quat_from_axis_angle((0, 1, 0), rng.uniform(0, 2 * math.pi))
            body = drop(world, model.Shape.box(self.SLAB), (x, a * x + b * z + 3.0, z), yaw)
            for _ in range(720):
                world.step(DT)
            lowest = min(plane_distance(p, a, b)
                         for p in world_points(world, body, corners(self.SLAB)))
            assert abs(lowest) <= 0.01 * 1.0, (angle, heading, lowest)
            assert np.linalg.norm(world.linear_velocity[body]) < 0.2

    def test_a_convex_hull_set_on_a_slope_comes_to_rest_on_it(self) -> None:
        rng = np.random.default_rng(RNG_SEED + 2)
        ring = [(3.0 * math.cos(t), 3.0 * math.sin(t)) for t in np.arange(6) * math.pi / 3]
        prism = np.array([(x, y, z) for (x, z) in ring for y in (-1.0, 1.0)])
        for _ in range(50):
            angle = rng.uniform(0.0, 60.0)
            heading = rng.uniform(0.0, 2 * math.pi)
            shape, a, b = tilted_plane(angle, heading)
            world, _ground = ground_world(shape)
            x, z = rng.uniform(-10, 10, 2)
            pose = slope_orientation(a, b, rng.uniform(0, 2 * math.pi))
            up = mathutil.quat_to_matrix(pose)[:, 1]
            start = np.array([x, a * x + b * z, z]) + up * 1.4
            body = drop(world, model.Shape.convex(prism), start, pose)
            for _ in range(480):
                world.step(DT)
            lowest = min(plane_distance(p, a, b) for p in world_points(world, body, prism))
            assert abs(lowest) <= 0.01 * 1.0, (angle, heading, lowest)
            assert np.linalg.norm(world.linear_velocity[body]) < 0.2

    def test_a_peak_under_the_middle_of_a_face_is_a_contact(self) -> None:
        """The corners clear the ground and a bump still presses into the underside."""
        bump = model.Shape.heightfield(
            sample=lambda x, z: 0.3 * math.exp(-(x * x + z * z) / 0.5), feature=0.25,
            extent=((-5.0, -5.0), (5.0, 5.0)))
        ground = make_proxy(bump, (0, 0, 0), (0, 0, 0, 1))
        slab = make_proxy(model.Shape.box((4.0, 0.4, 4.0)), (0.0, 0.48, 0.0), (0, 0, 0, 1))
        contacts = collide.collide(0, 1, ground, slab)
        assert contacts
        deepest = max(contacts, key=lambda c: c.depth)
        assert deepest.depth == pytest.approx(0.3 - 0.28, abs=1e-3)
        assert np.linalg.norm(deepest.point[[0, 2]]) < 0.5
        assert deepest.normal == pytest.approx([0, 1, 0], abs=1e-3)

    def test_a_box_sitting_on_a_bump_rests_on_it(self) -> None:
        """A slab laid across a bump and level ground is held by the bump and its corners."""
        bump = model.Shape.heightfield(
            sample=lambda x, z: 0.2 * math.exp(-(x * x + z * z) / 0.5), feature=0.25,
            extent=((-5.0, -5.0), (5.0, 5.0)))
        world, _ground = ground_world(bump)
        slab = drop(world, model.Shape.box((1.0, 0.2, 1.0)), (0.0, 1.0, 0.0))
        for _ in range(480):
            world.step(DT)
        assert world.position[slab][1] - 0.1 == pytest.approx(0.2, abs=0.01)


class TestOtherShapes:
    def test_a_capsule_lies_on_a_slope(self) -> None:
        shape, a, b = tilted_plane(25.0, 0.0)
        world, _ground = ground_world(shape)
        capsule = drop(world, model.Shape.capsule(height=1.0, radius=0.3), (0.0, 2.0, 0.0),
                       mathutil.quat_from_axis_angle((0, 0, 1), math.pi / 2))
        for _ in range(720):
            world.step(DT)
        rotation = mathutil.quat_to_matrix(world.orientation[capsule])
        axis = rotation[:, 1]
        ends = [world.position[capsule] + sign * 0.5 * axis for sign in (-1, 1)]
        gaps = [plane_distance(end, a, b) - 0.3 for end in ends]
        assert max(abs(gap) for gap in gaps) <= 0.01
        assert np.linalg.norm(world.linear_velocity[capsule]) < 0.2

    def test_a_standing_capsule_rests_on_its_lower_end(self) -> None:
        shape = model.Shape.heightfield(np.zeros((20, 20)), origin=(-10, 0, -10))
        world, _ground = ground_world(shape)
        capsule = drop(world, model.Shape.capsule(height=1.0, radius=0.3), (0.0, 3.0, 0.0))
        for _ in range(480):
            world.step(DT)
        assert world.position[capsule][1] == pytest.approx(0.5 + 0.3, abs=0.01)

    def test_a_cylinder_stands_on_its_flat_end(self) -> None:
        shape = model.Shape.heightfield(np.zeros((20, 20)), origin=(-10, 0, -10))
        world, _ground = ground_world(shape)
        post = drop(world, model.Shape.cylinder(height=2.0, radius=0.4), (0.0, 3.0, 0.0))
        for _ in range(480):
            world.step(DT)
        assert world.position[post][1] == pytest.approx(1.0, abs=0.01)

    def test_a_log_lies_on_a_slope(self) -> None:
        shape, a, b = tilted_plane(10.0, 0.0)
        world, _ground = ground_world(shape)
        log = drop(world, model.Shape.cylinder(height=3.0, radius=0.3), (0.0, 2.0, 0.0),
                   mathutil.quat_from_axis_angle((1, 0, 0), math.pi / 2))
        for _ in range(240):
            world.step(DT)
        axis = mathutil.quat_to_matrix(world.orientation[log])[:, 1]
        end_gaps = [plane_distance(world.position[log] + s * 1.5 * axis, a, b)
                    for s in (-1, 1)]
        # Both ends of a log lying across the fall line sit a radius above the plane.
        assert min(end_gaps) >= 0.3 * 0.97 - 0.02


class TestPlacedFields:
    def test_a_ground_body_that_is_moved_and_turned_carries_its_surface(self) -> None:
        flat = model.Shape.heightfield(np.zeros((40, 40)), origin=(-20, 0, -20))
        tilt = mathutil.quat_from_axis_angle((1, 0, 0), math.radians(30))
        world, _ground = ground_world(flat, position=(5.0, 2.0, -3.0), orientation=tilt)
        ball = drop(world, model.Shape.sphere(1.0), (5.0, 6.0, -3.0))
        for _ in range(240):
            world.step(DT)
        up = mathutil.quat_to_matrix(tilt)[:, 1]
        gap = float(np.dot(world.position[ball] - np.array([5.0, 2.0, -3.0]), up)) - 1.0
        assert abs(gap) <= 0.01

    def test_an_unbounded_function_field_holds_a_ball(self) -> None:
        shape = model.Shape.heightfield(sample=lambda x, z: 0.2 * x - 1.0)
        world, _ground = ground_world(shape)
        ball = drop(world, model.Shape.sphere(1.0), (300.0, 70.0, 5.0))
        for _ in range(360):
            world.step(DT)
        x, y, _z = world.position[ball]
        distance = (y - (0.2 * x - 1.0)) / math.sqrt(1 + 0.04)
        assert abs(distance - 1.0) <= 0.01

    def test_a_ball_rests_in_a_curved_valley(self) -> None:
        def valley(x, z):
            return 0.05 * (x * x + z * z)

        shape = model.Shape.heightfield(sample=valley, feature=0.25,
                                        extent=((-30.0, -30.0), (30.0, 30.0)))
        world, _ground = ground_world(shape, friction=1.0)
        ball = drop(world, model.Shape.sphere(1.0), (4.0, 3.0, 2.0), friction=1.0)
        for _ in range(1800):
            world.step(DT)
        centre = world.position[ball]
        axis = np.linspace(-2.0, 2.0, 801)
        gx, gz = np.meshgrid(centre[0] + axis, centre[2] + axis)
        surface = np.stack((gx, valley(gx, gz), gz), axis=-1)
        nearest = np.linalg.norm(surface - centre, axis=-1).min()
        assert nearest == pytest.approx(1.0, abs=0.01)

    def test_only_static_bodies_can_be_a_field(self) -> None:
        world = PhysicsWorld()
        index = world.add_shape(model.Shape.heightfield(np.zeros((2, 2))))
        with pytest.raises(ValueError, match='static'):
            world.add_body(model.Motion(type=model.DYNAMIC), collider=model.Collider(shape=index))
        with pytest.raises(ValueError, match='static'):
            world.add_body(model.Motion(type=model.KINEMATIC),
                           collider=model.Collider(shape=index))


class TestRays:
    def test_a_ray_hits_where_the_sampled_height_says(self) -> None:
        def surface(x, z):
            return 2.0 * math.sin(0.2 * x) * math.cos(0.15 * z)

        shape = model.Shape.heightfield(sample=surface, feature=0.5,
                                        extent=((-40.0, -40.0), (40.0, 40.0)))
        world, ground = ground_world(shape)
        rng = np.random.default_rng(RNG_SEED + 3)
        for _ in range(50):
            x, z = rng.uniform(-30, 30, 2)
            hit = raycast(world, (x, 20.0, z), (0, -1, 0))
            assert hit is not None and hit.body == ground
            assert hit.point[1] == pytest.approx(surface(x, z), abs=1e-6)
            assert hit.distance == pytest.approx(20.0 - surface(x, z), abs=1e-6)
            assert hit.normal[1] > 0.0

    def test_a_slanted_ray_meets_a_plane_at_its_intersection(self) -> None:
        shape, a, b = tilted_plane(40.0, 0.4)
        world, _ground = ground_world(shape)
        origin = np.array([2.0, 30.0, -5.0])
        direction = np.array([0.3, -1.0, 0.2])
        hit = raycast(world, origin, direction)
        direction = direction / np.linalg.norm(direction)
        # y = a x + b z along the ray gives the distance directly.
        t = (origin[1] - a * origin[0] - b * origin[2]) / (
            a * direction[0] + b * direction[2] - direction[1])
        assert hit is not None
        assert hit.distance == pytest.approx(t, abs=1e-6)
        assert hit.normal == pytest.approx(np.array([-a, 1.0, -b]) / math.sqrt(1 + a * a + b * b))

    def test_the_normal_faces_back_along_the_ray(self) -> None:
        shape, _a, _b = tilted_plane(30.0, 0.0)
        world, _ground = ground_world(shape)
        hit = raycast(world, (0.0, 20.0, 0.0), (1.0, -1.0, 0.0))
        assert hit is not None
        assert np.dot(hit.normal, [1.0, -1.0, 0.0]) < 0.0

    def test_a_ray_that_stops_short_misses(self) -> None:
        shape = model.Shape.heightfield(np.zeros((10, 10)), origin=(-5, 0, -5))
        world, _ground = ground_world(shape)
        assert raycast(world, (0.0, 10.0, 0.0), (0, -1, 0), max_distance=9.0) is None
        assert raycast(world, (0.0, 10.0, 0.0), (0, -1, 0), max_distance=11.0) is not None

    def test_a_ray_that_starts_below_the_surface_hits_at_zero(self) -> None:
        ramp = np.tile(np.linspace(0.0, 10.0, 11), (11, 1))
        shape = model.Shape.heightfield(ramp, origin=(-5, 0, -5))
        world, _ground = ground_world(shape)
        # The surface is 5 m up at x = 0 and the grid reaches 0 to 10 m.
        hit = raycast(world, (0.0, 2.0, 0.0), (0, -1, 0))
        assert hit is not None and hit.distance == 0.0

    def test_a_ray_beside_the_extent_misses(self) -> None:
        shape = model.Shape.heightfield(np.zeros((10, 10)), origin=(-5, 0, -5))
        world, _ground = ground_world(shape)
        assert raycast(world, (30.0, 5.0, 0.0), (0, -1, 0)) is None
        # Entering the column of the grid from the side, below the surface, is no hit.
        assert raycast(world, (-20.0, -1.0, 0.0), (1, 0, 0), max_distance=40.0) is None

    def test_a_horizontal_ray_skims_a_hill(self) -> None:
        shape = model.Shape.heightfield(
            sample=lambda x, z: max(0.0, 3.0 - abs(x)), feature=0.25,
            extent=((-10.0, -10.0), (10.0, 10.0)))
        world, _ground = ground_world(shape)
        hit = raycast(world, (-9.0, 1.5, 0.0), (1, 0, 0))
        assert hit is not None
        assert hit.point[0] == pytest.approx(-1.5, abs=1e-6)

    def test_rays_in_bulk_agree_with_single_casts(self) -> None:
        shape, _a, _b = tilted_plane(20.0, 1.0)
        world, _ground = ground_world(shape)
        rng = np.random.default_rng(RNG_SEED + 4)
        origins = np.column_stack((rng.uniform(-20, 20, 8), np.full(8, 40.0),
                                   rng.uniform(-20, 20, 8)))
        directions = np.tile((0.0, -1.0, 0.0), (8, 1))
        many = raycast_many(world, origins, directions)
        for origin, hit in zip(origins, many, strict=True):
            single = raycast(world, origin, (0, -1, 0))
            assert hit is not None and single is not None
            assert hit.distance == pytest.approx(single.distance)


class TestTheGridIsFixed:
    """A grid surface keeps its own copy of the heights, which nothing can change."""

    def test_an_edit_to_the_array_it_was_given_changes_nothing(self) -> None:
        heights = np.zeros((4, 4))
        surface = GridSurface(heights)
        heights[1, 1] = 5.0
        assert surface.height_at(1.0, 1.0) == 0.0
        assert surface.heights(np.array([1.0]), np.array([1.0]))[0] == 0.0
        assert float(surface.grid[1, 1]) == 0.0

    def test_a_write_to_the_grid_raises(self) -> None:
        surface = GridSurface(np.zeros((4, 4)))
        with pytest.raises(ValueError, match='read-only'):
            surface.grid[1, 1] = 5.0

    def test_the_grid_cannot_be_replaced(self) -> None:
        surface = GridSurface(np.zeros((4, 4)))
        with pytest.raises(AttributeError):
            surface.grid = np.zeros((1, 4))
