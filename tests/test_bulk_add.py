"""Adding many static bodies that share one shape in a single call.

``PhysicsWorld.add_bodies(shape, positions)`` fills the world's columns by
slice rather than one body at a time, which is what a forest of trunks or a
field of rocks needs: 10,000 of them arrive in a fraction of a second and
cost a step next to nothing.
"""
import time

import numpy as np
import pytest

from omi_physics import model
from omi_physics.raycast import raycast
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0


def forest_positions(count: int, size: float = 400.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = rng.uniform(-size / 2, size / 2, (count, 2))
    return np.column_stack((flat[:, 0], np.full(count, 2.0), flat[:, 1]))


def world_with_ground() -> PhysicsWorld:
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), fixed_dt=DT)
    floor = world.add_shape(model.Shape.box((1000.0, 1.0, 1000.0)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0.0, -0.5, 0.0))
    return world


class TestAddBodies:
    def test_returns_the_indices_in_position_order(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.3))
        positions = forest_positions(25)
        indices = world.add_bodies(trunk, positions)
        assert isinstance(indices, np.ndarray)
        assert indices.tolist() == list(range(1, 26))
        assert world.position[indices] == pytest.approx(positions)

    def test_fills_every_column_as_add_body_does(self) -> None:
        looped, bulk = world_with_ground(), world_with_ground()
        shapes = []
        for world in (looped, bulk):
            shapes.append(world.add_shape(model.Shape.cylinder(height=4.0, radius=0.3)))
        material = looped.add_material(model.Material(staticFriction=0.9))
        bulk.add_material(model.Material(staticFriction=0.9))
        positions = forest_positions(40)
        quaternion = (0.0, 0.3826834323650898, 0.0, 0.9238795325112867)
        for position in positions:
            looped.add_body(model.Motion(type=model.STATIC),
                            collider=model.Collider(shape=shapes[0], physicsMaterial=material),
                            position=position, orientation=quaternion)
        bulk.add_bodies(shapes[1], positions, collider=model.Collider(physicsMaterial=material),
                        orientation=quaternion)
        assert looped.body_count == bulk.body_count
        for column in ('position', 'orientation', 'prev_position', 'motion_type',
                       'collider_shape', 'collider_material', 'collider_filter',
                       'trigger_shape', 'aabb_min', 'aabb_max', 'inv_mass', 'mass',
                       'inv_inertia', 'linear_velocity', 'island', 'awake'):
            # A bound measured at the position and one measured at the origin and
            # offset differ in the last bits.
            assert np.allclose(getattr(looped, column), getattr(bulk, column),
                               rtol=0.0, atol=1e-12), column
        assert looped.bodies == bulk.bodies
        assert looped.live_body_count == bulk.live_body_count

    def test_slots_of_removed_bodies_are_used_first(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.3))
        first = world.add_bodies(trunk, forest_positions(10))
        world.remove_body(int(first[3]))
        world.remove_body(int(first[7]))
        second = world.add_bodies(trunk, forest_positions(5, seed=8))
        assert set(second[:2].tolist()) == {int(first[3]), int(first[7])}
        assert second[2:].tolist() == [11, 12, 13]
        assert world.live_body_count == 1 + 10 + 3
        assert len(set(second.tolist())) == 5

    def test_a_recycled_slot_has_a_new_generation(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.3))
        (slot,) = world.add_bodies(trunk, [(0.0, 2.0, 0.0)])
        old = world.ref(int(slot))
        world.remove_body(int(slot))
        (again,) = world.add_bodies(trunk, [(5.0, 2.0, 0.0)])
        assert again == slot
        assert not world.alive(old)
        assert world.alive(world.ref(int(again)))

    def test_collider_carries_material_and_filter_and_its_shape_is_ignored(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.3))
        material = world.add_material(model.Material(staticFriction=0.2))
        filt = world.add_filter(model.CollisionFilter(collisionSystems=('tree',)))
        indices = world.add_bodies(
            trunk, forest_positions(5),
            collider=model.Collider(shape=0, physicsMaterial=material, collisionFilter=filt))
        assert np.all(world.collider_shape[indices] == trunk)
        assert np.all(world.collider_material[indices] == material)
        assert np.all(world.collider_filter[indices] == filt)

    def test_an_empty_batch_adds_nothing(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.sphere(1.0))
        indices = world.add_bodies(trunk, np.zeros((0, 3)))
        assert len(indices) == 0
        assert world.body_count == 1

    def test_only_static_bodies_are_added_in_bulk(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.sphere(1.0))
        with pytest.raises(ValueError, match='static'):
            world.add_bodies(trunk, [(0, 0, 0)], motion=model.Motion(type=model.DYNAMIC))

    def test_bad_arguments_are_refused_before_anything_is_added(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.sphere(1.0))
        with pytest.raises(ValueError, match=r'\(N, 3\)'):
            world.add_bodies(trunk, np.zeros((4, 2)))
        with pytest.raises(IndexError, match='shape'):
            world.add_bodies(trunk + 5, [(0, 0, 0)])
        assert world.body_count == 1

    def test_the_bodies_collide(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.5))
        world.add_bodies(trunk, [(0.0, 2.0, 0.0)])
        ball_shape = world.add_shape(model.Shape.sphere(0.4))
        ball = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                              collider=model.Collider(shape=ball_shape),
                              position=(0.1, 8.0, 0.0))
        for _ in range(360):
            world.step(DT)
        assert world.position[ball][1] == pytest.approx(4.0 + 0.4, abs=0.02)

    def test_a_ray_meets_one_of_them(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.5))
        indices = world.add_bodies(trunk, forest_positions(300))
        target = int(indices[123])
        origin = world.position[target] + (-10.0, 0.0, 0.0)  # y = 2, level with the trunk's middle
        hit = raycast(world, origin, (1.0, 0.0, 0.0))
        assert hit is not None
        assert hit.distance <= 10.0 - 0.3 + 1e-6
        assert hit.body in indices.tolist()

    def test_bodies_moved_afterwards_are_measured_again(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.5))
        (post,) = world.add_bodies(trunk, [(0.0, 2.0, 0.0)])
        world.step(DT)
        world.place_body(int(post), position=(50.0, 2.0, 0.0))
        world.step(DT)
        assert world.aabb_min[post][0] == pytest.approx(49.5 - 0.05, abs=1e-6)
        assert world.aabb_max[post][0] == pytest.approx(50.5 + 0.05, abs=1e-6)


class TestCost:
    def test_ten_thousand_trunks_are_added_in_half_a_second(self) -> None:
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.3))
        positions = forest_positions(10_000)
        began = time.perf_counter()
        indices = world.add_bodies(trunk, positions)
        elapsed = time.perf_counter() - began
        assert len(indices) == 10_000
        assert elapsed < 0.5, elapsed

    def test_the_first_step_after_a_bulk_add_does_not_measure_each_body(self) -> None:
        """The bounds are filled in with the bodies, so the first refit has nothing to measure."""
        world = world_with_ground()
        trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.3))
        world.add_bodies(trunk, forest_positions(10_000))
        began = time.perf_counter()
        world.step(DT)
        assert time.perf_counter() - began < 0.25


def _median_step(world: PhysicsWorld, steps: int = 60) -> float:
    for _ in range(150):                        # long enough for the balls to land
        world.step(DT)
    times = []
    for _ in range(steps):
        began = time.perf_counter()
        world.step(DT)
        times.append(time.perf_counter() - began)
    return float(np.median(times))


def _world_with_movers(count: int) -> PhysicsWorld:
    """Balls over a 60 m square of floor that never fall asleep, so every step has work."""
    world = world_with_ground()
    world.sleep_enabled = False
    ball = world.add_shape(model.Shape.sphere(0.4))
    rng = np.random.default_rng(3)
    for position in rng.uniform(-30.0, 30.0, (count, 3)) * (1, 0, 1) + (0.0, 5.0, 0.0):
        world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                       collider=model.Collider(shape=ball), position=position)
    return world


@pytest.mark.serial
class TestStepCost:
    @pytest.mark.parametrize('movers', [1, 40])
    def test_ten_thousand_static_trunks_at_most_double_a_step(
            self, movers: int, monkeypatch: pytest.MonkeyPatch) -> None:
        # The check of every static pose is a cost of its own, which this measures without.
        monkeypatch.delenv('OMI_PHYSICS_CHECK_STATICS', raising=False)
        without = _world_with_movers(movers)
        baseline = _median_step(without)
        crowded = _world_with_movers(movers)
        trunk = crowded.add_shape(model.Shape.cylinder(height=4.0, radius=0.3))
        # The trunks stand around the arena, so the balls meet none of them.
        spots = forest_positions(10_000)
        outside = (np.abs(spots[:, 0]) > 40.0) | (np.abs(spots[:, 2]) > 40.0)
        crowded.add_bodies(trunk, spots[outside])
        with_trunks = _median_step(crowded)
        assert with_trunks < 2.0 * baseline, (baseline, with_trunks)
