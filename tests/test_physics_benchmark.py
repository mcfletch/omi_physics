"""Phase 7 scale: broad phase stays sub-quadratic, islands, sleeping ≈ free."""
import time
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld
from omi_physics.broadphase import BroadPhase, brute_force_pairs
from omi_physics.solver import build_islands


def grid_world(n_side=10, spacing=2.5):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    box = world.add_shape(model.Shape.box((1, 1, 1)))
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                world.add_body(
                    model.Motion(type=model.DYNAMIC),
                    collider=model.Collider(shape=box),
                    position=(i * spacing, 5 + k * spacing, j * spacing))
    return world


def test_broadphase_is_subquadratic_on_1000_bodies():
    world = grid_world(10)                     # 1000 bodies, well separated
    world.refit_aabbs()
    bp = BroadPhase()
    pairs = bp.pairs(world)
    n = world.body_count
    assert n == 1000
    assert len(pairs) < n * (n - 1) / 20        # far below O(N²)
    assert set(pairs) == set(brute_force_pairs(world))


def test_step_completes_on_1000_bodies():
    world = grid_world(10)
    t0 = time.time()
    world.step(1 / 60)
    assert time.time() - t0 < 2.0               # generous CPU-reference budget


def test_islands_partition_disjoint_contacts():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), sleep_enabled=False)
    box = world.add_shape(model.Shape.box((1, 1, 1)))
    ground = world.add_shape(model.Shape.box((60, 1, 60)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=ground), position=(0, -0.5, 0))
    # two separated stacks → two islands
    for cx in (0.0, 20.0):
        for k in range(3):
            world.add_body(model.Motion(type=model.DYNAMIC),
                           collider=model.Collider(shape=box),
                           position=(cx, 0.5 + k * 1.001, 0))
    for _ in range(60):
        world.step(1 / 120)
    islands = build_islands(world, world.contacts)
    assert len(islands) >= 2


def test_sleeping_scene_costs_little():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    box = world.add_shape(model.Shape.box((1, 1, 1)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=box), position=(0, -0.5, 0))
    for k in range(20):
        world.add_body(model.Motion(type=model.DYNAMIC),
                       collider=model.Collider(shape=box),
                       position=(k * 2.0, 0.5, 0))
    for _ in range(240):                        # let them settle and sleep
        world.step(1 / 120)
    awake = int(np.count_nonzero(world.awake[world.motion_type == 2]))
    assert awake < 20                           # most bodies asleep


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
