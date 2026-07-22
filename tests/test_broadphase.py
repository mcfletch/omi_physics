"""Phase 1 broad phase: dynamic AABB tree vs brute force, filters (no GL)."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld
from omi_physics.broadphase import BroadPhase, brute_force_pairs, DynamicAABBTree


def make_world(n, seed=0, spread=6.0):
    rng = np.random.RandomState(seed)
    world = PhysicsWorld()
    shape = world.add_shape(model.Shape.sphere(0.5))
    for _ in range(n):
        pos = rng.uniform(-spread, spread, 3)
        world.add_body(model.Motion(type=model.DYNAMIC),
                       collider=model.Collider(shape=shape), position=pos)
    world.refit_aabbs(margin=0.0)
    return world


def test_tree_pairs_match_brute_force():
    world = make_world(120, seed=1, spread=3.0)
    bp = BroadPhase(fatten=0.0)
    tree_pairs = set(bp.pairs(world))
    brute = set(brute_force_pairs(world))
    assert tree_pairs == brute
    assert len(brute) > 0                         # the scene actually has overlaps


def test_refit_equals_rebuild():
    world = make_world(60, seed=2)
    bp = BroadPhase()
    bp.pairs(world)
    for i in range(world.body_count):             # move everything
        world.position[i] += (0.3, -0.2, 0.1)
    world.refit_aabbs(margin=0.0)
    refit_pairs = set(bp.pairs(world))
    fresh = BroadPhase()
    rebuild_pairs = set(fresh.pairs(world))
    assert refit_pairs == rebuild_pairs == set(brute_force_pairs(world))


def test_add_and_remove_bodies():
    world = make_world(20, seed=3)
    bp = BroadPhase()
    bp.pairs(world)
    world.add_body(model.Motion(type=model.DYNAMIC),
                   collider=model.Collider(shape=0), position=(0, 0, 0))
    world.refit_aabbs(margin=0.0)
    pairs_after = set(bp.pairs(world))
    assert pairs_after == set(brute_force_pairs(world))


def test_filters_drop_non_interacting_pairs():
    world = PhysicsWorld()
    shape = world.add_shape(model.Shape.sphere(1.0))
    players = world.add_filter(model.CollisionFilter(
        collisionSystems=('player',), notCollideWithSystems=('player',)))
    a = world.add_body(model.Motion(type=model.DYNAMIC),
                       collider=model.Collider(shape=shape, collisionFilter=players),
                       position=(0, 0, 0))
    b = world.add_body(model.Motion(type=model.DYNAMIC),
                       collider=model.Collider(shape=shape, collisionFilter=players),
                       position=(0.5, 0, 0))
    world.refit_aabbs(margin=0.0)
    bp = BroadPhase()
    assert bp.pairs(world) == []                  # same group, excluded


def test_two_static_bodies_never_pair():
    world = PhysicsWorld()
    shape = world.add_shape(model.Shape.box((2, 2, 2)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape), position=(0, 0, 0))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape), position=(0.5, 0, 0))
    world.refit_aabbs(margin=0.0)
    assert BroadPhase().pairs(world) == []


def test_tree_query_finds_containing_leaf():
    tree = DynamicAABBTree(fatten=0.0)
    tree.insert(0, np.array([0.0, 0, 0]), np.array([1.0, 1, 1]))
    tree.insert(1, np.array([5.0, 5, 5]), np.array([6.0, 6, 6]))
    hits = tree.query(np.array([0.5, 0.5, 0.5]), np.array([0.6, 0.6, 0.6]))
    assert hits == [0]


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
