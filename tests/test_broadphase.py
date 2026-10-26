"""Phase 1 broad phase: dynamic AABB tree vs brute force, filters (no GL)."""
import numpy as np
import pytest

from omi_physics import broadphase, model
from omi_physics.broadphase import BroadPhase, DynamicAABBTree, brute_force_pairs, filtered_pairs
from omi_physics.world import PhysicsWorld


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
        world.pose_arrays()[0][i] += (0.3, -0.2, 0.1)
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
    world.add_body(model.Motion(type=model.DYNAMIC),
                   collider=model.Collider(shape=shape, collisionFilter=players),
                   position=(0, 0, 0))
    world.add_body(model.Motion(type=model.DYNAMIC),
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


# -- collision filters, tested together ----------------------------------------------

SYSTEMS = ('a', 'b', 'c', 'd')


def random_filter(rng) -> model.CollisionFilter:
    """A filter of random systems, with an allow list or none and a random exclude list."""
    def some():
        return tuple(s for s in SYSTEMS if rng.random_sample() < 0.4)
    allow = None if rng.random_sample() < 0.4 else some()
    return model.CollisionFilter(collisionSystems=some(), collideWithSystems=allow,
                                 notCollideWithSystems=some())


def world_with_filters(seed, n=80):
    """Overlapping bodies, some static, with random filter indices: none, real, and past the end."""
    rng = np.random.RandomState(seed)
    world = PhysicsWorld()
    for _ in range(rng.randint(0, 5)):
        world.add_filter(random_filter(rng))
    shape = world.add_shape(model.Shape.sphere(0.6))
    for _ in range(n):
        kind = model.STATIC if rng.random_sample() < 0.3 else model.DYNAMIC
        index = int(rng.randint(-1, len(world.filters) + 2))
        world.add_body(model.Motion(type=kind),
                       collider=model.Collider(shape=shape, collisionFilter=index),
                       position=rng.uniform(-3.0, 3.0, 3))
    world.refit_aabbs(margin=0.0)
    return world


@pytest.mark.parametrize('seed', range(60))
def test_pairs_filtered_together_are_those_filtered_one_by_one(seed):
    world = world_with_filters(seed)
    rng = np.random.RandomState(seed)
    count = 400
    lo = rng.randint(0, world.body_count, count)
    hi = rng.randint(0, world.body_count, count)
    one_by_one = [(p, q) for p, q in zip(lo.tolist(), hi.tolist(), strict=True)
                  if broadphase._filters_interact(world, p, q)]  # noqa: SLF001 the reference answer
    assert filtered_pairs(world, lo, hi) == one_by_one


@pytest.mark.parametrize('seed', range(20))
def test_pairs_with_random_filters_match_brute_force(seed):
    world = world_with_filters(seed)
    assert set(BroadPhase().pairs(world)) == set(brute_force_pairs(world))


def test_a_world_with_no_filters_checks_no_pair_one_by_one(monkeypatch):
    world = make_world(120, seed=1, spread=3.0)
    asked = []
    original = broadphase._filters_interact  # noqa: SLF001 the one-by-one test is what is counted
    monkeypatch.setattr(broadphase, '_filters_interact',
                        lambda w, a, b: asked.append((a, b)) or original(w, a, b))
    assert BroadPhase().pairs(world)
    assert asked == []
