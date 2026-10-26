"""A broad phase with a few moving bodies finds the pairs from them outward, in sweep order."""
import numpy as np
import pytest

from omi_physics import broadphase, model
from omi_physics.world import PhysicsWorld


def scattered(rng, movers: int) -> PhysicsWorld:
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    shapes = [world.add_shape(model.Shape.box((1.0, 1.0, 1.0))),
              world.add_shape(model.Shape.sphere(0.6)),
              world.add_shape(model.Shape.box((8.0, 0.5, 8.0)))]
    for _ in range(int(rng.integers(20, 120))):
        shape = int(rng.choice(shapes))
        place = tuple(np.round(rng.uniform(-6, 6, 3), 1))   # rounded, so minimum x ties
        if rng.random() < 0.15:
            world.add_body(model.Motion(type=model.STATIC), trigger=model.Trigger(shape=shape),
                           position=place)
        else:
            world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape),
                           position=place)
    for _ in range(movers):
        kind = model.DYNAMIC if rng.random() < 0.5 else model.KINEMATIC
        shape = int(rng.choice(shapes[:2]))
        sensing = kind == model.KINEMATIC and rng.random() < 0.3
        world.add_body(model.Motion(type=kind, mass=1.0),
                       collider=None if sensing else model.Collider(shape=shape),
                       trigger=model.Trigger(shape=shape) if sensing else None,
                       position=tuple(np.round(rng.uniform(-6, 6, 3), 1)))
    world.refit_aabbs()
    return world


@pytest.mark.parametrize('movers', [0, 1, 2, 5, 8])
@pytest.mark.parametrize('seed', range(10))
def test_the_pairs_from_the_movers_are_the_sweeps_in_its_order(seed, movers):
    world = scattered(np.random.default_rng(seed * 10 + movers), movers)
    wanted = broadphase._sweep(world, None)          # noqa: SLF001 the sweep is the reference
    moving = world._rows_by_motion()[2]              # noqa: SLF001 the rows the world keeps
    got = broadphase._sweep_from(world, moving)      # noqa: SLF001 the path under test
    assert got[0].tolist() == wanted[0].tolist()
    assert got[1].tolist() == wanted[1].tolist()


def test_a_world_with_one_mover_is_not_swept_whole(monkeypatch):
    world = scattered(np.random.default_rng(1), 1)
    swept = []
    original = broadphase._sweep                      # noqa: SLF001 the sweep is being counted
    monkeypatch.setattr(broadphase, '_sweep', lambda *a: swept.append(1) or original(*a))
    world.step(1 / 60)
    assert swept == []


def test_a_kinematic_body_with_nothing_to_meet_is_not_paired_or_sorted(monkeypatch):
    """Beside static colliders alone, a kinematic body has no pair, and none is ordered."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
    for x in range(10):
        world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=box),
                       position=(float(x), 0.0, 0.0))
    world.add_body(model.Motion(type=model.STATIC), trigger=model.Trigger(shape=box),
                   position=(30.0, 0.0, 0.0))
    world.add_body(model.Motion(type=model.KINEMATIC), collider=model.Collider(shape=box),
                   position=(4.5, 0.5, 0.0))
    world.refit_aabbs()
    asked = []
    original = broadphase._moves_against              # noqa: SLF001 the test of each pair, counted
    monkeypatch.setattr(broadphase, '_moves_against', lambda *a: asked.append(1) or original(*a))
    found = broadphase._sweep_from(world, world._rows_by_motion()[2])  # noqa: SLF001 the path under test
    assert found[0].tolist() == [] and asked == []
