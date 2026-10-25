"""A world of many static bodies is swept without them, and finds the same pairs in the same order.

Static bodies that carry no trigger pair with dynamic bodies only. Past
``SPLIT_MIN`` of them the broad phase indexes them once and asks, for each
dynamic body, for the few within its x range. The pairs reach the narrow phase
in the order a sweep of every body gives them, because the contact solver is
sequential and a different order is a different result.
"""
import numpy as np
import pytest

from omi_physics import broadphase, model
from omi_physics.broadphase import BroadPhase
from omi_physics.world import PhysicsWorld


def build(seed: int, gridded: bool) -> PhysicsWorld:
    rng = np.random.default_rng(seed)
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
    trunk = world.add_shape(model.Shape.cylinder(height=4.0, radius=0.3))
    ball = world.add_shape(model.Shape.sphere(0.6))
    floor = world.add_shape(model.Shape.box((200.0, 1.0, 200.0)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0.0, -0.5, 0.0))
    spots = rng.uniform(-40.0, 40.0, (600, 2))
    if gridded:
        spots = np.round(spots)
    shapes = rng.choice([box, trunk], 600)
    for (x, z), shape in zip(spots, shapes, strict=True):
        world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=int(shape)),
                       position=(float(x), 0.5, float(z)))
    zone = world.add_shape(model.Shape.box((6.0, 6.0, 6.0)))
    world.add_body(model.Motion(type=model.STATIC), trigger=model.Trigger(shape=zone),
                   position=(5.0, 2.0, 5.0))
    for x, z in rng.uniform(-40.0, 40.0, (25, 2)):
        world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                       collider=model.Collider(shape=ball),
                       position=(round(float(x)) if gridded else float(x), 0.5,
                                 round(float(z)) if gridded else float(z)))
    for position in ((5.0, 2.0, 4.0), (-20.0, 0.5, -20.0)):
        world.add_body(model.Motion(type=model.KINEMATIC, linearVelocity=(1.0, 0.0, 0.0)),
                       collider=model.Collider(shape=box), position=position)
    return world


def full_sweep(world: PhysicsWorld, monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    with monkeypatch.context() as patch:
        patch.setattr(broadphase, 'SPLIT_MIN', 10 ** 9)
        return BroadPhase().pairs(world)


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('gridded', [False, True])
def test_pairs_and_their_order_match_a_sweep_of_every_body(seed, gridded, monkeypatch) -> None:
    world = build(seed, gridded)
    split = BroadPhase()
    got = split.pairs(world)
    assert split._inert is not None            # noqa: SLF001 the index is what is being exercised
    assert got == full_sweep(world, monkeypatch)
    assert len(got) > 30


def test_the_same_after_inert_bodies_move_arrive_and_leave(monkeypatch) -> None:
    world = build(5, True)
    split = BroadPhase()
    split.pairs(world)
    world.place_body(40, position=(3.0, 0.5, 3.0))         # an inert body moves
    assert split.pairs(world) == full_sweep(world, monkeypatch)
    world.remove_body(60)
    world.remove_body(61)
    assert split.pairs(world) == full_sweep(world, monkeypatch)
    box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=box),
                   position=(1.0, 0.5, 1.0))
    assert split.pairs(world) == full_sweep(world, monkeypatch)


def test_a_dynamic_body_moving_does_not_rebuild_the_index() -> None:
    world = build(6, False)
    split = BroadPhase()
    split.pairs(world)
    built = split._inert                       # noqa: SLF001 the index is what is being exercised
    dynamic = int(np.flatnonzero(world.motion_type == 2)[0])
    world.place_body(dynamic, position=(2.0, 0.8, 2.0))
    split.pairs(world)
    assert split._inert is built               # noqa: SLF001 the index is what is being exercised


def test_collision_filters_still_apply(monkeypatch) -> None:
    world = build(7, False)
    team = world.add_filter(model.CollisionFilter(collisionSystems=('a',),
                                                  notCollideWithSystems=('a',)))
    other = world.add_filter(model.CollisionFilter(collisionSystems=('b',)))
    world.collider_filter[1:200] = team
    world.collider_filter[200:300] = other
    dynamic = np.flatnonzero(world.motion_type == 2)
    world.collider_filter[dynamic[:10]] = team
    assert BroadPhase().pairs(world) == full_sweep(world, monkeypatch)


def test_a_world_with_few_static_bodies_is_swept_whole() -> None:
    world = build(8, False)
    world.remove_body(1)
    small = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    box = small.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
    for x in range(10):
        small.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=box),
                       position=(float(x), 0.0, 0.0))
    split = BroadPhase()
    split.pairs(small)
    assert split._inert is None                # noqa: SLF001 the index is what is being exercised
