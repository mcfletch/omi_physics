"""A step with nothing that can collide skips the broad phase; one with something does not."""
import numpy as np
import pytest
from omi_physics import model
from omi_physics.broadphase import BroadPhase
from omi_physics.world import PhysicsWorld


def kinematic_over_ground() -> PhysicsWorld:
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    floor = world.add_shape(model.Shape.box((100.0, 1.0, 100.0)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0.0, -0.5, 0.0))
    mover = world.add_shape(model.Shape.capsule(height=1.0, radius=0.3))
    world.add_body(model.Motion(type=model.KINEMATIC, linearVelocity=(1.0, 0.0, 0.0)),
                   collider=model.Collider(shape=mover), position=(0.0, 0.81, 0.0))
    return world


def counted(monkeypatch) -> list[int]:
    asked: list[int] = []
    original = BroadPhase.pairs
    monkeypatch.setattr(BroadPhase, 'pairs',
                        lambda self, *a, **k: asked.append(1) or original(self, *a, **k))
    return asked


def test_a_kinematic_body_over_the_ground_asks_the_broad_phase_nothing(monkeypatch):
    world = kinematic_over_ground()
    asked = counted(monkeypatch)
    for _ in range(10):
        world.step(1 / 60)
    assert asked == []


def test_a_dynamic_body_or_a_trigger_brings_the_broad_phase_back(monkeypatch):
    world = kinematic_over_ground()
    ball = world.add_shape(model.Shape.sphere(0.2))
    world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                   collider=model.Collider(shape=ball), position=(5.0, 2.0, 0.0))
    asked = counted(monkeypatch)
    world.step(1 / 60)
    assert asked

    world = kinematic_over_ground()
    zone = world.add_shape(model.Shape.box((1.0, 2.0, 1.0)))
    trigger = world.add_body(model.Motion(type=model.STATIC), trigger=model.Trigger(shape=zone),
                             position=(0.5, 1.0, 0.0))
    entered = []
    world.trigger_listeners.append(lambda *event: entered.append(event))
    asked = counted(monkeypatch)
    for _ in range(30):
        world.step(1 / 60)
    assert asked
    assert any(trigger in event for event in entered)


def test_a_kinematic_body_wakes_a_sleeping_one_it_walks_into():
    world = kinematic_over_ground()
    crate = world.add_shape(model.Shape.box((0.8, 0.8, 0.8)))
    box = world.add_body(model.Motion(type=model.DYNAMIC, mass=20.0),
                         collider=model.Collider(shape=crate), position=(2.0, 0.4, 0.0))
    world.awake[box] = False
    for _ in range(150):
        world.step(1 / 60)
    assert float(world.position[box][0]) > 2.1


def test_a_kinematic_body_far_from_every_trigger_asks_the_broad_phase_nothing(monkeypatch):
    world = kinematic_over_ground()
    zone = world.add_shape(model.Shape.box((1.0, 2.0, 1.0)))
    world.add_body(model.Motion(type=model.STATIC), trigger=model.Trigger(shape=zone),
                   position=(30.0, 1.0, 0.0))
    asked = counted(monkeypatch)
    for _ in range(10):
        world.step(1 / 60)
    assert asked == []


def walkers_among_triggers(seed: int) -> PhysicsWorld:
    rng = np.random.default_rng(seed)
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    floor = world.add_shape(model.Shape.box((100.0, 1.0, 100.0)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0.0, -0.5, 0.0))
    zones = [world.add_shape(model.Shape.box(tuple(rng.uniform(0.3, 2.0, 3)))),
             world.add_shape(model.Shape.sphere(float(rng.uniform(0.2, 1.0))))]
    for _ in range(int(rng.integers(3, 15))):
        kind = model.KINEMATIC if rng.random() < 0.2 else model.STATIC
        world.add_body(model.Motion(type=kind, linearVelocity=tuple(rng.uniform(-1, 1, 3))
                                    if kind == model.KINEMATIC else (0.0, 0.0, 0.0)),
                       trigger=model.Trigger(shape=int(rng.choice(zones))),
                       position=tuple(rng.uniform((-6.0, 0.0, -6.0), (6.0, 2.0, 6.0))))
    mover = world.add_shape(model.Shape.capsule(height=1.0, radius=0.3))
    for _ in range(int(rng.integers(1, 4))):
        start = rng.uniform((-6.0, 0.0, -6.0), (6.0, 2.0, 6.0))
        velocity = rng.uniform((-3, -0.2, -3), (3, 0.2, 3))
        world.add_body(model.Motion(type=model.KINEMATIC, linearVelocity=tuple(velocity)),
                       collider=model.Collider(shape=mover), position=tuple(start))
        # A zone across its way, which it enters and leaves.
        world.add_body(model.Motion(type=model.STATIC), trigger=model.Trigger(shape=zones[0]),
                       position=tuple(start + velocity * 1.5))
    return world


@pytest.mark.parametrize('seed', range(20))
def test_triggers_fire_as_when_every_step_runs_the_collision_pass(seed, monkeypatch):
    events = []
    for always in (False, True):
        world = walkers_among_triggers(seed)
        heard = []
        world.trigger_listeners.append(lambda *event, keep=heard: keep.append(event))
        with monkeypatch.context() as patch:
            if always:
                patch.setattr(PhysicsWorld, '_nothing_to_collide', lambda self, stages: False)
            for _ in range(240):
                world.step(1 / 60)
        events.append(heard)
    assert events[0] == events[1]
    assert events[0], 'no trigger fired, so nothing was compared'
