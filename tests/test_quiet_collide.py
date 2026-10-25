"""A step with nothing that can collide skips the broad phase; one with something does not."""
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
