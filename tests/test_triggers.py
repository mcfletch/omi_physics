"""Phase 4 triggers: enter/stay/exit events, no impulse (no GL)."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld

DT = 1.0 / 60.0


def test_body_passing_through_trigger_fires_enter_and_exit():
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    zone_shape = world.add_shape(model.Shape.box((1, 1, 1)))
    world.add_body(model.Motion(type=model.STATIC),
                   trigger=model.Trigger(shape=zone_shape), position=(0, 0, 0))
    ball_shape = world.add_shape(model.Shape.sphere(0.2))
    ball = world.add_body(model.Motion(type=model.DYNAMIC,
                                       linearVelocity=(2.0, 0, 0)),
                          collider=model.Collider(shape=ball_shape),
                          position=(-3.0, 0, 0))
    events = []
    world.add_trigger_listener(lambda kind, t, o: events.append((kind, o)))
    for _ in range(120):
        world.step(DT)
    kinds = [e[0] for e in events]
    assert 'enter' in kinds
    assert 'exit' in kinds
    assert kinds.index('enter') < kinds.index('exit')


def test_trigger_does_not_perturb_trajectory():
    """A body crossing a trigger volume flies straight through (no impulse)."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    zone = world.add_shape(model.Shape.box((1, 1, 1)))
    world.add_body(model.Motion(type=model.STATIC),
                   trigger=model.Trigger(shape=zone), position=(0, 0, 0))
    ball_shape = world.add_shape(model.Shape.sphere(0.3))
    ball = world.add_body(model.Motion(type=model.DYNAMIC,
                                       linearVelocity=(2.0, 0, 0)),
                          collider=model.Collider(shape=ball_shape),
                          position=(-3.0, 0, 0))
    for _ in range(120):
        world.step(DT)
    assert np.allclose(world.linear_velocity[ball], (2.0, 0, 0))
    assert world.position[ball][1] == pytest.approx(0.0, abs=1e-9)
    assert world.position[ball][2] == pytest.approx(0.0, abs=1e-9)


def test_stay_events_while_inside():
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    zone = world.add_shape(model.Shape.box((2, 2, 2)))
    world.add_body(model.Motion(type=model.STATIC),
                   trigger=model.Trigger(shape=zone), position=(0, 0, 0))
    ball_shape = world.add_shape(model.Shape.sphere(0.2))
    world.add_body(model.Motion(type=model.DYNAMIC),   # sits still inside
                   collider=model.Collider(shape=ball_shape), position=(0, 0, 0))
    seen = []
    world.add_trigger_listener(lambda kind, t, o: seen.append(kind))
    for _ in range(5):
        world.step(DT)
    assert seen[0] == 'enter'
    assert seen.count('stay') >= 3


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
