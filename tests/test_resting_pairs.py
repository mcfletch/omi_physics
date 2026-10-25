"""Pairs of bodies at rest cost a step nothing, and wake as they did when something strikes them.

A pair with no awake dynamic body, no kinematic body, no trigger and no sensor
makes no contact the solver acts on, so the broad phase leaves it out of the
step. The worlds here are stepped with and without that, and must agree to the
bit.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.broadphase import BroadPhase
from omi_physics.world import PhysicsWorld

DT = 1.0 / 120.0


def stacks_world(columns: int = 4, height: int = 4) -> tuple[PhysicsWorld, list[int], int]:
    """Columns of boxes on a floor, a sensor and a trigger among them, and a ball held above one."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    floor = world.add_shape(model.Shape.box((40, 1, 40)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0, -0.5, 0))
    crate = world.add_shape(model.Shape.box((0.5, 0.5, 0.5)))
    boxes = [world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                            collider=model.Collider(shape=crate),
                            position=(column * 2.0, 0.25 + 0.5 * level, 0.0))
             for column in range(columns) for level in range(height)]
    zone = world.add_shape(model.Shape.box((1.0, 3.0, 1.0)))
    world.add_body(model.Motion(type=model.STATIC), trigger=model.Trigger(shape=zone),
                   position=(2.0, 1.0, 0.0))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=crate),
                   position=(4.0, 1.0, 0.6), sensor=True)
    world.log_events = True
    ball = world.add_shape(model.Shape.sphere(0.3))
    dropped = world.add_body(model.Motion(type=model.KINEMATIC),
                             collider=model.Collider(shape=ball), position=(0.0, 6.0, 0.0))
    return world, boxes, dropped


def run(world: PhysicsWorld, boxes: list[int], dropped: int) -> list:
    """Settle the stacks, then strike the first with the ball; the poses and events each step."""
    record = []
    for number in range(900):
        if number == 600:
            world.place_body(dropped, position=(0.0, 2.6, 0.0))
            world.linear_velocity[dropped] = (0.0, -6.0, 0.0)
        world.step(DT)
        record.append((world.position.copy(), world.orientation.copy(),
                       world.awake.copy(), [repr(event) for event in world.contact_log.drain()]))
    return record


def test_resting_pairs_leave_the_world_as_it_was_with_them(monkeypatch):
    world, boxes, dropped = stacks_world()
    pruned = run(world, boxes, dropped)
    original = BroadPhase.pairs
    monkeypatch.setattr(BroadPhase, 'pairs',
                        lambda self, w, active=None: original(self, w))
    world, boxes, dropped = stacks_world()
    every = run(world, boxes, dropped)
    assert any(not awake[boxes].any() for _, _, awake, _ in pruned)      # the stacks slept
    assert pruned[-1][2][boxes].any()                                  # and the ball woke them
    for (p1, q1, a1, e1), (p2, q2, a2, e2) in zip(pruned, every, strict=True):
        assert np.array_equal(p1, p2) and np.array_equal(q1, q2)
        assert np.array_equal(a1, a2) and e1 == e2


def test_a_step_of_sleeping_stacks_is_handed_no_resting_pair(monkeypatch):
    world, boxes, dropped = stacks_world()
    world.linear_velocity[dropped] = 0.0
    for _ in range(600):
        world.step(DT)
    assert not world.awake[boxes].any()
    seen = []
    original = BroadPhase.pairs

    def recording(self, w, active=None):
        found = original(self, w, active) if active is not None else original(self, w)
        seen.extend(found)
        return found

    monkeypatch.setattr(BroadPhase, 'pairs', recording)
    world.step(DT)
    special = set(np.flatnonzero((world.trigger_shape >= 0) | world.sensor).tolist())
    assert all(i in special or j in special or dropped in (i, j) for i, j in seen)


@pytest.mark.parametrize('seed', range(5))
def test_pairs_of_active_bodies_are_the_pairs_that_touch_one(seed):
    rng = np.random.RandomState(seed)
    world = PhysicsWorld()
    shape = world.add_shape(model.Shape.sphere(0.6))
    for _ in range(60):
        world.add_body(model.Motion(type=model.DYNAMIC), collider=model.Collider(shape=shape),
                       position=rng.uniform(-3.0, 3.0, 3))
    world.refit_aabbs(margin=0.0)
    active = rng.random_sample(world.body_count) < 0.3
    every = BroadPhase().pairs(world)
    assert BroadPhase().pairs(world, active) == [
        (i, j) for i, j in every if active[i] or active[j]]
