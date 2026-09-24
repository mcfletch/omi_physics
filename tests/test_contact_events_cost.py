"""What recording contact events costs a step (no GL).

The share of a step spent in the tracker, measured on a resting pile of boxes
kept awake so every pair is in contact on every step. A ratio rather than a
time, so the budget means the same on a fast machine and a slow one.
"""
import time

import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld

STEP = 1.0 / 120.0


def pile(side=10, layers=3):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    ground = world.add_shape(model.Shape.box((200, 1, 200)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=ground),
                   position=(0, -0.5, 0))
    box = world.add_shape(model.Shape.box((1, 1, 1)))
    for i in range(side):
        for j in range(side):
            for k in range(layers):
                world.add_body(model.Motion(type=model.DYNAMIC),
                               collider=model.Collider(shape=box),
                               position=(i * 1.02, 0.5 + k, j * 1.02))
    for _ in range(30):
        world.step(STEP)
    return world


def tracker_share(world, monkeypatch, steps=20):
    spent = [0.0]
    record = PhysicsWorld._record_contacts

    def timed(self, *args):
        start = time.perf_counter()
        record(self, *args)
        spent[0] += time.perf_counter() - start

    monkeypatch.setattr(PhysicsWorld, '_record_contacts', timed)
    start = time.perf_counter()
    for _ in range(steps):
        world.step(STEP)
    return spent[0] / (time.perf_counter() - start)


@pytest.mark.serial
def test_reporting_every_pair_costs_under_five_percent(monkeypatch) -> None:
    world = pile()
    world.contact_reporting = 'all'
    assert len(world.contacts) > 1000
    assert tracker_share(world, monkeypatch) < 0.05


def test_a_world_nobody_listens_to_runs_no_tracker(monkeypatch) -> None:
    world = pile(side=4, layers=1)
    assert world.contact_reporting == 'off'
    assert tracker_share(world, monkeypatch, steps=3) == 0.0
