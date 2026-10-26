"""A static body that arrives with a box is not refitted by the next step.

``add_body`` and ``place_body`` fit the body's box themselves, so the step has
nothing to find out about it. Finding out is a comparison of the pose of every
body in the world with the pose its box was fitted at, and a world of 230,000
trunks that streams a rock at a time pays that for each.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld


def world_with_statics(count: int = 300) -> tuple[PhysicsWorld, int, int]:
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    world.log_events = False
    box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
    for x in range(count):
        world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=box),
                       position=(float(x * 3), 0.5, 0.0))
    ball = world.add_shape(model.Shape.sphere(0.3))
    mover = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                           collider=model.Collider(shape=ball), position=(0.0, 5.0, 0.0))
    world.step(1 / 60)
    world.step(1 / 60)
    return world, box, mover


class Refits:
    """The rows each refit of a world's boxes was asked to work out."""

    def __init__(self, world: PhysicsWorld, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rows: list[list[int]] = []
        original = world._refit_rows           # noqa: SLF001 the refit is what is being counted

        def refit(rows, *rest):
            self.rows.append(rows.tolist())
            return original(rows, *rest)

        monkeypatch.setattr(world, '_refit_rows', refit)


def test_a_static_body_added_between_steps_is_not_refitted(monkeypatch) -> None:
    world, box, mover = world_with_statics()
    refits = Refits(world, monkeypatch)
    arrived = world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=box),
                             position=(7.0, 0.5, 4.0))
    world.step(1 / 60)
    assert refits.rows == [[mover]]
    assert arrived not in refits.rows[0]


def test_a_static_body_placed_between_steps_is_not_refitted(monkeypatch) -> None:
    world, _, mover = world_with_statics()
    refits = Refits(world, monkeypatch)
    world.place_body(10, position=(31.0, 0.5, 9.0))
    world.step(1 / 60)
    assert refits.rows == [[mover]]


def test_the_box_of_an_arrival_is_where_it_was_fitted() -> None:
    world, box, _ = world_with_statics()
    arrived = world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=box),
                             position=(7.0, 0.5, 4.0))
    world.step(1 / 60)
    assert world.aabb_min[arrived] == pytest.approx(np.array([6.5, 0.0, 3.5]) - 0.05)
    assert world.aabb_max[arrived] == pytest.approx(np.array([7.5, 1.0, 4.5]) + 0.05)


def test_a_static_body_whose_pose_is_written_and_marked_has_its_box_there(monkeypatch) -> None:
    world, _, mover = world_with_statics()
    refits = Refits(world, monkeypatch)
    world.pose_arrays()[0][20] = (61.0, 0.5, 2.0)
    world.moved([20])
    world.step(1 / 60)
    assert refits.rows == [[mover]]
    assert world.aabb_min[20] == pytest.approx(np.array([60.5, 0.0, 1.5]) - 0.05)


class TestTheShapeTableOfTheRefit:
    """The kind and half-extent of every body's shape are kept for the refit; a body that
    arrives into a freed slot or leaves one changes its own row and not the table."""

    def builds(self, world, monkeypatch) -> list[int]:
        made: list[int] = []
        original = world._build_refit_cache    # noqa: SLF001 the table is what is being counted

        def build():
            made.append(1)
            return original()

        monkeypatch.setattr(world, '_build_refit_cache', build)
        return made

    def test_a_body_that_leaves_and_one_that_takes_its_slot_do_not_rebuild_it(
            self, monkeypatch) -> None:
        world, box, _ = world_with_statics()
        made = self.builds(world, monkeypatch)
        world.remove_body(40)
        world.step(1 / 60)
        taken = world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=box),
                               position=(8.0, 0.5, 8.0))
        world.step(1 / 60)
        assert taken == 40
        assert made == []

    def test_the_row_of_a_body_that_takes_a_slot_is_its_own(self) -> None:
        world, box, _ = world_with_statics()
        world.remove_body(40)
        world.step(1 / 60)
        taken = world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=box),
                               position=(8.0, 0.5, 8.0))
        world.pose_arrays()[0][taken] = (8.0, 0.5, 9.0)
        world.moved([taken])
        world.step(1 / 60)
        assert world.aabb_min[taken] == pytest.approx(np.array([7.5, 0.0, 8.5]) - 0.05)

    def test_the_row_of_a_body_that_leaves_is_no_shape(self) -> None:
        world, _, _ = world_with_statics()
        world.step(1 / 60)
        world.remove_body(40)
        world.step(1 / 60)
        kind, _half = world._refit_shapes()    # noqa: SLF001 the table is what is being read
        assert kind[40] == world._RFIT_NONE    # noqa: SLF001 the table is what is being read
