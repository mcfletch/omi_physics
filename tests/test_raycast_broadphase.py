"""A cast should not read the whole world to answer about one body.

A streamed world holds hundreds of static colliders, and a vehicle casts a ray
per wheel per step. Testing every body in turn -- even cheaply -- is then the
frame: the cost has to fall with the bodies the ray could possibly reach, not
with the bodies that exist.

These count exact shape tests rather than timing them, so what is asserted is
the behaviour rather than a speed that depends on the machine.
"""
import numpy as np
import pytest

from omi_physics import model, raycast as raycast_module
from omi_physics.raycast import raycast
from omi_physics.world import PhysicsWorld


def _tiles(count=200, side=20.0):
    """A row of ground tiles, the shape a streamed world takes."""
    world = PhysicsWorld()
    points = np.array([(0, 0, 0), (side, 0, 0), (0, 0, side), (side, 0, side)], 'd')
    faces = np.array([(0, 1, 2), (1, 3, 2)], 'i')
    for index in range(count):
        shape = world.add_shape(model.Shape.trimesh(points, faces))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=shape),
                       position=(index * side, 0.0, 0.0))
    world.refit_aabbs()
    return world


@pytest.fixture
def tested(monkeypatch):
    """Count the bodies a cast tests exactly."""
    seen = []
    original = raycast_module._hit_body

    def counting(world, body, *args):
        seen.append(body)
        return original(world, body, *args)

    monkeypatch.setattr(raycast_module, '_hit_body', counting)
    return seen


class TestOnlyWhatTheRayCouldReach:
    def test_a_short_ray_tests_a_handful_of_bodies(self, tested) -> None:
        world = _tiles()
        hit = raycast(world, (100.0, 5.0, 10.0), (0, -1, 0), max_distance=10.0)
        assert hit is not None
        assert len(tested) <= 4, "%d bodies tested for one downward ray" % len(tested)

    def test_it_still_finds_the_right_one(self) -> None:
        world = _tiles()
        hit = raycast(world, (105.0, 5.0, 10.0), (0, -1, 0), max_distance=10.0)
        assert hit is not None
        assert float(hit.point[1]) == pytest.approx(0.0, abs=1e-6)
        assert hit.body == 5

    def test_a_long_ray_along_the_row_still_finds_the_nearest(self) -> None:
        """Grazing in from off the end, it must stop at the first tile it
        actually reaches rather than at whichever was tested first."""
        world = _tiles()
        hit = raycast(world, (-50.0, 2.0, 10.0), (1, -0.02, 0), max_distance=400.0)
        assert hit is not None
        assert hit.body <= 3, "it found a tile behind a nearer one"

    def test_a_ray_that_meets_nothing_says_so(self, tested) -> None:
        world = _tiles()
        assert raycast(world, (100.0, 50.0, 500.0), (0, -1, 0)) is None
        assert len(tested) == 0

    def test_the_cost_does_not_grow_with_the_world(self, tested) -> None:
        """The point: twice the tiles, the same work for one wheel's ray."""
        raycast(_tiles(count=50), (100.0, 5.0, 10.0), (0, -1, 0), max_distance=10.0)
        small = len(tested)
        tested.clear()
        raycast(_tiles(count=400), (100.0, 5.0, 10.0), (0, -1, 0), max_distance=10.0)
        assert len(tested) <= small


class TestItStillAnswersTheSameWay:
    def test_a_skipped_body_is_not_hit(self) -> None:
        """Well inside one tile, so skipping it leaves nothing underneath."""
        world = _tiles()
        under = raycast(world, (105.0, 5.0, 10.0), (0, -1, 0), max_distance=10.0)
        assert under is not None
        assert raycast(world, (105.0, 5.0, 10.0), (0, -1, 0), max_distance=10.0,
                       skip=(under.body,)) is None

    def test_the_nearest_of_two_stacked_bodies_wins(self) -> None:
        world = PhysicsWorld()
        box = world.add_shape(model.Shape.box((10.0, 1.0, 10.0)))
        low = world.add_body(model.Motion(type=model.STATIC),
                             collider=model.Collider(shape=box),
                             position=(0.0, 0.0, 0.0))
        high = world.add_body(model.Motion(type=model.STATIC),
                              collider=model.Collider(shape=box),
                              position=(0.0, 5.0, 0.0))
        world.refit_aabbs()
        hit = raycast(world, (0.0, 20.0, 0.0), (0, -1, 0))
        assert hit is not None and hit.body == high
        assert low != high

    def test_a_body_is_castable_before_the_world_has_ever_stepped(self) -> None:
        """A body arrives with a box, so building a world and asking it a
        question needs no step in between."""
        world = PhysicsWorld()
        box = world.add_shape(model.Shape.box((4.0, 1.0, 4.0)))
        placed = world.add_body(model.Motion(type=model.STATIC),
                                collider=model.Collider(shape=box),
                                position=(30.0, 0.0, 0.0))
        hit = raycast(world, (30.0, 9.0, 0.0), (0, -1, 0))
        assert hit is not None and hit.body == placed

    def test_a_body_moved_since_the_refit_is_found_after_one(self) -> None:
        """The boxes are the world's, and a refit is what refreshes them --
        the same rule collision follows."""
        world = _tiles(count=20)
        world.position[3] = (500.0, 0.0, 0.0)
        world.refit_aabbs()
        hit = raycast(world, (510.0, 5.0, 10.0), (0, -1, 0), max_distance=10.0)
        assert hit is not None and hit.body == 3

    def test_a_ray_from_inside_a_body_still_reports_it(self) -> None:
        world = PhysicsWorld()
        box = world.add_shape(model.Shape.box((10.0, 10.0, 10.0)))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=box))
        world.refit_aabbs()
        assert raycast(world, (0.0, 0.0, 0.0), (0, 1, 0)) is not None
