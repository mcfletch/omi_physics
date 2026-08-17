"""A body that has not moved keeps its collision proxy.

Building a proxy for a triangle mesh means transforming every vertex into the
world and indexing the result, which for a terrain tile is thousands of
triangles. A static tile never moves, so doing that again every step is the
whole cost of a step -- and a game streaming a landscape has a few of them
resident at all times.

These count constructions rather than timing them, so what they assert is the
behaviour ("it was not rebuilt") rather than a speed that depends on the
machine.
"""
import numpy as np
import pytest

from omi_physics import body, model
from omi_physics.world import PhysicsWorld

STEP = 1.0 / 120.0


def _grid_mesh(side=24, extent=40.0):
    """A flat triangle mesh with enough triangles to be worth not rebuilding."""
    axis = np.linspace(-extent, extent, side)
    x, z = np.meshgrid(axis, axis, indexing='ij')
    points = np.stack([x.ravel(), np.zeros(x.size), z.ravel()], axis=-1)
    faces = []
    for i in range(side - 1):
        for j in range(side - 1):
            a = i * side + j
            faces += [(a, a + 1, a + side), (a + 1, a + side + 1, a + side)]
    return points, np.asarray(faces, dtype='i')


@pytest.fixture
def counted(monkeypatch):
    """Count how many triangle-mesh proxies get built."""
    built = []
    original = body.TriangleMeshProxy

    class Counting(original):                    # type: ignore[misc, valid-type]
        def __init__(self, *args, **named):
            built.append(1)
            super().__init__(*args, **named)

    monkeypatch.setattr(body, 'TriangleMeshProxy', Counting)
    return built


def _world_with_terrain_and_a_box():
    world = PhysicsWorld()
    points, faces = _grid_mesh()
    terrain = world.add_shape(model.Shape.trimesh(points, faces))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=terrain))
    box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
    mover = world.add_body(model.Motion(type=model.DYNAMIC, mass=10.0),
                           collider=model.Collider(shape=box),
                           position=(0.0, 0.6, 0.0))
    return world, mover


class TestAStaticMeshIsBuiltOnce:
    def test_stepping_does_not_rebuild_it(self, counted) -> None:
        world, _ = _world_with_terrain_and_a_box()
        world.step(STEP)
        first = len(counted)
        assert first >= 1, "the mesh was never built at all"
        for _ in range(20):
            world.step(STEP)
        assert len(counted) == first, (
            "%d rebuilds of a mesh that never moved" % (len(counted) - first))

    def test_the_contacts_are_still_generated(self) -> None:
        """Reuse must not mean the box quietly stops colliding."""
        world, mover = _world_with_terrain_and_a_box()
        world.position[mover] = (0.0, 0.2, 0.0)      # sunk into the ground
        for _ in range(30):
            world.step(STEP)
        assert float(world.position[mover][1]) > 0.4, "the box sank through"

    def test_a_mesh_that_is_moved_collides_where_it_now_is(self) -> None:
        """Static does not mean immovable, and a proxy that was kept must not
        go on speaking for the place the mesh used to be."""
        world, mover = _world_with_terrain_and_a_box()
        for _ in range(120):
            world.step(STEP)
        resting = float(world.position[mover][1])
        world.position[0] = (0.0, -5.0, 0.0)         # the ground drops away
        # A body resting on it has gone to sleep by now, and the ground moving
        # out from under a sleeper is not something the engine notices for it.
        world.wake(mover)
        for _ in range(240):
            world.step(STEP)
        settled = float(world.position[mover][1])
        assert settled == pytest.approx(resting - 5.0, abs=0.2), (
            "the box is resting on where the ground used to be")

    def test_a_mesh_that_is_turned_is_rebuilt(self, counted) -> None:
        world, _ = _world_with_terrain_and_a_box()
        world.step(STEP)
        before = len(counted)
        world.orientation[0] = (0.0, 0.383, 0.0, 0.924)
        world.step(STEP)
        assert len(counted) > before


class TestTheAabbIsNotRecomputedEither:
    def test_refitting_a_still_mesh_costs_nothing(self, counted) -> None:
        world, _ = _world_with_terrain_and_a_box()
        world.refit_aabbs()
        before = len(counted)
        for _ in range(10):
            world.refit_aabbs()
        assert len(counted) == before

    def test_the_aabb_is_right(self) -> None:
        world, _ = _world_with_terrain_and_a_box()
        world.refit_aabbs()
        assert world.aabb_min[0][0] == pytest.approx(-40.0, abs=0.1)
        assert world.aabb_max[0][2] == pytest.approx(40.0, abs=0.1)

    def test_moving_the_mesh_moves_its_aabb(self) -> None:
        world, _ = _world_with_terrain_and_a_box()
        world.refit_aabbs()
        world.position[0] = (100.0, 0.0, 0.0)
        world.refit_aabbs()
        assert world.aabb_min[0][0] == pytest.approx(60.0, abs=0.1)


class TestManyTilesStayAffordable:
    def test_a_landscape_of_meshes_steps_at_a_playable_rate(self) -> None:
        """Not a timing gate: a count of the work done per step.

        Nine terrain tiles and a car-sized box, the shape of a streamed world.
        What must not happen is the per-step cost growing with the tiles the
        streamer happens to be holding.
        """
        world = PhysicsWorld()
        points, faces = _grid_mesh(side=20, extent=20.0)
        for tile_x in (-40.0, 0.0, 40.0):
            for tile_z in (-40.0, 0.0, 40.0):
                shape = world.add_shape(model.Shape.trimesh(points, faces))
                world.add_body(model.Motion(type=model.STATIC),
                               collider=model.Collider(shape=shape),
                               position=(tile_x, 0.0, tile_z))
        box = world.add_shape(model.Shape.box((1.8, 0.6, 4.0)))
        car = world.add_body(model.Motion(type=model.DYNAMIC, mass=1200.0),
                             collider=model.Collider(shape=box),
                             position=(0.0, 1.0, 0.0))
        built: list[int] = []
        original = body.TriangleMeshProxy

        class Counting(original):                # type: ignore[misc, valid-type]
            def __init__(self, *args, **named):
                built.append(1)
                super().__init__(*args, **named)

        body.TriangleMeshProxy = Counting        # type: ignore[misc]
        try:
            world.step(STEP)
            warmed = len(built)
            for _ in range(30):
                world.step(STEP)
            assert len(built) == warmed
        finally:
            body.TriangleMeshProxy = original    # type: ignore[misc]
        assert float(world.position[car][1]) > 0.2


class TestRefittingCostsWhatMoved:
    """A world's bounds are recomputed every step. A body that has not moved
    has the bounds it had, and a streamed landscape is dozens of bodies that
    never move -- so the recompute has to cost what moved rather than what is
    there.
    """

    def _world(self, tiles=24):
        world = PhysicsWorld()
        points, faces = _grid_mesh(side=8, extent=20.0)
        for index in range(tiles):
            shape = world.add_shape(model.Shape.trimesh(points, faces))
            world.add_body(model.Motion(type=model.STATIC),
                           collider=model.Collider(shape=shape),
                           position=(index * 40.0, 0.0, 0.0))
        box = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
        mover = world.add_body(model.Motion(type=model.DYNAMIC, mass=10.0),
                               collider=model.Collider(shape=box),
                               position=(0.0, 4.0, 0.0))
        return world, mover

    def _counted(self, monkeypatch):
        """Count the measuring, which is what a terrain tile's AABB costs."""
        from omi_physics import body
        measured = []
        real = body.world_aabb

        def counting(*args, **named):
            measured.append(1)
            return real(*args, **named)

        monkeypatch.setattr(body, 'world_aabb', counting)
        return measured

    def test_a_still_landscape_is_measured_once(self, monkeypatch) -> None:
        measured = self._counted(monkeypatch)
        world, _mover = self._world()
        world.refit_aabbs()
        first = len(measured)
        assert first, "the tiles were never measured at all"
        for _ in range(20):
            world.refit_aabbs()
        assert len(measured) == first

    def test_a_tile_that_moves_is_measured_again(self, monkeypatch) -> None:
        measured = self._counted(monkeypatch)
        world, _mover = self._world()
        world.refit_aabbs()
        before = len(measured)
        world.position[3] = (999.0, 0.0, 0.0)
        world.refit_aabbs()
        assert len(measured) == before + 1

    def test_only_that_one(self, monkeypatch) -> None:
        measured = self._counted(monkeypatch)
        world, _mover = self._world()
        world.refit_aabbs()
        before = len(measured)
        world.position[3] = (999.0, 0.0, 0.0)
        world.refit_aabbs()
        world.refit_aabbs()
        assert len(measured) == before + 1

    def test_a_slot_reused_by_another_body_is_measured_for_it(self) -> None:
        """A removed body\'s bounds must not answer for whatever takes its slot."""
        world, _mover = self._world(tiles=4)
        world.refit_aabbs()
        world.remove_body(1)
        points, faces = _grid_mesh(side=8, extent=5.0)
        shape = world.add_shape(model.Shape.trimesh(points, faces))
        fresh = world.add_body(model.Motion(type=model.STATIC),
                               collider=model.Collider(shape=shape),
                               position=(0.0, 0.0, 0.0))
        world.refit_aabbs()
        assert float(world.aabb_max[fresh][0]) == pytest.approx(5.0, abs=0.1)

    def test_the_bounds_follow_it(self) -> None:
        world, _mover = self._world()
        world.refit_aabbs()
        world.position[3] = (999.0, 0.0, 0.0)
        world.refit_aabbs()
        assert float(world.aabb_min[3][0]) > 900.0

    def test_the_bounds_of_a_still_tile_are_right(self) -> None:
        world, _mover = self._world()
        world.refit_aabbs()
        assert float(world.aabb_min[0][0]) == pytest.approx(-20.0, abs=0.1)
        assert float(world.aabb_max[0][2]) == pytest.approx(20.0, abs=0.1)

    def test_a_body_added_later_is_measured(self, monkeypatch) -> None:
        measured = self._counted(monkeypatch)
        world, _mover = self._world(tiles=4)
        world.refit_aabbs()
        before = len(measured)
        points, faces = _grid_mesh(side=8, extent=20.0)
        shape = world.add_shape(model.Shape.trimesh(points, faces))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=shape),
                       position=(500.0, 0.0, 0.0))
        world.refit_aabbs()
        assert len(measured) > before
