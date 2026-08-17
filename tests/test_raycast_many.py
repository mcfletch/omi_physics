"""Several rays at once, sharing the work they have in common.

A vehicle casts one ray per wheel every step, and they all start within a car's
length of each other: the same bodies are worth testing and the same part of the
same mesh comes back from each. Cast one at a time, that work is done once per
wheel, and for a raycast vehicle it is most of the step.

What is asserted here is that the bundle answers exactly what the rays would
have answered on their own, and that it does less work to do it.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.raycast import raycast, raycast_many
from omi_physics.world import PhysicsWorld


def _hilly(x, z):
    return 2.0 * np.sin(x * 0.15) + 1.5 * np.cos(z * 0.11)


def _mesh_world(side=40, extent=30.0):
    """A rolling triangle-mesh floor, as a terrain tile is."""
    world = PhysicsWorld()
    axis = np.linspace(-extent, extent, side)
    gx, gz = np.meshgrid(axis, axis, indexing='ij')
    points = np.stack([gx.ravel(), _hilly(gx, gz).ravel(), gz.ravel()], axis=-1)
    faces = []
    for i in range(side - 1):
        for j in range(side - 1):
            a = i * side + j
            faces += [(a, a + 1, a + side), (a + 1, a + side + 1, a + side)]
    shape = world.add_shape(
        model.Shape.trimesh(points, np.asarray(faces, dtype='i')))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape))
    world.refit_aabbs()
    return world


def _wheels(centre=(0.0, 6.0, 0.0), wheelbase=2.6, track=1.6):
    """Four hubs, as a car's are."""
    x, y, z = centre
    return [(x + dx, y, z + dz)
            for dx in (-track / 2, track / 2)
            for dz in (-wheelbase / 2, wheelbase / 2)]


DOWN = [(0.0, -1.0, 0.0)] * 4


class TestItAnswersWhatTheRaysWouldHave:
    def test_every_ray_gets_an_answer(self) -> None:
        world = _mesh_world()
        found = raycast_many(world, _wheels(), DOWN, max_distance=20.0)
        assert len(found) == 4
        assert all(hit is not None for hit in found)

    def test_each_answer_is_the_one_the_ray_alone_gives(self) -> None:
        world = _mesh_world()
        origins = _wheels()
        together = raycast_many(world, origins, DOWN, max_distance=20.0)
        for origin, hit in zip(origins, together, strict=True):
            alone = raycast(world, origin, (0.0, -1.0, 0.0), max_distance=20.0)
            assert (hit is None) == (alone is None)
            if hit is not None:
                assert hit.distance == pytest.approx(alone.distance, abs=1e-9)
                assert np.allclose(hit.point, alone.point, atol=1e-9)
                assert np.allclose(hit.normal, alone.normal, atol=1e-9)
                assert hit.triangle == alone.triangle
                assert hit.body == alone.body

    def test_the_answers_come_back_in_order(self) -> None:
        world = _mesh_world()
        origins = [(-10.0, 6.0, 0.0), (0.0, 6.0, 0.0), (10.0, 6.0, 0.0)]
        found = raycast_many(world, origins, [(0, -1, 0)] * 3, max_distance=20.0)
        for origin, hit in zip(origins, found, strict=True):
            assert hit.point[0] == pytest.approx(origin[0])

    def test_a_ray_that_meets_nothing_says_so(self) -> None:
        world = _mesh_world()
        origins = _wheels(centre=(0.0, 6.0, 0.0))[:2] + [(500.0, 6.0, 0.0)]
        found = raycast_many(world, origins, [(0, -1, 0)] * 3, max_distance=20.0)
        assert found[-1] is None
        assert found[0] is not None

    def test_a_bundle_that_meets_nothing_at_all(self) -> None:
        world = _mesh_world()
        found = raycast_many(world, [(500.0, 6.0, 0.0)], [(0, -1, 0)],
                             max_distance=20.0)
        assert found == [None]

    def test_a_ray_pointing_nowhere_is_a_miss_not_an_error(self) -> None:
        world = _mesh_world()
        found = raycast_many(world, [(0.0, 6.0, 0.0), (1.0, 6.0, 0.0)],
                             [(0, 0, 0), (0, -1, 0)], max_distance=20.0)
        assert found[0] is None and found[1] is not None

    def test_directions_need_not_be_normalised(self) -> None:
        world = _mesh_world()
        one = raycast_many(world, [(0.0, 6.0, 0.0)], [(0, -1, 0)],
                           max_distance=20.0)[0]
        many = raycast_many(world, [(0.0, 6.0, 0.0)], [(0, -37.0, 0)],
                            max_distance=20.0)[0]
        assert many.distance == pytest.approx(one.distance)

    def test_no_rays_is_no_answers(self) -> None:
        assert raycast_many(_mesh_world(), [], []) == []

    def test_a_body_can_be_skipped(self) -> None:
        world = _mesh_world()
        box = world.add_shape(model.Shape.box((40.0, 1.0, 40.0)))
        lid = world.add_body(model.Motion(type=model.STATIC),
                             collider=model.Collider(shape=box),
                             position=(0.0, 4.0, 0.0))
        world.refit_aabbs()
        through = raycast_many(world, [(0.0, 6.0, 0.0)], [(0, -1, 0)],
                               max_distance=20.0, skip=(lid,))[0]
        assert through.body != lid

    def test_the_nearest_of_several_bodies_wins(self) -> None:
        world = _mesh_world()
        box = world.add_shape(model.Shape.box((40.0, 1.0, 40.0)))
        lid = world.add_body(model.Motion(type=model.STATIC),
                             collider=model.Collider(shape=box),
                             position=(0.0, 4.0, 0.0))
        world.refit_aabbs()
        found = raycast_many(world, [(0.0, 8.0, 0.0)], [(0, -1, 0)],
                             max_distance=20.0)[0]
        assert found.body == lid

    def test_rays_in_different_directions_are_still_right(self) -> None:
        """Nothing about the bundle assumes the rays are parallel."""
        world = _mesh_world()
        origins = [(0.0, 6.0, 0.0)] * 3
        headings = [(0, -1, 0), (0.4, -1, 0.2), (-0.6, -1, 0.3)]
        together = raycast_many(world, origins, headings, max_distance=20.0)
        for origin, heading, hit in zip(origins, headings, together, strict=True):
            alone = raycast(world, origin, heading, max_distance=20.0)
            assert hit.distance == pytest.approx(alone.distance, abs=1e-9)


class TestItDoesLessWorkThanCastingEachOne:
    def _counted(self, monkeypatch):
        from omi_physics import raycast as module
        walks = []
        real = module._hit_triangles

        def counting(origin, heading, triangles, limit):
            walks.append(len(triangles))
            return real(origin, heading, triangles, limit)

        monkeypatch.setattr(module, '_hit_triangles', counting)
        return walks

    def test_the_mesh_is_walked_once_for_the_bundle(self, monkeypatch) -> None:
        from omi_physics import trigrid
        walked = []
        real = trigrid.TriangleGrid.ray

        def counting(self, origin, heading, limit):
            walked.append(1)
            return real(self, origin, heading, limit)

        monkeypatch.setattr(trigrid.TriangleGrid, 'ray', counting)
        world = _mesh_world()
        raycast_many(world, _wheels(), DOWN, max_distance=20.0)
        assert not walked, "the grid was walked per ray"

    def test_each_ray_is_tested_against_one_candidate_set(self, monkeypatch) -> None:
        walks = self._counted(monkeypatch)
        world = _mesh_world()
        raycast_many(world, _wheels(), DOWN, max_distance=20.0)
        assert len(walks) == 4
        assert len(set(walks)) == 1, "the rays were given different candidates"

    def test_the_shared_set_is_smaller_than_the_whole_mesh(self, monkeypatch) -> None:
        walks = self._counted(monkeypatch)
        world = _mesh_world(side=40)
        raycast_many(world, _wheels(), DOWN, max_distance=20.0)
        assert walks[0] < 0.2 * (39 * 39 * 2)


class TestAVehicleUsesIt:
    def test_a_car_casts_its_wheels_together(self, monkeypatch) -> None:
        from omi_physics import vehicle as module
        from omi_physics.vehicle import RaycastVehicle, VehicleTuning, car_wheels
        bundles = []
        real = module.raycast_many

        def counting(world, origins, directions, **named):
            bundles.append(len(origins))
            return real(world, origins, directions, **named)

        monkeypatch.setattr(module, 'raycast_many', counting)
        world = _mesh_world()
        body = world.add_body(model.Motion(type=model.DYNAMIC, mass=1200.0),
                              position=(0.0, 6.0, 0.0))
        car = RaycastVehicle(world, body, car_wheels(), VehicleTuning())
        car.update(1.0 / 120.0)
        assert bundles == [4]

    def test_it_still_holds_itself_up(self) -> None:
        from omi_physics.vehicle import RaycastVehicle, VehicleTuning, car_wheels
        world = _mesh_world()
        body = world.add_body(model.Motion(type=model.DYNAMIC, mass=1200.0),
                              position=(0.0, 6.0, 0.0))
        car = RaycastVehicle(world, body, car_wheels(), VehicleTuning())
        for _ in range(360):
            car.control()
            car.update(1.0 / 120.0)
            world.step(1.0 / 120.0)
        ground = float(_hilly(np.array([0.0]), np.array([0.0]))[0])
        assert float(world.position[body][1]) == pytest.approx(ground + 0.64,
                                                               abs=0.35)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


class TestEverythingAlongTheRay:
    """The first thing hit is not always the answer: a shot passes through a
    trigger volume on its way to a wall."""

    def _stack(self):
        world = PhysicsWorld()
        box = world.add_shape(model.Shape.box((10.0, 1.0, 10.0)))
        for height in (0.0, 3.0, 6.0):
            world.add_body(model.Motion(type=model.STATIC),
                           collider=model.Collider(shape=box),
                           position=(0.0, height, 0.0))
        world.refit_aabbs()
        return world

    def test_it_finds_every_one(self) -> None:
        from omi_physics.raycast import bodies_along
        found = bodies_along(self._stack(), (0.0, 20.0, 0.0), (0, -1, 0),
                             max_distance=40.0)
        assert len(found) == 3

    def test_nearest_first(self) -> None:
        from omi_physics.raycast import bodies_along
        found = bodies_along(self._stack(), (0.0, 20.0, 0.0), (0, -1, 0),
                             max_distance=40.0)
        assert [hit.distance for hit in found] == sorted(
            hit.distance for hit in found)

    def test_the_first_is_the_one_a_single_cast_gives(self) -> None:
        from omi_physics.raycast import bodies_along
        world = self._stack()
        found = bodies_along(world, (0.0, 20.0, 0.0), (0, -1, 0),
                             max_distance=40.0)
        alone = raycast(world, (0.0, 20.0, 0.0), (0, -1, 0), max_distance=40.0)
        assert found[0].body == alone.body

    def test_a_skipped_body_is_not_among_them(self) -> None:
        from omi_physics.raycast import bodies_along
        world = self._stack()
        found = bodies_along(world, (0.0, 20.0, 0.0), (0, -1, 0),
                             max_distance=40.0, skip=(1,))
        assert all(hit.body != 1 for hit in found)

    def test_a_ray_that_meets_nothing_finds_nothing(self) -> None:
        from omi_physics.raycast import bodies_along
        assert bodies_along(self._stack(), (500.0, 20.0, 0.0), (0, -1, 0),
                            max_distance=40.0) == []


class TestASlotThatIsReused:
    """A streaming world removes the ground behind it and adds the ground in
    front, and the freed slot is the one the new body lands in. Anything the
    caster remembered about the old body is then about the wrong mesh."""

    def _patch(self, world, x0):
        points = np.array([(x0, 0.0, -5.0), (x0 + 10.0, 0.0, -5.0),
                           (x0, 0.0, 5.0), (x0 + 10.0, 0.0, 5.0)], dtype='d')
        shape = world.add_shape(model.Shape.trimesh(
            points, np.array([(0, 1, 2), (1, 3, 2)], dtype='i')))
        body = world.add_body(model.Motion(type=model.STATIC),
                              collider=model.Collider(shape=shape))
        world.refit_aabbs()
        return body

    def _under(self, world, x):
        return raycast(world, (x, 10.0, 0.0), (0.0, -1.0, 0.0),
                       max_distance=40.0)

    def test_the_new_mesh_is_the_one_that_is_hit(self) -> None:
        world = PhysicsWorld()
        first = self._patch(world, 0.0)
        world.remove_body(first)
        second = self._patch(world, 100.0)
        assert second == first, "the test needs the slot to be reused"
        assert self._under(world, 105.0) is not None

    def test_the_old_mesh_is_not(self) -> None:
        world = PhysicsWorld()
        world.remove_body(self._patch(world, 0.0))
        self._patch(world, 100.0)
        assert self._under(world, 5.0) is None

    def test_it_survives_being_recycled_again_and_again(self) -> None:
        """What a lap of a streamed circuit does to one slot."""
        world = PhysicsWorld()
        body = self._patch(world, 0.0)
        for step in range(1, 12):
            world.remove_body(body)
            body = self._patch(world, step * 100.0)
            assert self._under(world, step * 100.0 + 5.0) is not None

    def test_a_removed_body_is_forgotten(self) -> None:
        """Or a world that streams for an hour remembers every mesh it ever
        held."""
        world = PhysicsWorld()
        body = self._patch(world, 0.0)
        self._under(world, 5.0)
        world.remove_body(body)
        assert body not in getattr(world, '_raycast_meshes', {})

    def test_a_bundle_of_rays_sees_the_new_mesh_too(self) -> None:
        world = PhysicsWorld()
        world.remove_body(self._patch(world, 0.0))
        self._patch(world, 100.0)
        found = raycast_many(world, [(103.0, 10.0, 0.0), (107.0, 10.0, 0.0)],
                             [(0, -1, 0)] * 2, max_distance=40.0)
        assert all(hit is not None for hit in found)
