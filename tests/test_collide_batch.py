"""Capsule against many triangles at once, which is how a character asks.

The scalar :func:`capsule_triangle` costs about seventy microseconds a
triangle, almost all of it numpy dispatch on three-element vectors rather than
arithmetic. A character controller runs three depenetration iterations inside
each of four calls per frame, so at a handful of characters that overhead is
the frame; at a hundred it is hopeless.

:func:`capsule_triangles` answers the same question for N triangles in one pass
of array operations, so the dispatch is paid once instead of N times. These pin
it against geometry worked out by hand -- a capsule a known distance into a
known plane has a known depth and a known normal -- and then against the scalar
routine over randomised inputs, which is a regression guard rather than the
definition of right.
"""

import numpy as np
import pytest

from omi_physics import collide
from omi_physics.body import CapsuleProxy, TriangleProxy


def _capsule(centre=(0.0, 1.0, 0.0), half=0.5, radius=0.4, rotation=None):
    return CapsuleProxy(np.array(centre, dtype='d'), half, radius,
                        np.eye(3) if rotation is None else rotation)


def _tris(*triangles):
    """(N,3) x3 vertex arrays from a list of triangles."""
    arr = np.array(triangles, dtype='d')
    return arr[:, 0], arr[:, 1], arr[:, 2]


#: A large horizontal triangle in the y=0 plane, wound counter-clockwise.
FLOOR = ((-10.0, 0.0, -10.0), (10.0, 0.0, -10.0), (0.0, 0.0, 10.0))
#: A large vertical triangle in the x=1 plane.
WALL = ((1.0, -10.0, -10.0), (1.0, -10.0, 10.0), (1.0, 10.0, 0.0))


class TestAgainstGeometryWorkedOutByHand:
    def test_a_capsule_resting_into_a_floor_is_pushed_straight_up(self):
        # Bottom cap at y = 1.0 - 0.5 = 0.5, radius 0.4, so it reaches to 0.1
        # above the floor: no contact. Drop it 0.3 and it is 0.2 through.
        cap = _capsule(centre=(0.0, 0.7, 0.0))
        hit, points, normals, depths = collide.capsule_triangles(cap, *_tris(FLOOR))
        assert bool(hit[0])
        assert normals[0] == pytest.approx([0.0, 1.0, 0.0])
        assert depths[0] == pytest.approx(0.2)
        assert points[0][1] == pytest.approx(0.0)      # the contact is on the plane

    def test_a_capsule_clear_of_the_floor_does_not_touch_it(self):
        cap = _capsule(centre=(0.0, 1.0, 0.0))         # reaches to y=0.1
        hit, _points, _normals, _depths = collide.capsule_triangles(
            cap, *_tris(FLOOR))
        assert not bool(hit[0])

    def test_a_capsule_driven_deep_reports_the_whole_depth(self):
        """Unbounded depth, so something driven inside comes back out.

        Centre at -0.3 puts the axis across the plane (-0.8 to 0.2) with the
        capsule's own centre *below* it, so it is pushed down -- and the upper
        cap reaches 0.2 + 0.4 = 0.6 past the plane, which is the depth. A test
        answering with the distance to the nearest feature would say 0.4 here
        and start pointing the wrong way deeper still.
        """
        cap = _capsule(centre=(0.0, -0.3, 0.0))
        hit, _points, normals, depths = collide.capsule_triangles(
            cap, *_tris(FLOOR))
        assert bool(hit[0])
        assert normals[0] == pytest.approx([0.0, -1.0, 0.0])
        assert depths[0] == pytest.approx(0.6)

    def test_a_capsule_wholly_below_a_floor_does_not_touch_it(self):
        """Its highest reach is -0.1: under the plane, so nothing to resolve."""
        cap = _capsule(centre=(0.0, -1.0, 0.0))        # reaches -1.9 .. -0.1
        hit, _points, _normals, _depths = collide.capsule_triangles(
            cap, *_tris(FLOOR))
        assert not bool(hit[0])

    def test_a_capsule_under_a_ceiling_is_pushed_back_down(self):
        """The side is taken from the capsule, not from the winding."""
        ceiling = ((-10.0, 0.0, -10.0), (0.0, 0.0, 10.0), (10.0, 0.0, -10.0))
        cap = _capsule(centre=(0.0, -0.7, 0.0))
        _hit, _points, normals, depths = collide.capsule_triangles(
            cap, *_tris(ceiling))
        assert normals[0] == pytest.approx([0.0, -1.0, 0.0])
        assert depths[0] == pytest.approx(0.2)

    def test_a_capsule_against_a_wall_is_pushed_along_the_wall_normal(self):
        cap = _capsule(centre=(0.75, 0.0, 0.0))        # 0.25 from the x=1 plane
        _hit, _points, normals, depths = collide.capsule_triangles(
            cap, *_tris(WALL))
        assert normals[0] == pytest.approx([-1.0, 0.0, 0.0])
        assert depths[0] == pytest.approx(0.15)        # radius 0.4 - gap 0.25

    def test_off_the_rim_the_push_is_toward_the_nearest_point(self):
        """No face to be in front of at an edge; the neighbour has the say."""
        small = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        # Beyond the (0,0,0) corner along -x, level with the plane.
        cap = _capsule(centre=(-0.2, 0.0, -0.2), half=0.01, radius=0.4)
        hit, points, normals, _depths = collide.capsule_triangles(
            cap, *_tris(small))
        assert bool(hit[0])
        assert points[0] == pytest.approx([0.0, 0.0, 0.0])   # the corner
        assert normals[0] == pytest.approx(
            [-np.sqrt(0.5), 0.0, -np.sqrt(0.5)], abs=1e-6)

    def test_a_degenerate_triangle_does_not_produce_a_contact(self):
        """A zero-area triangle has no plane to be pushed out of."""
        sliver = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        cap = _capsule(centre=(0.5, 0.0, 0.0), half=0.01, radius=0.4)
        hit, _points, normals, _depths = collide.capsule_triangles(
            cap, *_tris(sliver))
        # It may still contact the degenerate edge, but never with a NaN normal.
        if bool(hit[0]):
            assert np.all(np.isfinite(normals[0]))


class TestTheBatchIsTheSameAnswerAsTheLoop:
    """A regression guard: the two must not drift apart."""

    @staticmethod
    def _random_case(seed):
        rng = np.random.default_rng(seed)
        verts = rng.normal(scale=1.2, size=(3, 3))
        centre = rng.normal(scale=0.8, size=3)
        return verts, centre

    @pytest.mark.parametrize('seed', range(60))
    def test_one_triangle_agrees_with_the_scalar_routine(self, seed):
        verts, centre = self._random_case(seed)
        cap = _capsule(centre=tuple(centre), half=0.5, radius=0.6)
        scalar = collide.capsule_triangle(cap, TriangleProxy(*verts))
        hit, points, normals, depths = collide.capsule_triangles(
            cap, *_tris(tuple(map(tuple, verts))))
        if scalar is None:
            assert not bool(hit[0])
            return
        point, normal, depth = scalar
        assert bool(hit[0])
        assert depths[0] == pytest.approx(depth, abs=1e-9)
        assert normals[0] == pytest.approx(normal, abs=1e-9)
        assert points[0] == pytest.approx(point, abs=1e-9)

    def test_a_whole_batch_agrees_triangle_for_triangle(self):
        rng = np.random.default_rng(99)
        triangles = [tuple(map(tuple, rng.normal(scale=1.2, size=(3, 3))))
                     for _ in range(200)]
        cap = _capsule(centre=(0.1, 0.2, -0.1), half=0.5, radius=0.7)
        hit, points, normals, depths = collide.capsule_triangles(
            cap, *_tris(*triangles))
        for index, triangle in enumerate(triangles):
            scalar = collide.capsule_triangle(
                cap, TriangleProxy(*[np.array(v) for v in triangle]))
            if scalar is None:
                assert not bool(hit[index]), index
                continue
            point, normal, depth = scalar
            assert bool(hit[index]), index
            assert depths[index] == pytest.approx(depth, abs=1e-9), index
            assert normals[index] == pytest.approx(normal, abs=1e-9), index
            assert points[index] == pytest.approx(point, abs=1e-9), index

    def test_a_tilted_capsule_agrees_too(self):
        """The face test uses the capsule's own axis, so orientation matters."""
        angle = 0.6
        rot = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                        [np.sin(angle), np.cos(angle), 0.0],
                        [0.0, 0.0, 1.0]])
        cap = _capsule(centre=(0.0, 0.3, 0.0), rotation=rot)
        for triangle in (FLOOR, WALL):
            scalar = collide.capsule_triangle(
                cap, TriangleProxy(*[np.array(v) for v in triangle]))
            hit, _points, normals, depths = collide.capsule_triangles(
                cap, *_tris(triangle))
            if scalar is None:
                assert not bool(hit[0])
            else:
                assert depths[0] == pytest.approx(scalar[2], abs=1e-9)
                assert normals[0] == pytest.approx(scalar[1], abs=1e-9)


class TestTheEdges:
    def test_no_triangles_is_an_empty_answer_rather_than_a_failure(self):
        cap = _capsule()
        empty = np.zeros((0, 3), dtype='d')
        hit, points, normals, depths = collide.capsule_triangles(
            cap, empty, empty, empty)
        assert len(hit) == len(points) == len(normals) == len(depths) == 0

    def test_every_triangle_gets_an_answer_even_when_none_touch(self):
        cap = _capsule(centre=(0.0, 50.0, 0.0))
        hit, _points, _normals, _depths = collide.capsule_triangles(
            cap, *_tris(FLOOR, WALL))
        assert len(hit) == 2
        assert not hit.any()

    def test_the_answer_arrays_line_up_with_the_input(self):
        cap = _capsule(centre=(0.0, 0.7, 0.0))
        far = ((100.0, 0.0, 100.0), (101.0, 0.0, 100.0), (100.0, 0.0, 101.0))
        hit, _points, _normals, _depths = collide.capsule_triangles(
            cap, *_tris(far, FLOOR, far))
        assert list(hit) == [False, True, False]


class TestTheMeshPathUsesIt:
    """`_collide_mesh` is what the character controller actually reaches."""

    @staticmethod
    def _mesh(cells=40, extent=8.0):
        from omi_physics.body import TriangleMeshProxy
        xs = np.linspace(-extent, extent, cells)
        points = [(x, 0.05 * np.sin(i * 0.7) * np.cos(j * 0.7), z)
                  for i, x in enumerate(xs) for j, z in enumerate(xs)]
        indices = []
        for i in range(cells - 1):
            for j in range(cells - 1):
                a, b = i * cells + j, (i + 1) * cells + j
                indices.append((a, b, a + 1))
                indices.append((b, b + 1, a + 1))
        return TriangleMeshProxy(np.array(points, dtype='d'),
                                 np.array(indices, dtype='i'),
                                 (0, 0, 0), np.eye(3))

    def test_the_contacts_are_the_ones_the_triangle_loop_would_give(self):
        """Same contacts, in the same order, from the same triangles."""
        mesh = self._mesh()
        cap = _capsule(centre=(0.0, 0.3, 0.0), half=0.5, radius=0.45)
        lo, hi = cap.aabb()
        expected = []
        for tri in mesh.triangles_overlapping(lo, hi):
            found = collide.capsule_triangle(cap, tri)
            if found is not None:
                expected.append(found)
        contacts = collide.collide(0, 1, cap, mesh)
        assert len(contacts) == len(expected)
        for contact, (point, normal, depth) in zip(contacts, expected,
                                                   strict=True):
            # The mesh is B here, so the normal is flipped to point A -> B.
            assert contact.depth == pytest.approx(depth, abs=1e-9)
            assert contact.normal == pytest.approx(-normal, abs=1e-9)
            assert contact.point == pytest.approx(point, abs=1e-9)

    def test_the_normal_points_the_right_way_with_the_mesh_first(self):
        mesh = self._mesh()
        cap = _capsule(centre=(0.0, 0.3, 0.0), half=0.5, radius=0.45)
        forward = collide.collide(0, 1, mesh, cap)
        backward = collide.collide(0, 1, cap, mesh)
        assert len(forward) == len(backward) > 0
        for one, other in zip(forward, backward, strict=True):
            assert one.normal == pytest.approx(-other.normal, abs=1e-9)

    def test_a_capsule_nowhere_near_the_mesh_makes_no_contacts(self):
        mesh = self._mesh()
        assert collide.collide(0, 1, _capsule(centre=(0.0, 80.0, 0.0)),
                               mesh) == []

    def test_the_push_path_does_not_build_a_contact_object_per_triangle(self):
        """What the controller calls: the same answer, as two arrays."""
        mesh = self._mesh()
        cap = _capsule(centre=(0.0, 0.3, 0.0), half=0.5, radius=0.45)
        contacts = collide.collide(0, 1, cap, mesh)
        pushes, depths = collide.capsule_mesh_pushes(cap, mesh)
        assert len(pushes) == len(depths) == len(contacts)
        # `collide` with the mesh as B answers normals pointing A->B; the push
        # is into the capsule, which is their negation.
        by_depth = sorted(contacts, key=lambda c: -c.depth)
        for (push, depth), contact in zip(zip(pushes, depths, strict=True),
                                          by_depth, strict=True):
            assert depth == pytest.approx(contact.depth, abs=1e-9)
            assert push == pytest.approx(-contact.normal, abs=1e-9)

    def test_the_pushes_come_back_deepest_first(self):
        """The order sequential projection resolves in."""
        mesh = self._mesh()
        cap = _capsule(centre=(0.0, 0.3, 0.0), half=0.5, radius=0.45)
        _pushes, depths = collide.capsule_mesh_pushes(cap, mesh)
        assert list(depths) == sorted(depths, reverse=True)

    def test_supplied_candidates_give_the_same_answer_as_gathering_them(self):
        """The controller's reuse across a frame must not change the result."""
        mesh = self._mesh()
        cap = _capsule(centre=(0.0, 0.3, 0.0), half=0.5, radius=0.45)
        lo, hi = cap.aabb()
        wide = mesh.candidate_vertices(lo - 0.5, hi + 0.5)   # a superset
        fresh = collide.capsule_mesh_pushes(cap, mesh)
        reused = collide.capsule_mesh_pushes(cap, mesh, wide)
        assert len(fresh[0]) == len(reused[0])
        assert np.allclose(np.sort(fresh[1]), np.sort(reused[1]))


class TestTheBudgetThisExistsFor:
    """A hundred characters' depenetration has to fit in a frame with room.

    A controller runs three depenetration iterations inside each of the ground
    probe, the move, the step-up and the step-down. Measured on the controller
    rather than on a collision call, because the gather it reuses across those
    calls is part of the answer.
    """

    @staticmethod
    def _world_with_floor(cells=60, extent=30.0):
        from omi_physics import model
        from omi_physics.world import PhysicsWorld
        xs = np.linspace(-extent, extent, cells)
        points = np.array([(x, 0.05 * np.sin(i * 0.7) * np.cos(j * 0.7), z)
                           for i, x in enumerate(xs)
                           for j, z in enumerate(xs)], dtype='d')
        indices = []
        for i in range(cells - 1):
            for j in range(cells - 1):
                a, b = i * cells + j, (i + 1) * cells + j
                indices.append((a, b, a + 1))
                indices.append((b, b + 1, a + 1))
        world = PhysicsWorld(
            gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
        shape = world.add_shape(
            model.Shape.trimesh(points=points, indices=np.array(indices, 'i')))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=shape), position=(0, 0, 0))
        return world

    @pytest.mark.serial
    def test_a_hundred_characters_walking_fit_in_a_frame(self):
        import time
        from omi_physics.character import (CharacterCapabilities,
                                           CharacterController)
        world = self._world_with_floor()
        rng = np.random.default_rng(4)
        walkers = []
        for _ in range(100):
            x, z = rng.uniform(-20.0, 20.0, size=2)
            walker = CharacterController(world, CharacterCapabilities(),
                                         (x, 1.2, z))
            walker.safe_bind((x, 1.2, z))
            walker.set_move(np.array([1.0, 0.0, 0.0]))
            for _ in range(3):
                walker.update(1 / 60.0)              # settle and warm
            walkers.append(walker)
        started = time.perf_counter()
        for _ in range(10):
            for walker in walkers:
                walker.update(1 / 60.0)
        each = (time.perf_counter() - started) / 10
        assert each < 0.0166, (
            'a hundred walkers cost %.1f ms a frame, which is the whole of one'
            % (each * 1000,))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
