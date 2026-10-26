"""Casting a ray at the world: what it hits, where, and which way that faces.

The query a shooter needs for a hitscan weapon and a bot needs for line of
sight, and the two are the same question asked for different reasons — which is
why it belongs here rather than in either of them.

What a ray must get right is not "did it hit" but **which** thing it hit first.
A cast that returns any hit rather than the nearest shoots through walls, and
that is the defect these tests are shaped around.
"""

import numpy as np
import pytest

from omi_physics import model, raycast
from omi_physics.trigrid import TriangleGrid
from omi_physics.world import PhysicsWorld


def world():
    return PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))


def box(w, position=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0),
        motion=model.STATIC):
    shape = w.add_shape(model.Shape.box(size=size))
    return w.add_body(model.Motion(type=motion),
                      collider=model.Collider(shape=shape), position=position)


def sphere(w, position=(0.0, 0.0, 0.0), radius=1.0):
    shape = w.add_shape(model.Shape.sphere(radius=radius))
    return w.add_body(model.Motion(type=model.STATIC),
                      collider=model.Collider(shape=shape), position=position)


def capsule(w, position=(0.0, 0.0, 0.0), height=1.0, radius=0.5):
    shape = w.add_shape(model.Shape.capsule(height=height, radius=radius))
    return w.add_body(model.Motion(type=model.STATIC),
                      collider=model.Collider(shape=shape), position=position)


def wall(w, x=0.0, extent=10.0):
    """A trimesh plane across the x axis — what a map's geometry is."""
    e = extent
    points = np.array([(x, -e, -e), (x, e, -e), (x, e, e), (x, -e, e)], dtype='d')
    indices = np.array([(0, 1, 2), (0, 2, 3)], dtype='i')
    shape = w.add_shape(model.Shape.trimesh(points, indices))
    return w.add_body(model.Motion(type=model.STATIC),
                      collider=model.Collider(shape=shape), position=(0, 0, 0))


class TestHittingSomething:

    def test_a_ray_that_hits_a_box_says_so(self):
        w = world()
        box(w, position=(5.0, 0.0, 0.0))
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert hit is not None

    def test_a_ray_that_hits_nothing_returns_nothing(self):
        w = world()
        box(w, position=(5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (0, 1, 0)) is None

    def test_an_empty_world_is_a_miss_rather_than_an_error(self):
        assert raycast.raycast(world(), (0, 0, 0), (1, 0, 0)) is None

    def test_the_hit_names_the_body(self):
        w = world()
        target = box(w, position=(5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0)).body == target

    def test_the_hit_gives_the_distance_along_the_ray(self):
        w = world()
        box(w, position=(5.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert hit.distance == pytest.approx(4.0, abs=1e-6)

    def test_the_hit_gives_the_point(self):
        w = world()
        box(w, position=(5.0, 0.0, 0.0))
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert np.allclose(hit.point, (4.0, 0.0, 0.0), atol=1e-6)

    def test_the_hit_gives_a_normal_facing_the_ray(self):
        """What an impact decal is aligned to, and which way a splash bounces."""
        w = world()
        box(w, position=(5.0, 0.0, 0.0))
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert np.allclose(hit.normal, (-1.0, 0.0, 0.0), atol=1e-6)

    def test_an_unnormalised_direction_still_works(self):
        """A caller with a velocity rather than a heading should not have to care."""
        w = world()
        box(w, position=(5.0, 0.0, 0.0))
        hit = raycast.raycast(w, (0, 0, 0), (7.0, 0.0, 0.0))
        assert hit.distance == pytest.approx(4.0, abs=1e-6)

    def test_a_zero_direction_is_a_miss_rather_than_a_crash(self):
        w = world()
        box(w, position=(5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (0, 0, 0)) is None


class TestHittingTheNearestThing:
    """The property that separates a working shot from shooting through walls."""

    def test_the_nearest_of_two_boxes_is_the_one_hit(self):
        w = world()
        near = box(w, position=(5.0, 0.0, 0.0))
        box(w, position=(9.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0)).body == near

    def test_the_order_bodies_were_added_in_does_not_decide(self):
        w = world()
        box(w, position=(9.0, 0.0, 0.0))
        near = box(w, position=(5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0)).body == near

    def test_a_wall_between_the_ray_and_a_target_stops_it(self):
        """Line of sight, which is the whole reason a bot asks."""
        w = world()
        wall(w, x=3.0)
        target = box(w, position=(8.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0)).body != target


class TestHowFarItReaches:

    def test_a_target_beyond_the_range_is_not_hit(self):
        w = world()
        box(w, position=(50.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0), max_distance=10.0) is None

    def test_a_target_within_the_range_is(self):
        w = world()
        box(w, position=(5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0), max_distance=10.0)

    def test_a_ray_starting_past_a_target_does_not_hit_it_behind(self):
        """A ray is a half-line; a shot does not hit what is behind the gun."""
        w = world()
        box(w, position=(-5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0)) is None


class TestEachShape:

    def test_a_sphere_is_hit_at_its_surface(self):
        w = world()
        sphere(w, position=(5.0, 0.0, 0.0), radius=1.0)
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert hit.distance == pytest.approx(4.0, abs=1e-6)

    def test_a_sphere_normal_points_out_from_its_centre(self):
        w = world()
        sphere(w, position=(5.0, 0.0, 0.0), radius=1.0)
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert np.allclose(hit.normal, (-1.0, 0.0, 0.0), atol=1e-6)

    def test_a_ray_that_passes_beside_a_sphere_misses(self):
        w = world()
        sphere(w, position=(5.0, 0.0, 0.0), radius=1.0)
        assert raycast.raycast(w, (0, 2.0, 0), (1, 0, 0)) is None

    def test_a_capsule_is_hit_on_its_side(self):
        """What a shot at a standing character actually meets."""
        w = world()
        capsule(w, position=(5.0, 0.0, 0.0), height=1.0, radius=0.5)
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert hit.distance == pytest.approx(4.5, abs=1e-6)

    def test_a_capsule_is_hit_on_its_cap(self):
        w = world()
        capsule(w, position=(0.0, 5.0, 0.0), height=1.0, radius=0.5)
        hit = raycast.raycast(w, (0, 0, 0), (0, 1, 0))
        assert hit.distance == pytest.approx(4.0, abs=1e-6)

    def test_a_ray_beside_a_capsule_misses(self):
        w = world()
        capsule(w, position=(5.0, 0.0, 0.0), height=1.0, radius=0.5)
        assert raycast.raycast(w, (0, 2.0, 0), (1, 0, 0)) is None

    def test_a_trimesh_is_hit_at_its_triangle(self):
        w = world()
        wall(w, x=6.0)
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert hit.distance == pytest.approx(6.0, abs=1e-6)

    def test_a_trimesh_normal_faces_the_ray(self):
        """Whichever way its triangles were wound: a wall is not one-sided.

        A map's geometry has no reliable winding from a shooter's point of
        view, so a normal that pointed away half the time would put every
        second impact decal inside the wall.
        """
        w = world()
        wall(w, x=6.0)
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert float(np.dot(hit.normal, (1, 0, 0))) < 0.0

    def test_a_trimesh_hit_says_which_triangle_it_met(self):
        """So a caller can look up what that part of the mesh is made of.

        A level is one mesh of many materials, and the only thing that tells
        stone from metal at an impact point is which triangle was struck.
        """
        w = world()
        wall(w, x=6.0)
        # The near half of the wall's two triangles, chosen by aiming into it.
        assert raycast.raycast(w, (0, 4.0, 4.0), (1, 0, 0)).triangle == 0
        assert raycast.raycast(w, (0, -4.0, 4.0), (1, 0, 0)).triangle == 1

    def test_a_hit_on_a_shape_with_no_triangles_says_so(self):
        """A box has no triangle to name, and must not invent one."""
        w = world()
        box(w, position=(6.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0)).triangle == raycast.NO_TRIANGLE

    def test_a_ray_past_the_edge_of_a_trimesh_misses(self):
        w = world()
        wall(w, x=6.0, extent=1.0)
        assert raycast.raycast(w, (0, 5.0, 0), (1, 0, 0)) is None

    def test_a_large_trimesh_is_hit_correctly(self):
        """Big enough to build the spatial grid, which is a different path."""
        w = world()
        size = 40
        xs, ys = np.meshgrid(np.linspace(-10, 10, size),
                             np.linspace(-10, 10, size))
        points = np.column_stack([np.full(xs.size, 6.0), xs.ravel(), ys.ravel()])
        indices = []
        for row in range(size - 1):
            for col in range(size - 1):
                a = row * size + col
                indices.append((a, a + 1, a + size))
                indices.append((a + 1, a + size + 1, a + size))
        shape = w.add_shape(model.Shape.trimesh(points, np.array(indices, 'i')))
        w.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape), position=(0, 0, 0))
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert hit is not None
        assert hit.distance == pytest.approx(6.0, abs=1e-6)


class TestArithmeticTheRayIsAllowedToDo:
    """The slab and triangle tests divide by the ray's own numbers, and reach
    infinity on purpose when one of them is zero -- the infinities cancel and
    the comparisons come out right. What they must not do is *signal*: a game
    hunting a NaN through its physics runs under ``np.seterr(all='raise')``, and
    a cast that is behaving correctly cannot be what stops it."""

    @pytest.mark.parametrize('heading', [
        (0.0, -1.0, 0.0),                  # straight down: the ground check
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ])
    def test_a_cast_along_an_axis_signals_nothing(self, heading):
        w = world()
        box(w, position=(0.0, -5.0, 0.0))
        wall(w, x=3.0)
        with np.errstate(all='raise'):
            raycast.raycast(w, (0.0, 0.0, 0.0), heading, max_distance=100.0)
            raycast.raycast_many(w, [(0.0, 0.0, 0.0)], [heading], max_distance=100.0)

    def test_a_ray_that_starts_on_a_face_still_answers(self):
        """Exactly on the boundary and exactly parallel to it: every slab
        comparison is then a NaN, which is false however it is asked."""
        w = world()
        body = box(w, position=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
        with np.errstate(all='raise'):
            hit = raycast.raycast(w, (-5.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                                  max_distance=100.0)
        assert hit is not None and hit.body == body


class TestWhatToIgnore:

    def test_a_body_can_be_skipped(self):
        """A shooter must not hit their own body with their own shot."""
        w = world()
        own = box(w, position=(2.0, 0.0, 0.0))
        other = box(w, position=(6.0, 0.0, 0.0))
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0), skip=(own,))
        assert hit.body == other

    def test_several_bodies_can_be_skipped(self):
        w = world()
        first = box(w, position=(2.0, 0.0, 0.0))
        second = box(w, position=(4.0, 0.0, 0.0))
        last = box(w, position=(6.0, 0.0, 0.0))
        hit = raycast.raycast(w, (0, 0, 0), (1, 0, 0), skip=(first, second))
        assert hit.body == last

    def test_a_body_with_no_collider_is_not_hit(self):
        w = world()
        w.add_body(model.Motion(type=model.STATIC), position=(3.0, 0.0, 0.0))
        box(w, position=(6.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0)).distance == \
            pytest.approx(5.0, abs=1e-6)

    def test_moving_bodies_are_hit_too(self):
        """A bot is a body that moves, and a shot has to be able to reach it."""
        w = world()
        target = box(w, position=(5.0, 0.0, 0.0), motion=model.KINEMATIC)
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0)).body == target

    def test_every_shape_a_world_holds_can_be_cast_at(self):
        w = world()
        box(w)
        hull(w, CUBE)
        assert raycast.unsupported_shapes(w) == set()


CUBE = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]

#: A square pyramid, its base on y = 0 and its apex at (0, 2, 0).
PYRAMID = [(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1), (0, 2, 0)]


def hull(w, points, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)):
    shape = w.add_shape(model.Shape.convex(np.array(points, 'd')))
    return w.add_body(model.Motion(type=model.STATIC),
                      collider=model.Collider(shape=shape), position=position,
                      orientation=orientation)


class TestAConvexHull:
    """A hull is cast at by clipping the ray against its faces."""

    def test_a_ray_meets_the_face_it_enters_by(self):
        w = world()
        body = hull(w, CUBE, position=(5.0, 0.0, 0.0))
        found = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert found.body == body
        assert found.distance == pytest.approx(4.0)
        assert found.point == pytest.approx([4.0, 0.0, 0.0])
        assert found.normal == pytest.approx([-1.0, 0.0, 0.0])

    def test_a_ray_down_onto_an_apex_meets_the_apex(self):
        w = world()
        hull(w, PYRAMID, position=(0.0, 1.0, 0.0))
        found = raycast.raycast(w, (0, 10, 0), (0, -1, 0))
        assert found.distance == pytest.approx(10.0 - 3.0)

    def test_a_ray_down_the_slope_of_a_pyramid_meets_its_face(self):
        w = world()
        hull(w, PYRAMID)
        found = raycast.raycast(w, (0.5, 10, 0), (0, -1, 0))
        assert found.point[1] == pytest.approx(1.0)
        assert found.normal == pytest.approx(np.array([2.0, 1.0, 0.0]) / np.sqrt(5.0))

    def test_a_ray_past_the_hull_misses_it(self):
        w = world()
        hull(w, CUBE, position=(5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 3, 0), (1, 0, 0)) is None
        assert raycast.raycast(w, (0, 0, 0), (-1, 0, 0)) is None

    def test_a_ray_alongside_a_face_without_touching_it_misses(self):
        w = world()
        hull(w, CUBE, position=(5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 1.5, 0), (1, 0, 0)) is None

    def test_a_ray_along_a_face_inside_its_plane_is_a_hit_or_a_miss_but_not_an_error(self):
        w = world()
        hull(w, CUBE, position=(5.0, 0.0, 0.0))
        raycast.raycast(w, (0, 1.0, 0), (1, 0, 0))

    def test_a_ray_starting_inside_leaves_by_the_far_face(self):
        w = world()
        hull(w, CUBE)
        found = raycast.raycast(w, (0, 0, 0), (1, 0, 0))
        assert found.distance == pytest.approx(1.0)

    def test_the_limit_is_kept(self):
        w = world()
        hull(w, CUBE, position=(5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0, 0), (1, 0, 0), max_distance=3.0) is None

    def test_a_turned_hull_is_cast_at_where_it_is(self):
        w = world()
        half = np.sqrt(0.5)
        long_box = [(x, y, z) for x in (-3, 3) for y in (-1, 1) for z in (-1, 1)]
        hull(w, long_box, position=(0.0, 0.0, 0.0), orientation=(0.0, half, 0.0, half))
        assert raycast.raycast(w, (10, 0, 0), (-1, 0, 0)).distance == pytest.approx(9.0)
        assert raycast.raycast(w, (0, 0, 10), (0, 0, -1)).distance == pytest.approx(7.0)

    def test_points_that_enclose_no_volume_are_not_hit(self):
        w = world()
        hull(w, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], position=(5.0, 0.0, 0.0))
        assert raycast.raycast(w, (0, 0.5, 0.5), (1, 0, 0)) is None

    def test_a_bundle_of_rays_meets_it_as_each_would_alone(self):
        w = world()
        hull(w, PYRAMID, position=(0.0, 1.0, 0.0))
        origins = [(0.0, 10, 0.0), (0.5, 10, 0.0), (4.0, 10, 0.0)]
        many = raycast.raycast_many(w, origins, [(0, -1, 0)] * 3)
        alone = [raycast.raycast(w, o, (0, -1, 0)) for o in origins]
        assert many[2] is None and alone[2] is None
        for together, single in zip(many[:2], alone[:2], strict=True):
            assert together.distance == pytest.approx(single.distance)
            assert together.normal == pytest.approx(single.normal)


class TestAskingOnlyWhetherTheViewIsClear:
    """The cheaper question a bot asks many times a frame."""

    def test_a_clear_line_is_clear(self):
        assert raycast.line_of_sight(world(), (0, 0, 0), (10, 0, 0))

    def test_a_wall_blocks_it(self):
        w = world()
        wall(w, x=5.0)
        assert not raycast.line_of_sight(w, (0, 0, 0), (10, 0, 0))

    def test_something_past_the_target_does_not_block_it(self):
        """Only what is *between* the two points matters."""
        w = world()
        wall(w, x=15.0)
        assert raycast.line_of_sight(w, (0, 0, 0), (10, 0, 0))

    def test_the_two_bodies_can_be_ignored(self):
        """A looker and a target must not block their own line of sight."""
        w = world()
        looker = box(w, position=(0.0, 0.0, 0.0))
        target = box(w, position=(10.0, 0.0, 0.0))
        assert raycast.line_of_sight(w, (0, 0, 0), (10, 0, 0),
                                     skip=(looker, target))

    def test_two_points_in_the_same_place_can_see_each_other(self):
        assert raycast.line_of_sight(world(), (1, 2, 3), (1, 2, 3))


class TestTheCostOfACast:
    """A cast must not look at the whole level.

    A map is one trimesh of tens of thousands of triangles, and the things that
    cast at it do so several times a tick each: a bot checks what it can see and
    probes the step in front of it, and every shot is a cast.  Work proportional
    to the whole mesh is milliseconds per cast at a real map's size, so a
    handful of bots costs more than the entire frame budget.

    Asserted as the work a cast does, counted: the grid cells it walks and the
    candidate triangles those cells hand it. A time measures the machine and
    whatever else runs on it. Quadrupling the triangles without moving the
    geometry walks the same cells and must not quadruple the candidates: what a
    cast looks at is what is near it, and there is no more near it than there was.
    """

    def level(self, side):
        """A room spanning the same 60 m however finely it is divided.

        A floor **and a ceiling**, so the mesh encloses the caster the way a
        real level does.  A single flat plane would not test anything: its
        bounding box has no height, so a ray above it is rejected outright and
        never reaches the part whose cost is in question.
        """
        w = world()
        xs, zs = np.meshgrid(np.linspace(-30, 30, side),
                             np.linspace(-30, 30, side))
        flat = np.column_stack([xs.ravel(), np.zeros(xs.size), zs.ravel()])
        points = np.vstack([flat, flat + (0.0, 4.0, 0.0)])
        indices = []
        for storey in (0, len(flat)):
            for row in range(side - 1):
                for col in range(side - 1):
                    a = storey + row * side + col
                    indices.append((a, a + 1, a + side))
                    indices.append((a + 1, a + side + 1, a + side))
        shape = w.add_shape(model.Shape.trimesh(points,
                                                np.array(indices, 'i')))
        w.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape), position=(0, 0, 0))
        return w

    def work(self, w, origin, heading, limit, monkeypatch):
        """The cells one cast walks, the candidates they hand it, and the mesh's triangle count.

        The first cast builds the placed mesh and its grid, and is not counted.
        """
        raycast.raycast(w, origin, heading, max_distance=limit)
        counted = [0, 0]
        walk = TriangleGrid._walk                # noqa: SLF001 the walk is the work counted
        gathered = TriangleGrid._gathered        # noqa: SLF001 the candidates are the work counted

        def walking(grid, *arguments):
            for cell in walk(grid, *arguments):
                counted[0] += 1
                yield cell

        def gathering(grid, found, *arguments, **named):
            counted[1] += sum(len(cell) for cell in found)
            return gathered(grid, found, *arguments, **named)

        with monkeypatch.context() as patch:
            patch.setattr(TriangleGrid, '_walk', walking)
            patch.setattr(TriangleGrid, '_gathered', gathering)
            raycast.raycast(w, origin, heading, max_distance=limit)
        return counted[0], counted[1], len(w.shapes[0].indices)

    def scaling(self, origin, heading, limit, monkeypatch):
        """The work of a cast on a level, and on the same level with four times the triangles."""
        return (self.work(self.level(40), origin, heading, limit, monkeypatch),
                self.work(self.level(80), origin, heading, limit, monkeypatch))

    def test_a_short_probe_does_not_get_dearer_with_the_mesh(self, monkeypatch):
        """What a bot's step probe is: half a metre, on thousands of triangles."""
        (cells, small, _), (finer_cells, large, triangles) = self.scaling(
            (0.0, 0.5, 0.0), (1.0, 0.0, 0.0), 0.5, monkeypatch)
        assert cells == finer_cells == 1
        assert large < 2 * small
        assert large < triangles / 1000

    def test_a_cast_across_the_level_does_not_either(self, monkeypatch):
        """The case a bounding box cannot help with: the box is the level."""
        (cells, small, _), (finer_cells, large, triangles) = self.scaling(
            (-40.0, 0.5, 0.3), (1.0, 0.0, 0.0), 100.0, monkeypatch)
        assert cells == finer_cells > 1
        assert 0 < large < 2 * small
        # A line of cells across the floor: about 1 in 40 of the mesh's triangles.
        assert large < triangles / 20

    def test_it_still_finds_the_floor_it_is_aimed_at(self):
        """The narrowing must not have cost the cast its answer."""
        hit = raycast.raycast(self.level(60), (0.0, 2.5, 0.0), (0.0, -1.0, 0.0))
        assert hit is not None
        assert hit.distance == pytest.approx(2.5, abs=1e-6)
