"""Capsule against capsule, and capsule against sphere, without GJK/EPA.

The analytic contact is checked against the general convex path over random
pairs. EPA stops once the next support point is within 1e-6 of its nearest
face, so its depth is good to 1e-6 and its normal, on the curved Minkowski
surface of two capsules, only to about the square root of that over the radii.
The analytic normal is held to 1e-9 against the support functions EPA is built
from instead: the two capsules overlap along it by exactly the depth reported.
"""
import math

import numpy as np
import pytest

from omi_physics import collide, gjk, mathutil, model
from omi_physics.body import CapsuleProxy, SphereProxy
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld

I3 = np.eye(3)
DT = 1.0 / 60.0

#: How far EPA's normal may lie from the analytic one, as a chord length.
EPA_NORMAL_TOLERANCE = 5e-3


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return mathutil.quat_to_matrix(q / np.linalg.norm(q))


def _overlap_along(A, B, normal: np.ndarray) -> float:
    """How far ``A`` and ``B`` overlap along ``normal``, from their support functions."""
    return float(np.dot(A.support(normal) - B.support(-normal), normal))


def _epa(A, B):
    """EPA's ``(normal, depth)`` oriented from A to B, or None where it did not converge."""
    hit, simplex = gjk.gjk_intersect(A, B)
    if not hit:
        return None
    result = gjk.epa(A, B, simplex)
    if result is None:
        return None
    normal, depth, _point = result
    if _overlap_along(A, B, normal) - depth > 1e-6:
        normal = -normal
    if _overlap_along(A, B, normal) - depth > 1e-6:
        return None
    return normal, depth


def _random_pairs(seed: int, count: int):
    """Capsule pairs: crossing and parallel axes, touching, penetrating, deep, coincident axes."""
    rng = np.random.default_rng(seed)
    for index in range(count):
        RA = _rotation(rng)
        style = index % 5
        RB = RA.copy() if style == 1 else _rotation(rng)
        A = CapsuleProxy(rng.uniform(-0.5, 0.5, 3), rng.uniform(0.0, 1.0),
                         rng.uniform(0.1, 0.6), RA)
        centre = rng.uniform(-0.5, 0.5, 3)
        if style == 2:                       # deep: centres close together
            centre = A.center + rng.uniform(-0.05, 0.05, 3)
        elif style == 3:                     # on A's axis
            centre = A.center + RA[:, 1] * rng.uniform(-1.5, 1.5)
            RB = RA.copy()
        B = CapsuleProxy(centre, rng.uniform(0.0, 1.0), rng.uniform(0.1, 0.6), RB)
        if style == 4:                       # just touching
            for contact in collide.capsule_capsule(0, 1, A, B):
                shift = contact.normal * (contact.depth - 1e-4)
                B = CapsuleProxy(B.center + shift, B.half_height, B.radius, B.R)
        yield A, B


class TestAgreementWithTheGeneralPath:
    def test_depth_and_normal_agree_over_random_pairs(self):
        compared = penetrating = 0
        for A, B in _random_pairs(seed=7, count=600):
            contacts = collide.capsule_capsule(0, 1, A, B)
            if not contacts:
                general = gjk.collide_convex(0, 1, A, B)
                assert not general or general[0].depth < 1e-6
                continue
            penetrating += 1
            (contact,) = contacts
            assert np.linalg.norm(contact.normal) == pytest.approx(1.0, abs=1e-12)
            assert _overlap_along(A, B, contact.normal) == pytest.approx(contact.depth, abs=1e-9)
            reference = _epa(A, B)
            if reference is None:
                continue
            normal, depth = reference
            compared += 1
            assert contact.depth == pytest.approx(depth, abs=1e-6)
            assert np.linalg.norm(contact.normal - normal) < EPA_NORMAL_TOLERANCE
        assert penetrating > 400
        assert compared > 0.7 * penetrating

    def test_the_contact_point_is_halfway_between_the_two_surfaces(self):
        for A, B in _random_pairs(seed=11, count=300):
            for c in collide.capsule_capsule(0, 1, A, B):
                surface_a = A.support(c.normal)
                surface_b = B.support(-c.normal)
                assert np.dot(surface_a - surface_b, c.normal) == pytest.approx(c.depth, abs=1e-9)
                middle = 0.5 * (np.dot(surface_a, c.normal) + np.dot(surface_b, c.normal))
                assert np.dot(c.point, c.normal) == pytest.approx(middle, abs=1e-9)

    def test_collide_dispatches_both_orders(self):
        A = CapsuleProxy((0, 0, 0), 0.5, 0.3, I3)
        B = CapsuleProxy((0.5, 0.2, 0), 0.5, 0.3, I3)
        (ab,) = collide.collide(0, 1, A, B)
        (ba,) = collide.collide(1, 0, B, A)
        assert (ab.a, ab.b, ba.a, ba.b) == (0, 1, 1, 0)
        assert np.allclose(ab.normal, (1, 0, 0), atol=1e-12)
        assert np.allclose(ba.normal, (-1, 0, 0), atol=1e-12)
        assert ab.depth == pytest.approx(0.1, abs=1e-12)

    def test_apart_capsules_make_no_contact(self):
        A = CapsuleProxy((0, 0, 0), 0.5, 0.3, I3)
        B = CapsuleProxy((0.61, 0, 0), 0.5, 0.3, I3)
        assert collide.collide(0, 1, A, B) == []


class TestDegenerateCapsules:
    def test_parallel_overlapping_segments_touch_at_the_middle_of_the_overlap(self):
        A = CapsuleProxy((0, 0.9, 0), 0.55, 0.35, I3)
        B = CapsuleProxy((0.6, 1.3, 0), 0.55, 0.35, I3)
        (c,) = collide.capsule_capsule(0, 1, A, B)
        assert np.allclose(c.normal, (1, 0, 0), atol=1e-12)
        assert c.depth == pytest.approx(0.1, abs=1e-12)
        assert c.point[1] == pytest.approx(1.1, abs=1e-12)    # overlap runs 0.75 .. 1.45

    def test_zero_length_capsules_are_spheres(self):
        A = CapsuleProxy((0, 0, 0), 0.0, 0.4, I3)
        B = CapsuleProxy((0.3, 0.4, 0), 0.0, 0.2, _rotation(np.random.default_rng(3)))
        (c,) = collide.capsule_capsule(0, 1, A, B)
        (s,) = collide.sphere_sphere(0, 1, SphereProxy((0, 0, 0), 0.4),
                                     SphereProxy((0.3, 0.4, 0), 0.2))
        assert np.allclose(c.normal, s.normal, atol=1e-12)
        assert c.depth == pytest.approx(s.depth, abs=1e-12)
        assert np.allclose(c.point, s.point, atol=1e-12)

    def test_a_zero_length_capsule_against_a_long_one(self):
        A = CapsuleProxy((0, 0, 0), 1.0, 0.3, I3)
        B = CapsuleProxy((0.4, 0.5, 0), 0.0, 0.2, I3)
        (c,) = collide.capsule_capsule(0, 1, A, B)
        assert np.allclose(c.normal, (1, 0, 0), atol=1e-12)
        assert c.depth == pytest.approx(0.1, abs=1e-12)

    @pytest.mark.parametrize('half_heights', [(0.5, 0.5), (0.5, 0.2), (0.0, 0.0), (0.0, 0.7)])
    def test_exactly_coincident_capsules_have_a_unit_normal(self, half_heights):
        R = _rotation(np.random.default_rng(5))
        A = CapsuleProxy((1, 2, 3), half_heights[0], 0.3, R)
        B = CapsuleProxy((1, 2, 3), half_heights[1], 0.25, R)
        (c,) = collide.capsule_capsule(0, 1, A, B)
        assert np.linalg.norm(c.normal) == pytest.approx(1.0, abs=1e-12)
        assert c.depth == pytest.approx(0.55, abs=1e-12)
        if max(half_heights) > 0:            # out sideways, the shortest way
            assert abs(np.dot(c.normal, R[:, 1])) < 1e-12
        assert _overlap_along(A, B, c.normal) == pytest.approx(c.depth, abs=1e-9)

    def test_capsules_on_one_axis_overlapping_end_to_end_push_apart_along_it(self):
        A = CapsuleProxy((0, 0, 0), 0.5, 0.3, I3)
        B = CapsuleProxy((0, 1.5, 0), 0.5, 0.3, I3)
        (c,) = collide.capsule_capsule(0, 1, A, B)
        assert np.allclose(c.normal, (0, 1, 0), atol=1e-12)
        assert c.depth == pytest.approx(0.1, abs=1e-12)

    def test_capsules_on_one_axis_overlapping_along_it_push_apart_sideways(self):
        A = CapsuleProxy((0, 0, 0), 0.5, 0.3, I3)
        B = CapsuleProxy((0, 0.4, 0), 0.5, 0.3, I3)
        (c,) = collide.capsule_capsule(0, 1, A, B)
        assert abs(c.normal[1]) < 1e-12
        assert np.linalg.norm(c.normal) == pytest.approx(1.0, abs=1e-12)
        assert c.depth == pytest.approx(0.6, abs=1e-12)


class TestCapsuleSphere:
    def test_agrees_with_a_zero_length_capsule_and_the_general_path(self):
        rng = np.random.default_rng(13)
        compared = 0
        for _ in range(400):
            A = CapsuleProxy(rng.uniform(-0.5, 0.5, 3), rng.uniform(0.0, 1.0),
                             rng.uniform(0.1, 0.6), _rotation(rng))
            centre, radius = rng.uniform(-0.8, 0.8, 3), float(rng.uniform(0.1, 0.6))
            S = SphereProxy(centre, radius)
            expected = collide.capsule_capsule(0, 1, A, CapsuleProxy(centre, 0.0, radius, I3))
            got = collide.collide(0, 1, A, S)
            flipped = collide.collide(1, 0, S, A)
            assert len(got) == len(expected) == len(flipped)
            for c, e, f in zip(got, expected, flipped, strict=True):
                assert np.allclose(c.normal, e.normal, atol=1e-12)
                assert c.depth == pytest.approx(e.depth, abs=1e-12)
                assert np.allclose(f.normal, -c.normal, atol=1e-12)
                assert (f.a, f.b) == (1, 0)
                reference = _epa(A, S)
                if reference is not None and e.depth > 1e-6:
                    compared += 1
                    assert c.depth == pytest.approx(reference[1], abs=1e-6)
        assert compared > 100


class _CountingGeneralPath:
    def __init__(self, monkeypatch):
        self.calls = 0
        original = gjk.collide_convex

        def counted(*args):
            self.calls += 1
            return original(*args)

        monkeypatch.setattr(gjk, 'collide_convex', counted)


def _general_path_only(monkeypatch):
    """Take the capsule pairs out of collide's table, so they go through GJK/EPA."""
    table = {key: value for key, value in collide._DISPATCH.items()  # noqa: SLF001 the table is what is bypassed
             if 'capsule' not in key}
    monkeypatch.setattr(collide, '_DISPATCH', table)


class TestTheGeneralSolverIsNotCalled:
    def test_two_capsule_bodies_in_contact(self, monkeypatch):
        counter = _CountingGeneralPath(monkeypatch)
        world = PhysicsWorld(gravity=model.Gravity(gravity=0.0, direction=(0, -1, 0)),
                             fixed_dt=DT)
        shape = world.add_shape(model.Shape.capsule(height=1.1, radius=0.35))
        a = world.add_body(model.Motion(type=model.DYNAMIC, mass=80.0,
                                        linearVelocity=(0.5, 0, 0)),
                           collider=model.Collider(shape=shape), position=(0, 0.9, 0))
        b = world.add_body(model.Motion(type=model.DYNAMIC, mass=80.0,
                                        linearVelocity=(-0.5, 0, 0)),
                           collider=model.Collider(shape=shape), position=(0.68, 0.9, 0))
        for _ in range(10):
            world.step(DT)
        assert world.position[b][0] - world.position[a][0] > 0.68
        assert counter.calls == 0

    def test_a_character_against_a_capsule_body(self, monkeypatch):
        counter = _CountingGeneralPath(monkeypatch)
        world, character = _walk_into_a_standing_capsule()
        positions = _walk(world, character, 30)
        assert min(math.hypot(p[0], p[2]) for p in positions) < REACH + 0.01
        assert counter.calls == 0


def _walk_into_a_standing_capsule(start_z: float = -0.3,
                                  direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
                                  ) -> tuple[PhysicsWorld, CharacterController]:
    """A character 1.2 m from a heavy standing capsule, walking along ``direction``."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)), fixed_dt=DT)
    floor = world.add_shape(model.Shape.box((50, 1, 50)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0, -0.5, 0))
    pillar = world.add_shape(model.Shape.capsule(height=1.1, radius=0.35))
    world.add_body(model.Motion(type=model.DYNAMIC, mass=1000.0),
                   collider=model.Collider(shape=pillar), position=(0.0, 0.9, 0.0))
    character = CharacterController(world, CharacterCapabilities(radius=0.35, standHeight=1.8),
                                    position=(-1.2, 0.9, start_z), gravity=9.81)
    character.set_move(direction, 'walk')
    return world, character


def _walk(world: PhysicsWorld, character: CharacterController, steps: int) -> list[np.ndarray]:
    positions = []
    for _ in range(steps):
        character.update(DT)
        positions.append(character.position.copy())
    return positions


def _both_paths(monkeypatch, steps: int, *args) -> tuple[list[np.ndarray], list[np.ndarray]]:
    analytic = _walk(*_walk_into_a_standing_capsule(*args), steps)
    with monkeypatch.context() as patch:
        _general_path_only(patch)
        general = _walk(*_walk_into_a_standing_capsule(*args), steps)
    return analytic, general


#: Centre to centre, the character's capsule and the pillar's touch at this horizontal distance.
REACH = 0.35 + 0.35


class TestCharacterWalkingIntoACapsule:
    """Against the general path: the same to 1e-12 before contact, to EPA's
    depth tolerance at the first contact, and after that to the drift EPA's
    normal builds while the character stays pressed against the pillar."""

    def test_slides_along_it_as_the_general_path_does(self, monkeypatch):
        analytic, general = _both_paths(monkeypatch, 100)
        gaps = [float(np.linalg.norm(p - q)) for p, q in zip(analytic, general, strict=True)]
        touching = [i for i, p in enumerate(analytic) if math.hypot(p[0], p[2]) < REACH + 0.01]
        assert touching
        assert max(gaps[:touching[0]]) < 1e-12
        assert gaps[touching[0]] < 5e-5
        assert max(gaps) < 5e-4
        assert min(math.hypot(p[0], p[2]) for p in analytic) > REACH - 1e-6
        assert analytic[-1][0] > 2.0                    # went round it
        assert analytic[-1][2] < -0.3                   # on the side it arrived on

    def test_walking_straight_at_it_stops(self, monkeypatch):
        analytic, general = _both_paths(monkeypatch, 100, -0.3, (1.0, 0.0, 0.25))
        touching = [i for i, p in enumerate(analytic) if math.hypot(p[0], p[2]) < REACH + 0.01]
        first = touching[0]
        gaps = [float(np.linalg.norm(p - q)) for p, q in zip(analytic, general, strict=True)]
        assert max(gaps[:first]) < 1e-12
        assert gaps[first] < 5e-5
        assert max(gaps) < 2e-2
        stopped = analytic[first + 5]
        assert all(np.linalg.norm(p - stopped) < 1e-9 for p in analytic[first + 5:])
        assert math.hypot(stopped[0], stopped[2]) == pytest.approx(REACH, abs=1e-3)
