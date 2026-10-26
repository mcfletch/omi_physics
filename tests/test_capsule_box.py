"""A capsule against oriented boxes, many at once, and a character that walks among them.

The answers are held against GJK and EPA, which answer every convex pair, and
against an exact reference where EPA's own tolerance is the larger error: the
rounded part of a capsule is a curve that EPA's polytope approaches face by face.
"""
import numpy as np
import pytest

from omi_physics import collide, gjk, mathutil, model
from omi_physics.body import BoxProxy, CapsuleProxy, SphereProxy
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld


def turn(rng) -> np.ndarray:
    quaternion = rng.normal(size=4)
    return mathutil.quat_to_matrix((quaternion / np.linalg.norm(quaternion))[None])[0]


def random_pair(rng) -> tuple[CapsuleProxy, BoxProxy]:
    box = BoxProxy(rng.uniform(-1, 1, 3), rng.uniform(0.1, 1.0, 3), turn(rng))
    capsule = CapsuleProxy(rng.uniform(-1.5, 1.5, 3), rng.uniform(0.0, 0.8),
                           rng.uniform(0.05, 0.5), turn(rng))
    return capsule, box


def pushes(capsule: CapsuleProxy, *boxes: BoxProxy):
    return collide.capsule_box_pushes(capsule, np.array([b.center for b in boxes]),
                                      np.array([b.half for b in boxes]),
                                      np.array([b.R for b in boxes]))


def axis_distance(capsule: CapsuleProxy, box: BoxProxy) -> tuple[float, np.ndarray]:
    """The distance from the capsule's axis to the box and the direction from box to axis.

    Golden-section search on the distance along the axis, which is convex.
    """
    def nearest(t: float) -> tuple[float, np.ndarray]:
        point = capsule.p0 + t * (capsule.p1 - capsule.p0)
        local = box.R.T @ (point - box.center)
        gap = local - np.clip(local, -box.half, box.half)
        return float(np.linalg.norm(gap)), box.R @ gap

    low, high = 0.0, 1.0
    ratio = (5 ** 0.5 - 1) / 2
    for _ in range(200):
        first, second = high - ratio * (high - low), low + ratio * (high - low)
        if nearest(first)[0] < nearest(second)[0]:
            high = second
        else:
            low = first
    distance, gap = nearest(0.5 * (low + high))
    return distance, gap / distance if distance > 0.0 else gap


def test_a_capsule_overlaps_the_boxes_gjk_says_it_overlaps():
    rng = np.random.default_rng(1)
    for _ in range(1500):
        capsule, box = random_pair(rng)
        hit, _push, depth = pushes(capsule, box)
        found = gjk.collide_convex(0, 1, capsule, box)
        if found and found[0].depth > 1e-6:
            assert hit[0]
        if hit[0] and depth[0] > 1e-6:
            assert found


def test_a_capsule_whose_axis_is_outside_a_box_is_pushed_from_the_nearest_point():
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(1500):
        capsule, box = random_pair(rng)
        hit, push, depth = pushes(capsule, box)
        distance, direction = axis_distance(capsule, box)
        if not hit[0] or distance < 1e-6:
            continue
        checked += 1
        assert depth[0] == pytest.approx(capsule.radius - distance, abs=1e-9)
        assert np.allclose(push[0], direction, atol=1e-6)
        found = gjk.collide_convex(0, 1, capsule, box)
        assert found[0].depth == pytest.approx(depth[0], abs=1e-6)
    assert checked > 100


def test_a_capsule_whose_axis_reaches_into_a_box_agrees_with_epa():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(1500):
        capsule, box = random_pair(rng)
        hit, push, depth = pushes(capsule, box)
        if not hit[0] or axis_distance(capsule, box)[0] > 1e-9:
            continue
        found = gjk.collide_convex(0, 1, capsule, box)
        checked += 1
        assert found[0].depth == pytest.approx(depth[0], abs=1e-6)
        assert np.allclose(-found[0].normal, push[0], atol=1e-5)
    assert checked > 100


def test_a_capsule_inside_a_box_leaves_by_the_nearest_face():
    box = BoxProxy((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), np.eye(3))
    capsule = CapsuleProxy((1.0, 0.0, 4.0), 0.5, 0.25, np.eye(3))
    hit, push, depth = pushes(capsule, box)
    assert hit[0]
    assert np.allclose(push[0], (0.0, 0.0, 1.0))
    assert depth[0] == pytest.approx(1.0 + 0.25)


def test_a_capsule_lying_through_a_box_is_pushed_out_of_its_side():
    box = BoxProxy((0.0, 0.0, 0.0), (0.2, 2.0, 0.2), np.eye(3))
    quarter = np.array([[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    across = mathutil.quat_to_matrix(quarter)[0]
    capsule = CapsuleProxy((0.0, 0.0, 0.1), 1.0, 0.1, across)
    hit, push, depth = pushes(capsule, box)
    assert hit[0]
    assert np.allclose(push[0], (0.0, 0.0, 1.0))
    # The box's side past the axis, and the radius.
    assert depth[0] == pytest.approx((0.2 - 0.1) + 0.1)


def test_a_capsule_of_no_length_is_a_sphere():
    rng = np.random.default_rng(4)
    for _ in range(500):
        capsule, box = random_pair(rng)
        capsule = CapsuleProxy(capsule.center, 0.0, capsule.radius, capsule.R)
        hit, push, depth = pushes(capsule, box)
        found = collide.sphere_box(0, 1, SphereProxy(capsule.center, capsule.radius), box)
        assert bool(hit[0]) == bool(found)
        if found:
            assert depth[0] == pytest.approx(found[0].depth, abs=1e-12)
            assert np.allclose(-found[0].normal, push[0], atol=1e-12)


def test_many_boxes_are_answered_as_each_alone():
    rng = np.random.default_rng(5)
    capsule, _ = random_pair(rng)
    boxes = [random_pair(rng)[1] for _ in range(40)]
    hit, push, depth = pushes(capsule, *boxes)
    for k, box in enumerate(boxes):
        one_hit, one_push, one_depth = pushes(capsule, box)
        assert hit[k] == one_hit[0]
        assert np.array_equal(push[k], one_push[0]) and depth[k] == one_depth[0]


def test_a_character_among_boxes_asks_gjk_nothing(monkeypatch):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    floor = world.add_shape(model.Shape.box((200, 1, 200)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0, -0.5, 0))
    post = world.add_shape(model.Shape.box((0.4, 2.0, 0.4)))
    world.add_bodies(post, np.array([(x * 1.5, 1.0, 1.0) for x in range(10)]),
                     collider=model.Collider(shape=post))
    controller = CharacterController(world, CharacterCapabilities(), position=(0.0, 0.95, 0.0))
    asked = []
    original = gjk.collide_convex
    monkeypatch.setattr(gjk, 'collide_convex',
                        lambda *args: asked.append(args) or original(*args))
    controller.set_move((1.0, 0.0, 0.3), 'run')
    for _ in range(120):
        controller.update(1.0 / 60.0)
    assert controller.grounded
    assert asked == []
