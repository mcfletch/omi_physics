"""A capsule against one box with numbers pushes out as it does against many with arrays, and a
character among a few boxes is collided with them that way."""
import numpy as np
import pytest

from omi_physics import collide, model
from omi_physics.body import CapsuleProxy
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.mathutil import quat_from_axis_angle, quat_to_matrix
from omi_physics.world import PhysicsWorld


@pytest.mark.parametrize('seed', range(8))
def test_one_box_with_numbers_is_many_with_arrays(seed):
    rng = np.random.default_rng(seed)
    met = 0
    for _ in range(200):
        axis = rng.normal(size=3)
        turn = quat_to_matrix(quat_from_axis_angle(axis / np.linalg.norm(axis),
                                                   float(rng.uniform(0, 6))))
        cap_turn = quat_to_matrix(quat_from_axis_angle(rng.normal(size=3),
                                                       float(rng.uniform(0, 6))))
        capsule = CapsuleProxy(rng.uniform(-1.5, 1.5, 3), float(rng.uniform(0.0, 1.0)),
                               float(rng.uniform(0.05, 0.6)),
                               np.eye(3) if rng.random() < 0.5 else cap_turn)
        center = rng.uniform(-0.5, 0.5, 3)
        half = rng.uniform(0.05, 1.0, 3)
        hit, pushes, depths = collide.capsule_box_pushes(capsule, center[None], half[None],
                                                         turn[None])
        got = collide.capsule_box_push(tuple(capsule.p0.tolist()), tuple(capsule.p1.tolist()),
                                       capsule.radius, tuple(center.tolist()),
                                       tuple(half.tolist()),
                                       tuple(tuple(row) for row in turn.tolist()))
        assert (got is not None) == bool(hit[0])
        if got is not None:
            met += 1
            assert got[0] == pytest.approx(pushes[0], abs=1e-9)
            assert got[1] == pytest.approx(depths[0], abs=1e-9)
    assert met


def test_a_character_on_a_box_floor_is_collided_with_numbers(monkeypatch):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    floor = world.add_shape(model.Shape.box((200.0, 1.0, 200.0)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=floor),
                   position=(0.0, -0.5, 0.0))
    post = world.add_shape(model.Shape.box((1.0, 1.5, 1.0)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=post),
                   position=(3.0, 0.75, 0.0))
    character = CharacterController(world, CharacterCapabilities(), gravity=9.81)
    character.safe_bind((0.0, 1.0, 0.0))
    asked = []
    original = collide.capsule_box_pushes
    monkeypatch.setattr(collide, 'capsule_box_pushes', lambda *a: asked.append(1) or original(*a))
    character.set_move((1.0, 0.0, 0.0), 'walk')
    for _ in range(120):
        character.update(1 / 60)
    assert character.grounded
    assert float(character.position[0]) < 3.0 - 0.5 - character.caps.radius + 0.01
    assert asked == []
