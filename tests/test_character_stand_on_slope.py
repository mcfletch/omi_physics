"""A crouched character stands up again on a slope, where its capsule rests against the ground."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld

DT = 1.0 / 60.0


def sloped_world(rise):
    """Ground that rises ``rise`` metres for each metre of x, as a sampled height field."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    ground = model.Shape.heightfield(
        sample=lambda x, z: rise * np.asarray(x, dtype='d') + 0.0 * np.asarray(z, dtype='d'),
        vectorized=True, extent=((-50.0, -50.0), (50.0, 50.0)),
        height_range=(-50.0 * rise - 1.0, 50.0 * rise + 1.0), feature=1.0)
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=world.add_shape(ground)),
                   position=(0.0, 0.0, 0.0))
    return world


@pytest.mark.parametrize('rise', [0.0, 0.1, 0.27, 0.58, 1.0])
def test_a_crouched_character_stands_up_on_a_slope(rise):
    world = sloped_world(rise)
    caps = CharacterCapabilities()
    character = CharacterController(world, caps, gravity=9.81)
    character.safe_bind((0.0, 3.0, 0.0))
    for _ in range(90):
        character.update(DT)
    assert character.grounded
    character.set_crouch(True)
    for _ in range(30):
        character.update(DT)
    assert character.crouching
    assert character.set_crouch(False) is True
    assert character.height == pytest.approx(caps.standHeight)


@pytest.mark.parametrize('rise', [0.1, 0.58])
def test_a_slide_on_a_slope_ends_standing(rise):
    world = sloped_world(rise)
    caps = CharacterCapabilities()
    character = CharacterController(world, caps, gravity=9.81)
    character.safe_bind((0.0, 3.0, 0.0))
    for _ in range(90):
        character.update(DT)
    character.set_move((0.0, 0.0, 1.0), 'run')
    for _ in range(30):
        character.update(DT)
    assert character.slide(0.8)
    for _ in range(80):
        character.update(DT)
    assert not character.crouching
    assert character.height == pytest.approx(caps.standHeight)
