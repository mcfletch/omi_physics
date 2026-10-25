"""A bundle of rays, some parallel to the sides of a box, raises nothing under strict NumPy errors.

A ray along a triangle's plane has a zero determinant; the test that masks it out
must not do arithmetic on the infinities it produced before masking.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.raycast import raycast_many
from omi_physics.world import PhysicsWorld

CORNERS = np.array([[x, y, z] for x in (-2, 2) for y in (-0.5, 0.5) for z in (-2, 2)], dtype='d')
FACES = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]


def box_world():
    world = PhysicsWorld(sleep_enabled=False)
    shape = world.add_shape(model.Shape.trimesh(CORNERS, FACES))
    world.add_body(model.Motion(type=model.STATIC), model.Collider(shape=shape))
    return world


def test_rays_straight_down_past_the_boxs_sides_are_answered_without_an_error():
    world = box_world()
    origins = [(0, 5, 0), (1, 5, 1), (50, 5, 0), (2, 5, 0)]
    with np.errstate(all='raise'):
        hits = raycast_many(world, origins, [(0, -1, 0)] * 4)
    assert hits[0] is not None and hits[0].distance == pytest.approx(4.5)
    assert hits[2] is None
