"""Void gravity suppression — hover over a bottomless gap instead of dropping."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld
from omi_physics.character import CharacterController, CharacterCapabilities

DT = 1.0 / 60.0


def add_box(world, size, center):
    shape = world.add_shape(model.Shape.box(size))
    return world.add_body(model.Motion(type=model.STATIC),
                          collider=model.Collider(shape=shape), position=center)


def gapped_world():
    """Two floor slabs on either side of a bottomless gap around x == 0."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    add_box(world, (20, 1, 20), (-12, -0.5, 0))    # left slab: x in [-22, -2]
    add_box(world, (20, 1, 20), (12, -0.5, 0))     # right slab: x in [2, 22]
    return world


def character(world, suppress, at=(0, 0, 0)):
    caps = CharacterCapabilities(suppressOverVoid=suppress)
    ch = CharacterController(world, caps, gravity=9.81)
    ch.position = np.array([at[0], at[1], at[2]], dtype='d')
    return ch


def test_over_void_detects_gap_between_slabs():
    ch = character(gapped_world(), suppress=True, at=(0, 0.0, 0))
    assert ch._over_void()


def test_over_solid_ground_is_not_a_void():
    ch = character(gapped_world(), suppress=True, at=(-12, 0.0, 0))
    assert not ch._over_void()


def test_edge_footprint_partly_on_ground_is_not_a_void():
    # base centre just past the slab edge, but the radius ring still overlaps it
    ch = character(gapped_world(), suppress=True, at=(-1.85, 0.0, 0))
    assert ch.caps.radius > 0.15                    # ring reaches back onto the slab
    assert not ch._over_void()


def test_suppressed_character_hovers_over_void():
    ch = character(gapped_world(), suppress=True, at=(0, 0.0, 0))
    y0 = ch.position[1]
    for _ in range(120):                            # 2 s of updates
        ch.update(DT)
    assert ch.position[1] == pytest.approx(y0, abs=1e-6)
    assert ch.vy == pytest.approx(0.0)
    assert not ch.grounded


def test_unsuppressed_character_falls_into_void():
    ch = character(gapped_world(), suppress=False, at=(0, 0.0, 0))
    y0 = ch.position[1]
    for _ in range(120):
        ch.update(DT)
    assert ch.position[1] < y0 - 5.0                # in free fall, no floor to catch it


def test_suppression_does_not_block_jump_ascent_over_void():
    ch = character(gapped_world(), suppress=True, at=(0, 0.0, 0))
    ch.grounded = True                              # allow the jump to fire
    assert ch.jump()
    rose = False
    for _ in range(30):
        before = ch.position[1]
        ch.update(DT)
        if ch.position[1] > before + 1e-6:
            rose = True
    assert rose                                     # ascent was not clamped to a hover
