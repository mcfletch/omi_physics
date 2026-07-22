"""Phase 5 character controller states (no GL)."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld
from omi_physics.character import CharacterController, CharacterCapabilities

DT = 1.0 / 60.0


def add_box(world, size, center, static=True):
    shape = world.add_shape(model.Shape.box(size))
    mt = model.STATIC if static else model.DYNAMIC
    return world.add_body(model.Motion(type=mt),
                          collider=model.Collider(shape=shape), position=center)


def floor_world():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    add_box(world, (60, 1, 60), (0, -0.5, 0))
    return world


def standing_character(world, caps=None, at=(0, 0, 0)):
    ch = CharacterController(world, caps or CharacterCapabilities(), gravity=9.81)
    ch.safe_bind((at[0], 3.0, at[2]))              # drop onto the floor
    return ch


def walk_distance(world, ch, mode, seconds=1.0):
    ch.set_move((1, 0, 0), mode=mode)
    start = ch.position[0]
    for _ in range(int(seconds / DT)):
        ch.update(DT)
    return ch.position[0] - start


def test_character_seats_on_floor_after_bind():
    world = floor_world()
    ch = standing_character(world)
    assert ch.grounded
    assert ch.base()[1] == pytest.approx(0.0, abs=0.05)


def test_speed_tiers_walk_run_sprint():
    walk = walk_distance(*(lambda w: (w, standing_character(w)))(floor_world()), mode='walk')
    run = walk_distance(*(lambda w: (w, standing_character(w)))(floor_world()), mode='run')
    sprint = walk_distance(*(lambda w: (w, standing_character(w)))(floor_world()), mode='sprint')
    assert walk < run < sprint
    assert walk == pytest.approx(3.0, rel=0.2)


def test_jump_only_when_grounded():
    world = floor_world()
    ch = standing_character(world)
    assert ch.jump() is True
    assert ch.vy > 0
    ch.update(DT)
    assert ch.jump() is False                      # airborne: no double jump


def test_crouch_shrinks_capsule_height():
    world = floor_world()
    ch = standing_character(world)
    tall = ch.height
    ch.set_crouch(True)
    assert ch.height < tall
    assert ch.crouching


def test_cannot_stand_under_low_ceiling():
    world = floor_world()
    caps = CharacterCapabilities()
    ceiling_y = caps.crouchHeight + 0.2            # room to crouch, not to stand
    add_box(world, (4, 0.2, 4), (0, ceiling_y + 0.1, 0))
    ch = CharacterController(world, caps, gravity=9.81)
    ch.position = np.array([0.0, caps.crouchHeight * 0.5, 0.0])
    ch.crouching = True
    ch.grounded = True
    assert ch.set_crouch(False) is False           # blocked from standing
    assert ch.crouching


def test_fly_ignores_gravity():
    world = floor_world()
    ch = standing_character(world)
    ch.set_fly(True)
    y0 = ch.position[1]
    for _ in range(60):
        ch.update(DT)
    assert ch.position[1] == pytest.approx(y0, abs=1e-6)


def test_steps_up_small_ledge():
    world = floor_world()
    caps = CharacterCapabilities(stepHeight=0.4)
    add_box(world, (40, 0.3, 40), (21.0, 0.15, 0))   # 0.3 m step < stepHeight
    ch = standing_character(world, caps)
    ch.set_move((1, 0, 0), mode='walk')
    for _ in range(120):                              # ~2 s: onto and along the ledge
        ch.update(DT)
    assert ch.position[0] > 2.0                       # climbed onto the ledge
    assert ch.base()[1] == pytest.approx(0.3, abs=0.12)


def test_blocked_by_tall_step():
    world = floor_world()
    caps = CharacterCapabilities(stepHeight=0.35)
    add_box(world, (4, 2.0, 4), (3.0, 1.0, 0))     # 2 m wall > stepHeight
    ch = standing_character(world, caps)
    ch.set_move((1, 0, 0), mode='run')
    for _ in range(180):
        ch.update(DT)
    assert ch.position[0] < 2.8                     # stopped at the wall


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
