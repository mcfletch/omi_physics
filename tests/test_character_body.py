"""A walking character's body in the world: a sensor that follows it (no GL).

The controller moves its capsule itself and collides it with the world's
static geometry. Its body is a *sensor*: it enters trigger volumes and its
touches with moving bodies are reported, but the solver neither pushes it nor
pushes anything with it, rays pass through it, and the controller does not
collide with its own body.
"""
import numpy as np
import pytest

from omi_physics import model, raycast
from omi_physics.character import CharacterCapabilities, CharacterController
from omi_physics.world import PhysicsWorld

DT = 1.0 / 60.0


def floor_world():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         fixed_dt=DT)
    shape = world.add_shape(model.Shape.box((60, 1, 60)))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=shape),
                   position=(0, -0.5, 0))
    return world


def walker(world, body=True):
    character = CharacterController(world, CharacterCapabilities(), body=body)
    character.safe_bind((-4.0, 3.0, 0.0))
    return character


def walk(world, character, seconds, direction=(1, 0, 0)):
    character.set_move(direction)
    for _ in range(round(seconds / DT)):
        character.update(DT)
        world.step(DT)


def pad(world, at=(0.0, 0.5, 0.0)):
    shape = world.add_shape(model.Shape.box((1.0, 1.0, 1.0)))
    return world.add_body(model.Motion(type=model.STATIC),
                          trigger=model.Trigger(shape=shape), position=at)


class TestTheBody:
    def test_it_follows_the_character(self) -> None:
        world = floor_world()
        character = walker(world)
        walk(world, character, 0.5)
        character.update(DT)
        assert np.allclose(world.position[character.body], character.position)
        # It moves at the character's velocity, so a touch has a closing speed.
        assert np.allclose(world.linear_velocity[character.body], character.velocity)

    def test_it_is_a_sensor(self) -> None:
        world = floor_world()
        character = walker(world)
        assert world.is_sensor(character.body)
        assert world.motion_type[character.body] == 1

    def test_the_character_does_not_collide_with_it(self) -> None:
        with_body, without = floor_world(), floor_world()
        a, b = walker(with_body), walker(without, body=False)
        walk(with_body, a, 1.0)
        walk(without, b, 1.0)
        assert np.allclose(a.position, b.position)
        assert b.body is None

    def test_a_ray_passes_through_it(self) -> None:
        world = floor_world()
        character = walker(world)
        eye = character.position + np.array([-2.0, 0.0, 0.0])
        hit = raycast.raycast(world, eye, (1, 0, 0), max_distance=10.0)
        assert hit is None or hit.body != character.body


class TestWhatItMeets:
    def test_it_enters_and_leaves_a_trigger(self) -> None:
        world = floor_world()
        volume = pad(world)
        character = walker(world)
        heard = []
        world.add_trigger_listener(lambda kind, t, o: heard.append((kind, t, o)))
        walk(world, character, 3.0)
        assert [event for event in heard if event[0] != 'stay'] == [
            ('enter', volume, character.body), ('exit', volume, character.body)]

    def test_a_crate_is_touched_and_not_pushed(self) -> None:
        world = floor_world()
        shape = world.add_shape(model.Shape.box((0.6, 0.6, 0.6)))
        crate = world.add_body(model.Motion(type=model.DYNAMIC, mass=5.0),
                               collider=model.Collider(shape=shape),
                               position=(0.0, 0.3, 0.0))
        character = walker(world)
        heard = []
        world.add_contact_listener(heard.append)
        for _ in range(30):
            world.step(DT)
        start = world.position[crate].copy()
        walk(world, character, 1.5)
        touches = [event for event in heard
                   if character.body in (event.a.index, event.b.index)]
        assert touches and touches[0].phase == 'begin'
        assert not touches[0].solved
        assert np.allclose(world.position[crate], start, atol=1e-3)

    def test_a_removed_character_takes_its_body_with_it(self) -> None:
        world = floor_world()
        character = walker(world)
        body = character.body
        character.remove()
        assert character.body is None
        assert body in world._free


def test_a_sensor_body_is_never_solved() -> None:
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), fixed_dt=DT)
    shape = world.add_shape(model.Shape.sphere(0.5))
    sensor = world.add_body(model.Motion(type=model.KINEMATIC),
                            collider=model.Collider(shape=shape), sensor=True)
    ball = world.add_body(model.Motion(type=model.DYNAMIC, linearVelocity=(-2, 0, 0)),
                          collider=model.Collider(shape=shape), position=(0.8, 0, 0))
    for _ in range(30):
        world.step(DT)
    assert world.linear_velocity[ball][0] == pytest.approx(-2.0)
    assert world.contacts == []
    assert world.is_sensor(sensor)
