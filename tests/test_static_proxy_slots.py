"""The world-space proxy of a fixed body is the proxy of the body in its slot now."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.collisionquery import WorldQuery
from omi_physics.world import PhysicsWorld


def world_with_a_slab() -> tuple[PhysicsWorld, WorldQuery, int]:
    world = PhysicsWorld()
    slab = world.add_shape(model.Shape.box((10.0, 1.0, 10.0)))
    first = world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=slab),
                           position=(0.0, 0.0, 0.0))
    return world, WorldQuery(world), first


def test_a_slot_handed_to_a_new_fixed_body_answers_the_new_body_and_not_the_one_removed():
    world, query, first = world_with_a_slab()
    before = query.static_proxy(first)
    world.remove_body(first)
    ball = world.add_shape(model.Shape.sphere(0.5))
    second = world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=ball),
                            position=(40.0, 3.0, -7.0))
    assert second == first
    after = query.static_proxy(second)
    assert after is not before
    assert type(after).__name__ != type(before).__name__
    assert np.allclose(after.center, (40.0, 3.0, -7.0))


def test_a_fixed_body_placed_somewhere_else_is_answered_from_where_it_is():
    world, query, first = world_with_a_slab()
    query.static_proxy(first)
    world.place_body(first, position=(20.0, 2.0, 5.0))
    assert np.allclose(query.static_proxy(first).center, (20.0, 2.0, 5.0))


def test_a_body_that_stays_keeps_the_proxy_it_has():
    world, query, first = world_with_a_slab()
    proxy = query.static_proxy(first)
    other = world.add_shape(model.Shape.sphere(0.5))
    world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=other),
                   position=(30.0, 0.0, 0.0))
    assert query.static_proxy(first) is proxy


def test_a_vacant_slot_has_no_proxy():
    world, query, first = world_with_a_slab()
    world.add_shape(model.Shape.sphere(0.5))
    world.remove_body(first)
    with pytest.raises(LookupError):
        query.static_proxy(first)
