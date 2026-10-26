"""The columns the world keeps indices and caches over are read-only views, and change through
the world's methods."""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.collisionquery import WorldQuery, dynamic_rows
from omi_physics.world import PhysicsWorld


def world_with_a_post_and_a_ball() -> tuple[PhysicsWorld, int, int]:
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)))
    post = world.add_shape(model.Shape.box((0.4, 2.0, 0.4)))
    fixed = world.add_body(model.Motion(type=model.STATIC), collider=model.Collider(shape=post),
                           position=(0.0, 1.0, 0.0))
    ball = world.add_shape(model.Shape.sphere(0.3))
    rolling = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0),
                             collider=model.Collider(shape=ball), position=(5.0, 0.3, 0.0))
    return world, fixed, rolling


@pytest.mark.parametrize('column, value', [
    ('position', (1.0, 2.0, 3.0)),
    ('orientation', (0.0, 0.0, 0.0, 1.0)),
    ('motion_type', 1),
    ('collider_shape', -1),
    ('trigger_shape', 0),
    ('sensor', True),
])
def test_a_write_to_the_column_raises(column, value):
    world, fixed, _ball = world_with_a_post_and_a_ball()
    with pytest.raises(ValueError, match='read-only'):
        getattr(world, column)[fixed] = value


def test_the_pose_columns_read_the_current_pose():
    world, fixed, _ball = world_with_a_post_and_a_ball()
    view = world.position
    world.place_body(fixed, position=(7.0, 1.0, 0.0))
    assert tuple(view[fixed]) == (7.0, 1.0, 0.0)


def test_a_collider_taken_away_leaves_no_fixed_body_near():
    world, fixed, _ball = world_with_a_post_and_a_ball()
    query = WorldQuery(world)
    low, high = np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.0, 1.0])
    assert fixed in query.static_near(low, high)
    world.set_collider(fixed, None)
    assert fixed not in query.static_near(low, high)
    assert fixed not in query.static_bodies()


def test_a_collider_given_back_is_found_with_its_new_shape():
    world, fixed, _ball = world_with_a_post_and_a_ball()
    query = WorldQuery(world)
    query.static_proxy(fixed)
    wide = world.add_shape(model.Shape.box((6.0, 1.0, 6.0)))
    world.set_collider(fixed, model.Collider(shape=wide))
    assert query.static_proxy(fixed).kind == 'box'
    assert np.allclose(query.static_proxy(fixed).half, (3.0, 0.5, 3.0))
    assert world.aabb_max[fixed][0] == pytest.approx(3.05)
    assert fixed in query.static_near(np.array([2.5, 0.0, 2.5]), np.array([2.9, 2.0, 2.9]))


def test_a_dynamic_body_without_a_collider_is_not_a_moving_obstacle():
    world, _fixed, ball = world_with_a_post_and_a_ball()
    assert ball in dynamic_rows(world)
    world.set_collider(ball, None)
    assert ball not in dynamic_rows(world)


def test_a_collider_for_a_vacant_slot_is_refused():
    world, fixed, _ball = world_with_a_post_and_a_ball()
    world.remove_body(fixed)
    with pytest.raises(IndexError):
        world.set_collider(fixed, model.Collider(shape=0))
