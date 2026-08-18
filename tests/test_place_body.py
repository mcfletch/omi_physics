"""Moving a body between steps, and the world staying answerable about it.

A game moves a kinematic body itself -- an animated platform, a lift, a
combatant staged for one query -- and then asks the world where things are.
Whether the answer is right turns on the body's world AABB, which is what the
broadphase rejects against and what nothing recomputes on a bare write to the
position array.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.raycast import raycast
from omi_physics.world import PhysicsWorld


@pytest.fixture
def standing():
    """One upright capsule, a person's size, and the world it is in."""
    world = PhysicsWorld()
    shape = world.add_shape(model.Shape.capsule(height=0.61, radius=0.41))
    body = world.add_body(model.Motion(type=model.KINEMATIC),
                          collider=model.Collider(shape=shape))
    return world, body


class TestABodyThatWasMoved:
    def test_a_ray_meets_it_where_it_now_is(self, standing):
        """The property a staged body exists for: it can be shot at."""
        world, body = standing
        world.place_body(body, position=(10.0, 0.71, 0.0))
        hit = raycast(world, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        assert hit is not None and hit.body == body

    def test_a_ray_no_longer_meets_it_where_it_was(self, standing):
        world, body = standing
        world.place_body(body, position=(10.0, 20.0, 0.0))
        assert raycast(world, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) is None

    def test_it_does_not_arrive_moving(self, standing):
        """Placed, not thrown: a step must not read the jump as a velocity."""
        world, body = standing
        world.place_body(body, position=(10.0, 0.71, 0.0))
        assert np.allclose(world.prev_position[body], world.position[body])

    def test_turning_it_without_moving_it(self, standing):
        world, body = standing
        world.place_body(body, orientation=(0.0, 0.0, 0.7071068, 0.7071068))
        assert np.allclose(world.position[body], (0.0, 0.0, 0.0))
        # Laid on its side, the capsule is wider than it is tall.
        low, high = world.aabb_min[body], world.aabb_max[body]
        assert (high - low)[0] > (high - low)[1]

    def test_placing_it_nowhere_in_particular_leaves_it_alone(self, standing):
        world, body = standing
        world.place_body(body)
        assert np.allclose(world.position[body], (0.0, 0.0, 0.0))

    def test_a_body_with_no_collider_still_places(self, standing):
        """A trigger or a marker has no shape to fit a box to."""
        world, _body = standing
        marker = world.add_body(model.Motion(type=model.KINEMATIC))
        world.place_body(marker, position=(3.0, 4.0, 5.0))
        assert np.allclose(world.aabb_min[marker], (3.0, 4.0, 5.0))
