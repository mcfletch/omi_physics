"""A ray that says which collision systems it hits (no GL).

A shot passes through its own team and through anything its filter excludes,
decided by the same :class:`~omi_physics.model.CollisionFilter` a collider
carries, so what a weapon can hit is data rather than a list of body indices.
"""
from omi_physics import model, raycast
from omi_physics.world import PhysicsWorld


def lined_up():
    """A player, a teammate and a wall along +x, nearest first."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0))
    red = world.add_filter(model.CollisionFilter(collisionSystems=('red',)))
    walls = world.add_filter(model.CollisionFilter(collisionSystems=('wall',)))
    shape = world.add_shape(model.Shape.box((1, 2, 1)))
    bodies = [world.add_body(model.Motion(type=model.STATIC),
                             collider=model.Collider(shape=shape, collisionFilter=f),
                             position=(x, 0, 0))
              for x, f in ((2, red), (5, walls))]
    return world, bodies


RED_SHOT = model.CollisionFilter(collisionSystems=('shot',),
                                 notCollideWithSystems=('red',))


def test_a_shot_passes_through_what_its_filter_excludes() -> None:
    world, (teammate, wall) = lined_up()
    assert raycast.raycast(world, (0, 0, 0), (1, 0, 0)).body == teammate
    assert raycast.raycast(world, (0, 0, 0), (1, 0, 0), filter=RED_SHOT).body == wall


def test_the_filter_applies_to_a_bundle() -> None:
    world, (_teammate, wall) = lined_up()
    hits = raycast.raycast_many(world, [(0, 0, 0), (0, 0.5, 0)], [(1, 0, 0)] * 2,
                                filter=RED_SHOT)
    assert [hit.body for hit in hits] == [wall, wall]


def test_the_filter_applies_to_everything_along_a_ray() -> None:
    world, (_teammate, wall) = lined_up()
    hits = raycast.bodies_along(world, (0, 0, 0), (1, 0, 0), filter=RED_SHOT)
    assert [hit.body for hit in hits] == [wall]


def test_a_filter_that_collides_with_nothing_hits_nothing() -> None:
    world, _bodies = lined_up()
    blind = model.CollisionFilter(collideWithSystems=())
    assert raycast.raycast(world, (0, 0, 0), (1, 0, 0), filter=blind) is None
