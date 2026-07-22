"""Unit tests for the game-facing impulse API and pairwise friction override.

These two additions to :class:`~omi_physics.world.PhysicsWorld` are what
an interactive game (the marble demo) needs on top of the passive simulation:

* ``apply_impulse`` / ``apply_angular_impulse`` let per-frame input push a body
  (steering spin, ramp boosts, spring traps) with the mass/inertia and wake
  bookkeeping handled once, in the engine, instead of poked into the raw arrays.
* ``set_pair_friction`` overrides the material-combine rule for a *specific* pair
  of materials, so "rubber grips stone but slips on ice" can be expressed
  independently of what every other material does — which a single combine rule of
  per-material scalars cannot represent.
"""
import numpy as np

from omi_physics import model
from omi_physics.world import PhysicsWorld


def _dynamic_sphere(world, mass=2.0, material=-1, position=(0, 0, 0)):
    shape = world.add_shape(model.Shape.sphere(0.5))
    return world.add_body(
        model.Motion(type=model.DYNAMIC, mass=mass),
        collider=model.Collider(shape=shape, physicsMaterial=material),
        position=position)


# -- linear impulse ------------------------------------------------------

def test_apply_impulse_changes_velocity_by_impulse_over_mass():
    world = PhysicsWorld(sleep_enabled=False)
    i = _dynamic_sphere(world, mass=2.0)
    world.apply_impulse(i, (4.0, 0.0, 0.0))
    # dv = J / m
    assert np.allclose(world.linear_velocity[i], (2.0, 0.0, 0.0))


def test_apply_impulse_accumulates():
    world = PhysicsWorld(sleep_enabled=False)
    i = _dynamic_sphere(world, mass=1.0)
    world.apply_impulse(i, (1.0, 0.0, 0.0))
    world.apply_impulse(i, (0.0, 3.0, 0.0))
    assert np.allclose(world.linear_velocity[i], (1.0, 3.0, 0.0))


def test_apply_impulse_wakes_a_sleeping_body():
    world = PhysicsWorld(sleep_enabled=True)
    i = _dynamic_sphere(world, mass=1.0)
    world._awake[i] = False
    world.apply_impulse(i, (0.0, 0.0, 5.0))
    assert world.awake[i]


def test_apply_impulse_ignores_static_bodies():
    """A static body has infinite mass; an impulse must not move it (no divide)."""
    world = PhysicsWorld(sleep_enabled=False)
    shape = world.add_shape(model.Shape.box((2, 1, 2)))
    i = world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=shape))
    world.apply_impulse(i, (10.0, 0.0, 0.0))
    assert np.allclose(world.linear_velocity[i], (0.0, 0.0, 0.0))


# -- angular impulse -----------------------------------------------------

def test_apply_angular_impulse_spins_about_inertia():
    world = PhysicsWorld(sleep_enabled=False)
    i = _dynamic_sphere(world, mass=2.0)          # solid sphere I = 2/5 m r^2
    inertia = 0.4 * 2.0 * 0.5 ** 2
    world.apply_angular_impulse(i, (0.0, 0.0, inertia))   # L = I * 1 rad/s
    assert np.allclose(world.angular_velocity[i], (0.0, 0.0, 1.0))


def test_apply_angular_impulse_wakes_body():
    world = PhysicsWorld(sleep_enabled=True)
    i = _dynamic_sphere(world, mass=1.0)
    world._awake[i] = False
    world.apply_angular_impulse(i, (0.0, 1.0, 0.0))
    assert world.awake[i]


# -- pairwise friction ---------------------------------------------------

def test_pair_friction_override_is_returned_for_the_pair():
    world = PhysicsWorld()
    a = world.add_material(model.Material(dynamicFriction=0.4))
    b = world.add_material(model.Material(dynamicFriction=0.4))
    world.set_pair_friction(a, b, static_mu=0.9, dynamic_mu=0.8)
    assert world.pair_friction_for(a, b) == (0.9, 0.8)


def test_pair_friction_is_symmetric():
    world = PhysicsWorld()
    a = world.add_material(model.Material())
    b = world.add_material(model.Material())
    world.set_pair_friction(a, b, 0.5, 0.3)
    assert world.pair_friction_for(b, a) == (0.5, 0.3)


def test_pair_friction_absent_returns_none():
    world = PhysicsWorld()
    a = world.add_material(model.Material())
    b = world.add_material(model.Material())
    assert world.pair_friction_for(a, b) is None


def test_solver_uses_pairwise_dynamic_friction_when_present():
    """A sliding block decelerates by the *pairwise* mu, not the material combine.

    Two materials each declare dynamicFriction 0.9 (combine → 0.9), but the pair
    is overridden to 0.1.  After stepping, the block that used the override retains
    far more speed than the combine rule alone would leave it.
    """
    def slide(with_override):
        world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                             sleep_enabled=False)
        mat = world.add_material(model.Material(staticFriction=0.9, dynamicFriction=0.9))
        ground = world.add_shape(model.Shape.box((40, 1, 40)))
        world.add_body(model.Motion(type=model.STATIC),
                       collider=model.Collider(shape=ground, physicsMaterial=mat),
                       position=(0, -0.5, 0))
        box = world.add_shape(model.Shape.box((1, 1, 1)))
        b = world.add_body(model.Motion(type=model.DYNAMIC, mass=1.0,
                                        linearVelocity=(6.0, 0, 0)),
                           collider=model.Collider(shape=box, physicsMaterial=mat),
                           position=(0, 0.5, 0))
        if with_override:
            world.set_pair_friction(mat, mat, 0.1, 0.1)
        for _ in range(30):
            world.step(1 / 60.0)
        return abs(world.linear_velocity[b][0])

    fast = slide(with_override=True)
    slow = slide(with_override=False)
    assert fast > slow + 1.0
