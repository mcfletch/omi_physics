"""Phase 0 integrator: symplectic Euler, gravity, drag, gravityFactor.

Pure-CPU, no GL.  Analytic references pin the behaviour.
"""
import numpy as np
import pytest

from omi_physics import model
from omi_physics.world import PhysicsWorld


def make_world(**kw):
    return PhysicsWorld(**kw)


def test_freefall_matches_symplectic_closed_form():
    """v=0 start under gravity g: x_n = g dt^2 n(n+1)/2 (symplectic Euler)."""
    g = 9.81
    dt = 1.0 / 240.0
    world = make_world(gravity=model.Gravity(gravity=g, direction=(0, -1, 0)))
    body = world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 0, 0))
    n = 240
    for _ in range(n):
        world.step(dt)
    expected = -0.5 * g * dt * dt * n * (n + 1)
    assert world.position[body][1] == pytest.approx(expected, rel=1e-9)


def test_freefall_approaches_analytic_as_dt_shrinks():
    g = 9.81
    dt = 1.0 / 2000.0
    world = make_world(gravity=model.Gravity(gravity=g, direction=(0, -1, 0)))
    body = world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 0, 0))
    t = 1.0
    for _ in range(int(t / dt)):
        world.step(dt)
    analytic = -0.5 * g * t * t
    assert world.position[body][1] == pytest.approx(analytic, rel=2e-3)


def test_gravity_factor_scales_fall():
    g = 10.0
    dt = 1.0 / 240.0
    world = make_world(gravity=model.Gravity(gravity=g, direction=(0, -1, 0)))
    full = world.add_body(model.Motion(type=model.DYNAMIC, gravityFactor=1.0),
                          position=(0, 0, 0))
    half = world.add_body(model.Motion(type=model.DYNAMIC, gravityFactor=0.5),
                          position=(0, 0, 0))
    for _ in range(240):
        world.step(dt)
    assert world.position[half][1] == pytest.approx(0.5 * world.position[full][1], rel=1e-9)


def test_zero_net_force_keeps_constant_velocity():
    dt = 1.0 / 120.0
    world = make_world(gravity=model.Gravity(gravity=0.0))
    body = world.add_body(model.Motion(type=model.DYNAMIC,
                                       linearVelocity=(2.0, 0.0, -3.0)),
                          position=(0, 0, 0))
    for _ in range(120):
        world.step(dt)
    assert np.allclose(world.linear_velocity[body], (2.0, 0.0, -3.0))
    assert np.allclose(world.position[body], (2.0, 0.0, -3.0), atol=1e-9)


def test_static_body_never_moves():
    world = make_world(gravity=model.Gravity(gravity=9.81))
    body = world.add_body(model.Motion(type=model.STATIC), position=(1, 2, 3))
    for _ in range(100):
        world.step(1 / 60)
    assert np.allclose(world.position[body], (1, 2, 3))


def test_linear_damping_terminal_velocity():
    """Under gravity + linear damping c, |v| -> g/c."""
    g, c = 9.81, 2.0
    dt = 1.0 / 240.0
    world = make_world(gravity=model.Gravity(gravity=g, direction=(0, -1, 0)))
    body = world.add_body(model.Motion(type=model.DYNAMIC, linearDamping=c),
                          position=(0, 0, 0))
    for _ in range(4000):
        world.step(dt)
    assert abs(world.linear_velocity[body][1]) == pytest.approx(g / c, rel=1e-2)


def test_quadratic_drag_terminal_velocity():
    """Under gravity + quadratic drag k, |v| -> sqrt(g/k)."""
    g, k = 9.81, 0.5
    dt = 1.0 / 240.0
    world = make_world(gravity=model.Gravity(gravity=g, direction=(0, -1, 0)))
    body = world.add_body(model.Motion(type=model.DYNAMIC, quadraticDrag=k),
                          position=(0, 0, 0))
    for _ in range(6000):
        world.step(dt)
    assert abs(world.linear_velocity[body][1]) == pytest.approx(np.sqrt(g / k), rel=1e-2)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
