"""Phase 7 GPGPU parity — the GL compute backend vs the numpy reference.

These run only where a GL 4.3 compute context can be created (skipped otherwise),
so they need a real window: a hidden GLFW 4.3-core context, per the project's GL
guidance (``CLAUDE.md``).  The window is destroyed on teardown but ``glfw`` is
never terminated (Wayland teardown segfault — see project memory).

float32-on-GPU vs float64-on-CPU means trajectories match only *within tolerance*
(the plan's best-effort-on-GPU caveat), so parity assertions use a loose ``atol``
and stick to non-chaotic scenes — a resting single body and free integration —
rather than stacks, where float noise amplifies and neither backend is "wrong".
"""
import numpy as np
import pytest

from omi_physics import model, backend
from omi_physics.world import PhysicsWorld


@pytest.fixture(scope='module')
def gl_context():
    glfw = pytest.importorskip('glfw')
    if not glfw.init():
        pytest.skip('glfw init failed')
    # GLFW window hints are sticky/process-global; reset them so a prior
    # core-profile test's profile can't leak into this context.
    glfw.default_window_hints()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    win = glfw.create_window(64, 64, 'physics-parity', None, None)
    if not win:
        pytest.skip('no GL 4.3 context available')
    glfw.make_context_current(win)
    if not backend._has_gl_compute():
        glfw.destroy_window(win)
        pytest.skip('GL 4.3 compute shaders unavailable')
    yield win
    glfw.destroy_window(win)          # never glfw.terminate() here


@pytest.fixture
def gpu_backend(gl_context):
    return backend.select_backend('gpu')


def _free_scene(backend_obj):
    """Free integration only (no collider) — exercises the fused GPU kernel:
    a drag+spin faller, a kinematic drifter, and an immobile static body."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         backend=backend_obj, sleep_enabled=False)
    world.add_body(model.Motion(type=model.DYNAMIC, angularVelocity=(0, 2.0, 0),
                                quadraticDrag=0.1, linearDamping=0.05),
                   position=(0, 100, 0))
    world.add_body(model.Motion(type=model.KINEMATIC, linearVelocity=(1, 0, 0)),
                   position=(0, 0, 0))
    world.add_body(model.Motion(type=model.STATIC), position=(5, 0, 0))
    return world


def _rest_scene(backend_obj):
    """A sphere settling on a static floor — exercises the non-fused path
    (forces on GPU, contact solve on CPU, positions on GPU)."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         backend=backend_obj)
    mat = world.add_material(model.Material(restitution=0.0))
    ground = world.add_shape(model.Shape.box((40, 1, 40)))
    sphere = world.add_shape(model.Shape.sphere(0.5))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=ground, physicsMaterial=mat),
                   position=(0, -0.5, 0))
    world.add_body(model.Motion(type=model.DYNAMIC),
                   collider=model.Collider(shape=sphere, physicsMaterial=mat),
                   position=(0, 5, 0))
    return world


def _run(world, steps=120, dt=1 / 120):
    for _ in range(steps):
        world.step(dt)


# -- selection ----------------------------------------------------------
def test_auto_selects_gpu_when_compute_present(gl_context):
    assert backend.select_backend('auto').name == 'glcompute'


def test_world_uses_gpu_backend_from_env(gl_context, monkeypatch):
    monkeypatch.setenv('OPENGLCONTEXT_PHYSICS_BACKEND', 'gpu')
    world = PhysicsWorld()
    assert world.backend.name == 'glcompute'


def test_auto_stays_numpy_below_threshold(gl_context):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), gpu_threshold=100)
    for _ in range(10):
        world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 10, 0))
    world.step(1 / 60)
    assert world.backend.name == 'numpy'


def test_auto_switches_to_gpu_above_threshold(gl_context):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), gpu_threshold=5)
    for _ in range(10):
        world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 10, 0))
    world.step(1 / 60)
    assert world.backend.name == 'glcompute'


def test_auto_falls_back_to_numpy_when_bodies_sleep(gl_context):
    """Hysteresis: once the awake count drops well below the threshold the world
    returns to numpy (a mostly-asleep scene shouldn't pay GPU transfer)."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), gpu_threshold=5)
    for _ in range(10):
        world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 10, 0))
    world.step(1 / 60)
    assert world.backend.name == 'glcompute'
    for i in range(8):                       # sleep most bodies -> awake < 0.8*thr
        world._awake[i] = False
    world.step(1 / 60)
    assert world.backend.name == 'numpy'


# -- kernels ------------------------------------------------------------
def test_gpu_free_fall_matches_analytic(gpu_backend):
    """Semi-implicit Euler under gravity: yₙ = y₀ − g·dt²·Σk.  The GPU kernel
    must reproduce the closed form to float32 precision."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         backend=gpu_backend, sleep_enabled=False)
    world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 0, 0))
    dt, n = 1 / 240, 100
    _run(world, steps=n, dt=dt)
    expected = -9.81 * dt * dt * (n * (n + 1) / 2)
    assert world.position[0, 1] == pytest.approx(expected, abs=1e-2)


def test_gpu_constant_velocity_without_gravity(gpu_backend):
    world = PhysicsWorld(gravity=model.Gravity(gravity=0.0), backend=gpu_backend,
                         sleep_enabled=False)
    world.add_body(model.Motion(type=model.DYNAMIC, linearVelocity=(2, -3, 1)),
                   position=(0, 0, 0))
    _run(world, steps=60, dt=1 / 60)
    assert np.allclose(world.position[0], np.array([2, -3, 1]) * (60 / 60),
                       atol=1e-3)


def test_gpu_fused_path_matches_numpy(gpu_backend):
    cpu, gpu = _free_scene(backend.NumpyBackend()), _free_scene(gpu_backend)
    _run(cpu), _run(gpu)
    assert np.allclose(cpu.position, gpu.position, atol=1e-2)
    assert np.allclose(cpu.orientation, gpu.orientation, atol=1e-3)
    # the fused kernel actually ran (velocity round-trip skipped)
    assert gpu_backend._fused_done is False   # reset after integrate_positions


def test_gpu_collision_path_matches_numpy(gpu_backend):
    cpu, gpu = _rest_scene(backend.NumpyBackend()), _rest_scene(gpu_backend)
    _run(cpu, steps=240), _run(gpu, steps=240)
    assert np.allclose(cpu.position[1], gpu.position[1], atol=1e-2)
    # both settle just above the floor
    assert gpu.position[1, 1] == pytest.approx(0.5, abs=0.05)


def test_gpu_backend_survives_growth(gpu_backend):
    """Bodies added after the first step force buffer re-allocation."""
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), backend=gpu_backend,
                         sleep_enabled=False)
    world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 10, 0))
    world.step(1 / 60)
    for i in range(50):
        world.add_body(model.Motion(type=model.DYNAMIC), position=(i, 10, 0))
    world.step(1 / 60)
    assert world.body_count == 51
    assert np.all(world.position[:, 1] < 10)     # everything fell


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
