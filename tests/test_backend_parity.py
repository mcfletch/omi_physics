"""Phase 7 backend seam: capability probe, fallback, parity harness (no GL)."""
import logging

import numpy as np
import pytest

from omi_physics import backend, glcompute, model
from omi_physics.world import PhysicsWorld


def falling_scene(backend_obj):
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         backend=backend_obj, sleep_enabled=False)
    mat = world.add_material(model.Material(restitution=0.2))
    ground = world.add_shape(model.Shape.box((40, 1, 40)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=ground, physicsMaterial=mat),
                   position=(0, -0.5, 0))
    box = world.add_shape(model.Shape.box((1, 1, 1)))
    for k in range(6):
        world.add_body(model.Motion(type=model.DYNAMIC),
                       collider=model.Collider(shape=box, physicsMaterial=mat),
                       position=(0.01 * k, 1.0 + k * 1.1, 0))
    return world


def test_auto_falls_back_to_numpy_without_gl_compute(monkeypatch):
    monkeypatch.setattr(backend, '_has_gl_compute', lambda: False)
    assert isinstance(backend.select_backend('auto'), backend.NumpyBackend)


def test_explicit_numpy_never_touches_gl():
    assert isinstance(backend.select_backend('numpy'), backend.NumpyBackend)


def test_explicit_gpu_request_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(backend, '_has_gl_compute', lambda: False)
    with pytest.raises(RuntimeError):
        backend.select_backend('gpu')


def test_auto_stays_on_numpy_when_gpu_cannot_build(monkeypatch):
    """Crossing the threshold with no GL context must not wedge: the GPU build
    fails once and the world keeps stepping on numpy."""
    monkeypatch.setattr(backend, '_has_gl_compute', lambda: False)
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), gpu_threshold=2)
    for _ in range(5):
        world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 10, 0))
    world.step(1 / 60)
    assert isinstance(world.backend, backend.NumpyBackend)
    assert world._gpu_failed is True


def test_explicit_backend_object_is_never_switched():
    obj = backend.NumpyBackend()
    world = PhysicsWorld(backend=obj, gpu_threshold=1)
    assert world._backend_mode == 'fixed'
    for _ in range(3):
        world.add_body(model.Motion(type=model.DYNAMIC))
    world.step(1 / 60)
    assert world.backend is obj


class _BackendThatCannotBuild:
    def __init__(self):
        raise RuntimeError('compute shader compilation failed: no driver')


def test_auto_logs_why_the_gpu_backend_could_not_be_built(monkeypatch, caplog):
    monkeypatch.setattr(backend, '_has_gl_compute', lambda: True)
    monkeypatch.setattr(glcompute, 'GLComputeBackend', _BackendThatCannotBuild)
    with caplog.at_level(logging.WARNING, logger='omi_physics.backend'):
        chosen = backend.select_backend('auto')
    assert isinstance(chosen, backend.NumpyBackend)
    [record] = caplog.records
    assert record.exc_info[0] is RuntimeError


def test_a_world_logs_why_its_gpu_backend_could_not_be_built(monkeypatch, caplog):
    monkeypatch.setattr(backend, '_has_gl_compute', lambda: True)
    monkeypatch.setattr(glcompute, 'GLComputeBackend', _BackendThatCannotBuild)
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81), gpu_threshold=2)
    for _ in range(5):
        world.add_body(model.Motion(type=model.DYNAMIC), position=(0, 10, 0))
    with caplog.at_level(logging.WARNING, logger='omi_physics'):
        world.step(1 / 60)
        world.step(1 / 60)
    assert isinstance(world.backend, backend.NumpyBackend)
    assert world._gpu_failed is True
    [record] = caplog.records
    assert record.exc_info[0] is RuntimeError


class TestTheComputeProbe:
    """``_has_gl_compute`` answers False for what a missing or old GL reports."""

    @pytest.fixture
    def gl(self, monkeypatch):
        GL = pytest.importorskip('OpenGL.GL')
        monkeypatch.setattr(GL, 'glDispatchCompute', object())
        return GL

    def reporting(self, monkeypatch, gl, answer):
        def glGetString(name):
            if isinstance(answer, BaseException):
                raise answer
            return answer
        monkeypatch.setattr(gl, 'glGetString', glGetString)

    def test_a_current_version_with_compute(self, monkeypatch, gl):
        self.reporting(monkeypatch, gl, b'4.6.0 NVIDIA 550.0')
        assert backend._has_gl_compute() is True

    def test_an_old_version(self, monkeypatch, gl):
        self.reporting(monkeypatch, gl, b'3.3 Mesa 24.0')
        assert backend._has_gl_compute() is False

    @pytest.mark.parametrize('text', [b'', b'garbage', b'four.six', b'4'])
    def test_a_version_string_that_does_not_parse(self, monkeypatch, gl, text):
        self.reporting(monkeypatch, gl, text)
        assert backend._has_gl_compute() is False

    def test_no_context(self, monkeypatch, gl):
        error = pytest.importorskip('OpenGL.error')
        self.reporting(monkeypatch, gl, error.NullFunctionError('glGetString'))
        assert backend._has_gl_compute() is False

    def test_a_defect_in_the_caller_is_not_mistaken_for_no_gl(self, monkeypatch, gl):
        self.reporting(monkeypatch, gl, TypeError('wrong argument count'))
        with pytest.raises(TypeError):
            backend._has_gl_compute()


def test_two_numpy_runs_match_within_tolerance():
    a = falling_scene(backend.NumpyBackend())
    b = falling_scene(backend.NumpyBackend())
    for _ in range(300):
        a.step(1 / 120)
        b.step(1 / 120)
    assert np.allclose(a.position, b.position, atol=1e-9)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
