"""A column of spheres on a grid ground found by the compiled accelerator: the same contacts as
the Python finds, to the bit, and no Python sampling of the ground."""
import importlib.util

import numpy as np
import pytest

from omi_physics import collide, heightfield
from omi_physics._accel import accelerators_disabled
from omi_physics.body import HeightFieldProxy
from omi_physics.heightfield import GridSurface

pytestmark = pytest.mark.skipif(accelerators_disabled() or importlib.util.find_spec(
    'omi_physics._collide_native') is None, reason='the compiled accelerators are not in use')


def rough_grid(rng: np.random.Generator) -> GridSurface:
    """A grid with gentle stretches, steep ones past 0.55 and the odd cliff."""
    rows, columns = rng.integers(3, 40, size=2)
    spacing = (float(rng.uniform(0.2, 1.5)), float(rng.uniform(0.2, 1.5)))
    z, x = np.mgrid[0:rows, 0:columns]
    heights = (rng.uniform(0.0, 1.2) * np.sin(x * rng.uniform(0.1, 1.5))
               * np.cos(z * rng.uniform(0.1, 1.5)) + rng.uniform(-0.3, 0.3) * x)
    if rng.random() < 0.3:
        heights[rng.integers(rows), :] += rng.uniform(-2.0, 2.0)
    origin = tuple(float(v) for v in rng.uniform(-5.0, 5.0, size=3))
    return GridSurface(heights, origin=origin, spacing=spacing)


def python_column(surface: GridSurface, x: float, z: float, heights: list[float],
                  radius: float, monkeypatch: pytest.MonkeyPatch) -> list:
    with monkeypatch.context() as patch:
        patch.setattr(heightfield, '_native', None)
        return surface.column_contacts(x, z, heights, radius)


def column_on(surface: GridSurface, rng: np.random.Generator
              ) -> tuple[float, float, list[float], float]:
    """A column somewhere on or just off the grid, its lowest sphere at or near the ground."""
    low, high = surface.low, surface.high
    x = float(rng.uniform(low[0] - 1.0, high[0] + 1.0))
    z = float(rng.uniform(low[2] - 1.0, high[2] + 1.0))
    radius = float(rng.uniform(0.05, 0.6))
    count = int(rng.integers(1, 17))
    ground = surface.height_at(x, z)
    base = (ground if np.isfinite(ground) else 0.0) + float(rng.uniform(-0.6, 1.2))
    spacing = float(rng.uniform(0.0, 2.0 * radius))
    return x, z, [base + spacing * k for k in range(count)], radius


@pytest.mark.parametrize('seed', range(40))
def test_the_compiled_column_is_the_python_column_to_the_bit(seed, monkeypatch):
    rng = np.random.default_rng(9100 + seed)
    surface = rough_grid(rng)
    touched = 0
    for _ in range(60):
        x, z, heights, radius = column_on(surface, rng)
        compiled = surface.column_contacts(x, z, heights, radius)
        assert compiled == python_column(surface, x, z, heights, radius, monkeypatch)
        touched += len(compiled)
    assert touched, 'every column missed the ground, so nothing was compared'


@pytest.mark.parametrize('seed', range(10))
def test_a_contact_the_python_finds_is_never_missed(seed, monkeypatch):
    rng = np.random.default_rng(700 + seed)
    surface = rough_grid(rng)
    for _ in range(100):
        x, z, heights, radius = column_on(surface, rng)
        heights = [h - radius for h in heights]          # pressed into the ground
        expected = python_column(surface, x, z, heights, radius, monkeypatch)
        assert len(surface.column_contacts(x, z, heights, radius)) == len(expected)


def test_the_compiled_column_samples_nothing_in_python(monkeypatch):
    rng = np.random.default_rng(5)
    surface = rough_grid(rng)
    columns = [column_on(surface, rng) for _ in range(50)]
    calls = []
    for name in ('_sample_one', 'height_at', 'sphere_contact', '_slide_together', 'sample'):
        original = getattr(surface, name)
        monkeypatch.setattr(surface, name, lambda *a, _o=original, _n=name, **k:
                            calls.append(_n) or _o(*a, **k))
    for column in columns:
        surface.column_contacts(*column)
    assert calls == []


def test_a_column_of_more_than_sixteen_spheres_is_found_with_arrays(monkeypatch):
    surface = rough_grid(np.random.default_rng(11))
    x, z = (float(v) for v in (surface.low + surface.high)[[0, 2]] / 2.0)
    ground = surface.height_at(x, z)
    heights = [ground + 0.05 * k for k in range(20)]
    assert (surface.column_contacts(x, z, heights, 0.3)
            == python_column(surface, x, z, heights, 0.3, monkeypatch))


def field_of(surface: GridSurface, rng: np.random.Generator) -> HeightFieldProxy:
    return HeightFieldProxy(surface, tuple(float(v) for v in rng.uniform(-2.0, 2.0, 3)), np.eye(3))


@pytest.mark.parametrize('seed', range(30))
def test_a_capsule_pushed_out_of_a_grid_by_the_compiled_column_is_pushed_as_by_python(
        seed, monkeypatch):
    rng = np.random.default_rng(4400 + seed)
    surface = rough_grid(rng)
    field = field_of(surface, rng)
    found = 0
    for _ in range(60):
        x, z, heights, radius = column_on(surface, rng)
        half = float(rng.uniform(0.05, 1.0))
        x, y, z = x + field.center[0], heights[0] + half + field.center[1], z + field.center[2]
        compiled = collide.upright_capsule_heightfield_pushes(field, x, y, z, half, radius)
        with monkeypatch.context() as patch:
            patch.setattr(heightfield, '_native', None)
            assert compiled == collide.upright_capsule_heightfield_pushes(field, x, y, z, half,
                                                                          radius)
        found += len(compiled)
    assert found


def test_a_capsule_on_a_grid_asks_python_only_for_a_column_of_many_contacts(monkeypatch):
    """The few contacts of a standing or sunk capsule come from the compiled column alone; one
    buried with more contacts than a manifold keeps is reduced by the Python."""
    rng = np.random.default_rng(8)
    surface = rough_grid(rng)
    field = field_of(surface, rng)
    columns = [column_on(surface, rng) for _ in range(200)]
    found = []
    python_column = surface.column_contacts
    monkeypatch.setattr(surface, 'column_contacts',
                        lambda *a: found.append(len(contacts := python_column(*a))) or contacts)
    for x, z, heights, radius in columns:
        collide.upright_capsule_heightfield_pushes(
            field, x + field.center[0], heights[0] + 0.2 + field.center[1], z + field.center[2],
            0.2, radius)
    assert all(count > collide._MANIFOLD_POINTS for count in found)  # noqa: SLF001 the manifold's size
