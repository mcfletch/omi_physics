"""Spheres against a grid ground: the contacts are the same however they are found, and a sphere
far above the ground does not cost a slide down the tangent planes."""
import math

import numpy as np
import pytest

from omi_physics import heightfield
from omi_physics.heightfield import GridSurface

RNG = np.random.default_rng(20261026)


def bumpy_grid(rows: int = 40, columns: int = 40, spacing: float = 0.5) -> GridSurface:
    z, x = np.mgrid[0:rows, 0:columns] * spacing
    heights = 0.4 * np.sin(x * 1.3) * np.cos(z * 0.9) + 0.05 * x
    return GridSurface(heights, origin=(0.0, -1.0, 0.0), spacing=spacing)


def axis_spheres(foot: np.ndarray, count: int = 9, length: float = 1.4, radius: float = 0.15
                 ) -> np.ndarray:
    """Centres of the spheres a standing capsule is taken as, its base at ``foot``."""
    return foot[None, :] + np.array([0.0, radius, 0.0]) + np.linspace(0.0, length, count)[:, None] * [0, 1, 0]


class Spy(GridSurface):
    """A grid that counts what it is asked."""

    def __init__(self, *arguments, **named) -> None:
        super().__init__(*arguments, **named)
        self.samples = 0
        self.heights_asked = 0

    def sample(self, x, z):
        self.samples += 1
        return super().sample(x, z)

    def heights(self, x, z):
        self.heights_asked += 1
        return super().heights(x, z)


def as_a_spy(surface: GridSurface) -> Spy:
    return Spy(surface.grid, origin=surface.origin, spacing=tuple(surface.spacing))


def test_a_grid_knows_the_steepest_slope_it_has():
    plane = GridSurface(np.add.outer(np.arange(5) * 0.0, np.arange(5) * 0.5), spacing=1.0)
    assert plane.lipschitz == pytest.approx(0.5)
    both = GridSurface(np.add.outer(np.arange(5) * 0.3, np.arange(5) * 0.4), spacing=1.0)
    assert both.lipschitz == pytest.approx(0.5)
    assert bumpy_grid().lipschitz > 0.5


@pytest.mark.parametrize('seed', range(8))
def test_the_contacts_are_those_of_the_slide_down_the_tangent_planes_to_a_nanometre(seed):
    rng = np.random.default_rng(seed)
    surface = bumpy_grid()
    for _ in range(25):
        foot = np.array([rng.uniform(1.0, 18.0), 0.0, rng.uniform(1.0, 18.0)])
        foot[1] = float(surface.heights(foot[[0]], foot[[2]])[0]) + rng.uniform(-0.1, 0.4)
        centres = axis_spheres(foot)
        fast = heightfield.sphere_contacts(surface, centres, 0.15)
        slow = heightfield.sphere_contacts_by_sliding(surface, centres, 0.15)
        assert np.array_equal(fast[0], slow[0])
        touching = slow[0]
        for got, wanted in zip(fast[1:], slow[1:], strict=True):
            assert got[touching] == pytest.approx(wanted[touching], abs=1e-9)


def test_spheres_off_the_grid_or_beside_it_touch_nothing():
    surface = bumpy_grid()
    centres = np.array([[-5.0, 3.0, 4.0], [9.0, 0.05, 9.0], [30.0, 0.0, 5.0]])
    fast = heightfield.sphere_contacts(surface, centres, 0.2)
    slow = heightfield.sphere_contacts_by_sliding(surface, centres, 0.2)
    assert np.array_equal(fast[0], slow[0])
    assert not fast[0][0] and not fast[0][2]


def test_a_standing_capsule_costs_a_height_for_each_sphere_and_no_slide():
    surface = as_a_spy(bumpy_grid())
    foot = np.array([8.0, float(surface.heights(np.array([8.0]), np.array([8.0]))[0]) + 0.001, 8.0])
    surface.heights_asked = 0
    touching, *_ = heightfield.sphere_contacts(surface, axis_spheres(foot), 0.15)
    assert touching.any() and touching.sum() <= 2
    assert surface.samples == 0
    assert surface.heights_asked == 0          # each sphere's height is read with numbers


def test_spheres_all_through_the_ground_are_all_found_touching():
    surface = bumpy_grid()
    foot = np.array([8.0, float(surface.heights(np.array([8.0]), np.array([8.0]))[0]) - 3.0, 8.0])
    centres = axis_spheres(foot, length=0.6, count=3)
    fast = heightfield.sphere_contacts(surface, centres, 0.15)
    slow = heightfield.sphere_contacts_by_sliding(surface, centres, 0.15)
    assert np.array_equal(fast[0], slow[0])
    assert math.isclose(float(fast[3][0]), float(slow[3][0]), abs_tol=1e-12)


# -- a few spheres, with numbers --------------------------------------------------

def random_centres(rng, surface: GridSurface, count: int) -> np.ndarray:
    """Centres over the grid, near its surface, under it, above it and off its edges."""
    x = rng.uniform(-1.0, 21.0, count)
    z = rng.uniform(-1.0, 21.0, count)
    ground = surface.heights(np.clip(x, 0.0, 19.5), np.clip(z, 0.0, 19.5))
    return np.stack([x, ground + rng.uniform(-0.5, 0.8, count), z], axis=1)


@pytest.mark.parametrize('seed', range(10))
def test_a_height_with_numbers_is_the_height_with_arrays_to_the_bit(seed):
    rng = np.random.default_rng(seed)
    surface = bumpy_grid()
    x = np.concatenate([rng.uniform(-1.0, 21.0, 200), [0.0, 19.5, 19.5, 0.0]])
    z = np.concatenate([rng.uniform(-1.0, 21.0, 200), [0.0, 19.5, 0.0, 19.5]])
    wanted = surface.heights(x, z)
    got = [surface.height_at(float(a), float(b)) for a, b in zip(x, z, strict=True)]
    assert np.array_equal(np.array(got), wanted)


@pytest.mark.parametrize('count', [1, 2, 4, 5, 9, 12])
@pytest.mark.parametrize('seed', range(12))
def test_a_few_spheres_slid_together_with_numbers_are_those_slid_with_arrays_to_the_bit(seed, count):
    rng = np.random.default_rng(1000 * count + seed)
    surface = bumpy_grid()
    for _ in range(10):
        centres = random_centres(rng, surface, count)
        radius = float(rng.uniform(0.05, 0.5))
        wanted = heightfield.sphere_contacts_by_sliding(surface, centres, radius)
        got = surface.sphere_contacts_together(centres, radius)
        for one, other in zip(got, wanted, strict=True):
            assert np.array_equal(one, other)


@pytest.mark.parametrize('seed', range(8))
def test_the_spheres_of_a_capsule_are_found_as_with_arrays_to_the_bit(seed):
    rng = np.random.default_rng(seed)
    surface = bumpy_grid()
    for _ in range(25):
        foot = random_centres(rng, surface, 1)[0]
        centres = axis_spheres(foot)
        got = heightfield.sphere_contacts(surface, centres, 0.15)
        heights = surface.heights(centres[:, 0], centres[:, 2])
        reach = 2.0 * 0.15 * math.sqrt(1.0 + surface.lipschitz ** 2)
        near = np.flatnonzero(centres[:, 1] - heights < reach)
        if len(near) > heightfield._ONE_BY_ONE:
            wanted = heightfield.sphere_contacts_by_sliding(surface, centres[near], 0.15)
            for one, other in zip(got, wanted, strict=True):
                assert np.array_equal(one[near], other)


def test_a_capsule_lying_on_the_ground_is_found_with_no_array_lookup():
    surface = as_a_spy(bumpy_grid())
    centres = np.array([[4.0 + 0.1 * k, 0.0, 6.0] for k in range(9)])
    centres[:, 1] = [surface.height_at(x, 6.0) + 0.1 for x in centres[:, 0]]
    surface.heights_asked = surface.samples = 0
    touching, *_ = heightfield.sphere_contacts(surface, centres, 0.15)
    assert touching.sum() > heightfield._ONE_BY_ONE
    assert surface.samples == 0 and surface.heights_asked == 0
