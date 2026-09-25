"""``mathutil.cross``: numpy's cross product, for the shapes the engine uses.

A contact, a raycast and a wheel each take the cross product of two
3-vectors many times a step; numpy's general implementation spends most of
its time working out the axes. ``cross`` answers what ``numpy.cross`` does
for 3-vectors and stacks of them.
"""
import numpy as np
import pytest

from omi_physics.mathutil import cross

RNG = np.random.default_rng(5)


@pytest.mark.parametrize('a,b', [
    (RNG.normal(size=3), RNG.normal(size=3)),
    (RNG.normal(size=(7, 3)), RNG.normal(size=(7, 3))),
    (RNG.normal(size=3), RNG.normal(size=(7, 3))),
    (RNG.normal(size=(7, 3)), RNG.normal(size=3)),
    (RNG.normal(size=(2, 5, 3)), RNG.normal(size=(5, 3))),
    ([1, 0, 0], [0, 1, 0]),
    (np.array([1, 2, 3]), np.array([4, 5, 6])),
])
def test_it_is_numpys_cross_product(a, b):
    found = cross(a, b)
    expected = np.cross(a, b)
    assert found.shape == expected.shape
    np.testing.assert_allclose(found, expected)


def test_a_single_product_is_a_fresh_float_array():
    a = np.array([1.0, 0.0, 0.0])
    found = cross(a, [0.0, 1.0, 0.0])
    assert found.dtype == np.float64 and found.shape == (3,)
    found[0] = 5.0
    assert a[0] == 1.0


def test_float32_stays_float32():
    a = np.ones((4, 3), 'f4')
    assert cross(a, a).dtype == np.float32
