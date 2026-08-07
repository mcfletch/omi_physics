"""Pytest configuration for omi_physics tests."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip wall-clock performance tests in CI environments.

    CI runners are slower and have variable load, causing absolute-time
    assertions to fail unpredictably. The serial marker identifies tests
    that need a quiet machine with predictable timing.
    """
    if not os.environ.get('CI'):
        return

    skip_ci = pytest.mark.skip(reason='wall-clock budget test skipped in CI')
    for item in items:
        if 'serial' in item.keywords:
            item.add_marker(skip_ci)
