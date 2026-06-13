"""Tests for the conftest module."""

import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
import ml_switcheroo_compiler as ml_switcheroo


@pytest.fixture(autouse=True)
def switcheroo_config():
    """Provides the switcheroo_config fixture.

    Returns:
        The result of the test.
    """
    with ml_switcheroo.EagerMode():
        yield
