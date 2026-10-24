"""Pytest configuration and shared fixtures for zero_pax tests."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")),
)
from ml_switcheroo_compiler import core as _compiler_core


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Enable `ml_switcheroo_compiler` EagerMode for every test.

    Yields:
        None while `EagerMode` is active.
    """
    with _compiler_core.EagerMode():
        yield
