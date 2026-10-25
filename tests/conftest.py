"""Pytest configuration and shared fixtures for zero_pax tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from zero_pax import _tensor_ops


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Enable eager execution mode for every test.

    Yields:
        None while eager mode is active.
    """
    _tensor_ops.set_eager_mode(True)
    yield
