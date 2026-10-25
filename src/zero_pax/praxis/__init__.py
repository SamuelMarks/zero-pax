"""Praxis namespace package for zero_pax."""

from __future__ import annotations

import ml_switcheroo_ir as ml_switcheroo

from zero_pax import types as pax_fiddle
from zero_pax.praxis import base, layers
from zero_pax.praxis import base as base_layer

__all__ = [
    "base",
    "base_layer",
    "layers",
    "ml_switcheroo",
    "pax_fiddle",
]
