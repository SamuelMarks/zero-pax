"""Tests for activation layers in zero_pax.praxis.layers.activations."""

from __future__ import annotations

import math

from tests import conftest as np
from zero_pax import _tensor_ops as _ops
from zero_pax.praxis.layers import (
    ELU,
    GELU,
    BaseActivation,
    CubedReLU,
    Identity,
    LeakyReLU,
    ReLU,
    ReLU6,
    Sigmoid,
    SiLU,
    SquaredReLU,
    Swish,
    Tanh,
)


def test_base_and_identity_activation() -> None:
    """Verify `BaseActivation` and `Identity` return inputs unchanged across 2D and 3D tensors."""
    inputs = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(BaseActivation()(inputs), inputs, atol=1e-5)
    np.testing.assert_allclose(Identity()(inputs), inputs, atol=1e-5)
    grid_3d = np.arange(-12, 12, dtype=np.float32).reshape((2, 3, 4))
    np.testing.assert_allclose(ReLU()(grid_3d), np.maximum(0.0, grid_3d), atol=1e-5)
    np.testing.assert_allclose(Sigmoid()(np.zeros((2, 2), dtype=np.float32)), np.full((2, 2), 0.5))


def test_relu() -> None:
    """Verify `ReLU` matches analytical `max(0, x)`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    ref_out = np.maximum(0.0, inputs)
    zero_out = ReLU()(inputs)
    np.testing.assert_allclose(zero_out, ref_out, atol=1e-5)


def test_gelu() -> None:
    """Verify `GELU` matches analytical approximate and exact formulas."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    c = math.sqrt(2.0 / math.pi)
    ref_approx = 0.5 * inputs * (1.0 + np.tanh(c * (inputs + 0.044715 * (inputs**3))))
    zero_approx = GELU(approximate=True)(inputs)
    np.testing.assert_allclose(zero_approx, ref_approx, atol=1e-5)

    ref_exact = 0.5 * inputs * (1.0 + _ops.erf(inputs / math.sqrt(2.0)))
    zero_exact = GELU(approximate=False)(inputs)
    np.testing.assert_allclose(zero_exact, ref_exact, atol=1e-5)


def test_swish_silu() -> None:
    """Verify `Swish` and `SiLU` match analytical `x / (1 + exp(-x))`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    ref_out = inputs / (1.0 + np.exp(-inputs))
    zero_out_swish = Swish()(inputs)
    zero_out_silu = SiLU()(inputs)
    np.testing.assert_allclose(zero_out_swish, ref_out, atol=1e-5)
    np.testing.assert_allclose(zero_out_silu, ref_out, atol=1e-5)


def test_leaky_relu() -> None:
    """Verify `LeakyReLU` matches analytical piecewise linear formula."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    ref_out = _ops.where(inputs > 0.0, inputs, 0.2 * inputs)
    zero_out = LeakyReLU(negative_slope=0.2)(inputs)
    np.testing.assert_allclose(zero_out, ref_out, atol=1e-5)


def test_sigmoid() -> None:
    """Verify `Sigmoid` matches analytical `1 / (1 + exp(-x))`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    ref_out = 1.0 / (1.0 + np.exp(-inputs))
    zero_out = Sigmoid()(inputs)
    np.testing.assert_allclose(zero_out, ref_out, atol=1e-5)


def test_tanh() -> None:
    """Verify `Tanh` matches analytical `tanh(x)`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    ref_out = np.tanh(inputs)
    zero_out = Tanh()(inputs)
    np.testing.assert_allclose(zero_out, ref_out, atol=1e-5)


def test_elu() -> None:
    """Verify `ELU` matches analytical `where(x > 0, x, alpha * (exp(x) - 1))`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    ref_out = _ops.where(inputs > 0.0, inputs, 1.5 * (np.exp(inputs) - 1.0))
    zero_out = ELU(alpha=1.5)(inputs)
    np.testing.assert_allclose(zero_out, ref_out, atol=1e-5)


def test_cubed_squared_relu() -> None:
    """Verify `CubedReLU` and `SquaredReLU` match power-of-ReLU formulas."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    ref_relu = np.maximum(0.0, inputs)
    zero_sq = SquaredReLU()(inputs)
    zero_cu = CubedReLU()(inputs)
    np.testing.assert_allclose(zero_sq, ref_relu**2, atol=1e-5)
    np.testing.assert_allclose(zero_cu, ref_relu**3, atol=1e-5)


def test_relu6() -> None:
    """Verify `ReLU6` matches analytical `min(max(0, x), 6)`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    ref_out = np.minimum(np.maximum(0.0, inputs), 6.0)
    zero_out = ReLU6()(inputs)
    np.testing.assert_allclose(zero_out, ref_out, atol=1e-5)
