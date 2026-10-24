"""Tests for activation layers in zero_pax.praxis.layers.activations."""

from __future__ import annotations

import jax
import numpy as np

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
    """Verify `BaseActivation` and `Identity` return inputs unchanged."""
    inputs = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(BaseActivation()(inputs), inputs, atol=1e-5)
    np.testing.assert_allclose(Identity()(inputs), inputs, atol=1e-5)


def test_relu() -> None:
    """Verify `ReLU` matches `jax.nn.relu`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.relu(inputs)
    zero_out = ReLU()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_gelu() -> None:
    """Verify `GELU` matches `jax.nn.gelu` for both approximate and exact modes."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_approx = jax.nn.gelu(inputs, approximate=True)
    zero_approx = GELU(approximate=True)(inputs)
    np.testing.assert_allclose(zero_approx, np.array(jax_approx), atol=1e-5)

    jax_exact = jax.nn.gelu(inputs, approximate=False)
    zero_exact = GELU(approximate=False)(inputs)
    np.testing.assert_allclose(zero_exact, np.array(jax_exact), atol=1e-5)


def test_swish_silu() -> None:
    """Verify `Swish` and `SiLU` match `jax.nn.swish` and `jax.nn.silu`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.swish(inputs)
    zero_out_swish = Swish()(inputs)
    zero_out_silu = SiLU()(inputs)
    np.testing.assert_allclose(zero_out_swish, np.array(jax_out), atol=1e-5)
    np.testing.assert_allclose(zero_out_silu, np.array(jax_out), atol=1e-5)


def test_leaky_relu() -> None:
    """Verify `LeakyReLU` matches `jax.nn.leaky_relu`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.leaky_relu(inputs, negative_slope=0.2)
    zero_out = LeakyReLU(negative_slope=0.2)(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_sigmoid() -> None:
    """Verify `Sigmoid` matches `jax.nn.sigmoid`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.sigmoid(inputs)
    zero_out = Sigmoid()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_tanh() -> None:
    """Verify `Tanh` matches `jax.numpy.tanh`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.numpy.tanh(inputs)
    zero_out = Tanh()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_elu() -> None:
    """Verify `ELU` matches `jax.nn.elu`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.elu(inputs, alpha=1.5)
    zero_out = ELU(alpha=1.5)(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_cubed_squared_relu() -> None:
    """Verify `CubedReLU` and `SquaredReLU` match power-of-ReLU formulas."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_relu = jax.nn.relu(inputs)
    zero_sq = SquaredReLU()(inputs)
    zero_cu = CubedReLU()(inputs)
    np.testing.assert_allclose(zero_sq, np.array(jax_relu**2), atol=1e-5)
    np.testing.assert_allclose(zero_cu, np.array(jax_relu**3), atol=1e-5)


def test_relu6() -> None:
    """Verify `ReLU6` matches `jax.nn.relu6`."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.relu6(inputs)
    zero_out = ReLU6()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)
