"""Tests for the test_activations_port module."""

import numpy as np
import pytest
from zero_pax.praxis.layers import (
    ReLU,
    GELU,
    Swish,
    SiLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    ELU,
    CubedReLU,
    SquaredReLU,
    ReLU6,
)
import jax
import jax.numpy as jnp


def test_relu_equivalence():
    """Executes the test_relu_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.relu(inputs)
    zero_out = ReLU()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_leaky_relu_equivalence():
    """Executes the test_leaky_relu_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.leaky_relu(inputs, negative_slope=0.01)
    zero_out = LeakyReLU()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_gelu_equivalence():
    """Executes the test_gelu_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.gelu(inputs, approximate=True)
    zero_out = GELU()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_swish_equivalence():
    """Executes the test_swish_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.swish(inputs)
    zero_out = Swish()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_silu_equivalence():
    """Executes the test_silu_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.silu(inputs)
    zero_out = SiLU()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_sigmoid_equivalence():
    """Executes the test_sigmoid_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.sigmoid(inputs)
    zero_out = Sigmoid()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_tanh_equivalence():
    """Executes the test_tanh_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.tanh(inputs)
    zero_out = Tanh()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_elu_equivalence():
    """Executes the test_elu_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.elu(inputs)
    zero_out = ELU()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_cubed_relu_equivalence():
    """Executes the test_cubed_relu_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.relu(inputs) ** 3
    zero_out = CubedReLU()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_squared_relu_equivalence():
    """Executes the test_squared_relu_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.relu(inputs) ** 2
    zero_out = SquaredReLU()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)


def test_relu6_equivalence():
    """Executes the test_relu6_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    jax_out = jax.nn.relu6(inputs)
    zero_out = ReLU6()(inputs)
    np.testing.assert_allclose(zero_out, np.array(jax_out), atol=1e-5)
