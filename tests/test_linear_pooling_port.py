"""Tests for the test_linear_pooling_port module."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from zero_pax.praxis.layers import Linear, Pooling, Pooling1D, GlobalPooling


def test_linear_equivalence():
    """Executes the test_linear_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    w = np.random.normal(size=(8, 16)).astype(np.float32)

    # JAX ref (simplified einsum)
    out_jax = jnp.dot(inputs, w)

    layer_zero = Linear(input_dims=8, output_dims=16)
    out_zero = layer_zero(inputs, w=w)

    np.testing.assert_allclose(out_zero, np.array(out_jax), atol=1e-5)

    # test implicit init
    layer_zero2 = Linear(input_dims=8, output_dims=16)
    out_zero2 = layer_zero2(inputs)
    assert out_zero2.shape == (2, 4, 16)


def test_pooling_equivalence():
    """Executes the test_pooling_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)

    # zero pax SAME max
    layer_zero = Pooling(
        window_shape=(2, 2), window_stride=(2, 2), pooling_type="MAX", padding="SAME"
    )
    out_zero, _ = layer_zero(inputs)
    assert out_zero.shape == (2, 4, 4, 3)

    # zero pax VALID avg
    layer_zero_valid = Pooling(
        window_shape=(3, 3), window_stride=(2, 2), pooling_type="AVG", padding="VALID"
    )
    out_zero_valid, _ = layer_zero_valid(inputs)
    assert out_zero_valid.shape == (2, 3, 3, 3)


def test_pooling_raises():
    """Executes the test_pooling_raises test.

    Returns:
        The result of the test.
    """
    with pytest.raises(ValueError, match="sequences of length 2"):
        Pooling(window_shape=(2,))(np.zeros((1, 2, 2, 1)))

    with pytest.raises(ValueError, match="positive integers"):
        Pooling(window_shape=(0, 2))(np.zeros((1, 2, 2, 1)))

    with pytest.raises(ValueError, match="one of AVG or MAX"):
        Pooling(window_shape=(2, 2), window_stride=(2, 2), pooling_type="MIN")(
            np.zeros((1, 2, 2, 1))
        )

    with pytest.raises(ValueError, match="SAME or VALID"):
        Pooling(window_shape=(2, 2), window_stride=(2, 2), padding="INVALID")(
            np.zeros((1, 2, 2, 1))
        )


def test_pooling1d_equivalence():
    """Executes the test_pooling1d_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 8)).astype(np.float32)

    # stride 1 window 1 (identity)
    layer_zero_id = Pooling1D(stride=1, window=1)
    out_id, _ = layer_zero_id(inputs)
    np.testing.assert_allclose(out_id, inputs)

    # stride 2 window 1 (slice)
    layer_zero_slice = Pooling1D(stride=2, window=1)
    out_slice, _ = layer_zero_slice(inputs)
    np.testing.assert_allclose(out_slice, inputs[:, ::2, :])

    # max pooling
    layer_zero_max = Pooling1D(stride=2, window=2, pooling_type="MAX")
    out_max, _ = layer_zero_max(inputs)
    assert out_max.shape == (2, 5, 8)

    # avg pooling
    layer_zero_avg = Pooling1D(stride=3, window=3, pooling_type="AVG")
    out_avg, _ = layer_zero_avg(inputs)
    assert out_avg.shape == (2, 4, 8)


def test_pooling1d_raises():
    """Executes the test_pooling1d_raises test.

    Returns:
        The result of the test.
    """
    with pytest.raises(ValueError, match="positive integer"):
        Pooling1D(stride=0)(np.zeros((1, 2, 1)))
    with pytest.raises(ValueError, match="one of AVG or MAX"):
        Pooling1D(stride=1, pooling_type="MIN")(np.zeros((1, 2, 1)))


def test_global_pooling_equivalence():
    """Executes the test_global_pooling_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 8)).astype(np.float32)

    # AVG over dim 1
    layer_zero_avg = GlobalPooling(pooling_type="AVG", pooling_dims=[1])
    out_avg = layer_zero_avg(inputs)
    np.testing.assert_allclose(out_avg, np.mean(inputs, axis=(1,)))

    # MAX over dim 1
    layer_zero_max = GlobalPooling(pooling_type="MAX", pooling_dims=[1], keepdims=True)
    out_max = layer_zero_max(inputs)
    np.testing.assert_allclose(out_max, np.max(inputs, axis=(1,), keepdims=True))

    # with padding MAX
    paddings = np.zeros((2, 10, 1)).astype(np.float32)
    paddings[0, 5:, :] = 1.0  # pad second half
    out_max_pad = layer_zero_max(inputs, compatible_paddings=paddings)
    inputs_masked_max = np.where(paddings > 0, -np.inf, inputs)
    np.testing.assert_allclose(
        out_max_pad, np.max(inputs_masked_max, axis=(1,), keepdims=True)
    )

    # with padding AVG
    layer_zero_avg_pad = GlobalPooling(
        pooling_type="AVG", pooling_dims=[1], keepdims=True
    )
    out_avg_pad = layer_zero_avg_pad(inputs, compatible_paddings=paddings)
    mask = 1.0 - paddings
    expected_avg = np.sum(inputs * mask, axis=(1,), keepdims=True) / np.maximum(
        np.sum(mask, axis=(1,), keepdims=True), 1e-8
    )
    np.testing.assert_allclose(out_avg_pad, expected_avg)


def test_global_pooling_raises():
    """Executes the test_global_pooling_raises test.

    Returns:
        The result of the test.
    """
    with pytest.raises(ValueError, match="one of AVG or MAX"):
        GlobalPooling(pooling_type="MIN")(np.zeros((1, 2, 1)))
    with pytest.raises(ValueError, match="must be set as a list"):
        GlobalPooling(pooling_dims=None)(np.zeros((1, 2, 1)))
    with pytest.raises(ValueError, match="non-negative integers"):
        GlobalPooling(pooling_dims=[-1])(np.zeros((1, 2, 1)))
