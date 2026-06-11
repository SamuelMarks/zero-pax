"""Tests for the test_convolutions_port module."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from zero_pax.praxis.layers import (
    Conv2D,
    ConvBNAct,
    ConvBNActWithPadding,
    DepthwiseConv1D,
    CausalDepthwiseConv1D,
    LightConv1D,
)


def test_conv2d_equivalence():
    """Executes the test_conv2d_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    w = np.random.normal(size=(3, 3, 3, 4)).astype(np.float32)
    bias = np.random.normal(size=(4,)).astype(np.float32)

    # zero pax SAME
    layer_zero = Conv2D(filter_shape=(3, 3, 3, 4), bias=True, padding="SAME")
    out_zero = layer_zero(inputs, w=w, bias=bias)
    assert out_zero.shape == (2, 8, 8, 4)

    # VALID padding & dilation
    layer_zero_valid = Conv2D(
        filter_shape=(3, 3, 3, 4), dilations=(2, 2), padding="VALID"
    )
    out_zero_valid = layer_zero_valid(inputs, w=w)
    # eff_w = (3-1)*2+1 = 5. output = (8-5+1)/1 = 4
    assert out_zero_valid.shape == (2, 4, 4, 4)

    # is_causal = True
    layer_zero_causal = Conv2D(
        filter_shape=(3, 3, 3, 4), padding="SAME", is_causal=True
    )
    out_zero_causal = layer_zero_causal(inputs, w=w)
    assert out_zero_causal.shape == (2, 8, 8, 4)


def test_convbnact_equivalence():
    """Executes the test_convbnact_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    w = np.random.normal(size=(3, 3, 3, 4)).astype(np.float32)
    layer_zero = ConvBNAct(filter_shape=(3, 3, 3, 4), padding="SAME")
    out_zero = layer_zero(inputs, w=w)
    assert out_zero.shape == (2, 8, 8, 4)
    assert np.all(out_zero >= 0)  # relu check


def test_convbnactwithpadding_equivalence():
    """Executes the test_convbnactwithpadding_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    w = np.random.normal(size=(3, 3, 3, 4)).astype(np.float32)
    paddings = np.zeros((2, 8)).astype(np.float32)
    paddings[:, 6:] = 1.0

    layer_zero = ConvBNActWithPadding(filter_shape=(3, 3, 3, 4), filter_stride=(2, 2))
    out_zero, out_pad = layer_zero(inputs, paddings=paddings, w=w)
    assert out_zero.shape == (2, 4, 4, 4)
    assert out_pad.shape == (2, 4)

    # testing without padding provided
    out_zero_nopad, out_pad_nopad = layer_zero(inputs, w=w)
    assert out_pad_nopad is None


def test_depthwiseconv1d_equivalence():
    """Executes the test_depthwiseconv1d_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 4)).astype(np.float32)
    w = np.random.normal(size=(3, 4, 8)).astype(np.float32)

    layer_zero = DepthwiseConv1D(filter_shape=(3, 4, 8))
    out_zero = layer_zero(inputs, w=w)
    assert out_zero.shape == (2, 10, 8)

    # check implicit w init
    out_zero_init = DepthwiseConv1D(filter_shape=(3, 4, 8))(inputs)
    np.testing.assert_allclose(out_zero_init, np.zeros((2, 10, 8)))


def test_causaldepthwiseconv1d_equivalence():
    """Executes the test_causaldepthwiseconv1d_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 4)).astype(np.float32)
    w = np.random.normal(size=(3, 4, 8)).astype(np.float32)

    layer_zero = CausalDepthwiseConv1D(filter_shape=(3, 4, 8))
    out_zero = layer_zero(inputs, w=w)
    assert out_zero.shape == (2, 10, 8)

    # check implicit w init
    out_zero_init = CausalDepthwiseConv1D(filter_shape=(3, 4, 8))(inputs)
    np.testing.assert_allclose(out_zero_init, np.zeros((2, 10, 8)))


def test_lightconv1d_equivalence():
    """Executes the test_lightconv1d_equivalence test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 4)).astype(np.float32)
    layer_zero = LightConv1D(kernel_size=3)
    out_zero = layer_zero(inputs)
    assert out_zero.shape == (2, 10, 4)


def test_conv2d_no_bias():
    """Executes the test_conv2d_no_bias test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    layer_zero = Conv2D(filter_shape=(3, 3, 3, 4), bias=False)
    out = layer_zero(inputs)
    assert out.shape == (2, 8, 8, 4)


def test_convbnact_with_bn():
    """Executes the test_convbnact_with_bn test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    layer_zero = ConvBNAct(filter_shape=(3, 3, 3, 4))
    bn_gamma = np.random.normal(size=(4,)).astype(np.float32)
    bn_beta = np.random.normal(size=(4,)).astype(np.float32)
    out = layer_zero(inputs, bn_gamma=bn_gamma, bn_beta=bn_beta)
    assert out.shape == (2, 8, 8, 4)


def test_convbnactwithpadding_with_bn():
    """Executes the test_convbnactwithpadding_with_bn test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    paddings = np.zeros((2, 8)).astype(np.float32)
    layer_zero = ConvBNActWithPadding(filter_shape=(3, 3, 3, 4))
    bn_gamma = np.random.normal(size=(4,)).astype(np.float32)
    bn_beta = np.random.normal(size=(4,)).astype(np.float32)
    out, pad = layer_zero(inputs, paddings=paddings, bn_gamma=bn_gamma, bn_beta=bn_beta)
    assert out.shape == (2, 8, 8, 4)


def test_depthwiseconv1d_coverage():
    """Executes the test_depthwiseconv1d_coverage test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 4)).astype(np.float32)
    w = np.random.normal(size=(3, 4, 8)).astype(np.float32)
    layer_zero = DepthwiseConv1D(filter_shape=(3, 4, 8), filter_stride=(0,))
    out = layer_zero(inputs, w=w)
    assert out.shape == (2, 10, 8)


def test_causaldepthwiseconv1d_coverage():
    """Executes the test_causaldepthwiseconv1d_coverage test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 4)).astype(np.float32)
    w = np.random.normal(size=(3, 4, 8)).astype(np.float32)
    layer_zero = CausalDepthwiseConv1D(filter_shape=(3, 4, 8), filter_stride=(0,))
    out = layer_zero(inputs, w=w)
    assert out.shape == (2, 10, 8)


def test_lightconv1d_w_coverage():
    """Executes the test_lightconv1d_w_coverage test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 4)).astype(np.float32)
    w = np.random.normal(size=(3, 4, 4)).astype(np.float32)
    layer_zero = LightConv1D(kernel_size=3)
    out = layer_zero(inputs, w=w)
    assert out.shape == (2, 10, 4)


def test_conv2d_bias_none_coverage():
    """Executes the test_conv2d_bias_none_coverage test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    layer_zero = Conv2D(filter_shape=(3, 3, 3, 4), bias=True)
    out = layer_zero(inputs, bias=None)
    assert out.shape == (2, 8, 8, 4)
