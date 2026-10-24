"""Tests for convolution layers in zero_pax.praxis.layers.convolutions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from zero_pax.praxis.layers import (
    GELU,
    BatchNorm,
    CausalDepthwiseConv1D,
    Conv2D,
    ConvBNAct,
    ConvBNActWithPadding,
    DepthwiseConv1D,
    Identity,
    LightConv1D,
)


def test_conv2d_numerical() -> None:
    """Verify `Conv2D` matches `jax.lax.conv_general_dilated` for SAME, VALID, causal, and padded inputs."""
    inputs = np.random.normal(size=(2, 6, 6, 3)).astype(np.float32)
    w = np.random.normal(size=(3, 3, 3, 4)).astype(np.float32)
    b = np.random.normal(size=(4,)).astype(np.float32)

    conv_same = Conv2D(filter_shape=(3, 3, 3, 4), filter_stride=(1, 1), padding="SAME", bias=True)
    zero_same = conv_same(inputs, w=w, bias=b)
    dn = jax.lax.conv_dimension_numbers(inputs.shape, w.shape, ("NHWC", "HWIO", "NHWC"))
    jax_same = (
        jax.lax.conv_general_dilated(
            jnp.asarray(inputs), jnp.asarray(w), (1, 1), "SAME", dimension_numbers=dn
        )
        + b
    )
    np.testing.assert_allclose(zero_same, jax_same, atol=1e-4)

    conv_valid = Conv2D(
        filter_shape=(3, 3, 3, 4), filter_stride=(2, 2), padding="VALID", bias=False
    )
    zero_valid = conv_valid(inputs, w=w)
    jax_valid = jax.lax.conv_general_dilated(
        jnp.asarray(inputs), jnp.asarray(w), (2, 2), "VALID", dimension_numbers=dn
    )
    np.testing.assert_allclose(zero_valid, jax_valid, atol=1e-4)

    conv_causal = Conv2D(
        filter_shape=(3, 3, 3, 4),
        is_causal=True,
        mask_t=np.ones((3, 3, 3, 4), dtype=np.float32),
        weight_norm_tpl=Identity(),
    )
    out_causal = conv_causal(inputs, w=w)
    inputs_mod = inputs.copy()
    inputs_mod[:, 3:, :, :] += 10.0
    out_causal_mod = conv_causal(inputs_mod, w=w)
    np.testing.assert_allclose(out_causal[:, :3, :, :], out_causal_mod[:, :3, :, :], atol=1e-5)

    paddings = np.array(
        [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    out_padded = conv_same(inputs, paddings=paddings, w=w, bias=b)
    inputs_zeroed = inputs.copy()
    inputs_zeroed[0, 4:, :, :] = 0.0
    np.testing.assert_allclose(out_padded, conv_same(inputs_zeroed, w=w, bias=b), atol=1e-4)

    conv_grouped = Conv2D(filter_shape=(3, 3, 1, 3))
    w_grp = np.ones((3, 3, 1, 3), dtype=np.float32)
    out_grp = conv_grouped(np.ones((1, 4, 4, 3), dtype=np.float32), w=w_grp)
    assert float(out_grp[0, 1, 1, 0]) == 9.0

    np.testing.assert_allclose(Conv2D()(inputs), np.zeros_like(inputs), atol=1e-6)


def test_conv_bn_act_variants() -> None:
    """Verify `ConvBNAct` and `ConvBNActWithPadding` including strided padding downsampling."""
    inputs = np.ones((2, 4, 4, 3), dtype=np.float32)
    paddings = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    w = np.ones((3, 3, 3, 4), dtype=np.float32)

    layer = ConvBNAct(
        filter_shape=(3, 3, 3, 4),
        batch_norm_tpl=BatchNorm(dim=4),
        activation_tpl=GELU(),
    )
    out = layer(inputs, paddings=paddings, w=w)
    assert out.shape == (2, 4, 4, 4)
    assert np.all(np.asarray(out[0, 2:]) == 0.0)

    layer_pad = ConvBNActWithPadding(filter_shape=(3, 3, 3, 4), filter_stride=(2, 2))
    res = layer_pad(inputs, paddings=paddings, w=w)
    assert isinstance(res, tuple)
    assert res[0].shape == (2, 2, 2, 4)
    assert res[1] is not None and res[1].shape == (2, 2)
    np.testing.assert_allclose(res[1], np.array([[0.0, 1.0], [0.0, 0.0]]))

    layer_valid_pad = ConvBNActWithPadding(
        filter_shape=(2, 2, 3, 4), filter_stride=(1, 1), padding="VALID"
    )
    res_val = layer_valid_pad(inputs, paddings=paddings, w=w[:2, :2])
    assert isinstance(res_val, tuple) and res_val[1] is not None
    assert res_val[1].shape == (2, 3)

    out_no_pad = layer_pad(inputs, w=w)
    assert not isinstance(out_no_pad, tuple) and out_no_pad.shape == (2, 2, 2, 4)


def test_depthwise_and_light_conv1d() -> None:
    """Verify `DepthwiseConv1D`, `CausalDepthwiseConv1D`, and `LightConv1D` numerically."""
    inputs = np.ones((2, 6, 4), dtype=np.float32)
    paddings = np.array(
        [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    w = np.ones((3, 1, 4), dtype=np.float32)
    b = np.full((4,), 0.5, dtype=np.float32)

    dconv = DepthwiseConv1D(filter_shape=(3, 1, 4), bias=True)
    out = dconv(inputs, paddings=paddings, w=w, bias=b)
    assert out.shape == (2, 6, 4)
    np.testing.assert_allclose(out[1, 1, :], np.full((4,), 3.5), atol=1e-5)
    np.testing.assert_allclose(out[0, 4:, :], 0.0, atol=1e-5)

    dconv_strided = DepthwiseConv1D(filter_shape=(3, 1, 4), filter_stride=(2,))
    out_strided = dconv_strided(inputs, paddings=paddings, w=w)
    assert out_strided.shape == (2, 3, 4)
    np.testing.assert_allclose(out_strided[0, 2, :], 0.0, atol=1e-5)

    cdconv = CausalDepthwiseConv1D(kernel_size=3, hidden_dims=4, bias=True)
    x_seq = np.arange(1, 25, dtype=np.float32).reshape((1, 6, 4))
    out_causal = cdconv(x_seq, w=w, bias=b)
    x_seq_future_mod = x_seq.copy()
    x_seq_future_mod[:, 3:, :] += 50.0
    out_causal_mod = cdconv(x_seq_future_mod, w=w, bias=b)
    np.testing.assert_allclose(out_causal[:, :3, :], out_causal_mod[:, :3, :], atol=1e-5)

    lconv = LightConv1D(input_dims=4, kernel_size=3)
    w_start = np.concatenate(
        [np.eye(4, dtype=np.float32), np.zeros((4, 4), dtype=np.float32)], axis=-1
    )
    w_end = np.eye(4, dtype=np.float32)
    out_lconv = lconv(x_seq, paddings=paddings[:1], w_start=w_start, w_conv=w, w_end=w_end)
    assert out_lconv.shape == (1, 6, 4)
    np.testing.assert_allclose(out_lconv[0, 4:, :], 0.0, atol=1e-5)
    assert np.any(np.asarray(out_lconv[0, :4, :]) != 0.0)

    lconv_causal = LightConv1D(
        input_dims=4, kernel_size=3, is_causal=True, conv_activation_tpl=GELU()
    )
    out_lc = lconv_causal(x_seq, w_start=w_start, w_conv=w, w_end=w_end)
    assert out_lc.shape == (1, 6, 4)
    assert np.any(np.asarray(out_lc) != 0.0)
