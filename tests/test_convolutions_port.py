"""Tests for convolution layers in zero_pax.praxis.layers.convolutions."""

from __future__ import annotations

import pytest

from tests import conftest as np
from zero_pax import _tensor_ops as t_ops
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
    """Verify `Conv2D` matches analytical 2D convolution for SAME, VALID, causal, and padded inputs."""
    inputs = np.arange(1, 2 * 6 * 6 * 3 + 1, dtype=np.float32).reshape((2, 6, 6, 3)) * 0.05
    w = np.ones((3, 3, 3, 4), dtype=np.float32) * 0.1
    b = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)

    conv_same = Conv2D(filter_shape=(3, 3, 3, 4), filter_stride=(1, 1), padding="SAME", bias=True)
    zero_same = conv_same(inputs, w=w, bias=b)
    ref_same = np.ref_conv2d(inputs, w, stride=(1, 1), padding="SAME") + b
    np.testing.assert_allclose(zero_same, ref_same, atol=1e-4)

    conv_tf_pad = Conv2D(
        filter_shape=(3, 3, 3, 4),
        filter_stride=(2, 2),
        padding="SAME",
        tf_equivalent_padding=True,
    )
    assert conv_tf_pad(inputs, w=w).shape == (2, 3, 3, 4)

    conv_valid = Conv2D(
        filter_shape=(3, 3, 3, 4), filter_stride=(2, 2), padding="VALID", bias=False
    )
    zero_valid = conv_valid(inputs, w=w)
    ref_valid = np.ref_conv2d(inputs, w, stride=(2, 2), padding="VALID")
    np.testing.assert_allclose(zero_valid, ref_valid, atol=1e-4)
    center_patch_sum = np.sum(inputs[0, 0:3, 0:3, :]) * 0.1
    np.testing.assert_allclose(
        zero_valid[0, 0, 0, :], np.full((4,), float(center_patch_sum)), atol=1e-4
    )
    br_patch_sum = np.sum(inputs[0, 2:5, 2:5, :]) * 0.1
    np.testing.assert_allclose(
        zero_valid[0, 1, 1, :], np.full((4,), float(br_patch_sum)), atol=1e-4
    )

    conv_causal = Conv2D(
        filter_shape=(3, 3, 3, 4),
        is_causal=True,
        mask_t=np.ones((3, 3, 3, 4), dtype=np.float32),
        weight_norm_tpl=Identity(),
    )
    out_causal = conv_causal(inputs, w=w)
    inputs_Future_add = np.concatenate(
        [np.zeros((2, 3, 6, 3)), np.full((2, 3, 6, 3), 10.0)], axis=1
    )
    inputs_mod = inputs + inputs_Future_add
    out_causal_mod = conv_causal(inputs_mod, w=w)
    np.testing.assert_allclose(out_causal[:, :3, :, :], out_causal_mod[:, :3, :, :], atol=1e-5)

    paddings = np.array(
        [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    out_padded = conv_same(inputs, paddings=paddings, w=w, bias=b)
    keep_mask = np.concatenate([np.ones((1, 4, 6, 3)), np.zeros((1, 2, 6, 3))], axis=1)
    keep_mask_batch = np.concatenate([keep_mask, np.ones((1, 6, 6, 3))], axis=0)
    inputs_zeroed = inputs * keep_mask_batch
    np.testing.assert_allclose(out_padded, conv_same(inputs_zeroed, w=w, bias=b), atol=1e-4)

    conv_grouped = Conv2D(filter_shape=(3, 3, 1, 3))
    w_grp = np.ones((3, 3, 1, 3), dtype=np.float32)
    out_grp = conv_grouped(np.ones((1, 4, 4, 3), dtype=np.float32), w=w_grp)
    assert float(out_grp[0, 1, 1, 0]) == 9.0

    np.testing.assert_allclose(Conv2D()(inputs), np.zeros_like(inputs), atol=1e-6)
    with pytest.raises(ValueError, match="Conv2D expected 4D input"):
        Conv2D()(np.ones((2, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="Conv2D explicit filter shape"):
        Conv2D(filter_shape=(3, 3, 3, 4))(inputs, w=np.ones((2, 2, 3, 4), dtype=np.float32))


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
        filter_shape=(2, 2, 3, 4),
        filter_stride=(1, 1),
        padding="VALID",
        compat_with_lingvo=True,
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

    dconv_2d_w = DepthwiseConv1D(filter_shape=(3, 1, 4), use_2d_conv_weight_shape=True)
    assert dconv_2d_w(inputs).shape == (2, 6, 4)
    assert dconv_2d_w.params["w"]().shape == (3, 1, 4, 1)

    cdconv_seq_h = CausalDepthwiseConv1D(kernel_size=3, hidden_dims=(4,))
    assert cdconv_seq_h(inputs).shape == (2, 6, 4)
    cdconv_no_h = CausalDepthwiseConv1D(kernel_size=3, hidden_dims=0)
    assert cdconv_no_h(inputs).shape == (2, 6, 4)

    dconv_strided = DepthwiseConv1D(filter_shape=(3, 1, 4), filter_stride=(2,))
    out_strided = dconv_strided(inputs, paddings=paddings, w=w)
    assert out_strided.shape == (2, 3, 4)
    np.testing.assert_allclose(out_strided[0, 2, :], 0.0, atol=1e-5)

    cdconv = CausalDepthwiseConv1D(kernel_size=3, hidden_dims=4, bias=True)
    x_seq = np.arange(1, 25, dtype=np.float32).reshape((1, 6, 4))
    out_causal = cdconv(x_seq, w=w, bias=b)
    x_seq_future_mod = x_seq + np.concatenate(
        [np.zeros((1, 3, 4)), np.full((1, 3, 4), 50.0)], axis=1
    )
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
        input_dims=4,
        kernel_size=3,
        is_causal=True,
        dropout_prob=0.2,
        conv_activation_tpl=GELU(),
    )
    out_lc = lconv_causal(x_seq, w_start=w_start, w_conv=w, w_end=w_end)
    assert out_lc.shape == (1, 6, 4)
    np.testing.assert_allclose(np.mean(out_lc, axis=1), np.zeros((1, 4)), atol=0.5)
    assert np.any(np.asarray(out_lc) != 0.0)

    np.testing.assert_allclose(
        cdconv(x_seq, w=w), np.ref_depthwise_conv1d(x_seq, w, is_causal=True), atol=1e-5
    )

    cba_custom = ConvBNAct(
        filter_shape=(3, 3, 2, 2), batch_norm_tpl=Identity(), activation_tpl=Identity()
    )
    np.testing.assert_allclose(
        cba_custom(np.ones((1, 3, 3, 2), dtype=np.float32)), np.zeros((1, 3, 3, 2))
    )

    dconv_default = DepthwiseConv1D()
    np.testing.assert_allclose(
        dconv_default(np.ones((1, 4, 2), dtype=np.float32)), np.zeros((1, 4, 2))
    )
    np.testing.assert_allclose(
        dconv_default(np.ones((1, 4, 2), dtype=np.float32), w=np.ones((3, 2), dtype=np.float32)),
        np.array([[[2.0, 2.0], [3.0, 3.0], [3.0, 3.0], [2.0, 2.0]]]),
    )

    dconv_dilated = DepthwiseConv1D(filter_shape=(2, 1, 2), rhs_dilation_rate=2)
    np.testing.assert_allclose(
        dconv_dilated(np.ones((1, 4, 2), dtype=np.float32), w=np.ones((2, 1, 2), dtype=np.float32)),
        np.array([[[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]]]),
    )

    lconv_custom = LightConv1D(
        input_dims=2,
        kernel_size=3,
        ln_tpl=Identity(),
        linear_start_tpl=lambda x, *a, **kw: t_ops.concatenate([x, x], axis=-1),
        depthwise_conv_tpl=Identity(),
        conv_norm_layer_tpl=Identity(),
        conv_activation_tpl=Identity(),
        linear_end_tpl=Identity(),
    )
    x_lc = np.ones((1, 4, 2), dtype=np.float32)
    expected_lc = x_lc * (1.0 / (1.0 + np.exp(-x_lc)))
    np.testing.assert_allclose(lconv_custom(x_lc), expected_lc, atol=1e-5)

    lconv_2d_norm = LightConv1D(input_dims=4, kernel_size=3, use_2d_conv_norm=True)
    assert lconv_2d_norm(x_seq, w_start=w_start, w_conv=w, w_end=w_end).shape == (1, 6, 4)
    lconv_2d_norm_id = LightConv1D(
        input_dims=2,
        kernel_size=3,
        use_2d_conv_norm=True,
        conv_norm_layer_tpl=Identity(),
    )
    assert lconv_2d_norm_id(x_lc).shape == (1, 4, 2)
