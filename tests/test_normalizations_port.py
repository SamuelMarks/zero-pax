"""Tests for normalization layers in zero_pax.praxis.layers.normalizations."""

from __future__ import annotations

import pytest

from tests import conftest as np
from zero_pax.praxis.layers import (
    BaseNormalization,
    BatchNorm,
    GroupNorm,
    IdentityNorm,
    LayerNorm,
    RmsNorm,
    RmsNormNoScale,
)


def test_base_and_identity_norm() -> None:
    """Verify `BaseNormalization` raises NotImplementedError and `IdentityNorm` returns input."""
    with pytest.raises(NotImplementedError):
        BaseNormalization()([1.0, 2.0])

    x = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(IdentityNorm()(x), x)
    np.testing.assert_allclose(IdentityNorm().fprop(x), x)


def test_layer_norm() -> None:
    """Verify `LayerNorm` matches analytical mean/variance normalization across scale modes."""
    inputs = np.array([[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]], dtype=np.float32)
    scale = np.array([0.5, -0.2, 0.1, 0.0], dtype=np.float32)
    bias = np.array([0.1, 0.2, -0.1, 0.05], dtype=np.float32)

    layer = LayerNorm(dim=4, epsilon=1e-5, direct_scale=False)
    zero_out = layer(inputs, scale=scale, bias=bias)

    ref_out = np.ref_layer_norm(inputs, scale=scale, bias=bias, eps=1e-5, direct_scale=False)
    np.testing.assert_allclose(zero_out, ref_out, atol=1e-5)

    layer_direct = LayerNorm(dim=4, epsilon=1e-5, direct_scale=True)
    out_direct = layer_direct(inputs, scale=scale, bias=bias)
    ref_direct = np.ref_layer_norm(inputs, scale=scale, bias=bias, eps=1e-5, direct_scale=True)
    np.testing.assert_allclose(out_direct, ref_direct, atol=1e-5)

    layer_noscale = LayerNorm(dim=4, epsilon=1e-5, use_scale=False, use_bias=False)
    out_noscale = layer_noscale(inputs)
    ref_noscale = np.ref_layer_norm(inputs, scale=None, bias=None, eps=1e-5)
    np.testing.assert_allclose(out_noscale, ref_noscale, atol=1e-5)


def test_rms_norm_and_no_scale() -> None:
    """Verify `RmsNorm` and `RmsNormNoScale` match analytical RMS normalization."""
    inputs = np.array([[[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]], dtype=np.float32)
    scale = np.array([1.2, 0.8, 1.5, 0.5], dtype=np.float32)

    rms = RmsNorm(dim=4, epsilon=1e-6, direct_scale=True)
    zero_out = rms(inputs, scale=scale)
    ref_out = np.ref_rms_norm(inputs, scale=scale, eps=1e-6, direct_scale=True)
    np.testing.assert_allclose(zero_out, ref_out, atol=1e-5)

    rms_offset = RmsNorm(dim=4, epsilon=1e-6, direct_scale=False)
    out_offset = rms_offset(inputs, scale=scale)
    ref_offset = np.ref_rms_norm(inputs, scale=scale, eps=1e-6, direct_scale=False)
    np.testing.assert_allclose(out_offset, ref_offset, atol=1e-5)

    rms_no_scale = RmsNormNoScale(dim=4, epsilon=1e-6)
    np.testing.assert_allclose(
        rms_no_scale(inputs), np.ref_rms_norm(inputs, scale=None, eps=1e-6), atol=1e-5
    )


def test_batch_norm() -> None:
    """Verify `BatchNorm` training, evaluation, and padded statistics."""
    inputs = np.arange(1, 17, dtype=np.float32).reshape((2, 4, 2))
    paddings = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    bn = BatchNorm(dim=2, decay=0.5, epsilon=1e-3)
    out_train = bn(inputs)
    mean_ref = np.mean(inputs, axis=(0, 1), keepdims=True)
    var_ref = np.var(inputs, axis=(0, 1), keepdims=True)
    np.testing.assert_allclose(out_train, (inputs - mean_ref) / np.sqrt(var_ref + 1e-3), atol=1e-5)
    np.testing.assert_allclose(
        bn.params["moving_mean"](), 0.5 * np.mean(inputs, axis=(0, 1)), atol=1e-5
    )

    out_pad = bn(inputs, paddings=paddings)
    assert np.all(np.asarray(out_pad[0, 2:]) == 0.0)
    valid_vals = np.concatenate([inputs[0, :2], inputs[1, :]], axis=0)
    pad_mean = np.mean(valid_vals, axis=0)
    pad_var = np.var(valid_vals, axis=0)
    expected_00 = (inputs[0, 0] - pad_mean) / np.sqrt(pad_var + 1e-3)
    np.testing.assert_allclose(out_pad[0, 0], expected_00, atol=1e-5)

    bn_eval = BatchNorm(dim=2, force_eval_mode=True, epsilon=1e-3)
    mov_mean = np.array([1.0, 2.0], dtype=np.float32)
    mov_var = np.array([4.0, 9.0], dtype=np.float32)
    out_eval = bn_eval(inputs, moving_mean=mov_mean, moving_variance=mov_var)
    expected_eval = (inputs - mov_mean) / np.sqrt(mov_var + 1e-3)
    np.testing.assert_allclose(out_eval, expected_eval, atol=1e-5)


def test_group_norm() -> None:
    """Verify `GroupNorm` standard, cumulative, and padded modes numerically."""
    inputs = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 2, 4))
    paddings = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    layer = GroupNorm(dim=4, num_groups=2, epsilon=1e-5)
    out = layer(
        inputs,
        scale=np.full((4,), 0.5, dtype=np.float32),
        bias=np.full((4,), 0.1, dtype=np.float32),
    )
    grouped = inputs.reshape((2, 4, 2, 2, 2))
    g_mean = np.mean(grouped, axis=(1, 2, 4), keepdims=True)
    g_var = np.var(grouped, axis=(1, 2, 4), keepdims=True)
    expected = ((grouped - g_mean) / np.sqrt(g_var + 1e-5)).reshape((2, 4, 2, 4)) * 1.5 + 0.1
    np.testing.assert_allclose(out, expected, atol=1e-5)

    out_pad = layer(inputs, paddings=paddings)
    assert np.all(np.asarray(out_pad[0, 2:]) == 0.0)
    g0_valid = grouped[0, :2, :, 0, :]
    g0_mean = np.mean(g0_valid)
    g0_var = np.var(g0_valid)
    np.testing.assert_allclose(
        out_pad[0, 0, 0, :2], (grouped[0, 0, 0, 0, :] - g0_mean) / np.sqrt(g0_var + 1e-5), atol=1e-5
    )

    layer_cum = GroupNorm(
        dim=4, num_groups=2, cumulative=True, epsilon=1e-5, use_scale=False, use_bias=False
    )
    out_cum = layer_cum(inputs)
    f0_mean = np.mean(grouped[:, 0:1, :, :, :], axis=(2, 4), keepdims=True)
    f0_var = np.var(grouped[:, 0:1, :, :, :], axis=(2, 4), keepdims=True)
    expected_f0 = ((grouped[:, 0:1, :, :, :] - f0_mean) / np.sqrt(f0_var + 1e-5)).reshape(
        (2, 1, 2, 4)
    )
    np.testing.assert_allclose(out_cum[:, 0:1, :, :], expected_f0, atol=1e-4)

    out_cum_pad = layer_cum(inputs, paddings=paddings)
    assert np.all(np.asarray(out_cum_pad[0, 2:]) == 0.0)

    layer_odd = GroupNorm(num_groups=3, input_rank=None, set_padded_output_to_zero=False)
    out_odd = layer_odd(np.ones((2, 4, 4), dtype=np.float32), paddings=paddings)
    np.testing.assert_allclose(out_odd, np.zeros((2, 4, 4)), atol=1e-5)

    x_4d = np.arange(1, 33, dtype=np.float32).reshape((2, 2, 2, 4))
    pad_2d = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    bn = BatchNorm(dim=4, set_padded_output_to_zero=False)
    out_bn = bn(x_4d, paddings=pad_2d)
    assert out_bn.shape == (2, 2, 2, 4)
    assert np.any(np.asarray(out_bn[0, 1]) != 0.0)

    gn_no_aff = GroupNorm(
        dim=4, num_groups=2, use_scale=False, use_bias=False, set_padded_output_to_zero=False
    )
    out_gn = gn_no_aff(x_4d, paddings=pad_2d)
    assert np.any(np.asarray(out_gn[0, 1]) != 0.0)

    with pytest.raises(ValueError, match="Expected input channel dimension"):
        GroupNorm(dim=8)(x_4d)
    with pytest.raises(ValueError, match="Expected input rank"):
        GroupNorm(dim=4, input_rank=3)(x_4d)

    ln_fp32 = LayerNorm(dim=4, reductions_in_fp32=True)
    np.testing.assert_allclose(np.ref_mean(ln_fp32(x_4d), axis=-1), np.zeros((2, 2, 2)), atol=1e-5)

    rms_inter = RmsNorm(dim=4, intermediate_dtype="float32")
    assert rms_inter(x_4d).shape == (2, 2, 2, 4)

    gn_pad_4d = GroupNorm(dim=4, num_groups=2)
    out_gn_4d = gn_pad_4d(x_4d, paddings=np.zeros((2, 2, 2, 1), dtype=np.float32))
    np.testing.assert_allclose(
        np.mean(np.asarray(out_gn_4d).reshape((2, 2, 2, 2, 2)), axis=(1, 2, 4)),
        np.zeros((2, 2)),
        atol=1e-5,
    )
