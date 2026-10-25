"""Tests for linear, feedforward, einsum, and pooling layers in zero_pax.praxis.layers.linear_pooling."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from zero_pax.praxis.layers import (
    GELU,
    Bias,
    Einsum,
    EinsumOp,
    FeedForward,
    GlobalPooling,
    Identity,
    Linear,
    MLPBlock,
    PerDimScale,
    Pooling,
    Pooling1D,
)


def test_linear_and_bias() -> None:
    """Verify `Linear` and `Bias` layers match `jax.numpy` matrix multiplication and addition."""
    x = np.random.normal(size=(2, 3, 4)).astype(np.float32)
    w = np.random.normal(size=(4, 5)).astype(np.float32)
    b = np.random.normal(size=(5,)).astype(np.float32)

    linear = Linear(input_dims=4, output_dims=5)
    bias = Bias(dims=5)

    out_lin = linear(x, w=w)
    out_bias = bias(out_lin, b=b)

    ref = jnp.einsum("...d,df->...f", x, w) + b
    np.testing.assert_allclose(out_lin, jnp.einsum("...d,df->...f", x, w), atol=1e-5)
    np.testing.assert_allclose(out_bias, ref, atol=1e-5)

    np.testing.assert_allclose(Linear()(x), np.zeros((2, 3, 4)), atol=1e-6)
    np.testing.assert_allclose(Bias()(x), x, atol=1e-6)


def test_einsum_and_einsum_op() -> None:
    """Verify `EinsumOp` and `Einsum` match `jax.numpy.einsum` across configurations."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y = np.array([[1.0, 0.0], [0.5, 2.0], [-1.0, 1.0]], dtype=np.float32)

    op = EinsumOp(equation="bd,df->bf")
    np.testing.assert_allclose(op(x, y), jnp.einsum("bd,df->bf", x, y), atol=1e-5)
    np.testing.assert_allclose(
        EinsumOp()("bd,df->bf", x, y), jnp.einsum("bd,df->bf", x, y), atol=1e-5
    )
    np.testing.assert_allclose(EinsumOp()(), np.zeros((1,)), atol=1e-6)
    np.testing.assert_allclose(EinsumOp()(x), x, atol=1e-6)

    ein_t2 = Einsum(eqn="bd,df->bf")
    np.testing.assert_allclose(ein_t2(x, t2=y), jnp.einsum("bd,df->bf", x, y), atol=1e-5)

    b_vec = np.array([0.5, -0.5], dtype=np.float32)
    ein_w = Einsum(eqn="bd,df->bf", w_shape=(3, 2), use_bias=True, bias_shape=(2,))
    np.testing.assert_allclose(
        ein_w(x, w=y, b=b_vec), jnp.einsum("bd,df->bf", x, y) + b_vec, atol=1e-5
    )
    np.testing.assert_allclose(ein_w(x), np.zeros((2, 2)), atol=1e-6)

    ein_default = Einsum(equation="bd,df->bf", use_bias=True)
    np.testing.assert_allclose(
        ein_default(x), jnp.einsum("bd,df->bf", x, np.ones((3, 3))), atol=1e-5
    )
    np.testing.assert_allclose(Einsum()(x), x, atol=1e-6)


def test_feedforward_and_mlp_block() -> None:
    """Verify `FeedForward` and `MLPBlock` numerically against analytical compositions."""
    inputs = np.array([[1.0, -2.0, 0.5, 3.0], [-1.0, 1.5, 2.0, -0.5]], dtype=np.float32)
    w = np.ones((4, 6), dtype=np.float32) * 0.5
    b = np.full((6,), 0.25, dtype=np.float32)

    ff = FeedForward(
        input_dims=4,
        output_dims=6,
        has_bias=True,
        activation_tpl=GELU(approximate=False),
        dropout_tpl=Identity(),
    )
    expected_ff = GELU(approximate=False)(inputs @ w + b)
    np.testing.assert_allclose(ff(inputs, w=w, b=b), expected_ff, atol=1e-5)

    ff_cls_act = FeedForward(input_dims=4, output_dims=4, has_bias=False, activation_tpl=GELU)
    w_id = np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(ff_cls_act(inputs, w=w_id), GELU()(inputs), atol=1e-5)

    mlp = MLPBlock(num_layers=2, hidden_dims=4, activate_final=False)
    expected_mlp = np.maximum(0.0, inputs @ w_id) @ w_id
    np.testing.assert_allclose(mlp(inputs, w=w_id), expected_mlp, atol=1e-5)

    mlp_act = MLPBlock(num_layers=2, hidden_dims=4, activate_final=True)
    expected_mlp_act = np.maximum(0.0, np.maximum(0.0, inputs @ w_id) @ w_id)
    np.testing.assert_allclose(mlp_act(inputs, w=w_id), expected_mlp_act, atol=1e-5)

    mlp_custom = MLPBlock(num_layers=2, ff_tpl=Identity())
    np.testing.assert_allclose(mlp_custom(inputs), inputs, atol=1e-5)

    from zero_pax.types import WeightInit

    ff_inits = FeedForward(
        input_dims=4,
        output_dims=4,
        weight_init=WeightInit(method="ones", scale=1.0),
        bias_init=WeightInit(method="ones", scale=2.0),
    )
    np.testing.assert_allclose(
        ff_inits(np.ones((2, 4), dtype=np.float32)), np.full((2, 4), 6.0), atol=1e-5
    )


def test_per_dim_scale() -> None:
    """Verify `PerDimScale` multiplies each feature dimension by its scale parameter."""
    inputs = np.ones((2, 4), dtype=np.float32) * 2.0
    pds = PerDimScale(dim=4)
    np.testing.assert_allclose(pds(inputs), inputs, atol=1e-5)

    scale_vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scaled = PerDimScale(dims=4)(inputs, scale=scale_vec)
    np.testing.assert_allclose(scaled, inputs * scale_vec, atol=1e-5)


def test_pooling_2d() -> None:
    """Verify 2D `Pooling` MAX and AVG reductions for SAME/VALID and with `paddings`."""
    inputs = np.arange(1, 33, dtype=np.float32).reshape((1, 4, 4, 2))
    paddings = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

    pool_id = Pooling(window_shape=(1, 1), window_stride=(1, 1), pooling_type="MAX")
    out_pool_id = pool_id(inputs)
    assert not isinstance(out_pool_id, tuple)
    np.testing.assert_allclose(out_pool_id, inputs)
    res_id_pad = pool_id(inputs, paddings=paddings)
    assert isinstance(res_id_pad, tuple) and res_id_pad[1] is not None
    np.testing.assert_allclose(res_id_pad[0], inputs)
    np.testing.assert_allclose(res_id_pad[1], paddings)

    pool_max = Pooling(
        window_shape=(2, 2), window_stride=(2, 2), pooling_type="MAX", padding="VALID"
    )
    out_max = pool_max(inputs)
    assert not isinstance(out_max, tuple)
    expected_tl = np.max(inputs[:, 0:2, 0:2, :], axis=(1, 2))
    expected_br = np.max(inputs[:, 2:4, 2:4, :], axis=(1, 2))
    np.testing.assert_allclose(out_max[:, 0, 0, :], expected_tl, atol=1e-5)
    np.testing.assert_allclose(out_max[:, 1, 1, :], expected_br, atol=1e-5)

    pool_avg = Pooling(
        window_shape=(2, 2), window_stride=(2, 2), pooling_type="AVG", padding="SAME"
    )
    res_avg = pool_avg(inputs, paddings=paddings)
    assert isinstance(res_avg, tuple)
    np.testing.assert_allclose(
        res_avg[0][:, 0, 0, :], np.mean(inputs[:, 0:2, 0:2, :], axis=(1, 2)), atol=1e-5
    )
    np.testing.assert_allclose(res_avg[0][:, 1, :, :], 0.0, atol=1e-5)

    pool_max_same = Pooling(
        window_shape=(3, 3), window_stride=(1, 1), pooling_type="MAX", padding="SAME"
    )
    res_ms = pool_max_same(inputs, paddings=paddings)
    assert isinstance(res_ms, tuple)
    assert res_ms[0].shape == (1, 4, 4, 2)

    with pytest.raises(ValueError, match="positive integers"):
        Pooling(window_shape=(0, 2), window_stride=(1, 1))(inputs)
    with pytest.raises(ValueError, match="positive integers"):
        Pooling(window_shape=(2, 2), window_stride=(0, 1))(inputs)
    with pytest.raises(ValueError, match="MAX or AVG"):
        Pooling(window_shape=(2, 2), window_stride=(1, 1), pooling_type="SUM")(inputs)


def test_pooling_1d() -> None:
    """Verify `Pooling1D` temporal window reductions with and without `paddings`."""
    inputs = np.arange(1, 25, dtype=np.float32).reshape((1, 6, 4))
    paddings = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

    p1d_noop = Pooling1D(window=1, stride=1)
    out_p1d_noop = p1d_noop(inputs)
    assert not isinstance(out_p1d_noop, tuple)
    np.testing.assert_allclose(out_p1d_noop, inputs)
    res_noop = p1d_noop(inputs, paddings=paddings)
    assert isinstance(res_noop, tuple)
    np.testing.assert_allclose(res_noop[0], inputs)

    p1d = Pooling1D(window=2, stride=2, pooling_type="AVG", padding="SAME")
    out = p1d(inputs)
    assert not isinstance(out, tuple)
    np.testing.assert_allclose(out[:, 0, :], np.mean(inputs[:, 0:2, :], axis=1), atol=1e-5)

    res_pad = p1d(inputs, paddings=paddings)
    assert isinstance(res_pad, tuple)
    np.testing.assert_allclose(res_pad[0][:, 0, :], np.mean(inputs[:, 0:2, :], axis=1), atol=1e-5)
    np.testing.assert_allclose(res_pad[0][:, 2, :], 0.0, atol=1e-5)


def test_global_pooling() -> None:
    """Verify `GlobalPooling` AVG and MAX reductions and error handling."""
    x = np.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=np.float32)
    paddings = np.array([[[0.0], [0.0], [0.0], [1.0], [1.0]]], dtype=np.float32)

    gp_avg = GlobalPooling(pooling_type="AVG", pooling_dims=[1], keepdims=True)
    np.testing.assert_allclose(gp_avg(x), np.array([[[3.0]]]), atol=1e-5)
    np.testing.assert_allclose(
        gp_avg(x, compatible_paddings=paddings), np.array([[[2.0]]]), atol=1e-5
    )

    gp_max = GlobalPooling(pooling_type="MAX", pooling_dims=[1], keepdims=False)
    np.testing.assert_allclose(gp_max(x), np.array([[5.0]]), atol=1e-5)
    np.testing.assert_allclose(
        gp_max(x, compatible_paddings=paddings), np.array([[3.0]]), atol=1e-5
    )

    with pytest.raises(ValueError, match="must be set as a list"):
        GlobalPooling(pooling_dims=None)(x)
    with pytest.raises(ValueError, match="non-negative integers"):
        GlobalPooling(pooling_dims=[-1])(x)
    with pytest.raises(ValueError, match="smaller than input rank"):
        GlobalPooling(pooling_dims=[5])(x)
    with pytest.raises(ValueError, match="MAX or AVG"):
        GlobalPooling(pooling_dims=[1], pooling_type="BAD")(x)
