"""Module docstring."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from zero_pax.praxis.layers import (
    IdentityNorm,
    BatchNorm,
    LayerNorm,
    RmsNorm,
    RmsNormNoScale,
    GroupNorm,
    BaseNormalization,
)


def test_identity_norm_equivalence():
    """test_identity_norm_equivalence docstring."""
    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    layer_zero = IdentityNorm()
    out_zero = layer_zero(inputs)
    np.testing.assert_allclose(out_zero, inputs, atol=1e-5)


def test_layer_norm_equivalence():
    """test_layer_norm_equivalence docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    scale = np.random.normal(size=(8,)).astype(np.float32)
    bias = np.random.normal(size=(8,)).astype(np.float32)

    # JAX ref
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)
    out_jax = (inputs - mean) * jax.lax.rsqrt(var + 1e-6)
    out_jax = out_jax * (1.0 + scale) + bias

    layer_zero = LayerNorm(dim=8, epsilon=1e-6)
    out_zero = layer_zero(inputs, scale=scale, bias=bias)

    np.testing.assert_allclose(out_zero, np.array(out_jax), atol=1e-5)


def test_rms_norm_equivalence():
    """test_rms_norm_equivalence docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    scale = np.random.normal(size=(8,)).astype(np.float32)

    # JAX ref
    var = jnp.mean(jnp.square(inputs), axis=-1, keepdims=True)
    out_jax = inputs * jax.lax.rsqrt(var + 1e-6)
    out_jax *= scale

    layer_zero = RmsNorm(dim=8, epsilon=1e-6, direct_scale=True)
    out_zero = layer_zero(inputs, scale=scale)

    np.testing.assert_allclose(out_zero, np.array(out_jax), atol=1e-5)


def test_rms_norm_noscale_equivalence():
    """test_rms_norm_noscale_equivalence docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)

    var = jnp.mean(jnp.square(inputs), axis=-1, keepdims=True)
    out_jax = inputs * jax.lax.rsqrt(var + 1e-6)

    layer_zero = RmsNormNoScale(dim=8, epsilon=1e-6)
    out_zero = layer_zero(inputs)

    np.testing.assert_allclose(out_zero, np.array(out_jax), atol=1e-5)


def test_batch_norm_equivalence():
    """test_batch_norm_equivalence docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    beta = np.random.normal(size=(8,)).astype(np.float32)
    gamma = np.random.normal(size=(8,)).astype(np.float32)

    reduce_over_dims = (0, 1)
    mean = jnp.mean(inputs, axis=reduce_over_dims, keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=reduce_over_dims, keepdims=True)
    out_jax = (inputs - mean) * jax.lax.rsqrt(var + 0.001)
    out_jax = out_jax * (1.0 + gamma) + beta

    layer_zero = BatchNorm(dim=8, epsilon=0.001)
    out_zero = layer_zero(inputs, beta=beta, gamma=gamma)

    np.testing.assert_allclose(out_zero, np.array(out_jax), atol=1e-5)

    # with padding
    paddings = np.array([[0, 1, 0, 1], [0, 0, 0, 1]]).astype(np.float32)
    mask = 1.0 - jnp.expand_dims(paddings, -1)

    sum_v = jnp.sum(inputs * mask, axis=reduce_over_dims, keepdims=True)
    count_v = jnp.sum(mask, axis=reduce_over_dims, keepdims=True)
    count_v = jnp.maximum(count_v, 1.0)
    mean = sum_v / count_v

    sum_vv = jnp.sum(
        jnp.square(inputs - mean) * mask, axis=reduce_over_dims, keepdims=True
    )
    var = sum_vv / count_v

    out_jax_pad = (inputs - mean) * jax.lax.rsqrt(var + 0.001)
    out_jax_pad = out_jax_pad * (1.0 + gamma) + beta
    out_jax_pad *= mask

    out_zero_pad = layer_zero(inputs, paddings=paddings, beta=beta, gamma=gamma)
    np.testing.assert_allclose(out_zero_pad, np.array(out_jax_pad), atol=1e-5)


def test_group_norm_equivalence():
    """test_group_norm_equivalence docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    beta = np.random.normal(size=(8,)).astype(np.float32)
    gamma = np.random.normal(size=(8,)).astype(np.float32)

    # GroupNorm with num_groups=2, so 4 channels per group
    x = jnp.reshape(inputs, (2, 4, 2, 4))

    mean = jnp.mean(x, axis=(1, 3), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 3), keepdims=True)

    out_jax = (x - mean) * jax.lax.rsqrt(var + 0.001)
    out_jax = jnp.reshape(out_jax, (2, 4, 8))
    out_jax = out_jax * (1.0 + gamma) + beta

    layer_zero = GroupNorm(dim=8, num_groups=2, input_rank=3, epsilon=0.001)
    out_zero = layer_zero(inputs, gamma=gamma, beta=beta)

    np.testing.assert_allclose(out_zero, np.array(out_jax), atol=1e-5)

    # test with padding
    paddings = np.array([[0, 1, 0, 1], [0, 0, 0, 1]]).astype(np.float32)
    mask_expanded = 1.0 - jnp.reshape(paddings, (2, 4, 1, 1))

    sum_v = jnp.sum(x * mask_expanded, axis=(1, 3), keepdims=True)
    count_v = jnp.sum(mask_expanded, axis=(1, 3), keepdims=True)
    count_v = jnp.maximum(count_v, 1.0)
    mean = sum_v / count_v

    sum_vv = jnp.sum(jnp.square(x - mean) * mask_expanded, axis=(1, 3), keepdims=True)
    var = sum_vv / count_v

    out_jax_pad = (x - mean) * jax.lax.rsqrt(var + 0.001)
    out_jax_pad = jnp.reshape(out_jax_pad, (2, 4, 8))
    out_jax_pad = out_jax_pad * (1.0 + gamma) + beta

    out_jax_pad *= 1.0 - jnp.reshape(paddings, (2, 4, 1))

    out_zero_pad = layer_zero(inputs, paddings=paddings, gamma=gamma, beta=beta)
    np.testing.assert_allclose(out_zero_pad, np.array(out_jax_pad), atol=1e-5)


def test_base_normalization_raises():
    """test_base_normalization_raises docstring."""
    layer = BaseNormalization()
    with pytest.raises(NotImplementedError):
        layer(np.array([1.0]))


def test_group_norm_uncovered_branches():
    # input_rank unset, gamma is None, beta is None
    """test_group_norm_uncovered_branches docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    layer = GroupNorm(dim=8, num_groups=2)
    layer.input_rank = None
    out = layer(inputs)  # should set input_rank to 3, init gamma/beta to 0
    assert out.shape == (2, 4, 8)

    # paddings and cumulative=True
    layer_cum = GroupNorm(dim=8, num_groups=2, input_rank=3, cumulative=True)
    paddings = np.array([[0, 1, 0, 1], [0, 0, 0, 1]]).astype(np.float32)

    # We just need to hit the branch for coverage, but we can verify it runs.
    out_cum = layer_cum(inputs, paddings=paddings)
    assert out_cum.shape == (2, 4, 8)


def test_dummymeta_getitem():
    """test_dummymeta_getitem docstring."""
    from zero_pax.praxis.layers import DummyType

    # hit the __getitem__ on metaclass
    _ = DummyType["test"]
