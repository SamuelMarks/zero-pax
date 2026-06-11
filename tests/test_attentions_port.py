"""Tests for the test_attentions_port module."""

import numpy as np
import pytest
from zero_pax.praxis.layers import (
    DotProductAttention,
    DotProductAttentionWithContext,
    DotProductAttentionWithContextXL,
    DotProductAttentionXL,
    LocalSelfAttention,
    LocalSelfAttentionAlibi,
    LocalSelfAttentionRelativeBias,
    LocalSelfAttentionXL,
    SelfAttentionWithNormAndResidual,
    AttentionProjection,
    GroupedQueryAttention,
)


def test_attention_projection():
    """Executes the test_attention_projection test.

    Returns:
        The result of the test.
    """
    inputs_in = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    proj_in = AttentionProjection(
        input_dim=8, num_heads=2, dim_per_head=4, is_output_projection=False
    )
    out_in = proj_in(inputs_in)
    assert out_in.shape == (2, 4, 2, 4)

    # with bias
    proj_in_b = AttentionProjection(
        input_dim=8,
        num_heads=2,
        dim_per_head=4,
        is_output_projection=False,
        use_bias=True,
    )
    out_in_b = proj_in_b(inputs_in)
    assert out_in_b.shape == (2, 4, 2, 4)

    inputs_out = np.random.normal(size=(2, 4, 2, 4)).astype(np.float32)
    proj_out = AttentionProjection(
        input_dim=8, num_heads=2, dim_per_head=4, is_output_projection=True
    )
    out_out = proj_out(inputs_out)
    assert out_out.shape == (2, 4, 8)

    # with bias
    proj_out_b = AttentionProjection(
        input_dim=8,
        num_heads=2,
        dim_per_head=4,
        is_output_projection=True,
        use_bias=True,
    )
    out_out_b = proj_out_b(inputs_out)
    assert out_out_b.shape == (2, 4, 8)


def test_dot_product_attention():
    """Executes the test_dot_product_attention test.

    Returns:
        The result of the test.
    """
    query = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    key = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    value = np.random.normal(size=(2, 4, 8)).astype(np.float32)

    layer = DotProductAttention(num_heads=2, dim_per_head=4)
    out = layer(query, key, value)
    assert out.shape == (2, 4, 2, 4)

    # with atten_mask
    mask = np.zeros((2, 2, 4, 4))
    mask[:, :, :, 2:] = 1.0  # mask out last two elements
    out_masked = layer(query, key, value, atten_mask=mask)
    assert out_masked.shape == (2, 4, 2, 4)


def test_grouped_query_attention():
    """Executes the test_grouped_query_attention test.

    Returns:
        The result of the test.
    """
    query = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    key = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    value = np.random.normal(size=(2, 4, 8)).astype(np.float32)

    layer = GroupedQueryAttention(num_heads=4, num_kv_heads=2, dim_per_head=4)
    out = layer(query, key, value)
    assert out.shape == (2, 4, 4, 4)

    # with mask
    mask = np.zeros((2, 4, 4, 4))
    out_masked = layer(query, key, value, atten_mask=mask)
    assert out_masked.shape == (2, 4, 4, 4)


def test_wrappers_coverage():
    """Executes the test_wrappers_coverage test.

    Returns:
        The result of the test.
    """
    query = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    key = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    value = np.random.normal(size=(2, 4, 8)).astype(np.float32)

    for cls in [
        DotProductAttentionWithContext,
        DotProductAttentionWithContextXL,
        DotProductAttentionXL,
        LocalSelfAttention,
        LocalSelfAttentionAlibi,
        LocalSelfAttentionRelativeBias,
        LocalSelfAttentionXL,
    ]:
        layer = cls()
        out = layer(query, key, value)
        # Default num_heads=1, dim_per_head=1
        assert out.shape == (2, 4, 1, 1)


def test_self_attention_norm_residual():
    """Executes the test_self_attention_norm_residual test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    layer = SelfAttentionWithNormAndResidual(num_heads=2, dim_per_head=4)
    layer.force_no_residual = True
    layer.force_no_residual = True
    out = layer(inputs)
    assert out.shape == (2, 4, 8)

    # Check shape mismatch bypass
    layer2 = SelfAttentionWithNormAndResidual(num_heads=2, dim_per_head=2)
    out2 = layer2(inputs)
    assert out2.shape == (2, 4, 8)


def test_self_attention_norm_residual_miss():
    """Executes the test_self_attention_norm_residual_miss test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    layer = SelfAttentionWithNormAndResidual(num_heads=2, dim_per_head=4)
    layer.force_no_residual = True

    # mock the projection output to have a different shape
    class BadProj:
        """Executes the BadProj operation."""

        def __init__(self, *args, **kwargs):
            """Executes the __init__ test.

            Args:
                self: The self parameter.

            Returns:
                The result of the test.
            """
            pass

        def __call__(self, x):
            """Executes the __call__ test.

            Args:
                self: The self parameter.
                x: The x parameter.

            Returns:
                The result of the test.
            """
            return np.zeros((2, 4, 9))

    import zero_pax.praxis.layers as zpl

    orig = zpl.AttentionProjection
    zpl.AttentionProjection = BadProj
    try:
        out = layer(inputs)
        assert out.shape == (2, 4, 9)

    finally:
        zpl.AttentionProjection = orig


def test_self_attention_norm_residual_force_no_residual():
    """Executes the test_self_attention_norm_residual_force_no_residual test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    layer = SelfAttentionWithNormAndResidual(num_heads=2, dim_per_head=4)
    layer.force_no_residual = True
    out = layer(inputs)
    assert out.shape == (2, 4, 8)


def test_self_attention_norm_residual_shape_mismatch():
    """Executes the test_self_attention_norm_residual_shape_mismatch test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    layer = SelfAttentionWithNormAndResidual(num_heads=2, dim_per_head=4)

    # mock projection to return totally different shape
    class BadProj:
        """Executes the BadProj operation."""

        def __init__(self, *args, **kwargs):
            """Executes the __init__ test.

            Args:
                self: The self parameter.

            Returns:
                The result of the test.
            """
            pass

        def __call__(self, x):
            """Executes the __call__ test.

            Args:
                self: The self parameter.
                x: The x parameter.

            Returns:
                The result of the test.
            """
            return np.zeros((2, 4, 9))

    import zero_pax.praxis.layers as zpl

    orig = zpl.AttentionProjection
    zpl.AttentionProjection = BadProj
    try:
        out = layer(inputs)
        assert out.shape == (2, 4, 9)
    finally:
        zpl.AttentionProjection = orig
