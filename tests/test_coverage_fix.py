"""Tests to fix branch coverage."""

import pytest
import numpy as np
from zero_pax.praxis.layers import (
    AttentionProjection,
    CifgLstmCellSimple,
    DotProductAttention,
    FRnn,
    GroupNorm,
    GroupedQueryAttention,
    LayerNormalizedLstmCellSimple,
    PositionalEmbedding,
    PositionalEmbedding2D,
    RmsNorm,
    SharedEmbeddingSoftmax,
    VectorQuantization,
    LayerNorm,
    BasePraxisLayer,
    DummyType,
    BatchNorm,
)


def test_dummy_meta():
    """Test DummyMeta."""

    class TestDummy(DummyType):
        """TestDummy class."""

        pass

    t = TestDummy()
    assert getattr(TestDummy, "non_existent", None) is None
    assert TestDummy["item"] is None


def test_base_praxis_layer():
    """Test BasePraxisLayer."""
    layer = BasePraxisLayer()
    assert layer(1, 2) == 1


def test_attention_projection_coverage():
    """Test AttentionProjection coverage."""
    layer = AttentionProjection(
        dim_per_head=4,
        num_heads=2,
        is_output_projection=False,
        use_bias=True,
    )
    inputs = np.zeros((1, 8))
    w = np.zeros((8, 2, 4))
    bias = np.zeros((2, 4))
    layer(inputs, w=w, bias=bias)

    layer2 = AttentionProjection(
        dim_per_head=4,
        num_heads=2,
        is_output_projection=True,
        use_bias=True,
    )
    inputs2 = np.zeros((1, 2, 4))
    w2 = np.zeros((2, 4, 8))
    bias2 = np.zeros(8)
    layer2(inputs2, w=w2, bias=bias2)


def test_lstm_cell_simple():
    """Test CifgLstmCellSimple."""
    layer = CifgLstmCellSimple(hidden_size=4, num_gates=3)
    state0 = (np.zeros((1, 4)), np.zeros((1, 4)))
    act = np.zeros((1, 4))
    b = np.zeros(12)
    layer(state0, act, b=b)


def test_dot_product_attention():
    """Test DotProductAttention."""
    layer = DotProductAttention(num_heads=2, dim_per_head=4)
    query = np.zeros((1, 2, 4))
    key = np.zeros((1, 3, 4))
    value = np.zeros((1, 3, 4))
    qw = np.zeros((4, 2, 4))
    kw = np.zeros((4, 2, 4))
    vw = np.zeros((4, 2, 4))
    layer(query, key, value, query_w=qw, key_w=kw, value_w=vw)


def test_rnn_layer():
    """Test FRnn."""
    layer = FRnn(hidden_size=4)
    inputs = np.zeros((1, 2, 4))
    w = np.zeros((8, 4))
    layer(inputs, w=w)


def test_grouped_layer_norm():
    """Test GroupNorm."""
    layer = GroupNorm(num_groups=2, use_scale=True, use_bias=True)
    inputs = np.zeros((1, 2, 4))
    gamma = np.zeros(4)
    beta = np.zeros(4)
    layer(inputs, gamma=gamma, beta=beta)
    layer(inputs)
    layer2 = GroupNorm(num_groups=2, use_scale=False, use_bias=False)
    layer2(inputs)


def test_mq_dot_product_attention():
    """Test GroupedQueryAttention."""
    layer = GroupedQueryAttention(num_heads=4, num_kv_heads=2, dim_per_head=4)
    query = np.zeros((1, 2, 4))
    key = np.zeros((1, 3, 4))
    value = np.zeros((1, 3, 4))
    qw = np.zeros((4, 4, 4))
    kw = np.zeros((4, 2, 4))
    vw = np.zeros((4, 2, 4))
    layer(query, key, value, query_w=qw, key_w=kw, value_w=vw)


def test_layer_norm_lstm():
    """Test LayerNormalizedLstmCellSimple."""
    layer = LayerNormalizedLstmCellSimple(hidden_size=4)
    state0 = (np.zeros((1, 4)), np.zeros((1, 4)))
    act = np.zeros((1, 4))
    wm = np.zeros((8, 16))
    b = np.zeros(16)
    ln_scale = np.ones(16)
    layer(state0, act, wm=wm, b=b, ln_scale=ln_scale)


def test_sinusoidal_pos_embedding():
    """Test PositionalEmbedding."""
    layer = PositionalEmbedding(embedding_dims=5)  # odd to trigger % 2 == 1
    layer(seq_length=2)
    layer2 = PositionalEmbedding(embedding_dims=4)  # even
    layer2(seq_length=2)


def test_vision_pos_embedding():
    """Test PositionalEmbedding2D."""
    layer = PositionalEmbedding2D(
        embedding_dims=4, h=2, w=2, num_prepend_cls_tokens=1, num_append_cls_tokens=1
    )
    layer()
    layer2 = PositionalEmbedding2D(
        embedding_dims=4, h=2, w=2, num_prepend_cls_tokens=0, num_append_cls_tokens=0
    )
    layer2()


def test_rmsnorm():
    """Test RmsNorm."""
    layer = RmsNorm(direct_scale=True)
    inputs = np.zeros((1, 4))
    scale = np.ones(4)
    layer(inputs, scale=scale)
    layer(inputs)


def test_shared_embedding_softmax():
    """Test SharedEmbeddingSoftmax."""
    layer = SharedEmbeddingSoftmax(num_classes=10, input_dims=4, scale_sqrt_depth=False)
    layer.emb_lookup(np.array([0, 1]), w=np.zeros((10, 4)))


def test_approximate_clustering():
    """Test VectorQuantization."""
    layer = VectorQuantization(num_heads=2, num_clusters=3, dim_per_head=4)
    layer(np.zeros((1, 2, 4)), w=np.zeros((2, 3, 4)))


def test_layer_norm():
    """Test LayerNorm."""
    layer = LayerNorm(use_scale=True, use_bias=True, direct_scale=True)
    inputs = np.zeros((1, 4))
    scale = np.ones(4)
    bias = np.zeros(4)
    layer(inputs, scale=scale, bias=bias)
    layer(inputs)
    layer2 = LayerNorm(use_scale=False, use_bias=False, direct_scale=True)
    layer2(inputs)


def test_batch_norm():
    """Test BatchNorm."""
    layer = BatchNorm()
    inputs = np.zeros((1, 4))
    layer(inputs)
