"""Tests for the test_embeddings_port module."""

import numpy as np
import pytest
from zero_pax.praxis.layers import (
    Embedding,
    SharedEmbeddingSoftmax,
    GShardSharedEmbeddingSoftmax,
    PositionalEmbedding,
    PositionalEmbedding2D,
    TrainablePositionalEmbedding,
    VectorQuantization,
    VectorQuantizer,
    RandomVectorQuantizer,
)


def test_embedding():
    """Executes the test_embedding test.

    Returns:
        The result of the test.
    """
    layer = Embedding(num_classes=10, input_dims=4)
    w = np.random.normal(size=(10, 4)).astype(np.float32)
    ids = np.array([[1, 5], [9, 0]])

    out = layer(ids, w=w)
    assert out.shape == (2, 2, 4)
    np.testing.assert_allclose(out, w[ids])

    # default implicit w
    out_def = layer(ids)
    assert out_def.shape == (2, 2, 4)

    # NaN for OOB and scaling
    layer_nan = Embedding(
        num_classes=10, input_dims=4, scale_sqrt_depth=True, set_nan_for_oob_id=True
    )
    ids_oob = np.array([1, 15, -1])
    out_nan = layer_nan(ids_oob, w=w)
    assert np.isnan(out_nan[1]).all()
    assert np.isnan(out_nan[2]).all()
    assert not np.isnan(out_nan[0]).any()


def test_shared_embedding_softmax():
    """Executes the test_shared_embedding_softmax test.

    Returns:
        The result of the test.
    """
    layer = SharedEmbeddingSoftmax(num_classes=10, input_dims=4, scale_sqrt_depth=True)
    w = np.random.normal(size=(10, 4)).astype(np.float32)

    ids = np.array([1, 2])
    emb = layer.emb_lookup(ids, w=w)
    assert emb.shape == (2, 4)

    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    logits = layer(inputs, w=w)
    assert logits.shape == (2, 10)

    # default
    layer.emb_lookup(ids)
    layer(inputs)


def test_gshard_embedding_softmax():
    """Executes the test_gshard_embedding_softmax test.

    Returns:
        The result of the test.
    """
    layer = GShardSharedEmbeddingSoftmax(
        num_classes=10, input_dims=4, soft_cap_logits=2.0, logits_abs_max=1.0
    )
    w = np.random.normal(size=(10, 4)).astype(np.float32)

    ids = np.array([1])
    emb = layer.emb_lookup(ids, w=w)
    assert emb.shape == (1, 4)

    inputs = np.random.normal(size=(2, 4)).astype(np.float32)
    logits = layer(inputs, w=w)
    assert logits.shape == (2, 10)
    assert np.max(np.abs(logits)) <= 1.0

    layer.emb_lookup(ids)
    layer(inputs)


def test_positional_embedding():
    """Executes the test_positional_embedding test.

    Returns:
        The result of the test.
    """
    layer = PositionalEmbedding(embedding_dims=5)  # odd dim
    out = layer(seq_length=4)
    assert out.shape == (1, 4, 5)

    pos = np.array([[0, 2, 4]])
    out_pos = layer(position=pos)
    assert out_pos.shape == (1, 3, 5)


def test_positional_embedding_2d():
    """Executes the test_positional_embedding_2d test.

    Returns:
        The result of the test.
    """
    layer = PositionalEmbedding2D(
        h=2, w=3, embedding_dims=5, num_prepend_cls_tokens=1, num_append_cls_tokens=2
    )
    out = layer()
    # 2*3 + 1 + 2 = 9 tokens
    assert out.shape == (1, 9, 5)


def test_trainable_positional_embedding():
    """Executes the test_trainable_positional_embedding test.

    Returns:
        The result of the test.
    """
    layer = TrainablePositionalEmbedding(max_seq_length=10, embedding_dims=4)
    w = np.random.normal(size=(10, 4)).astype(np.float32)
    out = layer(seq_length=4, w=w)
    assert out.shape == (1, 4, 4)

    pos = np.array([[1, 3]])
    out_pos = layer(position=pos, w=w)
    assert out_pos.shape == (1, 2, 4)

    layer(seq_length=4)


def test_quantization_coverage():
    """Executes the test_quantization_coverage test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    vq = VectorQuantization()
    vqz = VectorQuantizer()
    rvqz = RandomVectorQuantizer()

    assert vq(inputs).shape == inputs.shape
    assert vqz(inputs)[0].shape == inputs.shape
    assert rvqz(inputs)[0].shape == inputs.shape


def test_gshard_warning_fix():
    """Executes the test_gshard_warning_fix test.

    Returns:
        The result of the test.
    """
    layer = GShardSharedEmbeddingSoftmax(input_dims=1, num_classes=2)
    layer(np.array([[1.0]]))
