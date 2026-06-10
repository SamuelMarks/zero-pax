"""Module docstring."""

import numpy as np
import pytest
from zero_pax.praxis.layers import (
    TransformerFeedForward,
    AdaptedTransformerFeedForward,
    TransformerFeedForwardMoe,
    Transformer,
    TransformerEncoderDecoder,
    TransformerLm,
    StackedTransformer,
    StackedTransformerRepeated,
    PipelinedTransformer,
    SSMTransformer,
    VisionTransformer,
    LanguageModel,
    LanguageModelContinuousBatching,
    LanguageModelDPO,
    SequenceModel,
    BertModel,
)


def test_transformer_feedforward():
    """test_transformer_feedforward docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    layer = TransformerFeedForward(input_dims=8, hidden_dims=16)

    w1 = np.random.normal(size=(8, 16)).astype(np.float32)
    w2 = np.random.normal(size=(16, 8)).astype(np.float32)

    out = layer(inputs, w1=w1, w2=w2)
    assert out.shape == (2, 4, 8)

    # default implicit params
    layer_def = TransformerFeedForward()
    out_def = layer_def(inputs)
    assert out_def.shape == (2, 4, 8)


def test_ff_wrappers_coverage():
    """test_ff_wrappers_coverage docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)

    for cls in [AdaptedTransformerFeedForward, TransformerFeedForwardMoe]:
        layer = cls()
        out = layer(inputs)
        assert out.shape == (2, 4, 8)


def test_transformer_block():
    """test_transformer_block docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    layer = Transformer()
    out = layer(inputs)
    assert out.shape == (2, 4, 8)


def test_stacked_transformer():
    """test_stacked_transformer docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)
    layer = StackedTransformer(num_layers=2)
    out = layer(inputs)
    assert out.shape == (2, 4, 8)

    layer_rep = StackedTransformerRepeated(num_layers=2)
    out_rep = layer_rep(inputs)
    assert out_rep.shape == (2, 4, 8)

    layer_zero = StackedTransformer(num_layers=0)  # fallbacks to 1
    out_zero = layer_zero(inputs)
    assert out_zero.shape == (2, 4, 8)


def test_transformer_model_wrappers():
    """test_transformer_model_wrappers docstring."""
    inputs = np.random.normal(size=(2, 4, 8)).astype(np.float32)

    for cls in [
        TransformerEncoderDecoder,
        TransformerLm,
        PipelinedTransformer,
        SSMTransformer,
        VisionTransformer,
        LanguageModel,
        LanguageModelContinuousBatching,
        LanguageModelDPO,
        SequenceModel,
        BertModel,
    ]:
        layer = cls()
        out = layer(inputs)
        assert out.shape == (2, 4, 8)
