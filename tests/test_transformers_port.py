"""Tests for Transformer and Language Model layers in zero_pax.praxis.layers.transformers."""

from __future__ import annotations

import numpy as np

from zero_pax.praxis.layers import (
    GELU,
    AdaptedTransformerFeedForward,
    BertModel,
    Bias,
    FullSoftmax,
    Identity,
    LanguageModel,
    LanguageModelContinuousBatching,
    LanguageModelDPO,
    LayerNorm,
    MultitaskResidualAdapter,
    PipelinedTransformer,
    ReLU,
    SequenceModel,
    StackedTransformer,
    StackedTransformerRepeated,
    Transformer,
    TransformerEncoderDecoder,
    TransformerFeedForward,
    TransformerFeedForwardMoe,
    TransformerLm,
)


def test_transformer_feed_forward_variants() -> None:
    """Verify `TransformerFeedForward`, `AdaptedTransformerFeedForward`, and `TransformerFeedForwardMoe`."""
    inputs = np.arange(1, 33, dtype=np.float32).reshape((2, 2, 8)) * 0.1
    paddings = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    w1 = np.arange(1, 129, dtype=np.float32).reshape((8, 16)) * 0.02
    w2 = np.arange(1, 129, dtype=np.float32).reshape((16, 8)) * 0.02

    ffn = TransformerFeedForward(
        input_dims=8,
        hidden_dims=16,
        activation_tpl=GELU(),
        use_gated_activation=True,
        apply_padding_first=True,
        norm_policy="primer_hybrid",
    )
    out_ffn = ffn(inputs, paddings=paddings, w1=w1, w1_gate=w1, w2=w2)
    assert out_ffn.shape == (2, 2, 8)
    np.testing.assert_allclose(out_ffn[0, 1], np.zeros((8,)), atol=1e-5)
    assert not np.allclose(out_ffn[0, 0], inputs[0, 0])

    ffn_post = TransformerFeedForward(
        input_dims=8,
        output_dims=4,
        hidden_dims=16,
        norm_policy="post",
        add_skip_connection=False,
        ln_tpl=LayerNorm(dim=4),
        relu_dropout_tpl=Identity(),
        residual_dropout_tpl=Identity(),
    )
    out_post = ffn_post(
        inputs, paddings=paddings, w1=w1, w2=np.arange(64, dtype=np.float32).reshape((16, 4)) * 0.1
    )
    assert out_post.shape == (2, 2, 4)
    np.testing.assert_allclose(out_post[0, 1], np.zeros((4,)), atol=1e-5)

    from zero_pax.types import WeightInit

    add_one = Bias(dims=8, bias_init=WeightInit(method="ones", scale=1.0))
    adapted_par = AdaptedTransformerFeedForward(
        input_dims=8,
        hidden_dims=16,
        adapter_tpl=add_one,
        mode="parallel",
    )
    np.testing.assert_allclose(adapted_par(inputs), inputs + 1.0, atol=1e-5)

    adapted_seq = AdaptedTransformerFeedForward(
        input_dims=8,
        hidden_dims=16,
        adapter_tpl=MultitaskResidualAdapter(input_dims=8, bottleneck_dims=2),
        mode="sequential",
    )
    np.testing.assert_allclose(adapted_seq(inputs), inputs, atol=1e-5)
    np.testing.assert_allclose(
        AdaptedTransformerFeedForward(input_dims=8)(inputs), inputs, atol=1e-5
    )

    moe = TransformerFeedForwardMoe(
        input_dims=8,
        hidden_dims=4,
        num_experts=2,
        gating_func="top1",
        gating_logit_cap=10.0,
        norm_policy="post",
    )
    gate_w = np.zeros((8, 2), dtype=np.float32)
    gate_w[:, 1] = 5.0
    wi_0 = np.zeros((2, 8, 4), dtype=np.float32)
    wi_0[1] = 0.5
    wo_0 = np.zeros((2, 4, 8), dtype=np.float32)
    wo_0[1] = np.arange(1, 33, dtype=np.float32).reshape((4, 8)) * 0.2
    out_moe = moe(inputs, paddings=paddings, gate_w=gate_w, wi_0=wi_0, wo_0=wo_0)
    assert out_moe.shape == (2, 2, 8)
    np.testing.assert_allclose(out_moe[0, 1], np.zeros((8,)), atol=1e-5)
    assert not np.allclose(out_moe[1, 0], LayerNorm(dim=8)(inputs)[1, 0])

    assert TransformerFeedForwardMoe(input_dims=8, hidden_dims=16)(inputs).shape == (2, 2, 8)


def test_transformer_and_stacks() -> None:
    """Verify `Transformer`, `StackedTransformer`, `StackedTransformerRepeated`, and `PipelinedTransformer`."""
    inputs = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 8)) * 0.1
    cross = np.ones((2, 3, 8), dtype=np.float32) * 0.5
    paddings = np.zeros((2, 4), dtype=np.float32)

    tr = Transformer(
        model_dims=8,
        hidden_dims=16,
        num_heads=2,
        use_cross_attention=True,
        mask_self_attention=True,
        norm_policy="pre",
        ngrammer_tpl=ReLU(),
    )
    out = tr(inputs, paddings=paddings, cross_inputs=cross)
    np.testing.assert_allclose(out, inputs, atol=1e-5)

    tr_post = Transformer(
        model_dims=8,
        hidden_dims=16,
        num_heads=2,
        use_cross_attention=True,
        norm_policy="post",
    )
    out_tr_post = tr_post(inputs, cross_inputs=cross)
    np.testing.assert_allclose(np.mean(out_tr_post, axis=-1), np.zeros((2, 4)), atol=1e-5)

    stack = StackedTransformer(
        num_layers=2,
        model_dims=8,
        hidden_dims=16,
        num_heads=2,
        use_cross_attention=True,
        moe_layers=[1],
        num_experts=2,
        ngrammer_tpls=[Identity(), Identity()],
    )
    np.testing.assert_allclose(
        stack(inputs, paddings=paddings, cross_inputs=cross), inputs, atol=1e-5
    )

    stack_seq_tpl = StackedTransformer(
        num_layers=2,
        transformer_layer_params_tpl=[ReLU(), Identity()],
    )
    np.testing.assert_allclose(stack_seq_tpl(inputs), np.maximum(0.0, inputs))

    rep = StackedTransformerRepeated(block=stack, x_times=2)
    np.testing.assert_allclose(
        rep(inputs, paddings=paddings, cross_inputs=cross), inputs, atol=1e-5
    )
    np.testing.assert_allclose(StackedTransformerRepeated()(inputs), inputs, atol=1e-5)

    pipe = PipelinedTransformer(pipeline_stage=stack, num_pipeline_stages=2, circular_repeat=1)
    np.testing.assert_allclose(pipe(inputs, paddings=paddings), inputs, atol=1e-5)
    np.testing.assert_allclose(PipelinedTransformer()(inputs), inputs, atol=1e-5)


def test_language_and_sequence_models() -> None:
    """Verify `TransformerLm`, `TransformerEncoderDecoder`, `LanguageModel`, `LanguageModelDPO`, `SequenceModel`, and `BertModel`."""
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    inputs = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 8)) * 0.1
    paddings = np.zeros((2, 4), dtype=np.float32)

    lm = TransformerLm(model_dims=8, vocab_size=10, ngrammer_tpl=Identity())
    assert lm(token_ids, paddings=paddings).shape == (2, 4, 10)
    assert lm(inputs, paddings=paddings).shape == (2, 4, 10)

    lm_sep = TransformerLm(
        model_dims=8,
        vocab_size=10,
        separate_embedding_tpl=lambda ids, *a, **kw: np.ones((2, 4, 8), dtype=np.float32),
        softmax_tpl=FullSoftmax(input_dims=8, num_classes=10),
    )
    probs_sep = lm_sep(token_ids)
    np.testing.assert_allclose(np.sum(probs_sep, axis=-1), np.ones((2, 4)), atol=1e-5)

    lm_skip = TransformerLm(model_dims=8, vocab_size=10, skip_compute_loss=True)
    assert lm_skip(inputs).shape == (2, 4, 8)
    assert TransformerLm()(inputs).shape == (2, 4, 8)

    enc_dec = TransformerEncoderDecoder(model_dims=8, softmax_tpl=ReLU())
    assert enc_dec(inputs, targets=inputs).shape == (2, 4, 8)
    assert TransformerEncoderDecoder(model_dims=8)(inputs).shape == (2, 4, 8)

    assert LanguageModel()(inputs).shape == (2, 4, 8)
    assert LanguageModel(lm_tpl=ReLU())(inputs).shape == (2, 4, 8)
    assert LanguageModelContinuousBatching()(inputs).shape == (2, 4, 8)

    dpo_sig = LanguageModelDPO(
        mdl=Identity(),
        ref_mdl=lambda x, *a, **kw: x * 0.5,
        beta=0.5,
        loss_type="sigmoid",
        label_smoothing=0.1,
    )
    np.testing.assert_allclose(dpo_sig(inputs), inputs)
    chosen = np.ones((2, 4, 8), dtype=np.float32) * 2.0
    rejected = np.zeros((2, 4, 8), dtype=np.float32)
    loss_sig = dpo_sig(chosen, rejected_inputs=rejected)
    assert loss_sig.shape == (2, 4) and np.all(np.asarray(loss_sig) > 0.0)

    dpo_ipo = LanguageModelDPO(mdl=Identity(), use_ref_model=False, beta=0.5, loss_type="ipo")
    loss_ipo = dpo_ipo(chosen, rejected_inputs=rejected)
    np.testing.assert_allclose(loss_ipo, np.zeros((2, 4)), atol=1e-5)

    dpo_hinge = LanguageModelDPO(mdl=Identity(), use_ref_model=False, beta=0.2, loss_type="hinge")
    loss_hinge = dpo_hinge(chosen, rejected_inputs=rejected)
    np.testing.assert_allclose(loss_hinge, np.full((2, 4), 0.6), atol=1e-5)

    assert SequenceModel()(inputs).shape == (2, 4, 8)
    assert SequenceModel(model_tpl=Identity())(inputs).shape == (2, 4, 8)
    assert BertModel()(inputs).shape == (2, 4, 8)
    assert BertModel(lm_tpl=Identity())(inputs).shape == (2, 4, 8)
