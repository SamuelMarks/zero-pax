"""Tests for Transformer and Language Model layers in zero_pax.praxis.layers.transformers."""

from __future__ import annotations

from tests import conftest as np
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
    MaskedLmDataAugmenter,
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
from zero_pax.types import DecoderHParams, WeightInit


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
    gate_w = np.concatenate([np.zeros((8, 1)), np.full((8, 1), 5.0)], axis=1)
    wi_0 = np.concatenate([np.zeros((1, 8, 4)), np.full((1, 8, 4), 0.5)], axis=0)
    wo_0 = np.concatenate(
        [np.zeros((1, 4, 8)), np.arange(1, 33, dtype=np.float32).reshape((1, 4, 8)) * 0.2],
        axis=0,
    )
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

    stack_active = StackedTransformer(
        num_layers=2,
        model_dims=8,
        hidden_dims=16,
        num_heads=2,
        use_cross_attention=True,
        params_init=WeightInit(method="xavier", scale=1.0),
    )
    out_stack_active = stack_active(inputs, paddings=paddings, cross_inputs=cross)
    assert out_stack_active.shape == (2, 4, 8)
    np.testing.assert_not_allclose(out_stack_active, inputs)

    stack_seq_tpl = StackedTransformer(
        num_layers=2,
        transformer_layer_params_tpl=[ReLU(), Identity()],
    )
    np.testing.assert_allclose(stack_seq_tpl(inputs), np.maximum(0.0, inputs))

    rep = StackedTransformerRepeated(block=stack, x_times=2)
    np.testing.assert_allclose(
        rep(inputs, paddings=paddings, cross_inputs=cross), inputs, atol=1e-5
    )
    rep_active = StackedTransformerRepeated(block=stack_active, x_times=2)
    np.testing.assert_not_allclose(
        rep_active(inputs, paddings=paddings, cross_inputs=cross), inputs
    )
    np.testing.assert_allclose(StackedTransformerRepeated()(inputs), inputs, atol=1e-5)

    pipe = PipelinedTransformer(pipeline_stage=stack, num_pipeline_stages=2, circular_repeat=1)
    np.testing.assert_allclose(pipe(inputs, paddings=paddings), inputs, atol=1e-5)
    pipe_active = PipelinedTransformer(
        pipeline_stage=stack_active, num_pipeline_stages=2, circular_repeat=1
    )
    np.testing.assert_not_allclose(pipe_active(inputs, paddings=paddings), inputs)
    np.testing.assert_allclose(PipelinedTransformer()(inputs), inputs, atol=1e-5)


def test_language_and_sequence_models() -> None:
    """Verify `TransformerLm`, `TransformerEncoderDecoder`, `LanguageModel`, `LanguageModelDPO`, `SequenceModel`, and `BertModel`."""
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    inputs = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 8)) * 0.1
    paddings = np.zeros((2, 4), dtype=np.float32)

    lm = TransformerLm(model_dims=8, vocab_size=10, ngrammer_tpl=Identity())
    out_lm_ids = lm(token_ids, paddings=paddings)
    assert out_lm_ids.shape == (2, 4, 10)
    np.testing.assert_allclose(out_lm_ids, np.zeros((2, 4, 10)), atol=1e-5)
    out_lm_inp = lm(inputs, paddings=paddings)
    assert out_lm_inp.shape == (2, 4, 10)

    lm_sep = TransformerLm(
        model_dims=8,
        vocab_size=10,
        separate_embedding_tpl=lambda ids, *a, **kw: np.ones((2, 4, 8), dtype=np.float32),
        softmax_tpl=FullSoftmax(input_dims=8, num_classes=10),
    )
    probs_sep = lm_sep(token_ids)
    np.testing.assert_allclose(np.sum(probs_sep, axis=-1), np.ones((2, 4)), atol=1e-5)

    lm_skip = TransformerLm(model_dims=8, vocab_size=10, skip_compute_loss=True)
    out_skip = lm_skip(inputs)
    assert out_skip.shape == (2, 4, 8)
    np.testing.assert_allclose(np.mean(out_skip, axis=-1), np.zeros((2, 4)), atol=1e-5)
    assert TransformerLm()(inputs).shape == (2, 4, 8)

    enc_dec = TransformerEncoderDecoder(model_dims=8, softmax_tpl=ReLU())
    out_ed = enc_dec(inputs, targets=inputs)
    assert out_ed.shape == (2, 4, 8)
    assert np.all(np.asarray(out_ed) >= 0.0)
    assert TransformerEncoderDecoder(model_dims=8)(inputs).shape == (2, 4, 8)

    assert LanguageModel()(inputs).shape == (2, 4, 8)
    np.testing.assert_allclose(
        LanguageModel(lm_tpl=ReLU())(inputs - 5.0), np.maximum(0.0, inputs - 5.0)
    )
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
    np.testing.assert_allclose(SequenceModel(model_tpl=Identity())(inputs), inputs)
    assert BertModel()(inputs).shape == (2, 4, 8)
    np.testing.assert_allclose(BertModel(lm_tpl=Identity())(inputs), inputs)

    slot_masked = LanguageModelContinuousBatching(lm_tpl=Identity())(
        inputs, slot_mask=np.array([1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(slot_masked[0], inputs[0])
    np.testing.assert_allclose(slot_masked[1], np.zeros((4, 8)))

    x_tr = np.arange(1, 25, dtype=np.float32).reshape((2, 3, 4)) * 0.1
    ffn_custom_ln = TransformerFeedForward(
        input_dims=4,
        norm_policy="post",
        ln_tpl=Identity(),
        fflayer_tpl=Identity(),
    )
    np.testing.assert_allclose(ffn_custom_ln(x_tr), x_tr * 2.0, atol=1e-5)

    ffn_gated_custom = TransformerFeedForward(
        input_dims=4,
        use_gated_activation=True,
        ffn0_tpl=Identity(),
        ffn1_tpl=Identity(),
        ffn2_tpl=Identity(),
        ln_tpl=Identity(),
    )
    np.testing.assert_allclose(ffn_gated_custom(x_tr), x_tr + np.square(x_tr), atol=1e-5)

    moe_top2 = TransformerFeedForwardMoe(
        input_dims=4,
        hidden_dims=4,
        num_experts=3,
        gating_func="top2",
        expert_capacity_dim=2,
        norm_policy="pre",
        add_skip_connection=False,
        use_gated_activation=True,
    )
    out_moe_top2 = moe_top2(
        x_tr,
        gate_w=np.ones((4, 3), dtype=np.float32),
        wi_0=np.ones((3, 4, 4), dtype=np.float32),
        wo_0=np.ones((3, 4, 4), dtype=np.float32),
    )
    assert out_moe_top2.shape == (2, 3, 4)

    moe_ec = TransformerFeedForwardMoe(
        input_dims=4,
        hidden_dims=4,
        num_experts=2,
        gating_func="expert_choice",
    )
    assert moe_ec(x_tr).shape == (2, 3, 4)

    moe_2d_w = TransformerFeedForwardMoe(input_dims=4, hidden_dims=4)
    np.testing.assert_allclose(
        moe_2d_w(
            x_tr,
            w1=np.zeros((4, 4), dtype=np.float32),
            w2=np.zeros((4, 4), dtype=np.float32),
        ),
        x_tr,
        atol=1e-5,
    )

    tr_custom = Transformer(
        model_dims=4,
        dim_per_head=2,
        use_cross_attention=True,
        tr_atten_tpl=Identity(),
        cross_atten_tpl=Identity(),
        tr_fflayer_tpl=Identity(),
        ln_tpl=Identity(),
    )
    np.testing.assert_allclose(tr_custom(x_tr, cross_inputs=x_tr), x_tr * 4.0, atol=1e-5)

    st_rep = StackedTransformerRepeated(sub_tpl=Identity(), x_times=1)
    np.testing.assert_allclose(st_rep(x_tr), x_tr)
    np.testing.assert_allclose(PipelinedTransformer(pipeline_stage=Identity())(x_tr), x_tr)

    lm_prefix = TransformerLm(
        model_dims=4,
        vocab_size=4,
        model_type="prefix",
        position_emb_tpl=Identity(),
        stacked_transformer_tpl=Identity(),
        softmax_tpl=Identity(),
        final_ln_tpl=Identity(),
    )
    np.testing.assert_allclose(
        lm_prefix(np.array([[0, 1, 2, 3]], dtype=np.int32)), np.zeros((1, 4, 4))
    )

    enc_dec_custom = TransformerEncoderDecoder(
        model_dims=4,
        encoder_embedding_tpl=Identity(),
        decoder_embedding_tpl=Identity(),
        encoder_ngrammer_tpl=Identity(),
        decoder_ngrammer_tpl=Identity(),
        position_emb_tpl=Identity(),
        encoder_stacked_transformer_tpl=Identity(),
        decoder_stacked_transformer_tpl=Identity(),
        encoder_ln_tpl=Identity(),
        decoder_ln_tpl=Identity(),
    )
    np.testing.assert_allclose(enc_dec_custom(x_tr), x_tr * 2.0, atol=1e-5)

    dpo_plain = LanguageModelDPO(mdl=Identity(), use_ref_model=False, label_smoothing=0.0)
    loss_dpo = dpo_plain(x_tr, rejected_inputs=x_tr * 0.5)
    assert loss_dpo.shape == (2, 3) and np.all(np.asarray(loss_dpo) > 0.0)

    bert_aug = BertModel(
        lm_tpl=Identity(),
        mlm_augmenter_tpl=MaskedLmDataAugmenter(vocab_size=10, mask_token_id=9, mask_prob=0.0),
        label_smoothing_prob=0.1,
        nsp_weight=0.5,
    )
    np.testing.assert_allclose(bert_aug(x_tr), x_tr)
    assert "bert_loss_meta" in bert_aug.params
    bert_force = BertModel(lm_tpl=Identity(), force_mask_generation=True, mask_token_id=9)
    assert bert_force(np.array([[1, 2, 3]], dtype=np.int32)).shape == (1, 3)

    ffn_drops = TransformerFeedForward(
        input_dims=4,
        hidden_dims=4,
        relu_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        residual_droppath_prob=0.1,
        internal_gshard_variance_scaling_fan_in_init=True,
        params_init=WeightInit(method="ones", scale=0.25),
    )
    assert ffn_drops(x_tr).shape == (2, 3, 4)
    assert "gshard_fan_in_init" in ffn_drops.params

    moe_full = TransformerFeedForwardMoe(
        input_dims=4,
        hidden_dims=4,
        num_experts=2,
        num_groups=2,
        apply_padding_first=True,
        relu_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        residual_droppath_prob=0.1,
        moe_load_balance_loss_weight=0.5,
        params_init=WeightInit(method="ones", scale=0.25),
    )
    assert moe_full(x_tr, paddings=np.zeros((2, 3), dtype=np.float32)).shape == (2, 3, 4)
    assert "aux_loss" in moe_full.params
    assert "moe_routing_meta" in moe_full.params
    moe_no_aux = TransformerFeedForwardMoe(
        input_dims=4, hidden_dims=4, num_experts=2, moe_load_balance_loss_weight=None
    )
    assert moe_no_aux(x_tr).shape == (2, 3, 4)

    tr_meta = Transformer(
        model_dims=4,
        num_heads=2,
        allow_at_least_one_headed_attention=True,
        packed_input=True,
        allow_skip_cross_attention=True,
    )
    assert tr_meta(x_tr).shape == (2, 3, 4)
    assert "transformer_meta" in tr_meta.params

    st_drop = StackedTransformer(
        num_layers=1,
        model_dims=4,
        num_heads=2,
        input_dropout_prob=0.1,
        enable_while_loop=False,
        packed_input=True,
    )
    assert st_drop(x_tr).shape == (2, 3, 4)
    assert "stack_meta" in st_drop.params

    st_rep_inter = StackedTransformerRepeated(
        sub_tpl=Identity(),
        x_times=0,
        nd_prefix_shape=(2,),
        unroll_in_decode=False,
        collect_intermediate_outputs=True,
        return_intermediate_outputs=True,
    )
    out_rep_inter = st_rep_inter(x_tr)
    assert out_rep_inter.shape == (2, 2, 3, 4)
    st_rep_collect_only = StackedTransformerRepeated(
        sub_tpl=Identity(),
        x_times=1,
        collect_intermediate_outputs=True,
        return_intermediate_outputs=False,
    )
    assert st_rep_collect_only(x_tr).shape == (2, 3, 4)

    pipe_mb = PipelinedTransformer(
        pipeline_stage=Identity(),
        num_pipeline_stages=2,
        num_pipeline_microbatches=2,
        stream_io=True,
    )
    assert pipe_mb(x_tr).shape == (2, 3, 4)
    assert "pipeline_tr_meta" in pipe_mb.params

    pipe_mb_sz = PipelinedTransformer(
        pipeline_stage=StackedTransformer(num_layers=1, model_dims=4),
        pipeline_microbatch_size=1,
    )
    assert pipe_mb_sz(x_tr).shape == (2, 3, 4)

    lm_opts = TransformerLm(
        model_dims=4,
        vocab_size=4,
        packed_input=True,
        skip_aux_loss=True,
        record_activations_in_xent_output=True,
        entropy_loss_weight=0.1,
        params_init=WeightInit(method="ones", scale=0.25),
    )
    loss_lm_opts = lm_opts(
        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        labels=np.array([[1, 2, 3], [2, 3, 0]], dtype=np.int32),
    )
    assert loss_lm_opts.shape == (2, 3)
    assert "activations" in lm_opts.params

    ed_packed = TransformerEncoderDecoder(model_dims=4, packed_input=True)
    assert ed_packed(x_tr).shape == (2, 3, 4)
    assert "enc_dec_packed" in ed_packed.params

    lm_task = LanguageModel(
        lm_tpl=Identity(),
        return_predictions=True,
        count_tokens=True,
        decoder_tpl=DecoderHParams(),
        label_smoothing_prob=0.1,
        z_loss_weight=0.01,
    )
    np.testing.assert_allclose(lm_task(x_tr), x_tr)
    assert "predictions" in lm_task.params and "num_tokens" in lm_task.params

    dpo_tc = LanguageModelDPO(
        mdl=Identity(),
        token_counter=Identity(),
        apply_eval_sample_weights=True,
    )
    np.testing.assert_allclose(dpo_tc(x_tr), x_tr)
    assert "dpo_eval_weights" in dpo_tc.params

    seq_task = SequenceModel(
        model_tpl=Identity(),
        return_predictions=True,
        decoder_tpl=DecoderHParams(),
        label_smoothing_prob=0.1,
    )
    np.testing.assert_allclose(seq_task(x_tr), x_tr)
    assert "predictions" in seq_task.params and "seq_model_meta" in seq_task.params
