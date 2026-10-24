"""Additional tests to ensure 100% function, line, and branch coverage across src/zero_pax/."""

from __future__ import annotations

import numpy as np

from zero_pax import _tensor_ops as t_ops
from zero_pax.praxis.base import BasePraxisLayer, instantiate_layer
from zero_pax.praxis.layers import (
    SSM,
    AttentionProjection,
    BatchNorm,
    BertModel,
    BregmanPCA,
    CausalDepthwiseConv1D,
    ClassificationMLPModel,
    ClassificationModel,
    Conformer,
    ConvBNAct,
    DepthwiseConv1D,
    DotProductAttention,
    DotProductAttentionWithContext,
    DotProductAttentionXL,
    Einsum,
    FeedForward,
    FRnn,
    FullSoftmax,
    GlobalPooling,
    GroupedQueryAttention,
    GroupNorm,
    GShardSharedEmbeddingSoftmax,
    Identity,
    LanguageModel,
    LanguageModelDPO,
    LayerNormalizedLstmCellSimple,
    LightConv1D,
    LocalSelfAttention,
    LocalSelfAttentionAlibi,
    LocalSelfAttentionRelativeBias,
    LstmCellSimple,
    LstmFrnn,
    MaskedLmDataAugmenter,
    MLPBlock,
    PipelinedTransformer,
    Pooling,
    Pooling1D,
    PositionalEmbedding,
    PositionalEmbedding2D,
    RandomVectorQuantizer,
    RelativeBias,
    ReLU,
    Repeat,
    ResNet,
    ResNetBlock,
    SelfAttentionWithNormAndResidual,
    SharedEmbeddingSoftmax,
    SigmoidCrossEntropy,
    SpectrumAugmenter,
    SSMGated,
    SSMTransformer,
    StackedTransformerRepeated,
    StackFrnn,
    StackingOverTime,
    StochasticResidual,
    TemporalShifting,
    TrainablePositionalEmbedding,
    Transformer,
    TransformerEncoderDecoder,
    TransformerFeedForward,
    TransformerFeedForwardMoe,
    TransformerLm,
    VanillaNet,
    VectorQuantizer,
    VitEntryLayers,
    VitExitLayers,
    VQNgrammer,
)
from zero_pax.praxis.layers.attentions import _apply_rotary_position_emb
from zero_pax.types import PaxConfig


def test_tensor_ops_and_base_branches() -> None:
    """Cover remaining branches in `_tensor_ops.py`, `base.py`, `normalizations.py`, `linear_pooling.py`, and `convolutions.py`."""
    t = t_ops.asarray([1.0, 2.0])
    assert t_ops.asarray(t) is t
    assert t_ops.transpose([[1.0, 2.0], [3.0, 4.0]], axes=(1, 0)).shape == (2, 2)
    assert t_ops.pad([[1.0, 2.0]], ((0, 0), (1, 1)), mode="edge").shape == (1, 4)
    assert t_ops.argmin([3.0, 1.0, 2.0]).shape == ()
    assert t_ops.argmax([3.0, 1.0, 2.0]).shape == ()

    class NonCallableModel(BasePraxisLayer):
        """Test subclass of BasePraxisLayer."""

    cfg_base = PaxConfig(cls=NonCallableModel)
    assert isinstance(instantiate_layer(cfg_base, Identity()), NonCallableModel)

    class DummyObj:
        """Non-callable test object."""

    cfg_dummy: PaxConfig[BasePraxisLayer] = PaxConfig(cls=DummyObj)  # type: ignore[arg-type]
    assert isinstance(instantiate_layer(cfg_dummy, Identity()), Identity)
    assert isinstance(instantiate_layer(DummyObj(), Identity()), Identity)  # type: ignore[arg-type]

    ein_bias = Einsum(eqn="bd,df->bf", use_bias=True)
    assert ein_bias(np.ones((2, 3), dtype=np.float32)).shape == (2, 3)
    ein_no_bias = Einsum(eqn="bd,df->bf", use_bias=False)
    assert ein_no_bias(np.ones((2, 3), dtype=np.float32)).shape == (2, 3)

    x_4d = np.ones((2, 4, 2, 4), dtype=np.float32)
    pad_2d = np.zeros((2, 4), dtype=np.float32)
    bn_no_zero = BatchNorm(dim=4, set_padded_output_to_zero=False)
    assert bn_no_zero(x_4d, paddings=pad_2d).shape == (2, 4, 2, 4)

    gn_cum = GroupNorm(
        num_groups=2,
        cumulative=True,
        set_padded_output_to_zero=False,
        use_scale=False,
        use_bias=False,
    )
    assert gn_cum(x_4d, paddings=pad_2d).shape == (2, 4, 2, 4)
    assert GroupNorm(num_groups=2)(np.ones((2, 4, 4), dtype=np.float32), paddings=pad_2d).shape == (
        2,
        4,
        4,
    )

    ff_custom = FeedForward(linear_tpl=Identity(), bias_tpl=Identity())
    assert ff_custom(np.ones((2, 4), dtype=np.float32)).shape == (2, 4)

    pool_max_pad = Pooling(
        window_shape=(2, 2), window_stride=(1, 1), pooling_type="MAX", padding="VALID"
    )
    res_pool = pool_max_pad(x_4d, paddings=pad_2d)
    assert isinstance(res_pool, tuple) and res_pool[0].shape == (2, 3, 1, 4)

    pool_same_strided_pad = Pooling(
        window_shape=(3, 3), window_stride=(2, 2), pooling_type="AVG", padding="SAME"
    )
    res_same_strided = pool_same_strided_pad(x_4d, paddings=pad_2d)
    assert isinstance(res_same_strided, tuple) and res_same_strided[0].shape == (2, 2, 1, 4)

    pool_stride_no_pad = Pooling(
        window_shape=(1, 1), window_stride=(2, 1), pooling_type="AVG", padding="SAME"
    )
    res_strided = pool_stride_no_pad(x_4d, paddings=pad_2d)
    assert isinstance(res_strided, tuple) and res_strided[0].shape == (2, 2, 2, 4)

    p1d_valid = Pooling1D(window=2, stride=1, pooling_type="MAX", padding="VALID")
    out_p1d_valid = p1d_valid(np.ones((2, 4, 4), dtype=np.float32))
    assert not isinstance(out_p1d_valid, tuple) and out_p1d_valid.shape == (2, 3, 4)

    gp_all_pad = GlobalPooling(pooling_type="AVG", pooling_dims=[1])
    out_all_pad = gp_all_pad(
        np.ones((1, 3, 2), dtype=np.float32),
        compatible_paddings=np.ones((1, 3, 1), dtype=np.float32),
    )
    np.testing.assert_allclose(out_all_pad, np.zeros((1, 2)))

    conv_bn_custom = ConvBNAct(
        filter_shape=(3, 3, 4, 4), batch_norm_tpl=Identity(), activation_tpl=ReLU()
    )
    assert conv_bn_custom(x_4d).shape == (2, 4, 2, 4)

    dconv_reshape = DepthwiseConv1D(filter_shape=(3, 1, 4))
    assert dconv_reshape(
        np.ones((2, 4, 4), dtype=np.float32), w=np.ones((3, 4), dtype=np.float32)
    ).shape == (2, 4, 4)

    cdconv_default = CausalDepthwiseConv1D(filter_shape=(1, 1, 4))
    assert cdconv_default(np.ones((2, 4, 4), dtype=np.float32)).shape == (2, 4, 4)

    lconv_custom = LightConv1D(
        input_dims=4,
        ln_tpl=Identity(),
        linear_start_tpl=lambda x, *a, **kw: np.concatenate([x, x], axis=-1),
        depthwise_conv_tpl=Identity(),
        conv_norm_layer_tpl=Identity(),
        linear_end_tpl=Identity(),
    )
    assert lconv_custom(np.ones((2, 4, 4), dtype=np.float32)).shape == (2, 4, 4)


def test_attentions_and_embeddings_branches() -> None:
    """Cover remaining branches in `attentions.py` and `embeddings.py`."""
    rope_tiny = _apply_rotary_position_emb(np.ones((1, 2, 1, 1), dtype=np.float32))
    assert rope_tiny.shape == (1, 2, 1, 1)
    rope_odd = _apply_rotary_position_emb(np.ones((1, 2, 1, 3), dtype=np.float32))
    assert rope_odd.shape == (1, 2, 1, 3)

    ap_out = AttentionProjection(
        input_dim=8, num_heads=1, dim_per_head=4, is_output_projection=True, use_bias=False
    )
    assert ap_out(
        np.ones((2, 3, 2, 4), dtype=np.float32), w=np.ones((8, 8), dtype=np.float32)
    ).shape == (2, 3, 8)

    ap_in = AttentionProjection(
        input_dim=8, num_heads=2, dim_per_head=4, is_output_projection=False
    )
    assert ap_in(
        np.ones((2, 3, 8), dtype=np.float32),
        w=np.ones((8, 8), dtype=np.float32),
        b=np.ones((8,), dtype=np.float32),
    ).shape == (2, 3, 2, 4)

    rb = RelativeBias(num_heads=2)
    assert rb(np.array([0, 1, 2])).shape == (1, 2, 3, 3)
    assert rb(np.array([0, 1, 2]), 3).shape == (1, 2, 3, 3)
    assert rb(3, np.array([0, 1, 2])).shape == (1, 2, 3, 3)

    dpa_no_scale = DotProductAttention(hidden_dim=8, num_heads=2, internal_enable_query_scale=False)
    assert dpa_no_scale(
        np.ones((2, 3, 8), dtype=np.float32),
        atten_mask=np.zeros((2, 2, 3, 3), dtype=np.float32),
    ).shape == (2, 3, 8)

    assert DotProductAttentionXL(rel_pos_emb_dim=0)(np.ones((2, 3, 8), dtype=np.float32)).shape == (
        2,
        3,
        8,
    )
    assert DotProductAttentionWithContext()(np.ones((2, 3, 8), dtype=np.float32)).shape == (2, 3, 8)
    assert DotProductAttentionWithContext(left_context=None, right_context=1)(
        np.ones((2, 3, 8), dtype=np.float32)
    ).shape == (2, 3, 8)
    assert LocalSelfAttention(block_size=0)(np.ones((2, 3, 8), dtype=np.float32)).shape == (2, 3, 8)
    assert LocalSelfAttentionRelativeBias(num_heads=2, relative_bias_tpl=RelativeBias(num_heads=2))(
        np.ones((2, 3, 8), dtype=np.float32)
    ).shape == (2, 3, 8)
    assert LocalSelfAttentionAlibi(num_heads=2)(np.ones((2, 3, 8), dtype=np.float32)).shape == (
        2,
        3,
        8,
    )

    gqa_no_bias = GroupedQueryAttention(
        input_dim=8, num_heads=2, num_kv_heads=2, dim_per_head=4, use_bias=False
    )
    assert gqa_no_bias(np.ones((2, 3, 8), dtype=np.float32)).shape == (2, 3, 8)

    sanr = SelfAttentionWithNormAndResidual(
        num_heads=2,
        dim_per_head=4,
        self_atten_tpl=DotProductAttention(input_dim=8, num_heads=2, dim_per_head=4),
    )
    assert sanr(np.ones((2, 3, 8), dtype=np.float32)).shape == (2, 3, 8)

    shared = SharedEmbeddingSoftmax(num_classes=5, input_dims=4)
    assert shared.emb_lookup(
        np.array([[0, 1]], dtype=np.int32), w=np.ones((5, 4), dtype=np.float32)
    ).shape == (1, 2, 4)

    gshard = GShardSharedEmbeddingSoftmax(
        num_classes=5, input_dims=4, logits_abs_max=0.0, soft_cap_logits=0.0
    )
    assert gshard.emb_lookup(
        np.array([[0, 1]], dtype=np.int32), w=np.ones((4, 5), dtype=np.float32)
    ).shape == (1, 2, 4)
    assert gshard(np.ones((1, 2, 4), dtype=np.float32)).shape == (1, 2, 5)

    pe = PositionalEmbedding(embedding_dims=4)
    assert pe(seq_length=np.array(3)).shape == (1, 3, 4)
    assert pe(seq_length=np.zeros((3,))).shape == (1, 3, 4)
    assert pe(position=np.array([0, 1, 2])).shape == (1, 3, 4)

    assert PositionalEmbedding2D(h=2, w=2, embedding_dims=2)().shape == (1, 4, 2)
    assert TrainablePositionalEmbedding(max_seq_length=4, embedding_dims=4)(
        position=np.array([[0, 1]])
    ).shape == (1, 2, 4)

    sm_ff = FullSoftmax(
        input_dims=4,
        num_classes=3,
        scale_before_logits=0.5,
        feed_forward_tpl=FeedForward(input_dims=4, output_dims=3),
    )
    assert sm_ff(
        np.ones((2, 4), dtype=np.float32),
        class_probabilities=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        class_weights=np.ones((2,), dtype=np.float32),
    ).shape == (2,)

    sm_proj_b = FullSoftmax(input_dims=4, num_classes=3, use_bias=True)
    assert sm_proj_b(
        np.ones((2, 4), dtype=np.float32),
        class_ids=np.array([0, 1], dtype=np.int32),
        b=np.ones((3,), dtype=np.float32),
    ).shape == (2,)

    sce_proj = SigmoidCrossEntropy(input_dims=4, num_classes=3, soft_cap_logits=0.0)
    assert sce_proj(np.ones((2, 4), dtype=np.float32)).shape == (2, 3)
    assert sce_proj(
        np.ones((2, 4), dtype=np.float32),
        labels=np.zeros((2, 3), dtype=np.float32),
    ).shape == (2, 3)


def test_utilities_recurrent_transformers_and_arch_branches() -> None:
    """Cover remaining branches in `quantization_utils.py`, `recurrent_ssm.py`, `transformers.py`, and `architectures.py`."""
    assert VectorQuantizer(num_latent_classes=1)(np.ones((1, 2, 2), dtype=np.float32))[0].shape == (
        1,
        2,
        2,
    )

    rvq = RandomVectorQuantizer(
        stack_ratio=1,
        stacking_tpl=StackingOverTime(stride=2, left_context=1),
        projection_dim=4,
        num_latent_classes=2,
    )
    q_rvq, _, _ = rvq(
        np.ones((1, 4, 2), dtype=np.float32),
        proj_w=np.ones((4, 4), dtype=np.float32),
        codebook=np.ones((2, 4), dtype=np.float32),
    )
    assert q_rvq.shape == (1, 2, 4)

    vq_ng = VQNgrammer(num_clusters=1)
    assert vq_ng(np.ones((1, 3, 4), dtype=np.float32)).shape == (1, 3, 4)

    pca_norm = BregmanPCA(num_components=2, input_dims=4, normalize_weight=False)
    assert pca_norm(
        np.ones((1, 4), dtype=np.float32), components=np.ones((2, 4), dtype=np.float32)
    )[0].shape == (1, 4)

    sr_no_mask = StochasticResidual(residual_weight=0.5, survival_prob=0.5)
    assert sr_no_mask(
        np.ones((1, 2), dtype=np.float32), residual=np.ones((1, 2), dtype=np.float32)
    ).shape == (1, 2)

    ts_neg = TemporalShifting(shift=1, axis=-1)
    assert ts_neg(np.ones((2, 3), dtype=np.float32)).shape == (2, 3)

    st_strided_pad = StackingOverTime(left_context=0, right_context=0, stride=2)
    res_st_sp = st_strided_pad(
        np.ones((1, 4, 2), dtype=np.float32),
        paddings=np.zeros((1, 4), dtype=np.float32),
    )
    assert isinstance(res_st_sp, tuple) and res_st_sp[0].shape == (1, 2, 2)

    mlm_no_pad = MaskedLmDataAugmenter(vocab_size=10, mask_token_id=9)
    assert mlm_no_pad(
        np.ones((1, 3), dtype=np.float32),
        mask_positions=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
    )[0].shape == (1, 3)

    spec_4d = SpectrumAugmenter(freq_mask_max_bins=0, time_mask_max_frames=0, augment_at_eval=True)
    assert spec_4d(np.ones((1, 4, 4, 2), dtype=np.float32)).shape == (1, 4, 4, 2)
    assert spec_4d(np.ones((4,), dtype=np.float32)).shape == (4,)
    spec_small = SpectrumAugmenter(
        freq_mask_max_bins=10, time_mask_max_frames=10, augment_at_eval=True
    )
    assert spec_small(np.ones((1, 1, 1), dtype=np.float32)).shape == (1, 1, 1)

    rep_sub = Repeat(sub_layer=ReLU(), x_times=0, num_repeats=1)
    assert rep_sub(np.ones((1, 2), dtype=np.float32)).shape == (1, 2)

    lstm_tuple_inferred = LstmCellSimple(num_output_nodes=0, hidden_size=0)
    assert lstm_tuple_inferred(
        np.ones((2, 3), dtype=np.float32),
        state0=(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)),
    )[0].shape == (2, 3)

    lstm_no_bias = LstmCellSimple(enable_lstm_bias=False, cell_value_cap=None)
    assert lstm_no_bias(
        np.ones((2, 4), dtype=np.float32),
        state0=np.ones((2, 4), dtype=np.float32),
        padding=np.array([0.0, 1.0], dtype=np.float32),
    )[0].shape == (2, 4)

    lstm_same_size = LstmCellSimple(hidden_size=2)
    assert lstm_same_size(
        np.ones((2, 2), dtype=np.float32),
        state0=np.ones((2, 2), dtype=np.float32),
    )[0].shape == (2, 2)

    ln_lstm_no_bias = LayerNormalizedLstmCellSimple(enable_lstm_bias=False)
    assert ln_lstm_no_bias(np.ones((2, 12), dtype=np.float32))[0].shape == (2, 4)

    assert FRnn()(np.ones((2, 4), dtype=np.float32)).shape == (2, 4)
    assert FRnn(cell_tpl=LstmCellSimple(num_input_nodes=3, num_output_nodes=3), reverse=False)(
        np.ones((2, 3, 3), dtype=np.float32)
    ).shape == (2, 3, 3)
    assert LstmFrnn()(np.ones((2, 4), dtype=np.float32)).shape == (2, 4)
    assert StackFrnn(frnn_tpl=ReLU())(np.ones((2, 3, 3), dtype=np.float32)).shape == (2, 3, 3)
    assert SSM()(np.ones((2, 4), dtype=np.float32)).shape == (2, 4)
    ssm_custom = SSM(hidden_size=3)
    assert ssm_custom(
        np.ones((2, 3, 3), dtype=np.float32),
        a_diag=np.array([-1.0]),
        b_mat=np.ones((3, 1)),
        d_vec=np.eye(3),
    ).shape == (2, 3, 3)
    assert SSMGated(hidden_size=2, ssm_tpl=Identity(), gss_fflayer_tpl=Identity())(
        np.ones((2, 3, 4), dtype=np.float32)
    ).shape == (2, 3, 4)
    assert SSMTransformer()(np.ones((2, 3, 3), dtype=np.float32)).shape == (2, 3, 3)

    x_tr = np.ones((2, 3, 4), dtype=np.float32)
    ffn_custom_ln = TransformerFeedForward(input_dims=4, norm_policy="post", ln_tpl=Identity())
    assert ffn_custom_ln(x_tr).shape == (2, 3, 4)

    moe_pre = TransformerFeedForwardMoe(
        input_dims=4,
        hidden_dims=4,
        num_experts=1,
        gating_func="top2",
        norm_policy="pre",
        add_skip_connection=False,
    )
    assert moe_pre(
        x_tr,
        gate_w=np.ones((4, 1), dtype=np.float32),
        w1=np.ones((1, 4, 4), dtype=np.float32),
        w2=np.ones((1, 4, 4), dtype=np.float32),
    ).shape == (2, 3, 4)

    tr_custom = Transformer(
        model_dims=4,
        dim_per_head=2,
        use_cross_attention=True,
        tr_atten_tpl=Identity(),
        cross_atten_tpl=Identity(),
        tr_fflayer_tpl=Identity(),
    )
    assert tr_custom(x_tr, cross_inputs=x_tr).shape == (2, 3, 4)

    st_rep = StackedTransformerRepeated(sub_tpl=Identity(), x_times=1)
    assert st_rep(x_tr).shape == (2, 3, 4)
    assert PipelinedTransformer(pipeline_stage=Identity())(x_tr).shape == (2, 3, 4)

    lm_custom = TransformerLm(
        model_dims=4,
        vocab_size=4,
        position_emb_tpl=Identity(),
        stacked_transformer_tpl=Identity(),
        softmax_tpl=Identity(),
    )
    assert lm_custom(np.array([[0, 1, 2]], dtype=np.int32)).shape == (1, 3, 4)

    enc_dec_custom = TransformerEncoderDecoder(
        model_dims=4,
        position_emb_tpl=Identity(),
        encoder_stacked_transformer_tpl=Identity(),
        decoder_stacked_transformer_tpl=Identity(),
    )
    assert enc_dec_custom(x_tr).shape == (2, 3, 4)

    dpo_plain = LanguageModelDPO(mdl=Identity(), use_ref_model=False, label_smoothing=0.0)
    assert dpo_plain(x_tr, rejected_inputs=x_tr * 0.5).shape == (2, 3)

    assert BertModel(lm_tpl=TransformerLm(model_dims=4))(x_tr).shape == (2, 3, 4)
    assert LanguageModel(lm_tpl=TransformerLm(model_dims=4))(x_tr).shape == (2, 3, 4)

    vnet_no_stem = VanillaNet(strides=(1,), channels=(4,), blocks=(2,), kernels=(3,))
    assert vnet_no_stem(np.ones((1, 4, 4, 4), dtype=np.float32)).shape == (1, 4, 4, 4)

    rnet_no_stem = ResNet(
        strides=(1,), channels=(1,), blocks=(2,), kernels=(3,), entry_max_pool=False
    )
    assert rnet_no_stem(np.ones((1, 4, 4, 4), dtype=np.float32)).shape == (1, 4, 4, 4)

    rnet_small_pool = ResNet(
        strides=(1,),
        channels=(1,),
        blocks=(1,),
        kernels=(3,),
        entry_max_pool=True,
        entryflow_conv_params=lambda x, *a, **kw: x,
    )
    assert rnet_small_pool(np.ones((1, 2, 2, 4), dtype=np.float32)).shape == (1, 2, 2, 4)

    assert ResNetBlock(input_dim=4, output_dim=0, stride=2)(
        np.ones((1, 4, 4, 4), dtype=np.float32)
    ).shape == (1, 2, 2, 4)
    assert Conformer(
        input_dims=4,
        layer_order="mhsa_before_conv",
        fflayer_start_tpl=Identity(),
        trans_atten_tpl=Identity(),
        lconv_tpl=Identity(),
        fflayer_end_tpl=Identity(),
    )(x_tr).shape == (2, 3, 4)
    assert Conformer(
        input_dims=4,
        layer_order="conv_before_mhsa",
        trans_atten_tpl=Identity(),
        lconv_tpl=Identity(),
    )(x_tr).shape == (2, 3, 4)
    assert Conformer(input_dims=4, layer_order="mhsa", trans_atten_tpl=Identity())(x_tr).shape == (
        2,
        3,
        4,
    )
    assert Conformer(input_dims=4, layer_order="conv", lconv_tpl=Identity())(x_tr).shape == (
        2,
        3,
        4,
    )
    assert Conformer(input_dims=4, layer_order="none")(x_tr).shape == (2, 3, 4)

    assert VitEntryLayers(
        patch_size=2,
        output_dims=4,
        pos_emb_tpl=PositionalEmbedding2D(h=5, w=5, embedding_dims=4),
    )(np.ones((1, 4, 4, 1), dtype=np.float32), w_patch=np.ones((4, 4), dtype=np.float32)).shape == (
        1,
        4,
        4,
    )

    assert VitExitLayers(
        hidden_dim=4, output_dim=4, pre_ln=False, output_fc_tanh=True, pooled=False
    )(x_tr).shape == (2, 3, 4)
    assert VitExitLayers(hidden_dim=4, output_dim=2, pooled=True)(
        x_tr, w_out=np.ones((4, 2), dtype=np.float32)
    ).shape == (2, 2)

    assert ClassificationModel(softmax_tpl=FullSoftmax(num_classes=4))(
        np.ones((2, 4), dtype=np.float32)
    ).shape == (2, 4)
    assert ClassificationMLPModel(
        mlp_tpl=MLPBlock(num_layers=1, hidden_dims=4),
        softmax_tpl=FullSoftmax(num_classes=4),
    )(np.ones((2, 4), dtype=np.float32)).shape == (2, 4)
