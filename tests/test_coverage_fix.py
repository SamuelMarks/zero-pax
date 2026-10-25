"""Numerical edge-case and branch verification tests across all zero_pax modules."""

from __future__ import annotations

import pytest

from tests import conftest as np
from zero_pax import _tensor_ops as t_ops
from zero_pax.praxis.base import BasePraxisLayer, instantiate_layer
from zero_pax.praxis.layers import (
    SSM,
    AttentionProjection,
    BatchNorm,
    BertModel,
    BiTemperedLoss,
    BregmanPCA,
    Conformer,
    ConvBNAct,
    DepthwiseConv1D,
    DotProductAttention,
    Einsum,
    EinsumOp,
    Embedding,
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
    LayerNorm,
    LayerNormalizedLstmCellSimple,
    LightConv1D,
    Linear,
    LocalSelfAttentionAlibi,
    LocalSelfAttentionRelativeBias,
    LstmCellSimple,
    LstmFrnn,
    MaskedLmDataAugmenter,
    MLPBlock,
    PipelinedTransformer,
    Pooling,
    PositionalEmbedding,
    PositionalEmbedding2D,
    RandomVectorQuantizer,
    RelativeBias,
    ReLU,
    Repeat,
    ResNet,
    ResNetBlock,
    RmsNorm,
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
    VectorQuantizer,
    VitEntryLayers,
    VitExitLayers,
    VQNgrammer,
)
from zero_pax.types import DecoderHParams, PaxConfig, WeightInit


def test_tensor_ops_and_base_numerical_branches() -> None:
    """Verify numerical edge cases in `_tensor_ops.py`, `base.py`, `normalizations.py`, `linear_pooling.py`, and `convolutions.py`."""
    t = t_ops.asarray([1.0, 2.0])
    assert t_ops.asarray(t) is t
    np.testing.assert_allclose(
        t_ops.transpose([[1.0, 2.0], [3.0, 4.0]], axes=(1, 0)),
        np.array([[1.0, 3.0], [2.0, 4.0]]),
    )
    np.testing.assert_allclose(
        t_ops.pad([[1.0, 2.0]], ((0, 0), (1, 1)), mode="edge"),
        np.array([[1.0, 1.0, 2.0, 2.0]]),
    )
    assert int(t_ops.argmin([3.0, 1.0, 2.0])) == 1
    assert int(t_ops.argmax([3.0, 1.0, 2.0])) == 0

    class CustomSubLayer(BasePraxisLayer):
        """Test subclass of BasePraxisLayer."""

    cfg_base = PaxConfig(cls=CustomSubLayer)
    assert isinstance(instantiate_layer(cfg_base, Identity()), CustomSubLayer)
    assert DecoderHParams().seqlen == 0

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

    ln_fp32 = LayerNorm(dim=4, reductions_in_fp32=True)
    np.testing.assert_allclose(np.mean(ln_fp32(x_4d), axis=-1), np.zeros((2, 2, 2)), atol=1e-5)

    rms_inter = RmsNorm(dim=4, intermediate_dtype="float32")
    assert rms_inter(x_4d).shape == (2, 2, 2, 4)

    gn_pad_4d = GroupNorm(dim=4, num_groups=2)
    out_gn_4d = gn_pad_4d(x_4d, paddings=np.zeros((2, 2, 2, 1), dtype=np.float32))
    np.testing.assert_allclose(
        np.mean(np.asarray(out_gn_4d).reshape((2, 2, 2, 2, 2)), axis=(1, 2, 4)),
        np.zeros((2, 2)),
        atol=1e-5,
    )

    ein_vec = Einsum(eqn="bd,d->b")
    np.testing.assert_allclose(ein_vec(np.array([[1.0, 2.0]], dtype=np.float32)), [3.0])

    ein_custom = Einsum(
        eqn="bd,df->bf",
        w_shape=(2, 2),
        use_bias=False,
        einsum_op_tpl=lambda x, w: t_ops.matmul(x, w),
    )
    np.testing.assert_allclose(
        ein_custom(np.ones((2, 2), dtype=np.float32), t2=np.eye(2, dtype=np.float32)),
        np.ones((2, 2)),
    )
    np.testing.assert_allclose(
        ein_custom(np.ones((2, 2), dtype=np.float32), w=np.eye(2, dtype=np.float32)),
        np.ones((2, 2)),
    )

    lin_custom = Linear(input_dims=2, output_dims=2, einsum_tpl=lambda x, w: t_ops.matmul(x, w))
    np.testing.assert_allclose(
        lin_custom(np.array([[2.0, 3.0]], dtype=np.float32), w=np.eye(2, dtype=np.float32)),
        np.array([[2.0, 3.0]]),
    )

    ff_custom = FeedForward(input_dims=2, output_dims=2, linear_tpl=Identity(), bias_tpl=Identity())
    np.testing.assert_allclose(
        ff_custom(np.array([[-1.0, 2.0]], dtype=np.float32)), np.array([[0.0, 2.0]])
    )

    pool_avg_valid = Pooling(
        window_shape=(2, 2), window_stride=(1, 1), pooling_type="AVG", padding="VALID"
    )
    res_pv = pool_avg_valid(
        np.ones((1, 3, 3, 2), dtype=np.float32) * 4.0,
        paddings=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    assert isinstance(res_pv, tuple)
    np.testing.assert_allclose(res_pv[0][:, 0, 0, :], np.array([[4.0, 4.0]]))

    gp_all_pad = GlobalPooling(pooling_type="AVG", pooling_dims=[1], keepdims=False)
    out_all_pad = gp_all_pad(
        np.ones((1, 3, 2), dtype=np.float32),
        compatible_paddings=np.ones((1, 3, 1), dtype=np.float32),
    )
    np.testing.assert_allclose(out_all_pad, np.zeros((1, 2)))

    cba_custom = ConvBNAct(
        filter_shape=(3, 3, 2, 2), batch_norm_tpl=Identity(), activation_tpl=Identity()
    )
    np.testing.assert_allclose(
        cba_custom(np.ones((1, 3, 3, 2), dtype=np.float32)), np.zeros((1, 3, 3, 2))
    )

    dconv_default = DepthwiseConv1D()
    np.testing.assert_allclose(
        dconv_default(np.ones((1, 4, 2), dtype=np.float32)), np.zeros((1, 4, 2))
    )
    np.testing.assert_allclose(
        dconv_default(np.ones((1, 4, 2), dtype=np.float32), w=np.ones((3, 2), dtype=np.float32)),
        np.array([[[2.0, 2.0], [3.0, 3.0], [3.0, 3.0], [2.0, 2.0]]]),
    )

    dconv_dilated = DepthwiseConv1D(filter_shape=(2, 1, 2), rhs_dilation_rate=2)
    np.testing.assert_allclose(
        dconv_dilated(np.ones((1, 4, 2), dtype=np.float32), w=np.ones((2, 1, 2), dtype=np.float32)),
        np.array([[[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [1.0, 1.0]]]),
    )

    lconv_custom = LightConv1D(
        input_dims=2,
        kernel_size=3,
        ln_tpl=Identity(),
        linear_start_tpl=lambda x, *a, **kw: t_ops.concatenate([x, x], axis=-1),
        depthwise_conv_tpl=Identity(),
        conv_norm_layer_tpl=Identity(),
        conv_activation_tpl=Identity(),
        linear_end_tpl=Identity(),
    )
    x_lc = np.ones((1, 4, 2), dtype=np.float32)
    expected_lc = x_lc * (1.0 / (1.0 + np.exp(-x_lc)))
    np.testing.assert_allclose(lconv_custom(x_lc), expected_lc, atol=1e-5)


def test_attentions_and_embeddings_numerical_branches() -> None:
    """Verify numerical edge cases in `attentions.py` and `embeddings.py`."""
    from zero_pax.praxis.layers.attentions import _apply_rotary_position_emb

    x_1d_head = t_ops.ones((1, 2, 1, 1))
    np.testing.assert_allclose(_apply_rotary_position_emb(x_1d_head), x_1d_head)
    x_odd_head = t_ops.ones((1, 2, 1, 3))
    out_odd_rope = _apply_rotary_position_emb(x_odd_head)
    np.testing.assert_allclose(out_odd_rope[:, :, :, 2], np.ones((1, 2, 1)))

    proj_out_no_bias = AttentionProjection(
        input_dim=6, num_heads=2, dim_per_head=3, is_output_projection=True, use_bias=False
    )
    np.testing.assert_allclose(
        proj_out_no_bias(np.ones((1, 2, 2, 3), dtype=np.float32)), np.zeros((1, 2, 6))
    )

    from zero_pax.praxis.layers import DotProductAttentionWithContext, DotProductAttentionXL

    np.testing.assert_allclose(
        DotProductAttentionXL(input_dim=4, num_heads=2, rel_pos_emb_dim=0)(
            np.ones((1, 2, 4), dtype=np.float32)
        ),
        np.zeros((1, 2, 4)),
    )
    np.testing.assert_allclose(
        DotProductAttentionWithContext(
            input_dim=4, num_heads=2, left_context=None, right_context=1
        )(np.ones((1, 2, 4), dtype=np.float32)),
        np.zeros((1, 2, 4)),
    )

    proj_out_auto = AttentionProjection(
        input_dim=0, num_heads=1, dim_per_head=0, is_output_projection=True, use_bias=True
    )
    x_nh = np.ones((1, 2, 2, 3), dtype=np.float32)
    np.testing.assert_allclose(proj_out_auto(x_nh), np.zeros((1, 2, 6)))
    np.testing.assert_allclose(
        proj_out_auto(
            x_nh,
            w=np.ones((6, 6), dtype=np.float32),
            b=np.ones((6,), dtype=np.float32),
        ),
        np.full((1, 2, 6), 7.0),
    )

    proj_in_auto = AttentionProjection(
        input_dim=0, num_heads=2, dim_per_head=0, is_output_projection=False, use_bias=True
    )
    x_d = np.ones((1, 2, 4), dtype=np.float32)
    np.testing.assert_allclose(proj_in_auto(x_d), np.zeros((1, 2, 2, 2)))
    np.testing.assert_allclose(
        proj_in_auto(
            x_d,
            w=np.ones((4, 4), dtype=np.float32),
            b=np.ones((4,), dtype=np.float32),
        ),
        np.full((1, 2, 2, 2), 5.0),
    )

    rb = RelativeBias(num_heads=2, rb_stddev=0.05)
    out_rb_1d = rb(np.array([0, 1], dtype=np.int32), 2)
    np.testing.assert_allclose(out_rb_1d[:, :, 0, 0], out_rb_1d[:, :, 1, 1], atol=1e-5)
    out_rb_none = rb(2, None)
    assert out_rb_none.shape == (1, 2, 2, 2)

    dpa_hidden = DotProductAttention(
        input_dim=4,
        hidden_dim=4,
        num_heads=2,
        internal_enable_query_scale=False,
        ln_tpl=Identity(),
        ngrammer_tpl=Identity(),
    )
    np.testing.assert_allclose(
        dpa_hidden(
            x_d,
            atten_mask=np.zeros((1, 2, 2, 2), dtype=np.float32),
        ),
        np.zeros((1, 2, 4)),
    )

    dpa_custom_proj = DotProductAttention(
        input_dim=4,
        num_heads=2,
        proj_tpl=lambda x, *a, **kw: t_ops.reshape(x, (1, 2, 2, 2)),
        qk_einsum_tpl=lambda q, k: t_ops.einsum("btnh,bsnh->bnts", q, k),
        pv_einsum_tpl=lambda p, v: t_ops.einsum("bnts,bsnh->btnh", p, v),
        per_dim_scale_tpl=Identity(),
        relative_bias_tpl=lambda q_len, *a, **kw: t_ops.zeros((1, 2, 2, 2)),
    )
    assert dpa_custom_proj(x_d).shape == (1, 2, 4)

    lsa_rb_custom = LocalSelfAttentionRelativeBias(
        input_dim=4, num_heads=2, relative_bias_tpl=RelativeBias(num_heads=2)
    )
    np.testing.assert_allclose(lsa_rb_custom(x_d), np.zeros((1, 2, 4)))
    lsa_alibi_no_ctx = LocalSelfAttentionAlibi(input_dim=4, num_heads=2)
    np.testing.assert_allclose(lsa_alibi_no_ctx(x_d), np.zeros((1, 2, 4)))

    gqa_no_bias = GroupedQueryAttention(
        input_dim=4, num_heads=2, num_kv_heads=2, use_bias=False, atten_temp=0.0
    )
    np.testing.assert_allclose(gqa_no_bias(x_d), np.zeros((1, 2, 4)))

    sanr = SelfAttentionWithNormAndResidual(
        num_heads=2, dim_per_head=2, self_atten_tpl=ReLU(), force_no_residual=False
    )
    np.testing.assert_allclose(sanr(x_d), x_d)

    with pytest.raises(ValueError, match="INDEX or MATMUL"):
        Embedding(lookup_style="INVALID")([[0, 1]])

    emb_custom_lookup = Embedding(
        num_classes=4,
        input_dims=2,
        lookup_style="INDEX",
        array_lookup_tpl=EinsumOp(equation="...v,vd->...d"),
    )
    np.testing.assert_allclose(
        emb_custom_lookup([[0, 1]], w=np.ones((4, 2), dtype=np.float32)),
        np.ones((1, 2, 2)),
    )
    emb_custom_fn = Embedding(
        num_classes=4,
        input_dims=2,
        lookup_style="INDEX",
        array_lookup_tpl=lambda ids: t_ops.ones((1, 2, 2)),
    )
    np.testing.assert_allclose(emb_custom_fn([[0, 1]]), np.ones((1, 2, 2)))
    emb_custom_ein = Embedding(
        num_classes=4,
        input_dims=2,
        lookup_style="MATMUL",
        einsum_tpl=lambda oh, w: t_ops.einsum("...v,vd->...d", oh, w),
    )
    np.testing.assert_allclose(
        emb_custom_ein([[0, 1]], w=np.ones((4, 2), dtype=np.float32)),
        np.ones((1, 2, 2)),
    )

    shared = SharedEmbeddingSoftmax(
        num_classes=4,
        input_dims=4,
        use_bias=False,
        soft_cap_logits=0.0,
        ais_temperature=2.0,
        feed_forward_tpl=Identity(),
    )
    np.testing.assert_allclose(
        shared(np.ones((1, 2, 4), dtype=np.float32)),
        np.full((1, 2, 4), 0.5),
    )

    gshard_loss = GShardSharedEmbeddingSoftmax(
        num_classes=3, input_dims=2, label_smoothing_prob=0.1, z_loss_weight=0.01
    )
    loss_gs = gshard_loss(
        np.ones((1, 2, 2), dtype=np.float32),
        w=np.ones((2, 3), dtype=np.float32),
        class_ids=np.array([[0, 1]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        loss_gs,
        np.full((1, 2), np.log(3.0) + 0.01 * ((2.0 + np.log(3.0)) ** 2)),
        atol=1e-4,
    )

    assert PositionalEmbedding(embedding_dims=4)(seq_length=t_ops.asarray(3)).shape == (1, 3, 4)
    assert PositionalEmbedding(embedding_dims=4)(seq_length=t_ops.zeros((3,))).shape == (1, 3, 4)
    assert PositionalEmbedding(embedding_dims=4)(position=t_ops.arange(3)).shape == (1, 3, 4)

    shared_no_ff = SharedEmbeddingSoftmax(num_classes=4, input_dims=2, use_bias=False)
    np.testing.assert_allclose(
        shared_no_ff.emb_lookup(
            np.array([[0, 1]], dtype=np.int32), w=np.ones((4, 2), dtype=np.float32)
        ),
        np.ones((1, 2, 2)),
    )
    np.testing.assert_allclose(
        shared_no_ff(np.ones((1, 2, 2), dtype=np.float32)), np.zeros((1, 2, 4))
    )

    pe2d_trunc = PositionalEmbedding2D(h=2, w=2, embedding_dims=3)()
    assert pe2d_trunc.shape == (1, 4, 3)

    pe2d_pad = PositionalEmbedding2D(h=2, w=2, embedding_dims=5)()
    assert pe2d_pad.shape == (1, 4, 5)
    np.testing.assert_allclose(pe2d_pad[:, :, 4:], 0.0)
    pe2d_pos = PositionalEmbedding2D(h=2, w=2, embedding_dims=4)(position=np.array([0, 2]))
    np.testing.assert_allclose(
        pe2d_pos[0, 0], PositionalEmbedding2D(h=2, w=2, embedding_dims=4)()[0, 0]
    )
    assert PositionalEmbedding2D(h=2, w=2, embedding_dims=4)(seq_length=2).shape == (1, 2, 4)
    assert PositionalEmbedding2D(h=2, w=2, embedding_dims=4)(seq_length=6).shape == (1, 6, 4)

    tpe_default = TrainablePositionalEmbedding(max_seq_length=4, embedding_dims=2)
    np.testing.assert_allclose(tpe_default(), np.zeros((1, 4, 2)))
    tpe_custom_ein = TrainablePositionalEmbedding(
        max_seq_length=4,
        embedding_dims=2,
        einsum_tpl=lambda oh, w: t_ops.einsum("btv,vd->btd", oh, w),
    )
    np.testing.assert_allclose(tpe_custom_ein(seq_length=2), np.zeros((1, 2, 2)))

    sm = FullSoftmax(
        input_dims=2,
        num_classes=3,
        use_bias=False,
        scale_before_logits=2.0,
        ais_temperature=2.0,
        bi_tempered_loss_tpl=BiTemperedLoss(t1=1.0, t2=1.0),
    )
    probs_2d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    loss_probs = sm(
        np.zeros((2, 2), dtype=np.float32),
        class_probabilities=probs_2d,
        class_weights=np.array([1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(loss_probs, np.full((2,), np.log(3.0)), atol=1e-5)

    sm_ff = FullSoftmax(
        input_dims=2, num_classes=3, feed_forward_tpl=Linear(input_dims=2, output_dims=3)
    )
    np.testing.assert_allclose(sm_ff(np.ones((2, 2), dtype=np.float32)), np.full((2, 3), 1.0 / 3.0))

    sce_proj = SigmoidCrossEntropy(input_dims=2, num_classes=3, soft_cap_logits=0.0)
    out_sce_proj = sce_proj(
        np.ones((2, 2), dtype=np.float32),
        w=np.zeros((2, 3), dtype=np.float32),
        b=np.zeros((3,), dtype=np.float32),
    )
    np.testing.assert_allclose(out_sce_proj, np.full((2, 3), 0.5), atol=1e-5)

    sce_no_b = SigmoidCrossEntropy(input_dims=2, num_classes=3, soft_cap_logits=0.0)
    np.testing.assert_allclose(
        sce_no_b(np.ones((2, 2), dtype=np.float32), w=np.zeros((2, 3), dtype=np.float32)),
        np.full((2, 3), 0.5),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sce_no_b(
            np.zeros((2, 3), dtype=np.float32),
            labels=np.ones((2, 3), dtype=np.float32),
            class_weights=np.full((2, 3), 2.0, dtype=np.float32),
        ),
        np.full((2, 3), 2.0 * np.log(2.0)),
        atol=1e-5,
    )

    sce_ff = SigmoidCrossEntropy(
        input_dims=2,
        num_classes=3,
        feed_forward_tpl=Linear(input_dims=2, output_dims=3),
        bias_init=WeightInit(method="zeros"),
    )
    np.testing.assert_allclose(
        sce_ff(np.ones((2, 2), dtype=np.float32)), np.full((2, 3), 0.5), atol=1e-5
    )
    np.testing.assert_allclose(
        SigmoidCrossEntropy()(
            np.zeros((2, 3), dtype=np.float32),
            labels=np.ones((2, 3), dtype=np.float32),
        ),
        np.full((2, 3), np.log(2.0)),
        atol=1e-5,
    )


def test_utilities_recurrent_transformers_and_arch_numerical_branches() -> None:
    """Verify numerical edge cases in `quantization_utils.py`, `recurrent_ssm.py`, `transformers.py`, and `architectures.py`."""
    q_vq, c_vq, _ = VectorQuantizer(num_latent_classes=1, normalize_latent_vector=False)(
        np.ones((1, 2, 2), dtype=np.float32)
    )
    np.testing.assert_allclose(q_vq, np.ones((1, 2, 2)))
    np.testing.assert_allclose(c_vq, np.zeros((1, 2)))

    rvq = RandomVectorQuantizer(
        stack_ratio=1,
        stacking_tpl=StackingOverTime(stride=2, left_context=1),
        projection_dim=4,
        num_latent_classes=2,
        normalize_latent_vector=True,
        normalize_codebook=True,
    )
    q_rvq, c_rvq, _ = rvq(
        np.ones((1, 4, 2), dtype=np.float32),
        proj_w=np.ones((4, 4), dtype=np.float32),
        codebook=np.ones((2, 4), dtype=np.float32),
    )
    assert q_rvq.shape == (1, 2, 4) and c_rvq.shape == (1, 2)
    np.testing.assert_allclose(np.sum(np.square(q_rvq), axis=-1), np.ones((1, 2)), atol=1e-5)

    vq_ng = VQNgrammer(num_clusters=1)
    x_134 = np.arange(1, 13, dtype=np.float32).reshape((1, 3, 4))
    np.testing.assert_allclose(vq_ng(x_134), x_134)

    pca_norm = BregmanPCA(num_components=2, input_dims=4, normalize_weight=False)
    recon_pca, _ = pca_norm(
        np.ones((1, 4), dtype=np.float32), components=np.eye(4, dtype=np.float32)[:2]
    )
    np.testing.assert_allclose(recon_pca, np.array([[1.0, 1.0, 0.0, 0.0]]))

    sr_no_mask = StochasticResidual(residual_weight=0.5, survival_prob=0.5)
    np.testing.assert_allclose(
        sr_no_mask(
            np.ones((1, 2), dtype=np.float32), residual=np.ones((1, 2), dtype=np.float32) * 2.0
        ),
        np.full((1, 2), 2.0),
    )

    ts_neg = TemporalShifting(shift=1, axis=-1)
    np.testing.assert_allclose(
        ts_neg(np.array([[1.0, 2.0, 3.0]], dtype=np.float32)),
        np.array([[0.0, 1.0, 2.0]]),
    )
    np.testing.assert_allclose(
        TemporalShifting(shift=1)([1.0, 2.0]),
        np.array([1.0, 2.0]),
    )

    st_strided_pad = StackingOverTime(left_context=0, right_context=0, stride=2)
    res_st_sp = st_strided_pad(
        np.arange(1, 9, dtype=np.float32).reshape((1, 4, 2)),
        paddings=np.zeros((1, 4), dtype=np.float32),
    )
    assert isinstance(res_st_sp, tuple)
    np.testing.assert_allclose(res_st_sp[0], np.array([[[1.0, 2.0], [5.0, 6.0]]]))

    mlm_no_pad = MaskedLmDataAugmenter(vocab_size=10, mask_token_id=9)
    aug_m, w_m = mlm_no_pad(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        mask_positions=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(aug_m, np.array([[9.0, 2.0, 9.0]]))
    np.testing.assert_allclose(w_m, np.array([[1.0, 0.0, 1.0]]))

    spec_4d = SpectrumAugmenter(freq_mask_max_bins=0, time_mask_max_frames=0, augment_at_eval=True)
    x_spec4 = np.ones((1, 4, 4, 2), dtype=np.float32)
    np.testing.assert_allclose(spec_4d(x_spec4), x_spec4)
    np.testing.assert_allclose(spec_4d(np.ones((4,), dtype=np.float32)), np.ones((4,)))
    spec_small = SpectrumAugmenter(
        freq_mask_max_bins=10, time_mask_max_frames=10, augment_at_eval=True
    )
    np.testing.assert_allclose(spec_small(np.ones((1, 1, 1), dtype=np.float32)), np.ones((1, 1, 1)))

    rep_sub = Repeat(sub_layer=ReLU(), x_times=0, num_repeats=1)
    np.testing.assert_allclose(
        rep_sub(np.array([[-1.0, 2.0]], dtype=np.float32)), np.array([[0.0, 2.0]])
    )

    lstm_tuple_inferred = LstmCellSimple(num_output_nodes=0, hidden_size=0)
    c_inf, m_inf = lstm_tuple_inferred(
        np.ones((2, 3), dtype=np.float32),
        state0=(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)),
    )
    np.testing.assert_allclose(c_inf, np.zeros((2, 3)))
    np.testing.assert_allclose(m_inf, np.zeros((2, 3)))

    lstm_no_bias = LstmCellSimple(enable_lstm_bias=False, cell_value_cap=None)
    c_nb, _ = lstm_no_bias(
        np.ones((2, 4), dtype=np.float32),
        state0=np.ones((2, 4), dtype=np.float32),
        padding=np.array([0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(c_nb[0], np.full((4,), 0.5))
    np.testing.assert_allclose(c_nb[1], np.ones((4,)))

    lstm_same_size = LstmCellSimple(hidden_size=2)
    c_ss, _ = lstm_same_size(
        np.ones((2, 2), dtype=np.float32),
        state0=np.ones((2, 2), dtype=np.float32),
    )
    np.testing.assert_allclose(c_ss, np.full((2, 2), 0.5))

    ln_lstm_no_bias = LayerNormalizedLstmCellSimple(enable_lstm_bias=False)
    c_ln_nb, _ = ln_lstm_no_bias(np.ones((2, 12), dtype=np.float32))
    np.testing.assert_allclose(c_ln_nb, np.zeros((2, 4)))

    np.testing.assert_allclose(
        FRnn()(np.ones((2, 4), dtype=np.float32)), np.tanh(np.ones((2, 4))), atol=1e-5
    )
    np.testing.assert_allclose(
        FRnn()(np.ones((4,), dtype=np.float32)), np.tanh(np.ones((4,))), atol=1e-5
    )
    out_frnn_lstm = FRnn(
        cell_tpl=LstmCellSimple(num_input_nodes=3, num_output_nodes=3), reverse=False
    )(np.ones((2, 3, 3), dtype=np.float32))
    np.testing.assert_allclose(out_frnn_lstm, np.zeros((2, 3, 3)))

    np.testing.assert_allclose(
        LstmFrnn()(np.ones((2, 4), dtype=np.float32)), np.zeros((2, 4)), atol=1e-5
    )
    np.testing.assert_allclose(
        StackFrnn(frnn_tpl=ReLU())(np.array([[[-1.0, 2.0]]], dtype=np.float32)),
        np.array([[[0.0, 2.0]]]),
    )
    np.testing.assert_allclose(SSM()(np.ones((2, 4), dtype=np.float32)), np.ones((2, 4)), atol=1e-5)
    ssm_bi = SSM(hidden_size=3, bidirectional=True)
    out_ssm_bi = ssm_bi(
        np.ones((2, 3, 3), dtype=np.float32),
        paddings=np.zeros((2, 3), dtype=np.float32),
        a_diag=np.array([-1.0]),
        b_mat=np.ones((3, 1)),
        c_mat=np.ones((1, 3)),
        d_vec=np.eye(3),
    )
    assert out_ssm_bi.shape == (2, 3, 3) and np.all(np.asarray(out_ssm_bi) > 1.0)

    out_ssm_bi_nopad = ssm_bi(
        np.ones((2, 3, 3), dtype=np.float32),
        a_diag=np.array([-1.0]),
        b_mat=np.ones((3, 1)),
        c_mat=np.ones((1, 3)),
        d_vec=np.eye(3),
    )
    np.testing.assert_allclose(out_ssm_bi_nopad, out_ssm_bi, atol=1e-5)

    rnet_no_stem = ResNet(strides=(1,), channels=(1,), blocks=(1,), kernels=(3,))
    np.testing.assert_allclose(
        rnet_no_stem(np.ones((1, 4, 4, 4), dtype=np.float32)),
        np.ones((1, 2, 2, 4)),
        atol=1e-5,
    )

    vit_entry_mismatch_pe = VitEntryLayers(
        pos_emb_shapes=(1, 3),
        output_dims=4,
        pos_emb_tpl=lambda x, *a, **kw: t_ops.zeros((1, 1, 1)),
    )
    np.testing.assert_allclose(
        vit_entry_mismatch_pe(np.ones((1, 3, 4), dtype=np.float32)),
        np.ones((1, 3, 4)),
    )

    out_ssmg = SSMGated(
        hidden_size=2,
        ssm_tpl=Identity(),
        gss_fflayer_tpl=Linear(input_dims=4, output_dims=2, weight_init=WeightInit(method="ones")),
    )(np.ones((2, 3, 4), dtype=np.float32))
    np.testing.assert_allclose(out_ssmg, np.zeros((2, 3, 2)))
    np.testing.assert_allclose(
        SSMTransformer()(np.ones((2, 3, 3), dtype=np.float32)), np.zeros((2, 3, 3))
    )

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
        mlm_augmenter_tpl=MaskedLmDataAugmenter(vocab_size=10, mask_token_id=9),
    )
    np.testing.assert_allclose(bert_aug(x_tr), x_tr)
    np.testing.assert_allclose(LanguageModel(lm_tpl=Identity())(x_tr), x_tr)

    rnet_small_pool = ResNet(
        strides=(1,),
        channels=(1,),
        blocks=(1,),
        kernels=(3,),
        entry_max_pool=True,
        entryflow_conv_params=Identity(),
    )
    x_img_2x2 = np.ones((1, 2, 2, 4), dtype=np.float32)
    np.testing.assert_allclose(rnet_small_pool(x_img_2x2), x_img_2x2, atol=1e-5)

    assert ResNetBlock(input_dim=4, output_dim=0, stride=2)(
        np.ones((1, 4, 4, 4), dtype=np.float32)
    ).shape == (1, 2, 2, 4)

    conf_custom = Conformer(
        input_dims=4,
        layer_order="mhsa_before_conv",
        fflayer_start_tpl=Identity(),
        trans_atten_tpl=Identity(),
        lconv_tpl=Identity(),
        fflayer_end_tpl=Identity(),
        final_ln_tpl=Identity(),
    )
    assert conf_custom(x_tr).shape == (2, 3, 4)
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

    vit_entry_proj = VitEntryLayers(pos_emb_shapes=(1, 3), output_dims=2, pos_emb_tpl=Identity())
    np.testing.assert_allclose(
        vit_entry_proj(np.ones((1, 3, 4), dtype=np.float32)),
        np.full((1, 3, 2), 8.0),
    )
    vit_exit_custom = VitExitLayers(
        hidden_dim=4,
        output_dim=2,
        pooled=True,
        output_fc_tanh=False,
        ln_tpl=Identity(),
        pooling_tpl=GlobalPooling(pooling_type="AVG", pooling_dims=[1]),
    )
    np.testing.assert_allclose(
        vit_exit_custom(
            np.ones((1, 3, 4), dtype=np.float32), w_out=np.ones((4, 2), dtype=np.float32)
        ),
        np.full((1, 2), 4.0),
    )

    mlp_custom = MLPBlock(
        num_layers=1,
        hidden_dims=4,
        has_bias=False,
        fflayer_tpl=Linear(input_dims=4, output_dims=4, weight_init=WeightInit(method="ones")),
    )
    np.testing.assert_allclose(mlp_custom(np.ones((1, 4), dtype=np.float32)), np.full((1, 4), 4.0))
