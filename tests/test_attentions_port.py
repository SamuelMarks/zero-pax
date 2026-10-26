"""Tests for attention layers in zero_pax.praxis.layers.attentions."""

from __future__ import annotations

from tests import conftest as np
from zero_pax import _tensor_ops as t_ops
from zero_pax.praxis.layers import (
    AttentionProjection,
    DotProductAttention,
    DotProductAttentionWithContext,
    DotProductAttentionWithContextXL,
    DotProductAttentionXL,
    GroupedQueryAttention,
    Identity,
    LayerNorm,
    LocalSelfAttention,
    LocalSelfAttentionAlibi,
    LocalSelfAttentionRelativeBias,
    LocalSelfAttentionXL,
    RelativeBias,
    ReLU,
    SelfAttentionWithNormAndResidual,
)
from zero_pax.praxis.layers.attentions import _apply_rotary_position_emb
from zero_pax.types import WeightInit


def test_attention_projection() -> None:
    """Verify `AttentionProjection` input and output modes numerically."""
    inputs = np.arange(1, 33, dtype=np.float32).reshape((2, 2, 8))
    w_in = np.ones((8, 2, 4), dtype=np.float32) * 0.25
    b_in = np.full((2, 4), 0.5, dtype=np.float32)

    proj_in = AttentionProjection(
        input_dim=8, num_heads=2, dim_per_head=4, is_output_projection=False, use_bias=True
    )
    out = proj_in(inputs, w=w_in, b=b_in)
    expected_in = np.einsum("btd,dnh->btnh", inputs, w_in) + b_in
    np.testing.assert_allclose(out, expected_in, atol=1e-5)

    proj_comb = AttentionProjection(
        input_dim=8, num_heads=2, dim_per_head=4, attention_combine_dims=True, use_bias=False
    )
    out_comb = proj_comb(inputs, w=w_in)
    np.testing.assert_allclose(out_comb, expected_in.reshape((2, 2, 8)) - 0.5, atol=1e-5)

    w_out = np.ones((2, 4, 8), dtype=np.float32) * 0.5
    b_out = np.full((8,), 0.1, dtype=np.float32)
    proj_out = AttentionProjection(
        input_dim=8,
        num_heads=2,
        dim_per_head=4,
        is_output_projection=True,
        use_bias=True,
        use_nhd_shape=True,
    )
    back = proj_out(out, w=w_out, b=b_out)
    expected_back = np.einsum("btnh,nhd->btd", np.asarray(out), w_out) + b_out
    np.testing.assert_allclose(back, expected_back, atol=1e-5)

    proj_out_dnh = AttentionProjection(
        input_dim=8,
        num_heads=2,
        dim_per_head=4,
        is_output_projection=True,
        use_bias=False,
        use_nhd_shape=False,
    )
    w_dnh = np.ones((8, 2, 4), dtype=np.float32) * 0.5
    np.testing.assert_allclose(proj_out_dnh(out, w=w_dnh), expected_back - 0.1, atol=1e-5)
    np.testing.assert_allclose(
        proj_out_dnh(out, w=w_dnh.reshape((64,))), expected_back - 0.1, atol=1e-5
    )


def test_relative_bias() -> None:
    """Verify `RelativeBias` bidirectional symmetry and causal bucket clipping."""
    w_table = np.arange(1, 17, dtype=np.float32).reshape((2, 8))
    rb_bi = RelativeBias(
        num_heads=2,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
        bidirectional=True,
    )
    out_bi = rb_bi(4, 4, w=w_table)
    assert out_bi.shape == (1, 2, 4, 4)
    np.testing.assert_allclose(out_bi[:, :, 0, 0], out_bi[:, :, 2, 2], atol=1e-5)
    assert not np.allclose(out_bi[:, :, 0, 2], out_bi[:, :, 2, 0])

    rb_uni = RelativeBias(
        num_heads=2,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
        bidirectional=False,
        use_xavier_init=True,
    )
    pos = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    out_uni = rb_uni(pos, pos, w=w_table)
    assert out_uni.shape == (2, 2, 4, 4)
    np.testing.assert_allclose(out_uni[:, :, 0, 1], out_uni[:, :, 0, 3], atol=1e-5)


def test_dot_product_attention_variants() -> None:
    """Verify `DotProductAttention` and its context, XL, RoPE, dconv, and local subclasses."""
    inputs = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 8)) * 0.1
    w_qkv = np.eye(8, dtype=np.float32).reshape((8, 2, 4))
    w_post = np.eye(8, dtype=np.float32).reshape((2, 4, 8))

    dpa = DotProductAttention(
        input_dim=8,
        num_heads=2,
        dim_per_head=4,
        atten_logit_cap=50.0,
        scale_logits_by_head_dims=True,
        relative_bias_tpl=RelativeBias(num_heads=2),
        dropout_tpl=Identity(),
        zero_fully_masked=True,
        use_rotary_position_emb=True,
        dconv_qkv=True,
        causal_depthwise_conv1d_tpl=Identity(),
        attention_extra_logit=0.0,
    )
    mask_2d = np.zeros((4, 4), dtype=np.float32)
    out = dpa(
        inputs,
        atten_mask=mask_2d,
        query_w=w_qkv,
        key_w=w_qkv,
        value_w=w_qkv,
        post_w=w_post,
        per_dim_scale=np.ones((4,), dtype=np.float32),
    )
    assert out.shape == (2, 4, 8)
    assert np.any(np.asarray(out) != 0.0)

    mask_3d = np.full((2, 4, 4), -1e9, dtype=np.float32)
    out_masked = dpa(
        inputs, atten_mask=mask_3d, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post
    )
    np.testing.assert_allclose(out_masked, np.zeros((2, 4, 8)), atol=1e-5)

    from zero_pax.types import WeightInit

    dpa_comb = DotProductAttention(
        input_dim=8,
        num_heads=2,
        dim_per_head=4,
        combine_qkv=True,
        params_init=WeightInit(method="ones", scale=0.25),
    )
    out_comb = dpa_comb(inputs)
    assert out_comb.shape == (2, 4, 8)
    assert np.any(np.asarray(out_comb) != 0.0)

    layer_ctx = DotProductAttentionWithContext(
        input_dim=8, num_heads=2, dim_per_head=4, left_context=1, right_context=0
    )
    out_ctx_1 = layer_ctx(inputs, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post)
    inputs_future_changed = inputs + np.concatenate(
        [np.zeros((2, 2, 8)), np.full((2, 2, 8), 100.0)], axis=1
    )
    out_ctx_2 = layer_ctx(
        inputs_future_changed, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post
    )
    np.testing.assert_allclose(out_ctx_1[:, :2, :], out_ctx_2[:, :2, :], atol=1e-5)

    xl_layer = DotProductAttentionXL(input_dim=8, num_heads=2, dim_per_head=4, rel_pos_emb_dim=4)
    out_xl = xl_layer(inputs, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post)
    out_no_xl = DotProductAttention(input_dim=8, num_heads=2, dim_per_head=4)(
        inputs, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post
    )
    assert not np.allclose(out_xl, out_no_xl)

    out_ctx_xl = DotProductAttentionWithContextXL(
        input_dim=8, num_heads=2, left_context=2, rel_pos_emb_dim=4
    )(inputs, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post)
    assert out_ctx_xl.shape == (2, 4, 8)
    assert not np.allclose(out_ctx_xl, out_xl)

    local_blk = LocalSelfAttention(input_dim=8, num_heads=2, dim_per_head=4, block_size=1)
    out_blk1 = local_blk(inputs, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post)
    inputs_far = inputs + np.concatenate([np.zeros((2, 3, 8)), np.full((2, 1, 8), 50.0)], axis=1)
    out_blk2 = local_blk(inputs_far, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post)
    np.testing.assert_allclose(out_blk1[:, 0, :], out_blk2[:, 0, :], atol=1e-5)

    out_lsa_xl = LocalSelfAttentionXL(input_dim=8, num_heads=2, rel_pos_emb_dim=4)(
        inputs, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post
    )
    np.testing.assert_allclose(out_lsa_xl, out_xl, atol=1e-5)

    out_lsa_rb = LocalSelfAttentionRelativeBias(input_dim=8, num_heads=2)(
        inputs, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post
    )
    np.testing.assert_allclose(out_lsa_rb, out_no_xl, atol=1e-5)

    out_alibi = LocalSelfAttentionAlibi(input_dim=8, num_heads=2, left_context=2)(
        inputs, query_w=w_qkv, key_w=w_qkv, value_w=w_qkv, post_w=w_post
    )
    assert out_alibi.shape == (2, 4, 8)
    assert not np.allclose(out_alibi, out_no_xl)


def test_grouped_query_attention() -> None:
    """Verify `GroupedQueryAttention` head broadcasting, temperature, bias, and RoPE."""
    inputs = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 8)) * 0.1
    w_q = np.ones((8, 4, 2), dtype=np.float32) * 0.2
    w_kv = np.ones((8, 2, 2), dtype=np.float32) * 0.2
    w_o = np.ones((4, 2, 8), dtype=np.float32) * 0.2
    b_o = np.full((8,), 0.5, dtype=np.float32)

    gqa = GroupedQueryAttention(
        input_dim=8,
        num_heads=4,
        num_kv_heads=2,
        dim_per_head=2,
        atten_logit_cap=20.0,
        atten_temp=0.5,
        rope_min_max_timescales=(1, 10000),
        dropout_tpl=Identity(),
    )
    out = gqa(
        inputs,
        atten_mask=np.zeros((2, 4, 4, 4), dtype=np.float32),
        w_q=w_q,
        w_k=w_kv,
        w_v=w_kv,
        w_o=w_o,
        b_q=np.zeros((4, 2), dtype=np.float32),
        b_k=np.zeros((2, 2), dtype=np.float32),
        b_v=np.zeros((2, 2), dtype=np.float32),
        b_o=b_o,
    )
    assert out.shape == (2, 4, 8)
    assert np.all(np.abs(np.asarray(out) - 0.5) > 0.0)

    gqa_odd = GroupedQueryAttention(input_dim=8, num_heads=4, num_kv_heads=3, dim_per_head=2)
    np.testing.assert_allclose(gqa_odd(inputs), np.zeros((2, 4, 8)), atol=1e-5)


def test_self_attention_with_norm_and_residual() -> None:
    """Verify `SelfAttentionWithNormAndResidual` pre/post-norm and residual weights."""
    inputs = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 8)) * 0.1

    pre = SelfAttentionWithNormAndResidual(
        num_heads=2, pre_layer_norm=True, norm_tpl=LayerNorm(dim=8), input_weight=2.0
    )
    np.testing.assert_allclose(pre(inputs), inputs * 2.0, atol=1e-5)

    pre_active = SelfAttentionWithNormAndResidual(
        num_heads=2,
        dim_per_head=4,
        pre_layer_norm=True,
        norm_tpl=LayerNorm(dim=8),
        input_weight=2.0,
        params_init=WeightInit(method="xavier", scale=1.0),
    )
    out_pre_active = pre_active(inputs)
    assert out_pre_active.shape == (2, 4, 8)
    np.testing.assert_not_allclose(out_pre_active, inputs * 2.0)

    post = SelfAttentionWithNormAndResidual(
        num_heads=2,
        pre_layer_norm=False,
        force_no_residual=True,
        self_atten_tpl=Identity(),
        residual_dropout_tpl=Identity(),
    )
    expected_post = np.ref_layer_norm(inputs, eps=1e-6)
    np.testing.assert_allclose(post(inputs), expected_post, atol=1e-5)

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
        input_dim=4,
        num_heads=2,
        num_kv_heads=2,
        use_bias=False,
        atten_temp=0.0,
        params_init=WeightInit(method="ones", scale=0.5),
    )
    out_gqa_nb = gqa_no_bias(x_d)
    assert out_gqa_nb.shape == (1, 2, 4)
    assert np.any(np.asarray(out_gqa_nb) != 0.0)

    # Verify strict causal masking invariance at t=0 when perturbing future token t=1
    dpa_causal = DotProductAttentionWithContext(
        input_dim=4,
        num_heads=2,
        dim_per_head=2,
        left_context=2,
        right_context=0,
        params_init=WeightInit(method="xavier", scale=1.0, seed=42),
    )
    x_c1 = np.array([[[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 1.0, -1.0]]], dtype=np.float32)
    x_c2 = np.array([[[1.0, 2.0, 3.0, 4.0], [9.0, 8.0, -7.0, 6.0]]], dtype=np.float32)
    np.testing.assert_allclose(dpa_causal(x_c1)[:, 0, :], dpa_causal(x_c2)[:, 0, :], atol=1e-5)

    sanr = SelfAttentionWithNormAndResidual(
        num_heads=2,
        dim_per_head=2,
        self_atten_tpl=ReLU(),
        force_no_residual=False,
        residual_dropout_prob=0.1,
    )
    assert sanr(x_d).shape == (1, 2, 4)

    proj_fan_out = AttentionProjection(
        input_dim=4,
        num_heads=2,
        dim_per_head=2,
        is_output_projection=True,
        explicit_fan_in_fan_out_axes=True,
        params_init=WeightInit(method="xavier", scale=1.0),
    )
    assert proj_fan_out(np.ones((1, 2, 2, 2), dtype=np.float32)).shape == (1, 2, 4)
    proj_fan_in = AttentionProjection(
        input_dim=4,
        num_heads=2,
        dim_per_head=2,
        is_output_projection=False,
        explicit_fan_in_fan_out_axes=True,
        params_init=WeightInit(method="xavier", scale=1.0),
    )
    assert proj_fan_in(x_d).shape == (1, 2, 2, 2)

    rb_len = RelativeBias(num_heads=2, use_length_as_position=True)
    assert rb_len(t_ops.asarray(2), t_ops.asarray(2)).shape == (1, 2, 2, 2)

    dpa_opts = DotProductAttention(
        input_dim=4,
        num_heads=2,
        dim_per_head=2,
        atten_dropout_prob=0.1,
        internal_gshard_gaussian_init=True,
        use_rotary_position_emb=True,
        rotary_position_emb_tpl=Identity(),
        cast_rotary_position_emb=True,
        fprop_dtype="float32",
        consolidate_rope_key_state=True,
        decode_cache=True,
        attention_mask_summary=True,
        decoding_maybe_shard_projections=True,
    )
    assert dpa_opts(x_d, atten_mask=np.zeros((1, 1, 2, 2), dtype=np.float32)).shape == (1, 2, 4)
    assert "rope_key_state" in dpa_opts.params
    assert "atten_mask_summary" in dpa_opts.params

    dpa_no_cache = DotProductAttention(input_dim=4, num_heads=2, decode_cache=False)
    assert dpa_no_cache(x_d).shape == (1, 2, 4)
    assert "key_state" not in dpa_no_cache.params

    dpa_ctx_xl_none = DotProductAttentionWithContextXL(
        input_dim=4, num_heads=2, rel_pos_emb_dim=None
    )
    assert dpa_ctx_xl_none(x_d).shape == (1, 2, 4)
    dpa_ctx_xl_pos = DotProductAttentionWithContextXL(input_dim=4, num_heads=2, rel_pos_emb_dim=4)
    assert dpa_ctx_xl_pos(x_d).shape == (1, 2, 4)

    lsa_sim = LocalSelfAttention(input_dim=4, num_heads=2, block_size=1, simulated=True)
    assert lsa_sim(x_d).shape == (1, 2, 4)
    lsa_xl_none = LocalSelfAttentionXL(input_dim=4, num_heads=2, rel_pos_emb_dim=0)
    assert lsa_xl_none(x_d).shape == (1, 2, 4)
    lsa_xl_pos = LocalSelfAttentionXL(input_dim=4, num_heads=2, rel_pos_emb_dim=4)
    assert lsa_xl_pos(x_d).shape == (1, 2, 4)

    gqa_proj = GroupedQueryAttention(
        input_dim=4,
        hidden_dim=4,
        num_heads=2,
        num_kv_heads=2,
        atten_dropout_prob=0.1,
        proj_tpl=AttentionProjection(input_dim=4, num_heads=2, dim_per_head=2),
    )
    assert gqa_proj(x_d).shape == (1, 2, 4)
