"""Tests for quantization, N-grammer, PCA, augmentation, and routing utilities in zero_pax.praxis.layers.quantization_utils."""

from __future__ import annotations

from tests import conftest as np
from zero_pax.praxis.layers import (
    Bias,
    BregmanPCA,
    Dropout,
    Identity,
    LayerNorm,
    LayerwiseShardablePipelined,
    MaskedLmDataAugmenter,
    MultitaskResidualAdapter,
    Ngrammer,
    RandomVectorQuantizer,
    ReLU,
    Repeat,
    Sequential,
    SpectrumAugmenter,
    StackingOverTime,
    StochasticResidual,
    TemporalShifting,
    VectorQuantization,
    VectorQuantizer,
    VQNgrammer,
)


def test_vector_quantization_variants() -> None:
    """Verify `VectorQuantization`, `VectorQuantizer`, and `RandomVectorQuantizer` snap to nearest centroids."""
    inputs = np.array([[[0.9, 0.1], [0.2, 1.8]]], dtype=np.float32)
    codebook = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    vq = VectorQuantization(num_clusters=2, dim_per_head=2)
    q, codes, info = vq(inputs, codebook=codebook)
    np.testing.assert_allclose(q, np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=np.float32))
    np.testing.assert_allclose(codes, np.array([[0, 1]]))
    expected_loss = 0.25 * np.mean(np.square(inputs - np.asarray(q)))
    np.testing.assert_allclose(info["loss"], expected_loss, atol=1e-6)

    vq_prenorm = VectorQuantization(num_clusters=2, dim_per_head=2, prenormalize=True)
    q_pn, codes_pn, _ = vq_prenorm(inputs, codebook=codebook)
    np.testing.assert_allclose(codes_pn, np.array([[0, 1]]))
    np.testing.assert_allclose(q_pn, np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=np.float32))

    q_auto, c_auto, _ = VectorQuantization(num_clusters=3)(inputs)
    assert q_auto.shape == (1, 2, 2) and c_auto.shape == (1, 2)
    q_single, c_single, info_single = VectorQuantization(num_clusters=1)(inputs)
    np.testing.assert_allclose(q_single, inputs)
    np.testing.assert_allclose(c_single, np.zeros((1, 2)))
    assert info_single["loss"] == 0.0

    vqz = VectorQuantizer(
        latent_dim=2,
        num_latent_classes=2,
        beta=0.5,
        normalize_latent_vector=False,
        normalize_codebook=False,
    )
    q2, c2, info2 = vqz(inputs, codebook=codebook)
    np.testing.assert_allclose(q2, np.array([[[1.0, 0.0], [0.0, 2.0]]]))
    np.testing.assert_allclose(c2, np.array([[0, 1]]))
    np.testing.assert_allclose(info2["loss"], expected_loss * 2.0, atol=1e-6)

    vqz_norm = VectorQuantizer(
        latent_dim=2, num_latent_classes=2, normalize_latent_vector=True, normalize_codebook=True
    )
    q_vn, c_vn, _ = vqz_norm(inputs)
    assert q_vn.shape == (1, 2, 2) and c_vn.shape == (1, 2)
    np.testing.assert_allclose(np.sum(np.square(q_vn), axis=-1), np.ones((1, 2)), atol=1e-5)

    rvq_default = RandomVectorQuantizer()
    q3, c3, info3 = rvq_default(inputs, proj_w=np.eye(2, dtype=np.float32), codebook=codebook)
    np.testing.assert_allclose(c3, np.array([[0, 1]]))
    assert q3.shape == (1, 2, 2) and float(info3["loss"]) >= 0.0

    rvq_active = RandomVectorQuantizer(
        latent_dim=2,
        projection_dim=4,
        num_latent_classes=3,
        stack_ratio=2,
        normalize_latent_vector=True,
        normalize_codebook=True,
    )
    q_rvq, c_rvq, info_rvq = rvq_active(np.arange(1, 17, dtype=np.float32).reshape((2, 4, 2)))
    assert q_rvq.shape == (2, 2, 4) and c_rvq.shape == (2, 2)
    np.testing.assert_allclose(np.sum(np.square(q_rvq), axis=-1), np.ones((2, 2)), atol=1e-5)
    assert float(info_rvq["loss"]) >= 0.0


def test_ngrammer_and_bregman_pca() -> None:
    """Verify `Ngrammer`, `VQNgrammer`, and `BregmanPCA` across activation and reconstruction modes."""
    inputs = np.ones((2, 4, 8), dtype=np.float32)
    ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    paddings = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    ngram_table = np.ones((16, 8), dtype=np.float32) * 0.5

    ng = Ngrammer(ngram_vocab_size=16, ngram_emb_dim=8, ln_tpl=LayerNorm(dim=8))
    out_ng = ng(inputs, input_ids=ids, paddings=paddings, ngram_table=ngram_table)
    np.testing.assert_allclose(out_ng[0, :3], np.full((3, 8), 0.5), atol=1e-5)
    np.testing.assert_allclose(out_ng[0, 3], np.zeros((8,)), atol=1e-5)
    np.testing.assert_allclose(ng(inputs), np.zeros((2, 4, 8)), atol=1e-5)

    ng_emb_tpl = Ngrammer(
        ngram_vocab_size=16,
        ngram_emb_dim=8,
        emb_layer_tpl=lambda b_ids, *a, **kw: np.ones((2, 4, 8), dtype=np.float32) * 2.0,
    )
    np.testing.assert_allclose(ng_emb_tpl(inputs, input_ids=ids), np.full((2, 4, 8), 3.0))

    vq_ng = VQNgrammer(ngram_vocab_size=16, ngram_emb_dim=8, num_clusters=2)
    codebook = np.concatenate([np.zeros((1, 8)), np.full((1, 8), 2.0)], axis=0)
    out_vqng = vq_ng(inputs, codebook=codebook, ngram_table=ngram_table)
    np.testing.assert_allclose(out_vqng, np.full((2, 4, 8), 1.5), atol=1e-5)

    pca_in = np.array([[3.0, 4.0, 0.0, -2.0]], dtype=np.float32)
    comps = np.eye(4, dtype=np.float32)[:2]
    pca_none = BregmanPCA(num_components=2, input_dims=4, activation_type="NONE", use_bias=False)
    recon_none, info_none = pca_none(pca_in, components=comps)
    np.testing.assert_allclose(recon_none, np.array([[3.0, 4.0, 0.0, 0.0]]), atol=1e-5)
    np.testing.assert_allclose(info_none["loss"], 1.0, atol=1e-5)

    pca_latent = BregmanPCA(
        num_components=2,
        input_dims=4,
        activation_type="NONE",
        use_bias=False,
        reconstruction=False,
    )
    coeffs_out, _ = pca_latent(pca_in, components=comps)
    np.testing.assert_allclose(coeffs_out, np.array([[3.0, 4.0]]), atol=1e-5)

    pca_relu = BregmanPCA(num_components=4, input_dims=4, activation_type="RELU")
    recon_relu, _ = pca_relu(pca_in)
    np.testing.assert_allclose(recon_relu, np.array([[3.0, 4.0, 0.0, 0.0]]), atol=1e-5)

    pca_leaky = BregmanPCA(
        num_components=4,
        input_dims=4,
        activation_type="LEAKY_RELU",
        negative_slope=0.5,
        use_bias=False,
    )
    recon_leaky, _ = pca_leaky(pca_in)
    np.testing.assert_allclose(recon_leaky, np.array([[3.0, 4.0, 0.0, -1.0]]), atol=1e-5)

    pca_sig = BregmanPCA(num_components=4, input_dims=4, activation_type="SIGMOID")
    recon_sig, _ = pca_sig(np.zeros((1, 4), dtype=np.float32))
    np.testing.assert_allclose(recon_sig, np.full((1, 4), 0.5), atol=1e-5)

    pca_tanh = BregmanPCA(num_components=4, input_dims=4, activation_type="TANH", is_training=False)
    recon_tanh, _ = pca_tanh(np.zeros((1, 4), dtype=np.float32))
    np.testing.assert_allclose(recon_tanh, np.zeros((1, 4)), atol=1e-5)


def test_dropout_stochastic_residual_and_adapter() -> None:
    """Verify `Dropout`, `StochasticResidual`, and `MultitaskResidualAdapter` numerically."""
    x = np.ones((2, 4), dtype=np.float32) * 2.0
    mask = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)

    assert np.all(np.asarray(Dropout(keep_prob=0.0)(x)) == 0.0)
    np.testing.assert_allclose(Dropout(keep_prob=1.0)(x), x)
    np.testing.assert_allclose(Dropout(keep_prob=0.5)(x, mask=mask), x * mask * 2.0)
    np.testing.assert_allclose(Dropout(keep_prob=0.5)(x), x)
    drop_eval = Dropout(keep_prob=0.5, dropout_at_eval=True, noise_shape=(2, 4))(x)
    assert np.any(np.asarray(drop_eval) == 0.0) and np.any(np.asarray(drop_eval) == 4.0)
    assert Dropout(keep_prob=0.5, dropout_at_eval=True)(x).shape == (2, 4)

    sr = StochasticResidual(residual_weight=0.5, survival_prob=0.5)
    np.testing.assert_allclose(sr(x, residual=x, survival_mask=mask), x + 0.5 * (x * mask * 2.0))
    np.testing.assert_allclose(StochasticResidual(survival_prob=0.0)(x, residual=x), x)

    x_adapt = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    adapter = MultitaskResidualAdapter(input_dims=4, bottleneck_dims=2, activation_tpl=ReLU())
    w_down = np.ones((4, 2), dtype=np.float32)
    w_up = np.ones((2, 4), dtype=np.float32)
    np.testing.assert_allclose(adapter(x_adapt, w_down=w_down, w_up=w_up), x_adapt, atol=1e-5)


def test_temporal_shifting_stacking_and_augmenters() -> None:
    """Verify `TemporalShifting`, `StackingOverTime`, `MaskedLmDataAugmenter`, and `SpectrumAugmenter`."""
    seq = np.arange(1, 13, dtype=np.float32).reshape((1, 4, 3))
    shifted = TemporalShifting(shift=2, axis=1)(seq)
    np.testing.assert_allclose(shifted[:, :2, :], 0.0)
    np.testing.assert_allclose(shifted[:, 2:, :], seq[:, :2, :])
    np.testing.assert_allclose(TemporalShifting(shift=0)(seq), seq)

    seq_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    shifted_2d = TemporalShifting(shift=1, axis=1)(seq_2d)
    np.testing.assert_allclose(shifted_2d, np.array([[0.0, 1.0, 2.0], [0.0, 4.0, 5.0]]))

    st_noop = StackingOverTime(left_context=0, right_context=0, stride=1)
    out_st_noop = st_noop(seq)
    assert not isinstance(out_st_noop, tuple)
    np.testing.assert_allclose(out_st_noop, seq)
    paddings = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    res_noop = st_noop(seq, paddings=paddings)
    assert isinstance(res_noop, tuple)
    np.testing.assert_allclose(res_noop[0], seq)

    st = StackingOverTime(
        left_context=1,
        right_context=1,
        stride=2,
        pad_with_left_frame=True,
        padding_reduce_option="reduce_max",
    )
    res_st = st(seq, paddings=paddings)
    assert isinstance(res_st, tuple)
    assert res_st[0].shape == (1, 2, 9)
    assert res_st[1] is not None
    np.testing.assert_allclose(res_st[0][0, 0, :3], seq[0, 0])
    np.testing.assert_allclose(res_st[0][0, 0, 3:6], seq[0, 0])
    np.testing.assert_allclose(res_st[0][0, 0, 6:9], seq[0, 1])
    np.testing.assert_allclose(res_st[1], np.array([[1.0, 1.0]]))

    st_min = StackingOverTime(
        left_context=1, right_context=0, stride=1, padding_reduce_option="reduce_min"
    )
    res_min = st_min(seq, paddings=paddings)
    assert isinstance(res_min, tuple)
    assert res_min[1] is not None
    np.testing.assert_allclose(res_min[1], np.array([[0.0, 0.0, 0.0, 1.0]]))

    mlm = MaskedLmDataAugmenter(vocab_size=100, mask_token_id=99)
    ids = np.array([[10, 20, 30, 40]], dtype=np.float32)
    aug_ids, weights = mlm(
        ids, paddings=paddings, mask_positions=np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(aug_ids, np.array([[99.0, 20.0, 30.0, 40.0]]))
    np.testing.assert_allclose(weights, np.array([[1.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(mlm(ids)[0], ids)

    spec_in = np.ones((1, 8, 6), dtype=np.float32)
    spec_eval = SpectrumAugmenter(
        freq_mask_max_bins=2,
        time_mask_max_frames=2,
        use_dynamic_time_mask_max_frames=True,
        time_mask_max_ratio=0.25,
        augment_at_eval=True,
    )
    out_spec = spec_eval(
        spec_in, paddings=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(out_spec[:, :2, :], 0.0)
    np.testing.assert_allclose(out_spec[:, :, :2], 0.0)
    np.testing.assert_allclose(out_spec[:, 6:, :], 0.0)
    np.testing.assert_allclose(out_spec[:, 2:6, 2:], 1.0)


def test_sequential_repeat_and_pipelined() -> None:
    """Verify `Sequential`, `Repeat`, and `LayerwiseShardablePipelined` containers."""
    from zero_pax.types import WeightInit

    inputs = np.array([[-1.0, 2.0]], dtype=np.float32)
    add1 = Bias(dims=2, bias_init=WeightInit(method="ones", scale=1.0))

    seq = Sequential(layers=[ReLU(), add1])
    np.testing.assert_allclose(seq(inputs), np.array([[1.0, 3.0]]))
    np.testing.assert_allclose(Sequential()(inputs), inputs)

    rep = Repeat(sub_tpl=add1, x_times=3)
    np.testing.assert_allclose(rep(inputs), inputs + 3.0)
    np.testing.assert_allclose(Repeat()(inputs), inputs)

    pipe = LayerwiseShardablePipelined(single_stage_body=add1, num_stages=2, CircularRepeat=2)
    np.testing.assert_allclose(pipe(inputs), inputs + 4.0)
    np.testing.assert_allclose(LayerwiseShardablePipelined()(inputs), inputs)

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

    ng_concat = Ngrammer(
        ngram_vocab_size=8,
        ngram_emb_dim=2,
        concat_ngrams=True,
        ln_tpl=Identity(),
    )
    assert ng_concat(
        np.ones((1, 3, 4), dtype=np.float32), input_ids=np.array([[1, 2, 3]])
    ).shape == (
        1,
        3,
        4,
    )

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

    vq_heads = VectorQuantization(num_clusters=2, num_heads=2, dim_per_head=2)
    q_h, c_h, _ = vq_heads(np.ones((1, 2, 4), dtype=np.float32))
    assert q_h.shape == (1, 2, 4) and c_h.shape == (1, 2)
    assert "ema_cluster_size" in vq_heads.params

    vqz_grp = VectorQuantizer(
        num_latent_classes=2,
        num_groups=2,
        normalize_latent_vector=True,
        normalize_latent_per_group=True,
        kmeans_init=True,
        EmptyCluster="reset",
    )
    q_g, _, _ = vqz_grp(np.ones((1, 2, 4), dtype=np.float32))
    assert q_g.shape == (1, 2, 4)

    rvq_lr = RandomVectorQuantizer(
        latent_dim=4,
        projection_dim=4,
        num_latent_classes=2,
        num_groups=2,
        low_rank_codebook=True,
        plot_codebook=True,
        aggregation_type="sum",
    )
    q_lr, _, _ = rvq_lr(np.ones((1, 2, 4), dtype=np.float32))
    assert q_lr.shape == (1, 2, 2)
    assert "codebook_norm" in rvq_lr.params

    rvq_custom_q = RandomVectorQuantizer(
        projection_dim=4,
        quantizer_tpl=VectorQuantizer(num_latent_classes=2),
    )
    assert rvq_custom_q(np.ones((1, 2, 4), dtype=np.float32))[0].shape == (1, 2, 4)
    rvq_fn_q = RandomVectorQuantizer(projection_dim=4, quantizer_tpl=Identity())
    assert rvq_fn_q(np.ones((1, 2, 4), dtype=np.float32))[0].shape == (1, 2, 4)

    ng_scores = Ngrammer(
        ngram_vocab_size=8,
        unigram_vocab_size=4,
        num_heads=2,
        dim_per_head=2,
        weight_sharing=True,
        ngram_using_attention_scores=True,
    )
    assert ng_scores(
        np.ones((1, 3, 4), dtype=np.float32), input_ids=np.array([[1, 2, 5]])
    ).shape == (1, 3, 4)

    vq_ng_full = VQNgrammer(
        num_clusters=2,
        causal_attention=False,
        use_cached_input_ids_to_cluster_ids=True,
        enable_cache_updates=True,
        ngram_layer_tpl=Ngrammer(ngram_vocab_size=8),
    )
    assert vq_ng_full(np.ones((1, 3, 4), dtype=np.float32)).shape == (1, 3, 4)
    assert "input_ids_to_cluster_ids" in vq_ng_full.params
    vq_ng_id = VQNgrammer(num_clusters=2, vq_layer_tpl=Identity(), ngram_layer_tpl=Identity())
    assert vq_ng_id(np.ones((1, 3, 4), dtype=np.float32)).shape == (1, 3, 4)

    pca_iter = BregmanPCA(
        num_components=2,
        input_dims=4,
        normalize_weight=True,
        normalize_scale=False,
        is_training=True,
        learnable_Coef=True,
        coefficients_steps=3,
        coefficients_lr=0.05,
        constant_lr_schedule=False,
        learnable_Component=True,
        components_lr=0.01,
    )
    x_pca = np.array([[2.0, 1.0, 0.5, -0.5], [1.0, 2.0, -0.5, 0.5]], dtype=np.float32)
    comp_init = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    recon_it, info_it = pca_iter(x_pca, components=comp_init)
    assert recon_it.shape == (2, 4) and float(info_it["loss"]) >= 0.0
    assert "running_mean" in pca_iter.params and "components_momentum" in pca_iter.params

    drop_bcast = Dropout(
        keep_prob=0.5,
        noise_shape_broadcast_dims=(-1,),
        dropout_at_eval=True,
        transpose_qk=True,
    )
    assert drop_bcast(np.ones((2, 2, 2), dtype=np.float32)).shape == (2, 2, 2)

    mra_tasks = MultitaskResidualAdapter(input_dims=4, bottleneck_dims=2, num_tasks=3)
    assert mra_tasks(np.ones((1, 2, 4), dtype=np.float32), task_ids=1).shape == (1, 2, 4)

    mlm_rand = MaskedLmDataAugmenter(
        vocab_size=20, mask_prob=0.8, random_prob=0.3, same_prob=0.2, mask_token_id=19
    )
    aug_r, mask_r = mlm_rand(np.arange(1, 11, dtype=np.float32).reshape((2, 5)))
    assert aug_r.shape == (2, 5) and mask_r.shape == (2, 5)

    spec_dyn = SpectrumAugmenter(
        augment_at_eval=True,
        unstack=True,
        stack_height=2,
        domain="time",
        time_masks_per_frame=0.5,
    )
    assert spec_dyn(np.ones((1, 6, 4), dtype=np.float32)).shape == (1, 6, 4)

    rep_inter = Repeat(
        sub_tpl=ReLU(),
        x_times=0,
        nd_prefix_shape=(2,),
        unroll_in_decode=True,
        collect_intermediate_outputs=True,
        return_intermediate_outputs=True,
    )
    assert rep_inter(np.array([[-1.0, 2.0]], dtype=np.float32)).shape == (2, 1, 2)
    rep_collect_only = Repeat(
        sub_tpl=ReLU(),
        x_times=1,
        collect_intermediate_outputs=True,
        return_intermediate_outputs=False,
    )
    assert rep_collect_only(np.array([[-1.0, 2.0]], dtype=np.float32)).shape == (1, 2)

    pipe_mb = LayerwiseShardablePipelined(
        single_stage_body=ReLU(),
        num_stages=2,
        num_microbatches=2,
        stream_io=True,
    )
    assert pipe_mb(np.ones((2, 2), dtype=np.float32)).shape == (2, 2)
    pipe_inter = LayerwiseShardablePipelined(
        single_stage_body=ReLU(),
        num_stages=2,
        microbatch_size=4,
        collect_intermediate_outputs=True,
    )
    assert pipe_inter(np.ones((2, 2), dtype=np.float32)).shape == (2, 2)
    assert "intermediate_outputs" in pipe_inter.params
