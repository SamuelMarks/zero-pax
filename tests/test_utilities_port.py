"""Tests for quantization, N-grammer, PCA, augmentation, and routing utilities in zero_pax.praxis.layers.quantization_utils."""

from __future__ import annotations

import numpy as np

from zero_pax.praxis.layers import (
    Bias,
    BregmanPCA,
    Dropout,
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
    assert q_pn.shape == (1, 2, 2)

    q_auto, _, _ = VectorQuantization(num_clusters=3)(inputs)
    assert q_auto.shape == (1, 2, 2)
    assert VectorQuantization(num_clusters=1)(inputs)[0].shape == (1, 2, 2)

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
    assert vqz_norm(inputs)[0].shape == (1, 2, 2)

    rvq_default = RandomVectorQuantizer()
    q3, _, info3 = rvq_default(inputs)
    np.testing.assert_allclose(q3, inputs)
    assert info3["loss"] == 0.0

    rvq_active = RandomVectorQuantizer(
        latent_dim=2, projection_dim=4, num_latent_classes=3, stack_ratio=2
    )
    q_rvq, c_rvq, info_rvq = rvq_active(np.ones((2, 4, 2), dtype=np.float32))
    assert q_rvq.shape == (2, 2, 4)
    assert c_rvq.shape == (2, 2)
    assert float(info_rvq["loss"]) >= 0.0


def test_ngrammer_and_bregman_pca() -> None:
    """Verify `Ngrammer`, `VQNgrammer`, and `BregmanPCA` across activation modes."""
    inputs = np.ones((2, 4, 8), dtype=np.float32)
    ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    paddings = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    ngram_table = np.ones((16, 8), dtype=np.float32) * 0.5

    ng = Ngrammer(ngram_vocab_size=16, ngram_emb_dim=8, ln_tpl=LayerNorm(dim=8))
    out_ng = ng(inputs, input_ids=ids, paddings=paddings, ngram_table=ngram_table)
    assert out_ng.shape == (2, 4, 8)
    np.testing.assert_allclose(out_ng[0, :3], np.full((3, 8), 0.5), atol=1e-5)
    np.testing.assert_allclose(out_ng[0, 3], np.zeros((8,)), atol=1e-5)
    assert ng(inputs).shape == (2, 4, 8)

    vq_ng = VQNgrammer(ngram_vocab_size=16, ngram_emb_dim=8, num_clusters=2)
    codebook = np.zeros((2, 8), dtype=np.float32)
    codebook[1, :] = 2.0
    out_vqng = vq_ng(inputs, codebook=codebook, ngram_table=ngram_table)
    np.testing.assert_allclose(out_vqng, np.full((2, 4, 8), 1.5), atol=1e-5)

    pca_in = np.array([[3.0, 4.0, 0.0, -2.0]], dtype=np.float32)
    comps = np.eye(4, dtype=np.float32)[:2]
    pca_none = BregmanPCA(num_components=2, input_dims=4, activation_type="NONE", use_bias=False)
    recon_none, info_none = pca_none(pca_in, components=comps)
    np.testing.assert_allclose(recon_none, np.array([[3.0, 4.0, 0.0, 0.0]]), atol=1e-5)
    np.testing.assert_allclose(info_none["loss"], 1.0, atol=1e-5)

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

    pca_tanh = BregmanPCA(num_components=4, input_dims=4, activation_type="TANH")
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
    assert isinstance(res_noop, tuple) and res_noop[0].shape == (1, 4, 3)

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
    np.testing.assert_allclose(res_st[0][0, 0, :3], seq[0, 0])
    np.testing.assert_allclose(res_st[0][0, 0, 3:6], seq[0, 0])
    np.testing.assert_allclose(res_st[0][0, 0, 6:9], seq[0, 1])
    np.testing.assert_allclose(res_st[1], np.array([[1.0, 1.0]]))

    st_min = StackingOverTime(
        left_context=1, right_context=0, stride=1, padding_reduce_option="reduce_min"
    )
    res_min = st_min(seq, paddings=paddings)
    assert isinstance(res_min, tuple)
    np.testing.assert_allclose(res_min[1], np.array([[0.0, 0.0, 0.0, 1.0]]))

    mlm = MaskedLmDataAugmenter(vocab_size=100, mask_token_id=99)
    ids = np.array([[10, 20, 30, 40]], dtype=np.float32)
    aug_ids, weights = mlm(
        ids, paddings=paddings, mask_positions=np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(aug_ids, np.array([[99.0, 20.0, 30.0, 40.0]]))
    np.testing.assert_allclose(weights, np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert mlm(ids)[0].shape == (1, 4)

    spec_in = np.ones((1, 8, 6), dtype=np.float32)
    spec_eval = SpectrumAugmenter(
        freq_mask_max_bins=2, time_mask_max_frames=2, augment_at_eval=True
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
