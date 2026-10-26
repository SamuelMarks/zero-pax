"""Tests for embedding, positional encoding, softmax, and loss layers in zero_pax.praxis.layers.embeddings."""

from __future__ import annotations

import pytest

from tests import conftest as np
from zero_pax import _tensor_ops as t_ops
from zero_pax.praxis.layers import (
    BiTemperedLoss,
    EinsumOp,
    Embedding,
    FullSoftmax,
    GShardSharedEmbeddingSoftmax,
    Identity,
    Linear,
    PositionalEmbedding,
    PositionalEmbedding2D,
    SharedEmbeddingSoftmax,
    SigmoidCrossEntropy,
    TrainablePositionalEmbedding,
)
from zero_pax.types import WeightInit


def test_embedding_and_shared_embedding() -> None:
    """Verify `Embedding`, `SharedEmbeddingSoftmax`, and `GShardSharedEmbeddingSoftmax`."""
    w = np.arange(20, dtype=np.float32).reshape((5, 4))
    ids = np.array([[0, 2, -1], [4, 1, 9]], dtype=np.int32)

    emb_idx = Embedding(
        num_classes=5,
        input_dims=4,
        lookup_style="INDEX",
        scale_sqrt_depth=True,
        set_nan_for_oob_id=False,
    )
    out_idx = emb_idx(ids, w=w)
    np.testing.assert_allclose(out_idx[0, 0], w[0] * 2.0)
    np.testing.assert_allclose(out_idx[0, 2], np.zeros((4,)))

    emb_mm = Embedding(
        num_classes=5,
        input_dims=4,
        lookup_style="MATMUL",
        scale_sqrt_depth=False,
        set_nan_for_oob_id=True,
    )
    out_mm = emb_mm(ids, w=w)
    np.testing.assert_allclose(out_mm[0, 1], w[2])
    assert np.all(np.isnan(np.asarray(out_mm[1, 2])))
    assert Embedding(num_classes=5, input_dims=4)(ids).shape == (2, 3, 4)

    shared = SharedEmbeddingSoftmax(
        num_classes=5, input_dims=4, soft_cap_logits=10.0, scale_before_logits=0.5
    )
    looked = shared.emb_lookup(np.array([[1, 3]], dtype=np.int32), w=w.T)
    np.testing.assert_allclose(looked[0, 0], w[1])
    assert shared.emb_lookup(np.array([[1, 3]], dtype=np.int32)).shape == (1, 2, 4)

    hidden = np.ones((2, 3, 4), dtype=np.float32)
    logits = shared(hidden, w=w, b=np.ones((5,), dtype=np.float32))
    raw_expected = np.einsum("btd,vd->btv", hidden * 0.5, w) + 1.0
    expected_logits = 10.0 * np.tanh(raw_expected / 10.0)
    np.testing.assert_allclose(logits, expected_logits, atol=1e-5)

    loss_shared = shared(hidden, class_ids=np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32), w=w)
    assert loss_shared.shape == (2, 3)
    assert np.all(np.asarray(loss_shared) > 0.0)

    gshard = GShardSharedEmbeddingSoftmax(
        num_classes=5, input_dims=4, logits_abs_max=5.0, soft_cap_logits=10.0
    )
    g_emb = gshard.emb_lookup(np.array([[1, 2]], dtype=np.int32), w=w)
    np.testing.assert_allclose(g_emb[0, 0], w[1] * 0.5, atol=1e-5)
    assert gshard.emb_lookup(np.array([[1, 2]], dtype=np.int32)).shape == (1, 2, 4)
    g_logits = gshard(hidden, w=w.T)
    assert g_logits.shape == (2, 3, 5)
    assert np.all(np.abs(np.asarray(g_logits)) <= 10.0)


def test_positional_embeddings() -> None:
    """Verify `PositionalEmbedding`, `PositionalEmbedding2D`, and `TrainablePositionalEmbedding`."""
    pe = PositionalEmbedding(embedding_dims=8, min_timescale=1, max_timescale=1000)
    out_len = pe(seq_length=4)
    assert out_len.shape == (1, 4, 8)
    np.testing.assert_allclose(out_len[0, 0, :4], np.zeros((4,)), atol=1e-6)
    np.testing.assert_allclose(out_len[0, 0, 4:], np.ones((4,)), atol=1e-6)

    out_pos = pe(position=np.array([[0, 1, 2, 3]], dtype=np.int32))
    np.testing.assert_allclose(out_len, out_pos, atol=1e-5)
    assert PositionalEmbedding(embedding_dims=7)(seq_length=np.zeros((2, 3))).shape == (1, 3, 7)
    assert PositionalEmbedding(embedding_dims=4)().shape == (1, 1, 4)

    pe2d = PositionalEmbedding2D(
        h=2, w=3, embedding_dims=8, num_prepend_cls_tokens=1, num_append_cls_tokens=1
    )
    out_2d = pe2d()
    assert out_2d.shape == (1, 8, 8)
    np.testing.assert_allclose(out_2d[0, 0], np.zeros((8,)), atol=1e-6)
    np.testing.assert_allclose(out_2d[0, -1], np.zeros((8,)), atol=1e-6)

    out_hw = PositionalEmbedding2D(h=2, w=2, embedding_dims=4, pos_transform="h_w")()
    out_wh = PositionalEmbedding2D(h=2, w=2, embedding_dims=5, pos_transform="w_h")()
    assert out_wh.shape == (1, 4, 5)
    np.testing.assert_allclose(out_wh[:, :, :2], out_hw[:, :, 2:4], atol=1e-5)
    np.testing.assert_allclose(out_wh[:, :, 2:4], out_hw[:, :, :2], atol=1e-5)

    tpe = TrainablePositionalEmbedding(max_seq_length=6, embedding_dims=4, lookup_style="INDEX")
    w_pos = np.arange(24, dtype=np.float32).reshape((6, 4))
    out_tpe_idx = tpe(seq_length=3, w=w_pos)
    np.testing.assert_allclose(out_tpe_idx[0], w_pos[:3], atol=1e-5)

    tpe_mm = TrainablePositionalEmbedding(max_seq_length=6, embedding_dims=4, lookup_style="MATMUL")
    out_tpe_mm = tpe_mm(position=np.array([0, 1, 2], dtype=np.int32), w=w_pos)
    np.testing.assert_allclose(out_tpe_mm[0], w_pos[:3], atol=1e-5)


def test_softmax_cross_entropy_and_bitempered() -> None:
    """Verify `FullSoftmax`, `SigmoidCrossEntropy`, and `BiTemperedLoss` against JAX references."""
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, -1.0]], dtype=np.float32)

    sm = FullSoftmax(
        num_classes=3, soft_cap_logits=20.0, label_smoothing_prob=0.1, z_loss_weight=0.01
    )
    probs = sm(logits, b=np.zeros((3,), dtype=np.float32))
    np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones((2,)), atol=1e-5)

    loss_ids = sm(
        logits,
        class_ids=np.array([[0], [1]], dtype=np.int32),
        class_weights=np.array([[1.0], [0.5]], dtype=np.float32),
    )
    assert loss_ids.shape == (2,)
    assert float(loss_ids[0]) > 0.0 and float(loss_ids[1]) > 0.0

    sm_proj = FullSoftmax(input_dims=2, num_classes=3, use_bias=True)
    proj_probs = sm_proj(
        np.ones((2, 2), dtype=np.float32),
        w=np.ones((2, 3), dtype=np.float32),
        b=np.zeros((3,), dtype=np.float32),
    )
    np.testing.assert_allclose(proj_probs, np.full((2, 3), 1.0 / 3.0), atol=1e-5)

    sce = SigmoidCrossEntropy(soft_cap_logits=10.0)
    sig_probs = sce(logits, b=np.zeros((3,), dtype=np.float32))
    capped = 10.0 * np.tanh(logits / 10.0)
    np.testing.assert_allclose(sig_probs, 1.0 / (1.0 + np.exp(-capped)), atol=1e-5)

    sce_loss = sce(
        logits,
        class_ids=np.array([0, 1], dtype=np.int32),
        class_weights=np.ones((2, 3), dtype=np.float32),
    )
    y_one_hot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    ref_bce = np.maximum(capped, 0.0) - capped * y_one_hot + np.log1p(np.exp(-np.abs(capped)))
    np.testing.assert_allclose(sce_loss, ref_bce, atol=1e-5)

    bt_default = BiTemperedLoss(t1=1.0, t2=1.0)
    bt_probs = bt_default(logits)
    ref_probs = np.ref_softmax(logits, axis=-1)
    np.testing.assert_allclose(bt_probs, ref_probs, atol=1e-5)

    labels_one_hot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    bt_ce_loss = bt_default(logits, labels=labels_one_hot)
    ref_ce_loss = -np.sum(labels_one_hot * np.ref_log_softmax(logits, axis=-1), axis=-1)
    np.testing.assert_allclose(bt_ce_loss, ref_ce_loss, atol=1e-5)

    bt_tempered = BiTemperedLoss(t1=0.8, t2=1.2, label_smoothing=0.1)
    bt_loss = bt_tempered(
        logits, labels=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    )
    assert bt_loss.shape == (2,)
    assert float(bt_loss[0]) > float(bt_loss[1]) > 0.0

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
        einsum_tpl=lambda oh, w_in: t_ops.einsum("...v,vd->...d", oh, w_in),
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
    assert PositionalEmbedding2D(
        h=2, w=2, embedding_dims=4, pos_transform="hwd->(wh)d"
    )().shape == (
        1,
        4,
        4,
    )
    assert PositionalEmbedding2D(h=2, w=2, embedding_dims=4, pos_transform="hwd->hwd")().shape == (
        1,
        2,
        2,
        4,
    )

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
        einsum_tpl=lambda oh, w_in: t_ops.einsum("btv,vd->btd", oh, w_in),
    )
    np.testing.assert_allclose(tpe_custom_ein(seq_length=2), np.zeros((1, 2, 2)))

    sm_bt = FullSoftmax(
        input_dims=2,
        num_classes=3,
        use_bias=False,
        scale_before_logits=2.0,
        ais_temperature=2.0,
        bi_tempered_loss_tpl=BiTemperedLoss(t1=1.0, t2=1.0),
    )
    probs_2d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    loss_probs = sm_bt(
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

    with pytest.raises(ValueError, match="Embedding weight shape"):
        Embedding(num_classes=4, input_dims=2)([[0, 1]], w=np.ones((3, 2), dtype=np.float32))
    assert Embedding(num_classes=0, input_dims=0)(
        [[0, 1]], w=np.ones((4, 3), dtype=np.float32)
    ).shape == (1, 2, 3)

    gs_cw = GShardSharedEmbeddingSoftmax(
        num_classes=3,
        input_dims=2,
        use_tgt_labels_size_as_loss_denominator=False,
    )
    loss_gs_cw = gs_cw(
        np.zeros((1, 2, 2), dtype=np.float32),
        w=np.zeros((2, 3), dtype=np.float32),
        class_ids=np.array([[0, 1]], dtype=np.int32),
        class_weights=np.array([[2.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(loss_gs_cw, np.full((1, 2), np.log(3.0) * 0.5), atol=1e-5)

    tpe_idx_lookup = TrainablePositionalEmbedding(
        max_seq_length=4,
        embedding_dims=2,
        lookup_style="INDEX",
        array_lookup_tpl=EinsumOp(equation="btv,vd->btd"),
    )
    assert tpe_idx_lookup(seq_length=2).shape == (1, 2, 2)
    tpe_idx_fn = TrainablePositionalEmbedding(
        max_seq_length=4,
        embedding_dims=2,
        lookup_style="INDEX",
        array_lookup_tpl=lambda p: t_ops.zeros((1, 2, 2)),
    )
    assert tpe_idx_fn(seq_length=2).shape == (1, 2, 2)

    sm_chunk = FullSoftmax(
        input_dims=2,
        num_classes=3,
        chunk_size=1,
        label_smoothing_prob=0.2,
        label_smoothing_apply_for_eval=False,
    )
    probs_chunk = sm_chunk(np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_allclose(probs_chunk, np.full((2, 3), 1.0 / 3.0), atol=1e-5)
    loss_eval = sm_chunk(
        np.zeros((2, 3), dtype=np.float32),
        class_ids=np.array([0, 1], dtype=np.int32),
        is_eval=True,
    )
    np.testing.assert_allclose(loss_eval, np.full((2,), np.log(3.0)), atol=1e-5)

    bt_sched = BiTemperedLoss(t1=0.8, t2=1.2, start_step=10, end_step=20)
    loss_step15 = bt_sched(
        np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        labels=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        step=15,
    )
    assert loss_step15.shape == (2,)
