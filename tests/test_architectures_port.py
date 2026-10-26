"""Tests for Vision, Audio, and Classification architectures in zero_pax.praxis.layers.architectures."""

from __future__ import annotations

import pytest

from tests import conftest as np
from zero_pax import _tensor_ops as t_ops
from zero_pax.praxis.layers import (
    GELU,
    ClassificationMLPModel,
    ClassificationModel,
    Conformer,
    GlobalPooling,
    Identity,
    ReLU,
    ResNet,
    ResNetBlock,
    VanillaBlock,
    VanillaNet,
    VisionTransformer,
    VitEntryLayers,
    VitExitLayers,
)
from zero_pax.types import WeightInit


def test_vanilla_and_resnet() -> None:
    """Verify `VanillaBlock`, `VanillaNet`, `ResNetBlock`, and `ResNet` multi-stage execution numerically."""
    img = np.arange(1, 257, dtype=np.float32).reshape((2, 4, 4, 8)) * 0.05

    vb = VanillaBlock(
        input_dim=8,
        output_dim=4,
        kernel_size=3,
        stride=2,
        negative_slope=0.1,
        params_init=WeightInit(method="xavier", scale=1.0),
    )
    out_vb = vb(img)
    assert out_vb.shape == (2, 2, 2, 4)
    assert np.any(np.asarray(out_vb) != 0.0)
    np.testing.assert_allclose(np.mean(out_vb, axis=(0, 1, 2)), np.zeros((4,)), atol=1e-4)

    vb_zero = VanillaBlock(input_dim=8, output_dim=4, kernel_size=3, stride=2, negative_slope=0.1)
    np.testing.assert_allclose(vb_zero(img), np.zeros((2, 2, 2, 4)), atol=1e-6)

    vb_body = VanillaBlock(body=[ReLU(), Identity()])
    np.testing.assert_allclose(vb_body(img - 5.0), np.maximum(0.0, img - 5.0), atol=1e-5)

    vnet_default = VanillaNet(
        strides=(1, 2),
        channels=(4, 8),
        blocks=(1, 1),
        kernels=(3, 3),
        params_init=WeightInit(method="xavier", scale=1.0),
    )
    out_vnet_def = vnet_default(img)
    assert out_vnet_def.shape == (2, 2, 2, 8)
    assert np.any(np.asarray(out_vnet_def) != 0.0)

    vnet_staged = VanillaNet(
        strides=(1, 2),
        channels=(4, 8),
        blocks=(1, 1),
        kernels=(3, 3),
        entryflow_conv_params=Identity(),
        output_spatial_pooling_params=GlobalPooling(pooling_dims=(1, 2)),
        params_init=WeightInit(method="xavier", scale=1.0),
    )
    out_vnet = vnet_staged(img)
    assert out_vnet.shape == (2, 8)
    assert np.any(np.asarray(out_vnet) != 0.0)
    np.testing.assert_allclose(np.mean(out_vnet, axis=0), np.zeros((8,)), atol=1e-4)

    rb_same = ResNetBlock(input_dim=8, output_dim=2, stride=1, zero_init_residual=True)
    np.testing.assert_allclose(rb_same(img), img, atol=1e-5)

    rb_body = ResNetBlock(body=[ReLU()])
    np.testing.assert_allclose(rb_body(img - 5.0), np.maximum(0.0, img - 5.0), atol=1e-5)

    rb_proj = ResNetBlock(
        input_dim=8,
        output_dim=4,
        stride=2,
        activation_tpl=GELU(),
        params_init=WeightInit(method="xavier", scale=1.0),
    )
    out_rb_proj = rb_proj(img)
    assert out_rb_proj.shape == (2, 2, 2, 16)
    assert np.any(np.asarray(out_rb_proj) != 0.0)

    rnet_staged = ResNet(
        strides=(1, 2),
        channels=(2, 4),
        blocks=(1, 1),
        kernels=(3, 3),
        entryflow_conv_params=Identity(),
        entry_max_pool=True,
        output_spatial_pooling_params=GlobalPooling(pooling_dims=(1, 2)),
        params_init=WeightInit(method="xavier", scale=1.0),
    )
    out_rnet = rnet_staged(img)
    assert out_rnet.shape == (2, 16)
    assert np.any(np.asarray(out_rnet) != 0.0)


def test_conformer() -> None:
    """Verify `Conformer` across all four `layer_order` configurations, weight sharing, and invalid order error."""
    seq = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 8)) * 0.1
    paddings = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    for order in ("mhsa_before_conv", "conv_before_mhsa", "mhsa", "conv"):
        conf = Conformer(
            input_dims=8,
            atten_num_heads=2,
            kernel_size=7,
            layer_order=order,
            ff_activation_tpl=GELU(),
            fflayer_weight_sharing=(order == "mhsa"),
        )
        out = conf(seq, paddings=paddings)
        assert out.shape == (2, 4, 8)
        np.testing.assert_allclose(out[0, 3], np.zeros((8,)), atol=1e-5)
        np.testing.assert_allclose(np.mean(out[1, 0]), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.var(out[1, 0]), 1.0, atol=1e-3)

    conf_model = Conformer(model_dims=8)
    out_model = conf_model(seq)
    np.testing.assert_allclose(np.mean(out_model, axis=-1), np.zeros((2, 4)), atol=1e-5)

    with pytest.raises(ValueError, match="Unsupported Conformer layer_order"):
        Conformer(input_dims=8, layer_order="invalid")(seq)


def test_vit_and_classification_models() -> None:
    """Verify `VitEntryLayers`, `VitExitLayers`, `VisionTransformer`, `ClassificationModel`, and `ClassificationMLPModel`."""
    img = np.ones((2, 4, 4, 3), dtype=np.float32)
    tokens = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 8)) * 0.1

    entry = VitEntryLayers(
        patch_size=2,
        output_dims=8,
        prepend_cls_tokens=1,
        append_cls_tokens=1,
    )
    cls_prep = np.full((1, 1, 8), 5.0, dtype=np.float32)
    cls_app = np.full((1, 1, 8), -5.0, dtype=np.float32)
    tok_out = entry(img, cls_prepend=cls_prep, cls_append=cls_app)
    assert tok_out.shape == (2, 6, 8)
    np.testing.assert_allclose(tok_out[:, 0, :], np.full((2, 8), 5.0), atol=1e-5)
    np.testing.assert_allclose(tok_out[:, -1, :], np.full((2, 8), -5.0), atol=1e-5)

    tok_3d = VitEntryLayers(pos_emb_shapes=(2, 2), prepend_cls_tokens=1, append_cls_tokens=1)(
        tokens
    )
    assert tok_3d.shape == (2, 6, 8)
    np.testing.assert_allclose(tok_3d[:, 0, :], np.zeros((2, 8)), atol=1e-6)
    np.testing.assert_allclose(tok_3d[:, -1, :], np.zeros((2, 8)), atol=1e-6)

    exit_pooled = VitExitLayers(
        hidden_dim=8,
        output_dim=4,
        pooled=True,
        pre_logits_Dims=6,
        output_fc_tanh=True,
    )
    out_exit = exit_pooled(tokens, b_out=np.ones((4,), dtype=np.float32))
    np.testing.assert_allclose(out_exit, np.tanh(np.ones((2, 4))), atol=1e-5)

    exit_unpooled = VitExitLayers(
        hidden_dim=8, output_dim=8, pooled=False, pre_ln=False, output_fc_tanh=False
    )
    np.testing.assert_allclose(exit_unpooled(tokens), tokens, atol=1e-5)

    vit = VisionTransformer()
    out_vit_img = vit(img)
    assert out_vit_img.shape == (2, 12)
    out_vit_tok = vit(tokens)
    assert out_vit_tok.shape == (2, 4, 8)
    np.testing.assert_allclose(np.mean(out_vit_tok, axis=-1), np.zeros((2, 4)), atol=0.1)

    cm = ClassificationModel()
    probs_cm = cm(tokens[:, 0, :])
    expected_probs_cm = np.exp(tokens[:, 0, :]) / np.sum(
        np.exp(tokens[:, 0, :]), axis=-1, keepdims=True
    )
    np.testing.assert_allclose(probs_cm, expected_probs_cm, atol=1e-5)
    np.testing.assert_allclose(
        ClassificationModel(network_tpl=Identity(), softmax_tpl=Identity())(tokens),
        tokens,
        atol=1e-5,
    )

    cmlp = ClassificationMLPModel()
    probs_cmlp = cmlp(tokens[:, 0, :])
    np.testing.assert_allclose(probs_cmlp, np.full((2, 8), 1.0 / 8.0), atol=1e-5)
    np.testing.assert_allclose(
        ClassificationMLPModel(mlp_tpl=Identity(), softmax_tpl=Identity())(tokens),
        tokens,
        atol=1e-5,
    )

    x_tr = np.arange(1, 25, dtype=np.float32).reshape((2, 3, 4)) * 0.1
    vnet_custom_stem = VanillaNet(
        strides=(1,),
        channels=(4,),
        blocks=(1,),
        kernels=(3,),
        entryflow_conv_kernel=(3, 3),
        entryflow_conv_stride=(1, 1),
    )
    assert vnet_custom_stem(np.ones((1, 4, 4, 4), dtype=np.float32)).shape == (1, 4, 4, 4)

    rnet_no_stem = ResNet(strides=(1,), channels=(1,), blocks=(1,), kernels=(3,))
    np.testing.assert_allclose(
        rnet_no_stem(np.ones((1, 4, 4, 4), dtype=np.float32)),
        np.ones((1, 2, 2, 4)),
        atol=1e-5,
    )
    rnet_custom_stem = ResNet(
        strides=(1,),
        channels=(1,),
        blocks=(1,),
        kernels=(3,),
        entryflow_conv_kernel=(3, 3, 3),
        entryflow_conv_stride=(1, 1),
        entry_max_pool=False,
    )
    assert rnet_custom_stem(np.ones((1, 4, 4, 4), dtype=np.float32)).shape == (1, 4, 4, 4)

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

    vit_entry_mismatch_pe = VitEntryLayers(
        pos_emb_shapes=(1, 3),
        output_dims=4,
        pos_emb_tpl=lambda x, *a, **kw: t_ops.zeros((1, 1, 1)),
    )
    np.testing.assert_allclose(
        vit_entry_mismatch_pe(np.ones((1, 3, 4), dtype=np.float32)),
        np.ones((1, 3, 4)),
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
        output_dropout_prob=0.2,
        ln_tpl=Identity(),
        pooling_tpl=GlobalPooling(pooling_type="AVG", pooling_dims=[1]),
    )
    np.testing.assert_allclose(
        vit_exit_custom(
            np.ones((1, 3, 4), dtype=np.float32), w_out=np.ones((4, 2), dtype=np.float32)
        ),
        np.full((1, 2), 4.0),
    )

    rblk_dp = ResNetBlock(input_dim=4, output_dim=1, residual_droppath_prob=0.2)
    assert rblk_dp(np.ones((1, 2, 2, 4), dtype=np.float32)).shape == (1, 2, 2, 4)

    rnet_feats = ResNet(
        strides=(1,), channels=(1,), blocks=(1,), kernels=(3,), return_block_features=True
    )
    assert rnet_feats(np.ones((1, 4, 4, 4), dtype=np.float32)).shape == (1, 2, 2, 4)
    assert "block_feature_0" in rnet_feats.params

    conf_drop = Conformer(
        input_dims=4,
        dropout_prob=0.1,
        remat=True,
    )
    assert conf_drop(x_tr).shape == (2, 3, 4)
    assert "remat_enabled" in conf_drop.params

    vit_entry_drop = VitEntryLayers(
        pos_emb_shapes=(1, 3),
        input_dims=4,
        output_dims=4,
        pos_emb_dropout_prob=0.1,
        input_fc_has_bias=False,
    )
    assert vit_entry_drop(np.ones((1, 3, 4), dtype=np.float32)).shape == (1, 3, 4)

    vit_dp = VisionTransformer(full_data_parallel_on_entry_exit=True)
    assert vit_dp(tokens).shape == (2, 4, 8)
    assert "full_data_parallel" in vit_dp.params

    cm_dict = ClassificationModel(input_field="image", label_field="label_probs")
    loss_dict = cm_dict(
        {
            "image": np.zeros((2, 4), dtype=np.float32),
            "label_probs": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        }
    )
    np.testing.assert_allclose(loss_dict, np.full((2,), np.log(4.0)), atol=1e-5)
    loss_kw = cm_dict(
        {"other": np.zeros((2, 4), dtype=np.float32)},
        label_probs=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert loss_kw.shape == (2,)
