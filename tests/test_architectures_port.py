"""Tests for Vision, Audio, and Classification architectures in zero_pax.praxis.layers.architectures."""

from __future__ import annotations

import numpy as np

from zero_pax.praxis.layers import (
    GELU,
    ClassificationMLPModel,
    ClassificationModel,
    Conformer,
    GlobalPooling,
    Identity,
    ResNet,
    ResNetBlock,
    VanillaBlock,
    VanillaNet,
    VisionTransformer,
    VitEntryLayers,
    VitExitLayers,
)


def test_vanilla_and_resnet() -> None:
    """Verify `VanillaBlock`, `VanillaNet`, `ResNetBlock`, and `ResNet` including multi-stage execution."""
    img = np.ones((2, 4, 4, 8), dtype=np.float32)

    vb = VanillaBlock(input_dim=8, output_dim=4, kernel_size=3, stride=2)
    assert vb(img).shape == (2, 2, 2, 4)

    vnet_default = VanillaNet()
    assert vnet_default(img).shape == (2, 4, 4, 8)

    vnet_staged = VanillaNet(
        strides=(1, 2),
        channels=(4, 8),
        blocks=(1, 1),
        kernels=(3, 3),
        entryflow_conv_params=Identity(),
        output_spatial_pooling_params=GlobalPooling(pooling_dims=(1, 2)),
    )
    out_vnet = vnet_staged(img)
    assert out_vnet.shape == (2, 8)

    rb_same = ResNetBlock(input_dim=8, output_dim=2, stride=1)
    np.testing.assert_allclose(rb_same(img), img, atol=1e-5)

    rb_proj = ResNetBlock(input_dim=8, output_dim=4, stride=2, activation_tpl=GELU())
    assert rb_proj(img).shape == (2, 2, 2, 16)

    rnet_default = ResNet()
    np.testing.assert_allclose(rnet_default(img), img, atol=1e-5)

    rnet_staged = ResNet(
        strides=(1, 2),
        channels=(2, 4),
        blocks=(1, 1),
        kernels=(3, 3),
        entryflow_conv_params=Identity(),
        entry_max_pool=True,
        output_spatial_pooling_params=GlobalPooling(pooling_dims=(1, 2)),
    )
    out_rnet = rnet_staged(img)
    assert out_rnet.shape == (2, 16)


def test_conformer() -> None:
    """Verify `Conformer` across all four `layer_order` configurations and kernel sizes."""
    seq = np.arange(1, 65, dtype=np.float32).reshape((2, 4, 8)) * 0.1
    paddings = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    for order in ("mhsa_before_conv", "conv_before_mhsa", "mhsa", "conv"):
        conf = Conformer(
            input_dims=8,
            atten_num_heads=2,
            kernel_size=7,
            layer_order=order,
            ff_activation_tpl=GELU(),
        )
        out = conf(seq, paddings=paddings)
        assert out.shape == (2, 4, 8)
        np.testing.assert_allclose(out[0, 3], np.zeros((8,)), atol=1e-5)
        assert np.any(np.asarray(out[0, :3]) != 0.0)

    assert Conformer(model_dims=8)(seq).shape == (2, 4, 8)


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

    assert VitEntryLayers(pos_emb_shapes=(2, 2), prepend_cls_tokens=1, append_cls_tokens=1)(
        tokens
    ).shape == (
        2,
        6,
        8,
    )

    exit_pooled = VitExitLayers(hidden_dim=8, output_dim=4, pooled=True)
    out_exit = exit_pooled(tokens, b_out=np.ones((4,), dtype=np.float32))
    assert out_exit.shape == (2, 4)

    exit_unpooled = VitExitLayers(hidden_dim=8, output_dim=8, pooled=False)
    assert exit_unpooled(tokens).shape == (2, 4, 8)

    vit = VisionTransformer()
    assert vit(img).shape == (2, 12)
    assert vit(tokens).shape == (2, 4, 8)

    cm = ClassificationModel()
    probs_cm = cm(tokens[:, 0, :])
    np.testing.assert_allclose(np.sum(probs_cm, axis=-1), np.ones((2,)), atol=1e-5)
    assert ClassificationModel(network_tpl=Identity(), softmax_tpl=Identity())(tokens).shape == (
        2,
        4,
        8,
    )

    cmlp = ClassificationMLPModel()
    probs_cmlp = cmlp(tokens[:, 0, :])
    np.testing.assert_allclose(np.sum(probs_cmlp, axis=-1), np.ones((2,)), atol=1e-5)
    assert ClassificationMLPModel(mlp_tpl=Identity(), softmax_tpl=Identity())(tokens).shape == (
        2,
        4,
        8,
    )
