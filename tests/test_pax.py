"""Tests for top-level zero_pax API and domain types."""

from __future__ import annotations

import numpy as np
import pytest

import zero_pax
from zero_pax import BaseParameter, LanguageModelType, Layer
from zero_pax import _tensor_ops as t_ops
from zero_pax.praxis.base import BasePraxisLayer, as_tensor_if_present, instantiate_layer
from zero_pax.praxis.layers import AutodiffCheckpointType, Identity, LayerNorm, Linear, ReLU
from zero_pax.types import PaxConfig, WeightInit


def test_top_level_api_and_parameter() -> None:
    """Verify `zero_pax.__call__`, `init_weights`, `BaseParameter`, and `Layer`."""
    assert zero_pax.__call__() is None
    t_direct = zero_pax.__call__([1.0, 2.0])
    assert t_direct is not None and t_direct.shape == (2,)
    np.testing.assert_allclose(t_direct, np.array([1.0, 2.0]))

    t_pos = zero_pax.__call__(None, [3.0, 4.0])
    assert t_pos is not None and t_pos.shape == (2,)
    np.testing.assert_allclose(t_pos, np.array([3.0, 4.0]))

    assert zero_pax.init_weights() == {}

    param = BaseParameter(value=3.5)
    assert param() == 3.5

    layer = Layer(name="root")
    assert layer.name == "root"
    layer.register_parameter("w", 2.0)
    assert layer.params["w"]() == 2.0
    np.testing.assert_allclose(layer([1.0, 2.0]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(layer.fprop([1.0, 2.0]), np.array([1.0, 2.0]))

    weights_dict = zero_pax.init_weights(layer)
    assert "w" in weights_dict
    assert float(weights_dict["w"]) == 2.0


def test_weight_init_and_base_helpers() -> None:
    """Verify `WeightInit.init_tensor`, `PaxConfig`, `instantiate_layer`, and `as_tensor_if_present`."""
    zeros_t = WeightInit(method="zeros").init_tensor((2, 3))
    np.testing.assert_allclose(zeros_t, np.zeros((2, 3)))

    ones_t = WeightInit(method="ones", scale=2.5).init_tensor((2, 2))
    np.testing.assert_allclose(ones_t, np.full((2, 2), 2.5))

    const_t = WeightInit(method="constant", scale=-1.5).init_tensor((2, 2))
    np.testing.assert_allclose(const_t, np.full((2, 2), -1.5))

    zero_scale_t = WeightInit(method="xavier", scale=0.0).init_tensor((2, 2))
    np.testing.assert_allclose(zero_scale_t, np.zeros((2, 2)))

    xavier_t = WeightInit(method="xavier", scale=1.0, seed=7).init_tensor((3, 4))
    assert xavier_t.shape == (3, 4)
    assert np.any(np.asarray(xavier_t) != 0.0)

    unif_t = WeightInit(method="uniform", scale=0.5, seed=3).init_tensor((2, 3))
    assert unif_t.shape == (2, 3)
    assert np.all(np.abs(np.asarray(unif_t)) <= 0.5 + 1e-5)

    kaiming_u = WeightInit(method="kaiming_uniform", scale=1.0).init_tensor((4,))
    assert kaiming_u.shape == (4,)

    normal_t = WeightInit(method="normal", scale=0.2).init_tensor((2, 2))
    assert normal_t.shape == (2, 2)
    assert np.any(np.asarray(normal_t) != 0.0)

    glorot_n = WeightInit(method="glorot_normal", scale=1.0).init_tensor((2, 3))
    assert glorot_n.shape == (2, 3)

    kaiming_n = WeightInit(method="kaiming_normal", scale=1.0).init_tensor((2, 3))
    assert kaiming_n.shape == (2, 3)

    custom_t = WeightInit(method="orthogonal", scale=1.0).init_tensor((2, 2))
    assert custom_t.shape == (2, 2)

    base = BasePraxisLayer()
    base.setup()
    np.testing.assert_allclose(base([1.0, 2.0]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(base.fprop([1.0, 2.0]), np.array([1.0, 2.0]))

    v_created = base.get_variable("k", (2, 2), init=WeightInit(method="ones", scale=3.0))
    np.testing.assert_allclose(v_created, np.full((2, 2), 3.0))
    v_cached = base.get_variable("k", (2, 2))
    np.testing.assert_allclose(v_cached, np.full((2, 2), 3.0))

    assert as_tensor_if_present(None) is None
    t_present = as_tensor_if_present([5.0])
    assert t_present is not None and float(t_present[0]) == 5.0

    cfg = PaxConfig(cls=LayerNorm, params={"dim": 4, "use_scale": False, "use_bias": False})
    cloned = cfg.clone()
    inst = cloned.instantiate()
    assert isinstance(inst, LayerNorm) and inst.dim == 4

    with pytest.raises(ValueError, match="cls is None"):
        PaxConfig().instantiate()

    resolved_cfg = instantiate_layer(cfg, Identity())
    assert isinstance(resolved_cfg, LayerNorm)
    resolved_cls = instantiate_layer(ReLU, Identity())
    assert isinstance(resolved_cls, ReLU)
    resolved_inst = instantiate_layer(Linear(input_dims=2, output_dims=2), Identity())
    assert isinstance(resolved_inst, Linear)
    resolved_none = instantiate_layer(None, Identity())
    assert isinstance(resolved_none, Identity)

    lm_type = LanguageModelType()
    assert lm_type.CAUSAL == "causal"
    assert lm_type.PREFIX == "prefix"
    assert lm_type.BIDIRECTIONAL == "bidirectional"
    assert "causal" in LanguageModelType.values()
    np.testing.assert_allclose(lm_type([1.0, 2.0]), np.array([1.0, 2.0]))

    ckpt_type = AutodiffCheckpointType()
    assert "save_nothing" in ckpt_type.values()
    np.testing.assert_allclose(ckpt_type([1.0, 2.0]), np.array([1.0, 2.0]))


def test_tensor_ops_wrappers() -> None:
    """Verify all tensor operation wrappers in `zero_pax._tensor_ops`."""
    assert t_ops.set_eager_mode(True) is True
    a = t_ops.array([[1.0, 3.0], [4.0, 2.0]])
    b = t_ops.asarray([[2.0, 0.0], [1.0, 5.0]])
    np.testing.assert_allclose(t_ops.matmul(a, b), np.array([[5.0, 15.0], [10.0, 10.0]]))
    np.testing.assert_allclose(t_ops.dot(a, b), np.array([[5.0, 15.0], [10.0, 10.0]]))
    np.testing.assert_allclose(t_ops.roll(a, 1, axis=-1), np.array([[3.0, 1.0], [2.0, 4.0]]))
    np.testing.assert_allclose(t_ops.min(a, axis=-1), np.array([1.0, 2.0]))
    np.testing.assert_allclose(t_ops.argmin(a, axis=-1), np.array([0, 1]))
    np.testing.assert_allclose(t_ops.argmin(a, axis=-1, keepdims=True), np.array([[0], [1]]))
    np.testing.assert_allclose(t_ops.argmax(a, axis=-1), np.array([1, 0]))
    np.testing.assert_allclose(t_ops.argmax(a, axis=-1, keepdims=True), np.array([[1], [0]]))
    np.testing.assert_allclose(t_ops.full((2, 2), 7.0), np.full((2, 2), 7.0))
    np.testing.assert_allclose(t_ops.ones_like(a), np.ones((2, 2)))
    np.testing.assert_allclose(t_ops.transpose(a), np.array([[1.0, 4.0], [3.0, 2.0]]))
