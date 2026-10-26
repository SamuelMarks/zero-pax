"""Tests for top-level zero_pax API, domain types, and static verification scripts."""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

import zero_pax
from scripts import (
    check_compliance,
    check_dependencies,
    check_docstrings,
    lint_typing_and_hacks,
)
from tests import conftest as np
from zero_pax import BaseParameter, LanguageModelType, Layer
from zero_pax import _tensor_ops as t_ops
from zero_pax.praxis.base import BasePraxisLayer, as_tensor_if_present, instantiate_layer
from zero_pax.praxis.layers import AutodiffCheckpointType, Identity, LayerNorm, Linear, ReLU
from zero_pax.types import (
    ActivationSharding,
    DecoderHParams,
    DType,
    PaxConfig,
    WeightInit,
    WeightSharding,
)


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
    sub_child = Layer(name="sub_child")
    sub_child.register_parameter("b", 5.0)
    layer.sublayers["sub_child"] = sub_child
    layer.sublayers["fn_sub"] = instantiate_layer(lambda x: x, Identity())
    assert layer.params["w"]() == 2.0
    np.testing.assert_allclose(layer([1.0, 2.0]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(layer.fprop([1.0, 2.0]), np.array([1.0, 2.0]))

    weights_dict = zero_pax.init_weights(layer)
    assert "w" in weights_dict and "sub_child/b" in weights_dict
    assert float(weights_dict["w"]) == 2.0
    assert float(weights_dict["sub_child/b"]) == 5.0


def test_weight_init_statistical_and_base_helpers() -> None:
    """Verify `WeightInit` distribution statistics, `PaxConfig`, `instantiate_layer`, and `Enum` types."""
    zeros_t = WeightInit(method="zeros").init_tensor((2, 3), dtype="float32")
    np.testing.assert_allclose(zeros_t, np.zeros((2, 3)))

    ones_t = WeightInit(method="ones", scale=2.5).init_tensor((2, 2))
    np.testing.assert_allclose(ones_t, np.full((2, 2), 2.5))

    const_t = WeightInit(method="constant", scale=-1.5).init_tensor((2, 2))
    np.testing.assert_allclose(const_t, np.full((2, 2), -1.5))

    zero_scale_t = WeightInit(method="xavier", scale=0.0).init_tensor((2, 2))
    np.testing.assert_allclose(zero_scale_t, np.zeros((2, 2)))

    xavier_t = WeightInit(method="xavier", scale=1.0, seed=7).init_tensor((3, 4))
    assert xavier_t.shape == (3, 4)
    np.testing.assert_allclose(np.mean(xavier_t), 0.0, atol=0.35)

    unif_t = WeightInit(method="uniform", scale=0.5, seed=3).init_tensor((100, 10))
    assert unif_t.shape == (100, 10)
    assert np.all(np.abs(np.asarray(unif_t)) <= 0.5 + 1e-5)
    np.testing.assert_allclose(np.mean(unif_t), 0.0, atol=0.05)
    np.testing.assert_allclose(np.var(unif_t), (0.5**2) / 3.0, atol=0.015)

    kaiming_u = WeightInit(method="kaiming_uniform", scale=1.0).init_tensor((4,))
    assert kaiming_u.shape == (4,)

    normal_t = WeightInit(method="normal", scale=1.0).init_tensor((100, 10))
    assert normal_t.shape == (100, 10)
    np.testing.assert_allclose(np.mean(normal_t), 0.0, atol=0.05)
    np.testing.assert_allclose(np.var(normal_t), 1.0, atol=0.05)

    glorot_n = WeightInit(method="glorot_normal", scale=1.0).init_tensor((20, 30))
    np.testing.assert_allclose(np.var(glorot_n), 2.0 / 50.0, atol=0.01)

    kaiming_n = WeightInit(method="kaiming_normal", scale=1.0).init_tensor((20, 30))
    np.testing.assert_allclose(np.var(kaiming_n), 2.0 / 20.0, atol=0.015)

    custom_t = WeightInit(method="orthogonal", scale=1.0).init_tensor((3, 3))
    assert custom_t.shape == (3, 3)
    np.testing.assert_allclose(
        t_ops.matmul(custom_t, t_ops.transpose(custom_t)), np.eye(3), atol=1e-5
    )

    wide_ortho = WeightInit(method="orthogonal", scale=2.0).init_tensor((2, 4))
    assert wide_ortho.shape == (2, 4)
    np.testing.assert_allclose(
        t_ops.matmul(wide_ortho, t_ops.transpose(wide_ortho)), np.eye(2) * 4.0, atol=1e-5
    )

    fallback_t = WeightInit(method="custom_unknown", scale=1.0).init_tensor((2, 2))
    assert fallback_t.shape == (2, 2)

    base = BasePraxisLayer(dtype="float64", fprop_dtype="float32")
    assert base.effective_fprop_dtype == "float32"
    sub_inherited = base.get_sublayer("sub_inh", None, Identity())
    assert (
        isinstance(sub_inherited, BasePraxisLayer)
        and sub_inherited.dtype == "float64"
        and sub_inherited.fprop_dtype == "float32"
    )
    base_default = BasePraxisLayer()
    assert base_default.effective_fprop_dtype == "float32"
    base.setup()
    np.testing.assert_allclose(base([1.0, 2.0]), np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(base.fprop([1.0, 2.0]), np.array([1.0, 2.0], dtype=np.float32))

    v_created = base.get_variable("k", (2, 2), init=WeightInit(method="ones", scale=3.0))
    np.testing.assert_allclose(v_created, np.full((2, 2), 3.0))
    v_cached = base.get_variable("k", (2, 2))
    np.testing.assert_allclose(v_cached, np.full((2, 2), 3.0))

    child = BasePraxisLayer(parent=base, name="child")
    child_v = child.get_variable("w", (2,), init=WeightInit(method="ones", scale=4.0))
    np.testing.assert_allclose(child_v, np.array([4.0, 4.0]))
    assert "child/w" in base.params
    grandchild = BasePraxisLayer(parent=child, name="grand")
    grand_v = grandchild.get_variable("g", (2,), init=WeightInit(method="ones", scale=6.0))
    np.testing.assert_allclose(grand_v, np.array([6.0, 6.0]))
    assert "child/grand/g" in base.params
    grandchild2 = BasePraxisLayer(parent=child, name="grand")
    np.testing.assert_allclose(grandchild2.get_variable("g", (2,)), np.array([6.0, 6.0]))
    child2 = BasePraxisLayer(parent=base, name="child")
    np.testing.assert_allclose(child2.get_variable("w", (2,)), np.array([4.0, 4.0]))
    np.testing.assert_allclose(
        child2.get_variable("w", (3,), init=WeightInit(method="ones", scale=2.0)),
        np.array([2.0, 2.0, 2.0]),
    )

    assert as_tensor_if_present(None) is None
    t_present = as_tensor_if_present([5.0])
    assert t_present is not None and float(t_present[0]) == 5.0

    cfg = PaxConfig(cls=LayerNorm, params={"dim": 4, "use_scale": False, "use_bias": False})
    cfg.set(epsilon=1e-4)
    assert cfg.params["epsilon"] == 1e-4
    cloned = cfg.clone()
    inst = cloned.instantiate()
    assert isinstance(inst, LayerNorm) and inst.dim == 4 and inst.epsilon == 1e-4

    list_param = [1, 2]
    tensor_param = t_ops.array([1.0, 2.0])
    sub_cfg: PaxConfig[LayerNorm] = PaxConfig(cls=LayerNorm, params={"dim": 4})
    dict_param: dict[str, int] = {"a": 1}
    model_param = WeightInit(method="ones", scale=1.0)
    cfg_nested: PaxConfig[BasePraxisLayer] = PaxConfig(
        cls=BasePraxisLayer,
        params={
            "ici_mesh_shape": list_param,
            "dcn_mesh_shape": (1, 2),
            "name": tensor_param,
            "sub": sub_cfg,
            "map": dict_param,
            "params_init": model_param,
        },
    )
    cloned_nested = cfg_nested.clone()
    list_param.append(3)
    dict_param["a"] = 99
    assert cloned_nested.params["ici_mesh_shape"] == [1, 2]
    assert cloned_nested.params["dcn_mesh_shape"] == (1, 2)
    assert isinstance(cloned_nested.params["sub"], PaxConfig)
    assert cloned_nested.params["map"] == {"a": 1}
    assert isinstance(cloned_nested.params["params_init"], WeightInit)

    class CustomSubLayer(BasePraxisLayer):
        """Test subclass of BasePraxisLayer."""

    cfg_base = PaxConfig(cls=CustomSubLayer)
    assert isinstance(instantiate_layer(cfg_base, Identity()), CustomSubLayer)
    assert DecoderHParams().seqlen == 0

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

    assert LanguageModelType.CAUSAL.value == "causal"
    assert LanguageModelType.PREFIX.value == "prefix"
    assert LanguageModelType.BIDIRECTIONAL.value == "bidirectional"
    assert LanguageModelType.values() == ("causal", "prefix", "bidirectional")
    assert "save_nothing" in AutodiffCheckpointType.values()


def test_tensor_ops_wrappers() -> None:
    """Verify all tensor operation wrappers and `resolve_dtype` in `zero_pax._tensor_ops`."""
    assert t_ops.set_eager_mode(True) is True
    compiler_f32 = t_ops.resolve_dtype("float32")
    assert compiler_f32 is not None
    assert t_ops.resolve_dtype(compiler_f32) == compiler_f32
    with pytest.raises(ValueError, match="Unsupported dtype"):
        t_ops.resolve_dtype("invalid_dtype")

    a = t_ops.array([[1.0, 3.0], [4.0, 2.0]], dtype="float32")
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

    assert t_ops.resolve_dtype("fp32") == t_ops.DType.float32
    assert t_ops.resolve_dtype(None) is None
    assert t_ops.zeros(3).shape == (3,)
    assert t_ops.asarray(None).shape == ()
    assert t_ops.asarray([t_ops.asarray(1.0), t_ops.asarray(2.0)]).shape == (2,)
    assert t_ops.asarray(a, dtype="float64").dtype == t_ops.DType.float64
    assert t_ops.asarray(a).dtype == t_ops.DType.float32
    assert a.size == 4
    assert a.tolist() == [[1.0, 3.0], [4.0, 2.0]]
    assert a.copy().tolist() == [[1.0, 3.0], [4.0, 2.0]]
    assert a.__array_interface__["shape"] == (2, 2)

    class _MockArrayLike:
        """Array-like stub implementing `ToListProtocol`."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Return the array shape.

            Returns:
                Shape tuple `(2,)`.
            """
            return (2,)

        def tolist(self) -> list[float]:
            """Return nested list elements.

            Returns:
                List of floats.
            """
            return [9.0, 10.0]

    assert t_ops.asarray(_MockArrayLike()).tolist() == [9.0, 10.0]
    assert t_ops.asarray(5.0).tolist() == 5.0
    assert a.astype("int32").dtype == t_ops.DType.int32
    assert a.reshape(4).shape == (4,)
    assert a.reshape((1, 4)).shape == (1, 4)
    assert int(t_ops.asarray(3.0)) == 3
    assert bool(t_ops.asarray([0.0, 1.0])) is True
    assert bool(t_ops.asarray([0.0, 0.0])) is False
    assert len(a) == 2
    assert [row.shape for row in a] == [(2,), (2,)]
    assert "Tensor(shape=(2, 2)" in repr(a)

    with pytest.raises(TypeError, match="0D tensor"):
        len(t_ops.asarray(1.0))
    with pytest.raises(TypeError, match="0-d tensor"):
        list(t_ops.asarray(1.0))

    # Empty sequence / 0-size tensor handling
    empty_t = t_ops.asarray([])
    assert empty_t.shape == (0,)
    assert t_ops.broadcast_to(empty_t, (2, 0)).shape == (2, 0)

    # Broadcasting error branches
    with pytest.raises(ValueError, match="could not be broadcast"):
        _ = t_ops.ones((2, 3)) + t_ops.ones((2, 4))
    with pytest.raises(ValueError, match="smaller rank"):
        t_ops.broadcast_to(t_ops.ones((2, 3)), (3,))
    with pytest.raises(ValueError, match="Cannot broadcast shape"):
        t_ops.broadcast_to(t_ops.ones((2, 3)), (2, 4))

    # Indexing branches: fancy indexing, negative indices, out-of-bounds, multiple ellipsis, None, sequence axis
    table = t_ops.arange(6).reshape(3, 2)
    np.testing.assert_allclose(table[[0, -1]], np.array([[0.0, 1.0], [4.0, 5.0]]))
    with pytest.raises(IndexError, match="out of bounds"):
        _ = table[[5]]
    with pytest.raises(IndexError, match="out of bounds"):
        _ = table[5, 0]
    np.testing.assert_allclose(table[..., ...], np.asarray(table))
    np.testing.assert_allclose(table[:, [1, -2]], np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]))
    scalar_t = t_ops.asarray(4.0)
    assert scalar_t[None].shape == (1,)

    # Unary, binary, division, floordiv, pow, comparison, and logical operators
    np.testing.assert_allclose(+a, np.asarray(a))
    np.testing.assert_allclose(abs(-a), np.asarray(a))
    np.testing.assert_allclose(a / 2.0, np.asarray(a) / 2.0)
    np.testing.assert_allclose(a // 2.0, np.asarray(a) // 2.0)
    np.testing.assert_allclose(10.0 // a, 10.0 // np.asarray(a))
    np.testing.assert_allclose(2.0**a, 2.0 ** np.asarray(a))
    assert float(t_ops.asarray(0.0) / 0.0) == 0.0
    assert math.isinf(float(t_ops.asarray(1.0) / 0.0))
    assert float(t_ops.asarray(-4.0) ** 0.5) == 0.0
    np.testing.assert_allclose(a @ b, np.array([[5.0, 15.0], [10.0, 10.0]]))
    np.testing.assert_allclose((a <= 2.0) & (a != 4.0), np.array([[1.0, 0.0], [0.0, 1.0]]))

    # arange with start/stop and negative step
    np.testing.assert_allclose(t_ops.arange(5.0, 1.0, -2.0), np.array([5.0, 3.0]))
    assert t_ops.arange(1.0, 5.0, 0.0).shape == (0,)

    # reshape error, transpose 1D, split unequal & sequence split, roll identity
    with pytest.raises(ValueError, match="Cannot reshape"):
        t_ops.reshape(a, (3, 3))
    np.testing.assert_allclose(t_ops.transpose(t_ops.arange(3)), np.arange(3))
    with pytest.raises(ValueError, match="equal division"):
        t_ops.split(t_ops.arange(5), 2)
    parts_seq = t_ops.split(t_ops.arange(5), [2, 4])
    assert [p.shape for p in parts_seq] == [(2,), (2,), (1,)]
    np.testing.assert_allclose(t_ops.roll(t_ops.arange(3), 0), np.arange(3))

    # matmul 1D/2D branches and error branches
    v1 = t_ops.array([1.0, 2.0])
    v2 = t_ops.array([3.0, 4.0])
    assert float(t_ops.matmul(v1, v2)) == 11.0
    np.testing.assert_allclose(t_ops.matmul(a, v1), np.array([7.0, 8.0]))
    np.testing.assert_allclose(t_ops.matmul(v1, a), np.array([9.0, 7.0]))
    with pytest.raises(ValueError, match="1D dot"):
        t_ops.matmul(v1, t_ops.arange(3))
    with pytest.raises(ValueError, match="Shape mismatch in matmul"):
        t_ops.matmul(a, t_ops.arange(3))
    with pytest.raises(ValueError, match="Shape mismatch in matmul"):
        t_ops.matmul(t_ops.arange(3), a)
    with pytest.raises(ValueError, match="Contraction dimension mismatch"):
        t_ops.matmul(t_ops.ones((2, 3)), t_ops.ones((4, 2)))

    # einsum implicit output, broadcast dimension, and error branches
    np.testing.assert_allclose(t_ops.einsum("ij,jk", a, b), np.array([[5.0, 15.0], [10.0, 10.0]]))
    np.testing.assert_allclose(
        t_ops.einsum("ij,ij->ij", t_ops.ones((1, 2)), t_ops.ones((3, 2))),
        np.ones((3, 2)),
    )
    np.testing.assert_allclose(
        t_ops.einsum("ij,ij->ij", t_ops.ones((3, 2)), t_ops.ones((1, 2))),
        np.ones((3, 2)),
    )
    with pytest.raises(ValueError, match="Number of einsum subscripts"):
        t_ops.einsum("ij,jk->ik", a)
    with pytest.raises(ValueError, match="does not match operand shape"):
        t_ops.einsum("ijk->ijk", a)
    with pytest.raises(ValueError, match="Dimension size mismatch"):
        t_ops.einsum("ij,jk->ik", t_ops.ones((2, 3)), t_ops.ones((4, 2)))

    # Reduction empty axis, axis=None argmin/argmax
    np.testing.assert_allclose(t_ops.sum(a, axis=()), np.asarray(a))
    assert float(t_ops.argmin(t_ops.array([3.0, 1.0, 2.0]), axis=None)) == 1.0
    assert t_ops.argmin(t_ops.array([[3.0, 1.0]]), axis=None, keepdims=True).shape == (1, 1)
    assert float(t_ops.argmax(t_ops.array([1.0, 5.0, 2.0]), axis=None)) == 1.0
    assert t_ops.argmax(t_ops.array([[1.0, 5.0]]), axis=None, keepdims=True).shape == (1, 1)
    assert int(t_ops.argmin([3.0, 1.0, 2.0])) == 1
    assert int(t_ops.argmax([3.0, 1.0, 2.0])) == 0

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
    with pytest.raises(ValueError, match="Unsupported pad mode"):
        t_ops.pad([[1.0, 2.0]], ((0, 0), (1, 1)), mode="reflect")

    # Repeated einsum diagonal / trace indices and uppercase ellipsis equations
    np.testing.assert_allclose(t_ops.einsum("ii->", a), 3.0)
    np.testing.assert_allclose(t_ops.einsum("ii->i", a), np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        t_ops.einsum("...AB,BC->...AC", t_ops.ones((2, 3, 4)), t_ops.ones((4, 5))),
        np.full((2, 3, 5), 4.0),
    )

    # Bool and integer tolist and promotion branches
    assert t_ops.asarray(1.0, dtype="bool_").tolist() is True
    assert t_ops.asarray(0.0, dtype="bool_").tolist() is False
    assert t_ops.asarray(3.7, dtype="int32").tolist() == 3
    assert t_ops.asarray([0.0, 2.0], dtype="bool_").tolist() == [False, True]
    assert t_ops.asarray([1.9, -2.8], dtype="int32").tolist() == [1, -2]
    assert (
        t_ops.asarray([1.0], dtype="float64") + t_ops.asarray([2.0], dtype="float32")
    ).dtype == t_ops.DType.float64
    assert (
        t_ops.asarray([1], dtype="int32") + t_ops.asarray([2.0], dtype="float32")
    ).dtype == t_ops.DType.float32
    assert (
        t_ops.asarray([1], dtype="int32") + t_ops.asarray([1], dtype="bool_")
    ).dtype == t_ops.DType.int32
    assert (
        t_ops.asarray([1], dtype="bool_") + t_ops.asarray([1], dtype="int32")
    ).dtype == t_ops.DType.int32
    assert (
        t_ops.asarray([1], dtype="int32") + t_ops.asarray([2], dtype="int64")
    ).dtype == t_ops.DType.float32

    # Non-eager LogicalGraph tracing across all ops
    t_ops.reset_active_graph("TestTraceGraph")
    t_ops.set_eager_mode(False)
    try:
        assert t_ops.get_eager_mode() is False
        tr_x = t_ops.array([[1.0, 2.0], [3.0, 4.0]])
        tr_y = (tr_x + 1.0) @ t_ops.eye(2)
        tr_z = t_ops.exp(tr_y).astype("float32")
        assert tr_z.node_id is not None
        assert t_ops.reshape(tr_z, (4,)).node_id is not None
        assert t_ops.transpose(tr_z).node_id is not None
        assert t_ops.broadcast_to(t_ops.array([1.0, 2.0]), (2, 2)).node_id is not None
        assert t_ops.expand_dims(tr_z, 0).node_id is not None
        assert t_ops.concatenate([tr_z, tr_z], axis=0).node_id is not None
        split_parts = t_ops.split(tr_z, 2, axis=0)
        assert all(p.node_id is not None for p in split_parts)
        assert t_ops.roll(tr_z, 1, axis=1).node_id is not None
        assert t_ops.where(tr_z > 0.0, tr_z, 0.0).node_id is not None
        assert t_ops.sum(tr_z, axis=-1).node_id is not None
        assert t_ops.mean(tr_z, axis=-1).node_id is not None
        assert t_ops.var(tr_z, axis=-1).node_id is not None
        assert t_ops.max(tr_z, axis=-1).node_id is not None
        assert t_ops.min(tr_z, axis=-1).node_id is not None
        assert t_ops.argmin(tr_z, axis=-1).node_id is not None
        assert t_ops.argmax(tr_z, axis=-1).node_id is not None
        assert t_ops.cumsum(tr_z, axis=-1).node_id is not None
        assert tr_z[0].node_id is not None
        assert tr_z[t_ops.array([0, 1], dtype="int32")].node_id is not None
        shared_l1 = Linear(
            input_dims=2,
            output_dims=2,
            shared_weight_layer_id="shared_lin_trace",
            skip_lp_regularization=True,
            ici_mesh_shape=(1, 1),
            dcn_mesh_shape=(1, 1),
            contiguous_submeshes=True,
            mesh_axis_names=("data", "mdl"),
            weight_split_dims_mapping=WeightSharding(wt=["data", "mdl"]),
            activation_split_dims_mapping=ActivationSharding(out=["data", "mdl"]),
            weight_init=WeightInit(method="uniform", scale=0.5, seed=7),
        )
        shared_l2 = Linear(
            input_dims=2,
            output_dims=2,
            shared_weight_layer_id="shared_lin_trace",
        )
        out_s1 = shared_l1(tr_x)
        out_s2 = shared_l2(tr_x)
        np.testing.assert_allclose(out_s1, out_s2)
        assert out_s1.node_id is not None
        graph = t_ops.get_active_graph()
        assert graph.name == "TestTraceGraph"
        assert len(graph.nodes) >= 15
        for node in graph.nodes.values():
            assert node.source_ast_ref is not None
    finally:
        t_ops.set_eager_mode(True)


def test_zero_numpy_runtime_isolation() -> None:
    """Verify `zero_pax` executes tensor and layer math with `numpy` blocked in `sys.meta_path`."""
    import importlib.abc
    import importlib.machinery
    import sys

    class _ForbidNumpyAndCompilerFinder(importlib.abc.MetaPathFinder):
        """MetaPathFinder raising `ImportError` if `numpy` or `ml_switcheroo_compiler` is imported."""

        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None,
            target: ModuleType | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            """Intercept module resolution for forbidden runtime packages.

            Args:
                fullname: Dotted module name being imported.
                path: Optional parent package search path.
                target: Optional target module object.

            Returns:
                None for allowed modules.

            Raises:
                ImportError: If `fullname` starts with `numpy` or `ml_switcheroo_compiler`.
            """
            top = fullname.split(".")[0]
            if top in ("numpy", "ml_switcheroo_compiler"):
                raise ImportError(f"Forbidden runtime import attempted: {fullname}")
            return None

    finder = _ForbidNumpyAndCompilerFinder()
    saved_np = sys.modules.pop("numpy", None)
    sys.meta_path.insert(0, finder)
    try:
        with pytest.raises(ImportError, match="Forbidden runtime import"):
            __import__("numpy")
        x = t_ops.array([[1.0, -2.0, 3.0, 4.0], [0.5, 1.5, -0.5, 2.5]])
        linear = Linear(
            input_dims=4, output_dims=2, weight_init=WeightInit(method="ones", scale=0.5)
        )
        norm = LayerNorm(dim=2)
        act = ReLU()
        out = act(norm(linear(x)))
        assert out.shape == (2, 2)
        assert isinstance(out.tolist(), list)
    finally:
        sys.meta_path.remove(finder)
        if saved_np is not None:
            sys.modules["numpy"] = saved_np


def test_static_lint_and_docstring_scripts(tmp_path: Path) -> None:
    """Verify AST/comment detection in `lint_typing_and_hacks.py`, `check_docstrings.py`, `check_dependencies.py`, and `check_compliance.py`.

    Args:
        tmp_path: Temporary directory fixture.
    """
    bad_src = (
        "from typing import Any\n"
        "from typing import int as object\n"
        "class Meta(metaclass=type):\n"
        "    def __getattr__(self, k: str) -> int:\n"
        "        return self.__dict__[k]\n"
        "x = vars(y)\n"
    )
    bad_tree = ast.parse(bad_src)
    violations = lint_typing_and_hacks.check_ast(bad_tree, Path("dummy.py"))
    assert len(violations) >= 5
    assert any("Any" in v for v in violations)
    assert any("vars" in v for v in violations)
    assert any("metaclass" in v for v in violations)

    ignore_comment = "x: int = 1  #" + " type: ignore[assignment]"
    comment_violations = lint_typing_and_hacks.check_comments(ignore_comment, Path("dummy.py"))
    assert len(comment_violations) == 1

    bad_lint_file = tmp_path / "bad_lint.py"
    bad_lint_file.write_text(bad_src + "\n" + ignore_comment, encoding="utf-8")
    assert len(lint_typing_and_hacks.check_file(bad_lint_file)) >= 6
    assert lint_typing_and_hacks.collect_python_files([str(bad_lint_file)]) == [bad_lint_file]
    assert lint_typing_and_hacks.collect_python_files([str(tmp_path)]) == [bad_lint_file]
    assert lint_typing_and_hacks.collect_python_files([str(tmp_path / "nonexistent")]) == []
    assert lint_typing_and_hacks.main([str(bad_lint_file)]) == 1
    assert lint_typing_and_hacks.main(["src"]) == 0

    assert check_dependencies.is_allowed_module("pydantic")
    assert not check_dependencies.is_allowed_module("torch")
    assert not check_dependencies.is_allowed_module("grain", is_test=True)
    assert not check_dependencies.is_allowed_module("chex", is_test=True)
    assert not check_dependencies.is_allowed_module("orbax", is_test=True)
    assert not check_dependencies.is_allowed_module("optax", is_test=True)
    assert check_dependencies.is_allowed_module("pytest", is_test=True)

    bad_dep_file = tmp_path / "bad_dep.py"
    bad_dep_file.write_text(
        "import torch\nfrom numpy import array\nimport importlib\nimportlib.import_module('scipy')\n",
        encoding="utf-8",
    )
    dep_errs = check_dependencies.check_file(bad_dep_file)
    assert len(dep_errs) == 3
    syn_err_file = tmp_path / "syn_err.py"
    syn_err_file.write_text("def broken(:\n", encoding="utf-8")
    assert len(check_dependencies.check_file(syn_err_file)) == 1
    bad_req = tmp_path / "requirements-bad.txt"
    bad_req.write_text("# comment\n\ntorch>=2.0\npydantic>=2.0\n", encoding="utf-8")
    assert len(check_dependencies.check_requirements_file(bad_req)) == 1
    assert check_dependencies.check_requirements_file(tmp_path / "missing_req.txt") == []
    assert check_dependencies.main([str(bad_dep_file)]) == 1
    assert check_dependencies.main([str(tmp_path / "nonexistent")]) == 0
    assert check_dependencies.main([]) == 0

    assert check_compliance.verify_module_hierarchy() == []
    assert check_compliance.verify_module_hierarchy(submodules=("praxis.nonexistent_mod",)) != []
    assert check_compliance.verify_module_hierarchy(forbidden_siblings=("layers",)) != []
    assert check_compliance._find_snapshot_path(tmp_path) is None
    fake_snap = tmp_path / "pax_v1.4.0.json"
    fake_snap.write_text(
        '{"categories": {"neural": ['
        '{"name": "MissingLayer", "api_path": "praxis.layers.MissingLayer", "params": []},'
        '{"name": "LanguageModelType", "api_path": "praxis.layers.LanguageModelType", "params": ['
        '{"name": "MISSING_ENUM", "default": "\'x\'"},'
        '{"name": "CAUSAL", "default": "\'wrong_val\'"}'
        "]},"
        '{"name": "ReLU", "api_path": "praxis.layers.ReLU", "params": ['
        '{"name": "missing_field", "annotation": "int", "default": "1"},'
        '{"name": "dtype", "annotation": "Optional[str]", "default": "\'wrong_default\'"},'
        '{"name": "name", "annotation": "Sequence[int]", "default": "(1, 2)"}'
        "]},"
        '{"name": "Linear", "api_path": "praxis.layers.Linear", "params": ['
        '{"name": "params", "annotation": "Optional[int]", "default": "None"},'
        '{"name": "input_dims", "annotation": "Optional[int]", "default": "0"}'
        "]}"
        "]}}",
        encoding="utf-8",
    )
    assert check_compliance._find_snapshot_path(tmp_path) == fake_snap
    snap_errs = check_compliance.verify_snapshot_compliance(fake_snap)
    assert len(snap_errs) >= 5
    assert check_compliance._parse_expected_default(None) == (False, None)
    assert check_compliance._parse_expected_default("'```(None)```'") == (True, None)
    assert check_compliance._parse_expected_default("'(1, 2)'") == (True, (1, 2))
    assert check_compliance._parse_expected_default("'(1)'") == (True, "(1)")
    assert check_compliance._parse_expected_default("'ActivationType'") == (False, None)
    assert check_compliance._parse_expected_default("'768 * 256'") == (True, 196608)
    assert check_compliance._parse_expected_default("[1, 2]") == (False, None)
    assert check_compliance._parse_expected_default("'template_field(None)'") == (False, None)
    assert check_compliance._parse_expected_default("invalid syntax ???") == (False, None)
    assert check_compliance.verify_semantic_plan(tmp_path / "missing_plan.md") != []
    bad_plan = tmp_path / "bad_plan.md"
    bad_plan.write_text("* [x] Implement `NonExistentLayer` semantics\n", encoding="utf-8")
    assert check_compliance.verify_semantic_plan(bad_plan) != []
    assert check_compliance.main(plan_path=bad_plan, snapshot_path=fake_snap) == 1
    assert (
        check_compliance.main(
            plan_path=Path("SEMANTIC_PLAN.md"),
            snapshot_path=None,
            cache_root=tmp_path / "empty_cache",
        )
        == 1
    )
    assert check_compliance.main() == 0

    doc_args = check_docstrings._extract_documented_args(
        "\n".join(["Summary.", "", "Args:", "    alpha: First.", "", "Returns:", "    Int."])
    )
    assert doc_args == {"alpha"}
    doc_attrs = check_docstrings._extract_documented_attributes(
        "\n".join(["Summary.", "", "Attributes:", "    beta: Field."])
    )
    assert doc_attrs == {"beta"}
    bad_doc_file = tmp_path / "bad_doc.py"
    bad_doc_file.write_text(
        "class NoDoc:\n    pass\n"
        "class MissingAttr:\n    '''Doc.'''\n    x: int = 1\n"
        "def no_fn_doc(a: int) -> int:\n    return a\n"
        "def bad_fn(a: int, *args: int, **kwargs: int) -> int:\n"
        "    '''Doc.'''\n"
        "    def inner() -> None:\n        pass\n"
        "    raise ValueError('bad')\n",
        encoding="utf-8",
    )
    doc_errs = check_docstrings.check_file_docstrings(bad_doc_file)
    assert len(doc_errs) >= 5
    assert check_docstrings.main([str(bad_doc_file)]) == 1
    assert check_docstrings.main([str(tmp_path / "nonexistent")]) == 0
    assert check_docstrings.main([]) == 0

    assert t_ops.DType.float32.value == "float32" and DType is not None
    np.testing.assert_not_allclose(np.array([1.0, 2.0]), np.array([1.0, 5.0]))
    assert np.random.uniform(size=(2, 2), seed=11).shape == (2, 2)


def test_ir_tracing_and_eager_equivalence() -> None:
    """Verify end-to-end layer execution under both eager and `LogicalGraph` IR tracing modes."""
    x = np.array([[1.0, -2.0, 3.0, 4.0], [0.5, 1.5, -0.5, 2.5]], dtype=np.float32)
    linear = Linear(input_dims=4, output_dims=4, weight_init=WeightInit(method="ones", scale=0.25))
    norm = LayerNorm(dim=4, epsilon=1e-5)
    act = ReLU()

    t_ops.reset_active_graph("IntegrationPipelineGraph")
    t_ops.set_eager_mode(False)
    try:
        traced_out = act(norm(linear(x)))
        graph = t_ops.get_active_graph()
        assert graph.name == "IntegrationPipelineGraph"
        assert len(graph.nodes) > 0
        assert traced_out.node_id is not None
    finally:
        t_ops.set_eager_mode(True)

    eager_out = act(norm(linear(x)))
    np.testing.assert_allclose(traced_out, eager_out, atol=1e-6)
