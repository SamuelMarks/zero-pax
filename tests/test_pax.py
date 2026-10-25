"""Tests for top-level zero_pax API, domain types, and static verification scripts."""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import zero_pax
from scripts import check_dependencies, check_docstrings, lint_typing_and_hacks
from zero_pax import BaseParameter, LanguageModelType, Layer
from zero_pax import _tensor_ops as t_ops
from zero_pax.praxis.base import BasePraxisLayer, as_tensor_if_present, instantiate_layer
from zero_pax.praxis.layers import AutodiffCheckpointType, Identity, LayerNorm, Linear, ReLU
from zero_pax.types import DType, PaxConfig, WeightInit


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

    custom_t = WeightInit(method="orthogonal", scale=1.0).init_tensor((2, 2))
    assert custom_t.shape == (2, 2)

    base = BasePraxisLayer(fprop_dtype="float32")
    assert base.effective_fprop_dtype == "float32"
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


def test_static_lint_and_docstring_scripts() -> None:
    """Verify AST/comment detection in `lint_typing_and_hacks.py`, `check_docstrings.py`, and `check_dependencies.py`."""
    bad_src = "from typing import Any; x = vars(y)"
    bad_tree = ast.parse(bad_src)
    violations = lint_typing_and_hacks.check_ast(bad_tree, Path("dummy.py"))
    assert len(violations) == 2
    assert any("Any" in v for v in violations)
    assert any("vars" in v for v in violations)

    ignore_comment = "x: int = 1  #" + " type: ignore[assignment]"
    comment_violations = lint_typing_and_hacks.check_comments(ignore_comment, Path("dummy.py"))
    assert len(comment_violations) == 1

    assert check_dependencies.is_allowed_module("pydantic")
    assert not check_dependencies.is_allowed_module("torch")

    doc_args = check_docstrings._extract_documented_args(
        "\n".join(["Summary.", "", "Args:", "    alpha: First."])
    )
    assert doc_args == {"alpha"}
    doc_attrs = check_docstrings._extract_documented_attributes(
        "\n".join(["Summary.", "", "Attributes:", "    beta: Field."])
    )
    assert doc_attrs == {"beta"}
    assert DType is not None
