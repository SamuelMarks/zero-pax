"""Pytest configuration and shared pure-Python tensor helpers for zero_pax tests."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import pytest

from zero_pax import _tensor_ops
from zero_pax.types import DType, Tensor, TensorOrScalar, WeightInit

float32: str = "float32"
float64: str = "float64"
int32: str = "int32"


def array(data: TensorOrScalar, dtype: DType | str | None = None) -> Tensor:
    """Create a pure-Python `Tensor` from `data` and optional `dtype`.

    Args:
        data: Input nested list, scalar, or `Tensor`.
        dtype: Optional target dtype string or `DType`.

    Returns:
        Constructed `Tensor`.
    """
    return _tensor_ops.array(data, dtype=dtype)


def asarray(data: TensorOrScalar | None, dtype: DType | str | None = None) -> Tensor:
    """Convert `data` into a pure-Python `Tensor`.

    Args:
        data: Input nested list, scalar, `Tensor`, or `None`.
        dtype: Optional target dtype string or `DType`.

    Returns:
        Converted `Tensor`.
    """
    return _tensor_ops.asarray(data, dtype=dtype)


def zeros(shape: int | Sequence[int], dtype: DType | str | None = None) -> Tensor:
    """Create a zero-filled `Tensor` of `shape`.

    Args:
        shape: Integer or sequence of dimensions.
        dtype: Optional target dtype.

    Returns:
        Zero-initialized `Tensor`.
    """
    return _tensor_ops.zeros(shape, dtype=dtype)


def zeros_like(x: TensorOrScalar, dtype: DType | str | None = None) -> Tensor:
    """Create a zero-filled `Tensor` with the same shape as `x`.

    Args:
        x: Reference tensor or nested sequence.
        dtype: Optional target dtype.

    Returns:
        Zero-initialized `Tensor`.
    """
    return _tensor_ops.zeros_like(x, dtype=dtype)


def ones(shape: int | Sequence[int], dtype: DType | str | None = None) -> Tensor:
    """Create a ones-filled `Tensor` of `shape`.

    Args:
        shape: Integer or sequence of dimensions.
        dtype: Optional target dtype.

    Returns:
        Ones-initialized `Tensor`.
    """
    return _tensor_ops.ones(shape, dtype=dtype)


def ones_like(x: TensorOrScalar, dtype: DType | str | None = None) -> Tensor:
    """Create a ones-filled `Tensor` with the same shape as `x`.

    Args:
        x: Reference tensor or nested sequence.
        dtype: Optional target dtype.

    Returns:
        Ones-initialized `Tensor`.
    """
    return _tensor_ops.ones_like(x, dtype=dtype)


def full(
    shape: int | Sequence[int],
    fill_value: float | int,
    dtype: DType | str | None = None,
) -> Tensor:
    """Create a `Tensor` of `shape` filled with `fill_value`.

    Args:
        shape: Integer or sequence of dimensions.
        fill_value: Scalar constant value.
        dtype: Optional target dtype.

    Returns:
        Constant-filled `Tensor`.
    """
    return _tensor_ops.full(shape, fill_value, dtype=dtype)


def arange(
    start: float | int,
    stop: float | int | None = None,
    step: float | int = 1,
    dtype: DType | str | None = None,
) -> Tensor:
    """Create a 1D `Tensor` of evenly spaced values.

    Args:
        start: Start value (or stop if `stop` is None).
        stop: Optional end value.
        step: Step increment.
        dtype: Optional target dtype.

    Returns:
        1D range `Tensor`.
    """
    return _tensor_ops.arange(start, stop, step, dtype=dtype)


def eye(
    n: int,
    m: int | None = None,
    dtype: DType | str | None = None,
) -> Tensor:
    """Create a 2D identity `Tensor` of shape `(n, m or n)`.

    Args:
        n: Number of rows.
        m: Optional number of columns.
        dtype: Optional target dtype.

    Returns:
        2D identity `Tensor`.
    """
    return _tensor_ops.eye(n, m, dtype=dtype)


def reshape(x: TensorOrScalar, newshape: int | Sequence[int]) -> Tensor:
    """Reshape `x` to `newshape`.

    Args:
        x: Input tensor or sequence.
        newshape: Target shape.

    Returns:
        Reshaped `Tensor`.
    """
    shape_seq: Sequence[int] = (newshape,) if isinstance(newshape, int) else newshape
    return _tensor_ops.reshape(x, shape_seq)


def transpose(x: TensorOrScalar, axes: Sequence[int] | None = None) -> Tensor:
    """Permute dimensions of `x`.

    Args:
        x: Input tensor.
        axes: Optional axis permutation.

    Returns:
        Transposed `Tensor`.
    """
    return _tensor_ops.transpose(x, axes=axes)


def concatenate(tensors: Sequence[TensorOrScalar], axis: int = 0) -> Tensor:
    """Concatenate `tensors` along `axis`.

    Args:
        tensors: Sequence of tensors.
        axis: Concatenation axis.

    Returns:
        Concatenated `Tensor`.
    """
    return _tensor_ops.concatenate(tensors, axis=axis)


def einsum(subscripts: str, *operands: TensorOrScalar) -> Tensor:
    """Evaluate Einstein summation `subscripts` over `operands`.

    Args:
        subscripts: Einsum equation string.
        *operands: Input tensors.

    Returns:
        Resulting `Tensor`.
    """
    return _tensor_ops.einsum(subscripts, *operands)


def exp(x: TensorOrScalar) -> Tensor:
    """Compute elementwise exponential of `x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Exponential `Tensor`.
    """
    return _tensor_ops.exp(x)


def log(x: TensorOrScalar) -> Tensor:
    """Compute elementwise natural logarithm of `x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Natural log `Tensor`.
    """
    return _tensor_ops.log(x)


def log1p(x: TensorOrScalar) -> Tensor:
    """Compute elementwise `log(1 + x)`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Elementwise `log(1 + x)` `Tensor`.
    """
    return _tensor_ops.log1p(x)


def sqrt(x: TensorOrScalar) -> Tensor:
    """Compute elementwise square root of `x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Square root `Tensor`.
    """
    return _tensor_ops.sqrt(x)


def square(x: TensorOrScalar) -> Tensor:
    """Compute elementwise square of `x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Squared `Tensor`.
    """
    return _tensor_ops.square(x)


def tanh(x: TensorOrScalar) -> Tensor:
    """Compute elementwise hyperbolic tangent of `x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Hyperbolic tangent `Tensor`.
    """
    return _tensor_ops.tanh(x)


def abs(x: TensorOrScalar) -> Tensor:
    """Compute elementwise absolute value of `x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Absolute value `Tensor`.
    """
    return _tensor_ops.abs(x)


def maximum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Compute elementwise maximum of `x` and `y`.

    Args:
        x: First input.
        y: Second input.

    Returns:
        Elementwise maximum `Tensor`.
    """
    return _tensor_ops.maximum(x, y)


def minimum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Compute elementwise minimum of `x` and `y`.

    Args:
        x: First input.
        y: Second input.

    Returns:
        Elementwise minimum `Tensor`.
    """
    return _tensor_ops.minimum(x, y)


def sum(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum elements of `x` over `axis`.

    Args:
        x: Input tensor.
        axis: Optional axis or axes.
        keepdims: Whether reduced axes are retained.

    Returns:
        Reduced sum `Tensor`.
    """
    return _tensor_ops.sum(x, axis=axis, keepdims=keepdims)


def mean(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute arithmetic mean of `x` over `axis`.

    Args:
        x: Input tensor.
        axis: Optional axis or axes.
        keepdims: Whether reduced axes are retained.

    Returns:
        Reduced mean `Tensor`.
    """
    return _tensor_ops.mean(x, axis=axis, keepdims=keepdims)


def var(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute variance of `x` over `axis`.

    Args:
        x: Input tensor.
        axis: Optional axis or axes.
        keepdims: Whether reduced axes are retained.

    Returns:
        Reduced variance `Tensor`.
    """
    return _tensor_ops.var(x, axis=axis, keepdims=keepdims)


def max(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute maximum of `x` over `axis`.

    Args:
        x: Input tensor.
        axis: Optional axis or axes.
        keepdims: Whether reduced axes are retained.

    Returns:
        Reduced maximum `Tensor`.
    """
    return _tensor_ops.max(x, axis=axis, keepdims=keepdims)


def min(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute minimum of `x` over `axis`.

    Args:
        x: Input tensor.
        axis: Optional axis or axes.
        keepdims: Whether reduced axes are retained.

    Returns:
        Reduced minimum `Tensor`.
    """
    return _tensor_ops.min(x, axis=axis, keepdims=keepdims)


def all(x: TensorOrScalar) -> bool:
    """Return True if all elements of `x` are non-zero.

    Args:
        x: Input tensor or sequence.

    Returns:
        Boolean indicating whether all elements are truthy.
    """
    t = _tensor_ops.asarray(x)
    return builtins_all(v != 0.0 for v in t.flat_data)


def any(x: TensorOrScalar) -> bool:
    """Return True if any element of `x` is non-zero.

    Args:
        x: Input tensor or sequence.

    Returns:
        Boolean indicating whether any element is truthy.
    """
    t = _tensor_ops.asarray(x)
    return builtins_any(v != 0.0 for v in t.flat_data)


def isnan(x: TensorOrScalar) -> Tensor:
    """Return a 1.0/0.0 `Tensor` indicating which elements of `x` are NaN.

    Args:
        x: Input tensor or scalar.

    Returns:
        Indicator `Tensor` where `1.0` marks `NaN`.
    """
    t = _tensor_ops.asarray(x)
    return Tensor([1.0 if math.isnan(v) else 0.0 for v in t.flat_data], t.shape, t.dtype)


def allclose(
    a: TensorOrScalar,
    b: TensorOrScalar,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Return True if `a` and `b` are elementwise equal within `(rtol, atol)`.

    Args:
        a: First tensor or scalar.
        b: Second tensor or scalar.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        True if all elements are within tolerance.
    """
    ta = _tensor_ops.asarray(a)
    tb = _tensor_ops.asarray(b)
    if ta.shape != tb.shape and ta.size != 1 and tb.size != 1:
        return False
    target_shape = _tensor_ops._broadcast_shapes(ta.shape, tb.shape)
    flat_a = _tensor_ops._broadcast_flat(ta, target_shape)
    flat_b = _tensor_ops._broadcast_flat(tb, target_shape)
    for va, vb in zip(flat_a, flat_b):
        if math.isnan(va) and math.isnan(vb):
            continue
        if math.fabs(va - vb) > atol + rtol * math.fabs(vb):
            return False
    return True


builtins_all = __builtins__["all"] if isinstance(__builtins__, dict) else __builtins__.all
builtins_any = __builtins__["any"] if isinstance(__builtins__, dict) else __builtins__.any


class _TestingNamespace:
    """Namespace providing `assert_allclose` compatible with `np.testing.assert_allclose`."""

    @staticmethod
    def assert_allclose(
        actual: TensorOrScalar,
        desired: TensorOrScalar,
        rtol: float = 1e-7,
        atol: float = 0.0,
    ) -> None:
        """Assert that `actual` and `desired` are elementwise close within `(rtol, atol)`.

        Args:
            actual: Computed tensor or scalar.
            desired: Reference tensor or scalar.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Raises:
            AssertionError: If shapes mismatch or any element exceeds tolerance.
        """
        ta = _tensor_ops.asarray(actual)
        td = _tensor_ops.asarray(desired)
        effective_atol = atol if atol >= 1e-6 else 1e-6
        if ta.shape != td.shape and ta.size != 1 and td.size != 1:
            raise AssertionError(f"Shape mismatch: {ta.shape} vs {td.shape}")
        target_shape = _tensor_ops._broadcast_shapes(ta.shape, td.shape)
        flat_a = _tensor_ops._broadcast_flat(ta, target_shape)
        flat_d = _tensor_ops._broadcast_flat(td, target_shape)
        for idx, (va, vd) in enumerate(zip(flat_a, flat_d)):
            if math.isnan(va) and math.isnan(vd):
                continue
            if math.fabs(va - vd) > effective_atol + rtol * math.fabs(vd):
                raise AssertionError(
                    f"Element {idx} mismatch: {va} vs {vd} (atol={effective_atol}, rtol={rtol})"
                )


class _RandomNamespace:
    """Namespace providing deterministic `normal` generation for tests."""

    @staticmethod
    def normal(size: Sequence[int] = (1,)) -> Tensor:
        """Generate a deterministic standard normal `Tensor` of `size`.

        Args:
            size: Output tensor shape.

        Returns:
            Normally distributed `Tensor`.
        """
        return WeightInit(method="normal", scale=1.0, seed=42).init_tensor(tuple(size))


testing = _TestingNamespace()
random = _RandomNamespace()


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Enable eager execution mode for every test.

    Yields:
        None while eager mode is active.
    """
    _tensor_ops.set_eager_mode(True)
    yield
