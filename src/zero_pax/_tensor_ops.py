"""Typed tensor operations wrapper routing to ml_switcheroo_compiler."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ml_switcheroo_compiler import Tensor, ops
from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import as_tensor_obj

pi: float = math.pi
inf: float = math.inf
nan: float = math.nan

_RESHAPE_OP = ops.reshape.Reshape()
_BROADCAST_TO_OP = ops.reshape.BroadcastTo()
_TRANSPOSE_OP = ops.reshape.Transpose()


def set_eager_mode(enabled: bool = True) -> bool:
    """Configure whether `ml_switcheroo_compiler` evaluates tensor ops eagerly.

    Args:
        enabled: True to enable eager execution, False for graph tracing.

    Returns:
        The new eager mode state.
    """
    _compiler_config.eager_mode = enabled
    return bool(_compiler_config.eager_mode)


set_eager_mode(True)


def asarray(x: TensorOrScalar, dtype: DType | None = None) -> Tensor:
    """Convert an input array, tensor, scalar, or sequence into a compiler `Tensor`.

    Args:
        x: Input tensor, array, scalar, or nested sequence.
        dtype: Optional target compiler data type.

    Returns:
        Converted `Tensor` instance.
    """
    if isinstance(x, Tensor) and dtype is None:
        return x
    return ops.asarray(x, dtype=dtype)


def array(x: TensorOrScalar, dtype: DType | None = None) -> Tensor:
    """Create a compiler `Tensor` from an input value.

    Args:
        x: Input tensor, array, scalar, or nested sequence.
        dtype: Optional target compiler data type.

    Returns:
        Constructed `Tensor` instance.
    """
    return ops.array(x, dtype=dtype)


def zeros(shape: int | Sequence[int], dtype: DType | None = None) -> Tensor:
    """Create a tensor filled with zeros.

    Args:
        shape: Output tensor shape.
        dtype: Optional data type.

    Returns:
        Zero-initialized `Tensor`.
    """
    return ops.zeros(shape, dtype=dtype)


def zeros_like(x: TensorOrScalar, dtype: DType | None = None) -> Tensor:
    """Create a tensor of zeros with the same shape as `x`.

    Args:
        x: Reference tensor or array.
        dtype: Optional data type override.

    Returns:
        Zero-initialized `Tensor` matching `x.shape`.
    """
    return ops.zeros_like(asarray(x), dtype=dtype)


def ones(shape: int | Sequence[int], dtype: DType | None = None) -> Tensor:
    """Create a tensor filled with ones.

    Args:
        shape: Output tensor shape.
        dtype: Optional data type.

    Returns:
        One-initialized `Tensor`.
    """
    return ops.ones(shape, dtype=dtype)


def ones_like(x: TensorOrScalar, dtype: DType | None = None) -> Tensor:
    """Create a tensor of ones with the same shape as `x`.

    Args:
        x: Reference tensor or array.
        dtype: Optional data type override.

    Returns:
        One-initialized `Tensor` matching `x.shape`.
    """
    return ops.ones_like(asarray(x), dtype=dtype)


def full(shape: int | Sequence[int], fill_value: float, dtype: DType | None = None) -> Tensor:
    """Create a tensor of `shape` filled with `fill_value`.

    Args:
        shape: Output tensor shape.
        fill_value: Scalar fill value.
        dtype: Optional data type.

    Returns:
        Filled `Tensor`.
    """
    return ops.full(shape, fill_value, dtype=dtype)


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    dtype: DType | None = None,
) -> Tensor:
    """Return evenly spaced values within a given interval.

    Args:
        start: Start of interval (or stop if `stop` is None).
        stop: End of interval.
        step: Spacing between values.
        dtype: Optional output data type.

    Returns:
        1D `Tensor` of evenly spaced values.
    """
    return as_tensor_obj(ops.arange(start, stop, step, dtype=dtype))


def eye(n: int, m: int | None = None, dtype: DType | None = None) -> Tensor:
    """Return a 2D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n: Number of rows.
        m: Optional number of columns (defaults to `n`).
        dtype: Optional output data type.

    Returns:
        2D identity `Tensor`.
    """
    return as_tensor_obj(ops.eye(n, m, dtype=dtype))


def reshape(x: TensorOrScalar, newshape: Sequence[int]) -> Tensor:
    """Give a new shape to a tensor without changing its data.

    Args:
        x: Tensor to be reshaped.
        newshape: Target shape tuple or sequence.

    Returns:
        Reshaped `Tensor`.
    """
    return _RESHAPE_OP(asarray(x), tuple(newshape))


def transpose(x: TensorOrScalar, axes: Sequence[int] | None = None) -> Tensor:
    """Permute the dimensions of a tensor.

    Args:
        x: Input tensor.
        axes: Optional permutation of dimension indices.

    Returns:
        Transposed `Tensor`.
    """
    t = asarray(x)
    if axes is None:
        return _TRANSPOSE_OP(t)
    return _TRANSPOSE_OP(t, tuple(axes))


def broadcast_to(x: TensorOrScalar, shape: Sequence[int]) -> Tensor:
    """Broadcast a tensor to a new shape.

    Args:
        x: Input tensor or scalar.
        shape: Target shape tuple.

    Returns:
        Broadcasted `Tensor`.
    """
    return _BROADCAST_TO_OP(asarray(x), tuple(shape))


def expand_dims(x: TensorOrScalar, axis: int) -> Tensor:
    """Expand the shape of a tensor by inserting a new axis of size 1.

    Args:
        x: Input tensor.
        axis: Position in the expanded axes where the new axis is placed.

    Returns:
        `Tensor` with an additional dimension.
    """
    return as_tensor_obj(ops.expand_dims(asarray(x), axis=axis))


def concatenate(tensors: Sequence[TensorOrScalar], axis: int = 0) -> Tensor:
    """Join a sequence of tensors along an existing axis.

    Args:
        tensors: Sequence of tensors with matching shapes outside `axis`.
        axis: The axis along which the tensors will be joined.

    Returns:
        Concatenated `Tensor`.
    """
    converted = [asarray(t) for t in tensors]
    return as_tensor_obj(ops.concatenate(converted, axis=axis))


def split(
    x: TensorOrScalar, indices_or_sections: int | Sequence[int], axis: int = 0
) -> list[Tensor]:
    """Split a tensor into multiple sub-tensors along `axis`.

    Args:
        x: Input tensor to split.
        indices_or_sections: Number of equal sections or split indices.
        axis: Axis along which to split.

    Returns:
        List of split `Tensor` objects.
    """
    raw_parts = ops.splitting.split(asarray(x), indices_or_sections, axis=axis)
    return [as_tensor_obj(p) for p in raw_parts]


def pad(
    x: TensorOrScalar,
    pad_width: Sequence[tuple[int, int]],
    mode: str = "constant",
    constant_values: float = 0.0,
) -> Tensor:
    """Pad a tensor along its dimensions.

    Args:
        x: Input tensor to pad.
        pad_width: Number of values padded to the edges of each axis.
        mode: Padding mode ('constant', 'edge', etc.).
        constant_values: Fill value for 'constant' mode.

    Returns:
        Padded `Tensor`.
    """
    t = asarray(x)
    pw = tuple((int(lo), int(hi)) for lo, hi in pad_width)
    if mode == "constant":
        return as_tensor_obj(ops.misc.pad(t, pw, mode=mode, constant_values=constant_values))
    return as_tensor_obj(ops.misc.pad(t, pw, mode=mode))


def where(condition: TensorOrScalar, x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Return elements chosen from `x` or `y` depending on `condition`.

    Args:
        condition: Boolean condition tensor.
        x: Values selected where `condition` is True.
        y: Values selected where `condition` is False.

    Returns:
        Selected `Tensor`.
    """
    return as_tensor_obj(ops.indexing.where(asarray(condition), asarray(x), asarray(y)))


def einsum(equation: str, *operands: TensorOrScalar) -> Tensor:
    """Evaluate the Einstein summation convention on the operands.

    Args:
        equation: Einstein summation subscript specification string.
        *operands: Input tensors for the summation.

    Returns:
        Resulting `Tensor` from the Einstein summation.
    """
    converted = [asarray(op) for op in operands]
    return as_tensor_obj(ops.einsum(equation, *converted))


def dot(a: TensorOrScalar, b: TensorOrScalar) -> Tensor:
    """Compute the dot product of two tensors.

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        Dot product `Tensor`.
    """
    return as_tensor_obj(ops.dot(asarray(a), asarray(b)))


def matmul(a: TensorOrScalar, b: TensorOrScalar) -> Tensor:
    """Compute the matrix product of two tensors.

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        Matrix product `Tensor`.
    """
    return as_tensor_obj(ops.matmul(asarray(a), asarray(b)))


def roll(x: TensorOrScalar, shift: int, axis: int = -1) -> Tensor:
    """Roll tensor elements along a given axis.

    Args:
        x: Input tensor.
        shift: Number of places by which elements are shifted.
        axis: Axis along which elements are shifted.

    Returns:
        Rolled `Tensor`.
    """
    return as_tensor_obj(ops.roll(asarray(x), shift, axis=axis))


def abs(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise absolute value.

    Args:
        x: Input tensor.

    Returns:
        Elementwise absolute value `Tensor`.
    """
    return ops.abs(asarray(x))


def exp(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise exponential.

    Args:
        x: Input tensor.

    Returns:
        Elementwise exponential `Tensor`.
    """
    return ops.exp(asarray(x))


def log(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise natural logarithm.

    Args:
        x: Input tensor.

    Returns:
        Elementwise natural logarithm `Tensor`.
    """
    return ops.log(asarray(x))


def log1p(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise natural logarithm of one plus the input.

    Args:
        x: Input tensor.

    Returns:
        Elementwise `log(1 + x)` `Tensor`.
    """
    return ops.log1p(asarray(x))


def sqrt(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise non-negative square root.

    Args:
        x: Input tensor.

    Returns:
        Elementwise square root `Tensor`.
    """
    return ops.sqrt(asarray(x))


def square(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise square of the input.

    Args:
        x: Input tensor.

    Returns:
        Elementwise squared `Tensor`.
    """
    return ops.square(asarray(x))


def sin(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise trigonometric sine.

    Args:
        x: Input tensor in radians.

    Returns:
        Elementwise sine `Tensor`.
    """
    return ops.sin(asarray(x))


def cos(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise trigonometric cosine.

    Args:
        x: Input tensor in radians.

    Returns:
        Elementwise cosine `Tensor`.
    """
    return ops.cos(asarray(x))


def tanh(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise hyperbolic tangent.

    Args:
        x: Input tensor.

    Returns:
        Elementwise hyperbolic tangent `Tensor`.
    """
    return ops.tanh(asarray(x))


def erf(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise Gauss error function.

    Args:
        x: Input tensor.

    Returns:
        Elementwise error function `Tensor`.
    """
    return ops.erf(asarray(x))


def floor(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise floor of the input.

    Args:
        x: Input tensor.

    Returns:
        Elementwise floor `Tensor`.
    """
    return ops.floor(asarray(x))


def maximum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Compute the elementwise maximum of two operands.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.

    Returns:
        Elementwise maximum `Tensor`.
    """
    return ops.maximum(x, y)


def minimum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Compute the elementwise minimum of two operands.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.

    Returns:
        Elementwise minimum `Tensor`.
    """
    return ops.minimum(x, y)


def clip(x: TensorOrScalar, a_min: float | TensorOrScalar, a_max: float | TensorOrScalar) -> Tensor:
    """Clip (limit) the values in a tensor to `[a_min, a_max]`.

    Args:
        x: Input tensor.
        a_min: Minimum threshold value.
        a_max: Maximum threshold value.

    Returns:
        Clipped `Tensor`.
    """
    return ops.clip(asarray(x), a_min, a_max)


def _normalize_axis(axis: int | Sequence[int] | None) -> int | tuple[int, ...] | None:
    """Convert an optional sequence of axes into an immutable tuple of ints.

    Args:
        axis: Single integer axis, sequence of integer axes, or None.

    Returns:
        Normalized integer axis, tuple of integer axes, or None.
    """
    if isinstance(axis, Sequence):
        return tuple(int(a) for a in axis)
    return axis


def sum(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum of tensor elements over a given axis or axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which a sum is performed.
        keepdims: If True, reduced axes are left with size one.

    Returns:
        Reduced sum `Tensor`.
    """
    return ops.sum(asarray(x), axis=_normalize_axis(axis), keepdims=keepdims)


def mean(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the arithmetic mean along the specified axis or axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the means are computed.
        keepdims: If True, reduced axes are left with size one.

    Returns:
        Reduced mean `Tensor`.
    """
    return ops.mean(asarray(x), axis=_normalize_axis(axis), keepdims=keepdims)


def var(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the variance along the specified axis or axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the variance is computed.
        keepdims: If True, reduced axes are left with size one.

    Returns:
        Reduced variance `Tensor`.
    """
    return ops.variance(asarray(x), axis=_normalize_axis(axis), keepdims=keepdims)


def max(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the maximum of a tensor or maximum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate.
        keepdims: If True, reduced axes are left with size one.

    Returns:
        Reduced maximum `Tensor`.
    """
    return ops.max(asarray(x), axis=_normalize_axis(axis), keepdims=keepdims)


def min(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the minimum of a tensor or minimum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate.
        keepdims: If True, reduced axes are left with size one.

    Returns:
        Reduced minimum `Tensor`.
    """
    return ops.min(asarray(x), axis=_normalize_axis(axis), keepdims=keepdims)


def argmin(x: TensorOrScalar, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """Return the indices of the minimum values along an axis.

    Args:
        x: Input tensor.
        axis: Optional axis along which to find minimum indices.
        keepdims: If True, the reduced axis is left in the result with size one.

    Returns:
        `Tensor` of integer indices.
    """
    res = ops.argmin(asarray(x), axis=axis)
    if keepdims and axis is not None:
        return expand_dims(res, axis=axis)
    return res


def argmax(x: TensorOrScalar, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """Return the indices of the maximum values along an axis.

    Args:
        x: Input tensor.
        axis: Optional axis along which to find maximum indices.
        keepdims: If True, the reduced axis is left in the result with size one.

    Returns:
        `Tensor` of integer indices.
    """
    res = ops.argmax(asarray(x), axis=axis)
    if keepdims and axis is not None:
        return expand_dims(res, axis=axis)
    return res


def cumsum(x: TensorOrScalar, axis: int = 0) -> Tensor:
    """Return the cumulative sum of the elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative sum is computed.

    Returns:
        Cumulative sum `Tensor`.
    """
    return ops.cumsum(asarray(x), axis=axis)
