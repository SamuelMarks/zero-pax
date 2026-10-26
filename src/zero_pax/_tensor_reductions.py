"""Reduction, argmin/argmax, and cumulative scan operations for zero_pax with IR tracing."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

from ml_switcheroo_ir.types import DType

from zero_pax._tensor_core import (
    Tensor,
    TensorOrScalar,
    _compute_strides,
    _float_max,
    _float_min,
    _float_sum,
    _record_ir_op,
    asarray,
)


def _resolve_reduction_axes(ndim: int, axis: int | Sequence[int] | None) -> tuple[int, ...]:
    """Normalize `axis` into a sorted tuple of non-negative axis indices in `[0, ndim)`.

    Args:
        ndim: Rank of the input tensor.
        axis: Optional integer axis or sequence of axes.

    Returns:
        Sorted tuple of unique reduced axes.
    """
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis + ndim if axis < 0 else axis,)
    return tuple(sorted({ax + ndim if ax < 0 else ax for ax in axis}))


def _reduce_axes(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None,
    keepdims: bool,
    reducer: Callable[[Sequence[float]], float],
    op_type: str = "Reduce",
) -> Tensor:
    """Apply `reducer` over the specified `axis` (or axes) of `x` and record an IR node.

    Args:
        x: Input tensor or scalar.
        axis: Axis or axes to reduce over (`None` reduces all axes).
        keepdims: Whether reduced axes are retained with size `1`.
        reducer: Function reducing a non-empty sequence of floats to a single float.
        op_type: IR operation name recorded when non-eager tracing is active.

    Returns:
        Reduced `Tensor`.
    """
    t = asarray(x)
    red_axes = _resolve_reduction_axes(t.ndim, axis)
    if not red_axes:
        out_noop = Tensor(t.flat_data, t.shape, dtype=t.dtype)
        return _record_ir_op(
            op_type,
            (t,),
            out_noop,
            attributes={"axes": [], "keepdims": keepdims},
        )

    kept_axes = [ax for ax in range(t.ndim) if ax not in red_axes]
    out_shape_no_keep = tuple(t.shape[ax] for ax in kept_axes)
    out_shape = (
        tuple(1 if ax in red_axes else t.shape[ax] for ax in range(t.ndim))
        if keepdims
        else out_shape_no_keep
    )

    src_strides = _compute_strides(t.shape)
    src_flat = t.flat_data
    kept_ranges = [range(t.shape[ax]) for ax in kept_axes]
    red_ranges = [range(t.shape[ax]) for ax in red_axes]

    out_flat: list[float] = []
    for kept_coord in itertools.product(*kept_ranges):
        base_idx = 0
        for ax, c in zip(kept_axes, kept_coord):
            base_idx += c * src_strides[ax]
        bucket: list[float] = []
        for red_coord in itertools.product(*red_ranges):
            idx = base_idx
            for ax, c in zip(red_axes, red_coord):
                idx += c * src_strides[ax]
            bucket.append(src_flat[idx])
        out_flat.append(reducer(bucket) if bucket else 0.0)

    out = Tensor(out_flat, out_shape, dtype=t.dtype)
    return _record_ir_op(
        op_type,
        (t,),
        out,
        attributes={"axes": list(red_axes), "keepdims": keepdims},
    )


def sum(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum of tensor elements over a given axis or axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which a sum is performed.
        keepdims: Whether to retain reduced dimensions with length 1.

    Returns:
        Summed `Tensor`.
    """
    return _reduce_axes(x, axis, keepdims, _float_sum, op_type="ReduceSum")


def mean(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the arithmetic mean along the specified axis or axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the means are computed.
        keepdims: Whether to retain reduced dimensions with length 1.

    Returns:
        Mean `Tensor`.
    """
    return _reduce_axes(
        x,
        axis,
        keepdims,
        lambda vals: _float_sum(vals) / len(vals),
        op_type="ReduceMean",
    )


def var(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the variance along the specified axis or axes.

    Args:
        x: Input tensor.
        axis: Axis or axes along which the variance is computed.
        keepdims: Whether to retain reduced dimensions with length 1.

    Returns:
        Variance `Tensor`.
    """

    def _var_fn(vals: Sequence[float]) -> float:
        """Compute population variance of `vals`.

        Args:
            vals: Sequence of float values.

        Returns:
            Variance value.
        """
        mu = _float_sum(vals) / len(vals)
        return _float_sum((v - mu) ** 2 for v in vals) / len(vals)

    return _reduce_axes(x, axis, keepdims, _var_fn, op_type="ReduceVariance")


def _seq_max(vals: Sequence[float]) -> float:
    """Return the largest element in `vals`.

    Args:
        vals: Non-empty sequence of floats.

    Returns:
        Maximum float value.
    """
    best = vals[0]
    for v in vals[1:]:
        best = _float_max(best, v)
    return best


def _seq_min(vals: Sequence[float]) -> float:
    """Return the smallest element in `vals`.

    Args:
        vals: Non-empty sequence of floats.

    Returns:
        Minimum float value.
    """
    best = vals[0]
    for v in vals[1:]:
        best = _float_min(best, v)
    return best


def max(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the maximum of a tensor or maximum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate.
        keepdims: Whether to retain reduced dimensions with length 1.

    Returns:
        Maximum `Tensor`.
    """
    return _reduce_axes(x, axis, keepdims, _seq_max, op_type="ReduceMax")


def min(
    x: TensorOrScalar,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the minimum of a tensor or minimum along an axis.

    Args:
        x: Input tensor.
        axis: Axis or axes along which to operate.
        keepdims: Whether to retain reduced dimensions with length 1.

    Returns:
        Minimum `Tensor`.
    """
    return _reduce_axes(x, axis, keepdims, _seq_min, op_type="ReduceMin")


def argmin(
    x: TensorOrScalar,
    axis: int | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the indices of the minimum values along an axis.

    Args:
        x: Input tensor.
        axis: Optional axis along which to find minimum indices.
        keepdims: Whether to retain the reduced axis with size 1.

    Returns:
        Integer index `Tensor`.
    """
    t = asarray(x)
    if axis is None:
        best_i = 0
        best_v = t.flat_data[0]
        for idx, v in enumerate(t.flat_data):
            if v < best_v:
                best_v = v
                best_i = idx
        out_shape = (1,) * t.ndim if keepdims else ()
        out_flat = Tensor((float(best_i),), out_shape, dtype=DType.int32)
        return _record_ir_op(
            "ArgMin",
            (t,),
            out_flat,
            attributes={"axis": None, "keepdims": keepdims},
        )

    def _argmin_fn(vals: Sequence[float]) -> float:
        """Find index of minimum value in `vals`.

        Args:
            vals: Sequence of floats.

        Returns:
            Index of minimum value as float.
        """
        best_idx = 0
        best_val = vals[0]
        for i, v in enumerate(vals):
            if v < best_val:
                best_val = v
                best_idx = i
        return float(best_idx)

    res = _reduce_axes(t, axis, keepdims, _argmin_fn, op_type="ArgMin")
    out = Tensor(res.flat_data, res.shape, dtype=DType.int32)
    return _record_ir_op(
        "ArgMin",
        (t,),
        out,
        attributes={"axis": axis, "keepdims": keepdims},
    )


def argmax(
    x: TensorOrScalar,
    axis: int | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the indices of the maximum values along an axis.

    Args:
        x: Input tensor.
        axis: Optional axis along which to find maximum indices.
        keepdims: Whether to retain the reduced axis with size 1.

    Returns:
        Integer index `Tensor`.
    """
    t = asarray(x)
    if axis is None:
        best_i = 0
        best_v = t.flat_data[0]
        for idx, v in enumerate(t.flat_data):
            if v > best_v:
                best_v = v
                best_i = idx
        out_shape = (1,) * t.ndim if keepdims else ()
        out_flat = Tensor((float(best_i),), out_shape, dtype=DType.int32)
        return _record_ir_op(
            "ArgMax",
            (t,),
            out_flat,
            attributes={"axis": None, "keepdims": keepdims},
        )

    def _argmax_fn(vals: Sequence[float]) -> float:
        """Find index of maximum value in `vals`.

        Args:
            vals: Sequence of floats.

        Returns:
            Index of maximum value as float.
        """
        best_idx = 0
        best_val = vals[0]
        for i, v in enumerate(vals):
            if v > best_val:
                best_val = v
                best_idx = i
        return float(best_idx)

    res = _reduce_axes(t, axis, keepdims, _argmax_fn, op_type="ArgMax")
    out = Tensor(res.flat_data, res.shape, dtype=DType.int32)
    return _record_ir_op(
        "ArgMax",
        (t,),
        out,
        attributes={"axis": axis, "keepdims": keepdims},
    )


def cumsum(x: TensorOrScalar, axis: int = 0) -> Tensor:
    """Return the cumulative sum of the elements along a given axis.

    Args:
        x: Input tensor.
        axis: Axis along which the cumulative sum is computed.

    Returns:
        Cumulative sum `Tensor`.
    """
    t = asarray(x)
    ndim = t.ndim
    norm_ax = axis + ndim if axis < 0 else axis
    dim_len = t.shape[norm_ax]
    src_strides = _compute_strides(t.shape)
    ax_stride = src_strides[norm_ax]
    other_axes = [ax for ax in range(ndim) if ax != norm_ax]
    other_ranges = [range(t.shape[ax]) for ax in other_axes]

    out_flat = list(t.flat_data)
    for other_coord in itertools.product(*other_ranges):
        base_idx = 0
        for ax, c in zip(other_axes, other_coord):
            base_idx += c * src_strides[ax]
        running = 0.0
        for k in range(dim_len):
            pos = base_idx + k * ax_stride
            running += out_flat[pos]
            out_flat[pos] = running
    out = Tensor(out_flat, t.shape, dtype=t.dtype)
    return _record_ir_op("CumSum", (t,), out, attributes={"axis": norm_ax})
