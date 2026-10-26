"""Pure-Python typed tensor operations and Tensor engine backed by ml_switcheroo_ir."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from ml_switcheroo_ir.types import DType

from zero_pax._tensor_core import (
    IndexAtom,
    IndexSpec,
    NestedSequence,
    Scalar,
    SliceAtom,
    Tensor,
    TensorOrScalar,
    ToListProtocol,
    TypedVariadicCallable,
    VariadicCallableProtocol,
    _binary_elementwise,
    _broadcast_flat,
    _broadcast_shapes,
    _compute_strides,
    _float_max,
    _float_min,
    _int_sum,
    _normalize_shape,
    _record_ir_op,
    _shape_size,
    _unary_elementwise,
    asarray,
    get_active_graph,
    get_eager_mode,
    inf,
    nan,
    pi,
    register_graph_input,
    reset_active_graph,
    reshape,
    resolve_dtype,
    set_eager_mode,
    transpose,
)
from zero_pax._tensor_linalg import (
    dot,
    einsum,
    extract_patches_1d,
    extract_patches_2d,
    matmul,
)
from zero_pax._tensor_reductions import (
    argmax,
    argmin,
    cumsum,
    max,
    mean,
    min,
    sum,
    var,
)


def array(x: TensorOrScalar, dtype: DType | str | None = None) -> Tensor:
    """Create a `Tensor` from `x` with optional `dtype`.

    Args:
        x: Input tensor, scalar, or nested sequence.
        dtype: Optional data type or string name.

    Returns:
        Constructed `Tensor`.
    """
    return asarray(x, dtype=dtype)


def zeros(shape: int | Sequence[int], dtype: DType | str | None = None) -> Tensor:
    """Create a tensor filled with `0.0`.

    Args:
        shape: Output tensor shape.
        dtype: Optional data type or string name.

    Returns:
        Zero-initialized `Tensor`.
    """
    norm_shape = _normalize_shape(shape)
    return Tensor((0.0,) * _shape_size(norm_shape), norm_shape, dtype=resolve_dtype(dtype))


def zeros_like(x: TensorOrScalar, dtype: DType | str | None = None) -> Tensor:
    """Create a tensor of zeros with the same shape as `x`.

    Args:
        x: Reference tensor or array.
        dtype: Optional override data type or string name.

    Returns:
        Zero-initialized `Tensor` matching `x.shape`.
    """
    t = asarray(x)
    return zeros(t.shape, dtype=resolve_dtype(dtype) or t.dtype)


def ones(shape: int | Sequence[int], dtype: DType | str | None = None) -> Tensor:
    """Create a tensor filled with `1.0`.

    Args:
        shape: Output tensor shape.
        dtype: Optional data type or string name.

    Returns:
        One-initialized `Tensor`.
    """
    norm_shape = _normalize_shape(shape)
    return Tensor((1.0,) * _shape_size(norm_shape), norm_shape, dtype=resolve_dtype(dtype))


def ones_like(x: TensorOrScalar, dtype: DType | str | None = None) -> Tensor:
    """Create a tensor of ones with the same shape as `x`.

    Args:
        x: Reference tensor or array.
        dtype: Optional override data type or string name.

    Returns:
        One-initialized `Tensor` matching `x.shape`.
    """
    t = asarray(x)
    return ones(t.shape, dtype=resolve_dtype(dtype) or t.dtype)


def full(
    shape: int | Sequence[int],
    fill_value: float,
    dtype: DType | str | None = None,
) -> Tensor:
    """Create a tensor of `shape` filled with `fill_value`.

    Args:
        shape: Output tensor shape.
        fill_value: Scalar constant to fill the tensor with.
        dtype: Optional data type or string name.

    Returns:
        Constant-filled `Tensor`.
    """
    norm_shape = _normalize_shape(shape)
    return Tensor(
        (float(fill_value),) * _shape_size(norm_shape),
        norm_shape,
        dtype=resolve_dtype(dtype),
    )


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    dtype: DType | str | None = None,
) -> Tensor:
    """Return evenly spaced values within a given interval.

    Args:
        start: Start of interval (or `stop` when `stop` is `None`).
        stop: End of interval (exclusive).
        step: Spacing between values.
        dtype: Optional data type or string name.

    Returns:
        1D `Tensor` of evenly spaced values.
    """
    if stop is None:
        actual_start = 0.0
        actual_stop = float(start)
    else:
        actual_start = float(start)
        actual_stop = float(stop)
    actual_step = float(step)
    vals: list[float] = []
    cur = actual_start
    if actual_step > 0.0:
        while cur < actual_stop - 1e-12:
            vals.append(cur)
            cur += actual_step
    elif actual_step < 0.0:
        while cur > actual_stop + 1e-12:
            vals.append(cur)
            cur += actual_step
    return Tensor(vals, (len(vals),), dtype=resolve_dtype(dtype))


def eye(
    n: int,
    m: int | None = None,
    dtype: DType | str | None = None,
) -> Tensor:
    """Return a 2D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n: Number of rows in the output.
        m: Number of columns in the output (defaults to `n`).
        dtype: Optional data type or string name.

    Returns:
        Identity/diagonal 2D `Tensor` of shape `(n, m)`.
    """
    cols = n if m is None else m
    flat = [1.0 if r == c else 0.0 for r in range(n) for c in range(cols)]
    return Tensor(flat, (n, cols), dtype=resolve_dtype(dtype))


def broadcast_to(x: TensorOrScalar, shape: Sequence[int]) -> Tensor:
    """Broadcast a tensor to a new shape.

    Args:
        x: Input tensor or scalar.
        shape: Target shape sequence.

    Returns:
        Broadcasted `Tensor`.
    """
    t = asarray(x)
    target = tuple(int(d) for d in shape)
    out = Tensor(_broadcast_flat(t, target), target, dtype=t.dtype)
    return _record_ir_op("Expand", (t,), out, attributes={"shape": list(target)})


def expand_dims(x: TensorOrScalar, axis: int) -> Tensor:
    """Expand the shape of a tensor by inserting a new axis of size 1.

    Args:
        x: Input tensor.
        axis: Position in the expanded axes where the new axis is placed.

    Returns:
        `Tensor` with an additional unit dimension.
    """
    t = asarray(x)
    new_ndim = t.ndim + 1
    norm_ax = axis + new_ndim if axis < 0 else axis
    new_shape = t.shape[:norm_ax] + (1,) + t.shape[norm_ax:]
    out = Tensor(t.flat_data, new_shape, dtype=t.dtype)
    return _record_ir_op("Unsqueeze", (t,), out, attributes={"axes": [norm_ax]})


def concatenate(tensors: Sequence[TensorOrScalar], axis: int = 0) -> Tensor:
    """Join a sequence of tensors along an existing axis.

    Args:
        tensors: Non-empty sequence of tensors with matching shapes except along `axis`.
        axis: Axis along which the tensors will be joined.

    Returns:
        Concatenated `Tensor`.
    """
    converted = [asarray(t) for t in tensors]
    first = converted[0]
    ndim = first.ndim
    norm_ax = axis + ndim if axis < 0 else axis
    out_axis_dim = _int_sum(t.shape[norm_ax] for t in converted)
    out_shape = first.shape[:norm_ax] + (out_axis_dim,) + first.shape[norm_ax + 1 :]
    outer_size = _shape_size(first.shape[:norm_ax])
    inner_size = _shape_size(first.shape[norm_ax + 1 :])

    out_flat: list[float] = []
    for outer_idx in range(outer_size):
        for t in converted:
            chunk_len = t.shape[norm_ax] * inner_size
            start = outer_idx * chunk_len
            out_flat.extend(t.flat_data[start : start + chunk_len])
    out = Tensor(out_flat, out_shape, dtype=first.dtype)
    return _record_ir_op("Concat", converted, out, attributes={"axis": norm_ax})


def split(
    x: TensorOrScalar, indices_or_sections: int | Sequence[int], axis: int = 0
) -> list[Tensor]:
    """Split a tensor into multiple sub-tensors along `axis`.

    Args:
        x: Tensor to be divided into sub-tensors.
        indices_or_sections: Number of equal sections (`int`) or split index sequence.
        axis: Axis along which to split `x`.

    Returns:
        List of split `Tensor` objects.

    Raises:
        ValueError: If `indices_or_sections` is an integer that does not evenly divide the axis.
    """
    t = asarray(x)
    ndim = t.ndim
    norm_ax = axis + ndim if axis < 0 else axis
    axis_len = t.shape[norm_ax]

    if isinstance(indices_or_sections, int):
        if axis_len % indices_or_sections != 0:
            raise ValueError(
                f"array split does not result in an equal division: {axis_len} % {indices_or_sections}"
            )
        sec_size = axis_len // indices_or_sections
        bounds = [i * sec_size for i in range(indices_or_sections + 1)]
    else:
        split_pts = [int(idx) for idx in indices_or_sections]
        bounds = [0, *split_pts, axis_len]

    parts: list[Tensor] = []
    for idx in range(len(bounds) - 1):
        start_idx = bounds[idx]
        end_idx = bounds[idx + 1]
        slicer: tuple[SliceAtom, ...] = (
            (slice(None),) * norm_ax
            + (slice(start_idx, end_idx),)
            + (slice(None),) * (ndim - norm_ax - 1)
        )
        part = t[slicer]
        parts.append(
            _record_ir_op(
                "Split",
                (t,),
                part,
                attributes={"axis": norm_ax, "index": idx},
            )
        )
    return parts


def pad(
    x: TensorOrScalar,
    pad_width: Sequence[tuple[int, int]],
    mode: str = "constant",
    constant_values: float = 0.0,
) -> Tensor:
    """Pad a tensor with constant or edge-replicated values.

    Args:
        x: Input tensor to pad.
        pad_width: Sequence of `(before_i, after_i)` padding pairs for each axis.
        mode: Padding mode (`'constant'` or `'edge'`).
        constant_values: Fill value used for `'constant'` padding.

    Returns:
        Padded `Tensor`.

    Raises:
        ValueError: If `mode` is not `'constant'` or `'edge'`.
    """
    norm_mode = mode.strip().lower()
    if norm_mode not in ("constant", "edge"):
        raise ValueError(f"Unsupported pad mode {mode!r}; expected 'constant' or 'edge'.")
    t = asarray(x)
    pads = [(int(b), int(a)) for b, a in pad_width]
    out_shape = tuple(t.shape[i] + pads[i][0] + pads[i][1] for i in range(t.ndim))
    src_strides = _compute_strides(t.shape)
    src_flat = t.flat_data
    is_edge = norm_mode == "edge"
    fill_val = float(constant_values)

    out_flat: list[float] = []
    for out_coords in itertools.product(*(range(d) for d in out_shape)):
        in_bounds = True
        flat_idx = 0
        for ax, c in enumerate(out_coords):
            src_c = c - pads[ax][0]
            dim_len = t.shape[ax]
            if is_edge:
                clamped = 0 if src_c < 0 else (dim_len - 1 if src_c >= dim_len else src_c)
                flat_idx += clamped * src_strides[ax]
            else:
                if src_c < 0 or src_c >= dim_len:
                    in_bounds = False
                    break
                flat_idx += src_c * src_strides[ax]
        out_flat.append(src_flat[flat_idx] if (is_edge or in_bounds) else fill_val)
    out = Tensor(out_flat, out_shape, dtype=t.dtype)
    return _record_ir_op("Pad", (t,), out, attributes={"mode": norm_mode})


def roll(x: TensorOrScalar, shift: int, axis: int = -1) -> Tensor:
    """Roll tensor elements cyclically along a given axis.

    Args:
        x: Input tensor.
        shift: Number of places by which elements are shifted.
        axis: Axis along which elements are shifted.

    Returns:
        Rolled `Tensor`.
    """
    t = asarray(x)
    ndim = t.ndim
    norm_ax = axis + ndim if axis < 0 else axis
    dim_len = t.shape[norm_ax]
    if dim_len <= 1 or shift % dim_len == 0:
        out_noop = Tensor(t.flat_data, t.shape, dtype=t.dtype)
        return _record_ir_op(
            "Roll",
            (t,),
            out_noop,
            attributes={"shift": shift, "axis": norm_ax},
        )
    src_strides = _compute_strides(t.shape)
    src_flat = t.flat_data
    out_flat: list[float] = []
    for out_coords in itertools.product(*(range(d) for d in t.shape)):
        flat_idx = 0
        for ax, c in enumerate(out_coords):
            src_c = (c - shift) % dim_len if ax == norm_ax else c
            flat_idx += src_c * src_strides[ax]
        out_flat.append(src_flat[flat_idx])
    out = Tensor(out_flat, t.shape, dtype=t.dtype)
    return _record_ir_op(
        "Roll",
        (t,),
        out,
        attributes={"shift": shift, "axis": norm_ax},
    )


def where(condition: TensorOrScalar, x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Select elements from `x` or `y` depending on `condition`.

    Args:
        condition: Boolean or numeric mask tensor (`!= 0` selects `x`, `0` selects `y`).
        x: Values selected where `condition` is truthy.
        y: Values selected where `condition` is falsy.

    Returns:
        Selected `Tensor`.
    """
    tc = asarray(condition)
    tx = asarray(x)
    ty = asarray(y)
    shape_cx = _broadcast_shapes(tc.shape, tx.shape)
    target_shape = _broadcast_shapes(shape_cx, ty.shape)
    flat_c = _broadcast_flat(tc, target_shape)
    flat_x = _broadcast_flat(tx, target_shape)
    flat_y = _broadcast_flat(ty, target_shape)
    out_flat = [vx if vc != 0.0 else vy for vc, vx, vy in zip(flat_c, flat_x, flat_y)]
    out = Tensor(out_flat, target_shape, dtype=tx.dtype)
    return _record_ir_op("Where", (tc, tx, ty), out)


def abs(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise absolute value.

    Args:
        x: Input tensor or scalar.

    Returns:
        Absolute value `Tensor`.
    """
    return _unary_elementwise(x, math.fabs, op_type="Abs")


def exp(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise exponential.

    Args:
        x: Input tensor or scalar.

    Returns:
        Exponential `Tensor`.
    """
    return _unary_elementwise(
        x,
        lambda v: math.exp(_float_min(88.0, _float_max(-88.0, v))),
        op_type="Exp",
    )


def log(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise natural logarithm.

    Args:
        x: Input tensor or scalar.

    Returns:
        Natural logarithm `Tensor`.
    """
    return _unary_elementwise(
        x,
        lambda v: math.log(v) if v > 0.0 else -math.inf,
        op_type="Log",
    )


def log1p(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise natural logarithm of `1 + x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Elementwise `log(1 + x)` `Tensor`.
    """
    return _unary_elementwise(
        x,
        lambda v: math.log1p(v) if v > -1.0 else -math.inf,
        op_type="Log1p",
    )


def sqrt(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise non-negative square root.

    Args:
        x: Input tensor or scalar.

    Returns:
        Square root `Tensor`.
    """
    return _unary_elementwise(
        x,
        lambda v: math.sqrt(_float_max(0.0, v)),
        op_type="Sqrt",
    )


def square(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise square `x * x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Squared `Tensor`.
    """
    return _unary_elementwise(x, lambda v: v * v, op_type="Square")


def sin(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise trigonometric sine.

    Args:
        x: Input tensor in radians.

    Returns:
        Sine `Tensor`.
    """
    return _unary_elementwise(x, math.sin, op_type="Sin")


def cos(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise trigonometric cosine.

    Args:
        x: Input tensor in radians.

    Returns:
        Cosine `Tensor`.
    """
    return _unary_elementwise(x, math.cos, op_type="Cos")


def tanh(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise hyperbolic tangent.

    Args:
        x: Input tensor or scalar.

    Returns:
        Hyperbolic tangent `Tensor`.
    """
    return _unary_elementwise(x, math.tanh, op_type="Tanh")


def erf(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise Gauss error function.

    Args:
        x: Input tensor or scalar.

    Returns:
        Error function `Tensor`.
    """
    return _unary_elementwise(x, math.erf, op_type="Erf")


def floor(x: TensorOrScalar) -> Tensor:
    """Return the elementwise floor of the input.

    Args:
        x: Input tensor or scalar.

    Returns:
        Floored `Tensor`.
    """
    return _unary_elementwise(x, lambda v: float(math.floor(v)), op_type="Floor")


def maximum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Compute the elementwise maximum of two tensors.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.

    Returns:
        Elementwise maximum `Tensor`.
    """
    return _binary_elementwise(x, y, _float_max, op_type="Max")


def minimum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Compute the elementwise minimum of two tensors.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.

    Returns:
        Elementwise minimum `Tensor`.
    """
    return _binary_elementwise(x, y, _float_min, op_type="Min")


def clip(
    x: TensorOrScalar,
    a_min: float | TensorOrScalar,
    a_max: float | TensorOrScalar,
) -> Tensor:
    """Clip (limit) the values in a tensor to `[a_min, a_max]`.

    Args:
        x: Input tensor to clip.
        a_min: Minimum threshold value.
        a_max: Maximum threshold value.

    Returns:
        Clipped `Tensor`.
    """
    return minimum(maximum(x, a_min), a_max)


__all__ = [
    "DType",
    "IndexAtom",
    "IndexSpec",
    "NestedSequence",
    "Scalar",
    "Tensor",
    "TensorOrScalar",
    "ToListProtocol",
    "TypedVariadicCallable",
    "VariadicCallableProtocol",
    "_broadcast_flat",
    "_broadcast_shapes",
    "_compute_strides",
    "abs",
    "arange",
    "argmax",
    "argmin",
    "array",
    "asarray",
    "broadcast_to",
    "clip",
    "concatenate",
    "cos",
    "cumsum",
    "dot",
    "einsum",
    "erf",
    "exp",
    "expand_dims",
    "extract_patches_1d",
    "extract_patches_2d",
    "eye",
    "floor",
    "full",
    "get_active_graph",
    "get_eager_mode",
    "inf",
    "log",
    "log1p",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "nan",
    "ones",
    "ones_like",
    "pad",
    "pi",
    "register_graph_input",
    "reset_active_graph",
    "reshape",
    "resolve_dtype",
    "roll",
    "set_eager_mode",
    "sin",
    "split",
    "sqrt",
    "square",
    "sum",
    "tanh",
    "transpose",
    "var",
    "where",
    "zeros",
    "zeros_like",
]
