"""Pytest configuration and shared pure-Python tensor helpers for zero_pax tests."""

from __future__ import annotations

import builtins
import itertools
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
    return builtins.all(v != 0.0 for v in t.flat_data)


def any(x: TensorOrScalar) -> bool:
    """Return True if any element of `x` is non-zero.

    Args:
        x: Input tensor or sequence.

    Returns:
        Boolean indicating whether any element is truthy.
    """
    t = _tensor_ops.asarray(x)
    return builtins.any(v != 0.0 for v in t.flat_data)


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


def ref_mean(
    x: TensorOrScalar,
    axis: int = -1,
    keepdims: bool = False,
) -> Tensor:
    """Compute arithmetic mean along `axis` using independent pure-Python list arithmetic.

    Args:
        x: Input tensor.
        axis: Reduction axis.
        keepdims: Whether reduced dimension is kept as size 1.

    Returns:
        Independent reference mean `Tensor`.
    """
    t = _tensor_ops.asarray(x)
    ax = axis + t.ndim if axis < 0 else axis
    dim_len = t.shape[ax]
    out_shape = tuple(1 if i == ax else d for i, d in enumerate(t.shape))
    final_shape = out_shape if keepdims else tuple(d for i, d in enumerate(t.shape) if i != ax)
    strides = _tensor_ops._compute_strides(t.shape)
    flat = t.flat_data
    out_vals: list[float] = []
    coord_ranges = [range(1) if i == ax else range(d) for i, d in enumerate(t.shape)]
    for base_coord in itertools.product(*coord_ranges):
        base_idx = builtins.sum(c * s for c, s in zip(base_coord, strides))
        acc = math.fsum(flat[base_idx + k * strides[ax]] for k in range(dim_len))
        out_vals.append(acc / float(dim_len))
    return Tensor(out_vals, final_shape, dtype=t.dtype)


def ref_var(
    x: TensorOrScalar,
    axis: int = -1,
    keepdims: bool = False,
) -> Tensor:
    """Compute variance along `axis` using independent pure-Python list arithmetic.

    Args:
        x: Input tensor.
        axis: Reduction axis.
        keepdims: Whether reduced dimension is kept as size 1.

    Returns:
        Independent reference variance `Tensor`.
    """
    t = _tensor_ops.asarray(x)
    ax = axis + t.ndim if axis < 0 else axis
    dim_len = t.shape[ax]
    out_shape = tuple(1 if i == ax else d for i, d in enumerate(t.shape))
    final_shape = out_shape if keepdims else tuple(d for i, d in enumerate(t.shape) if i != ax)
    strides = _tensor_ops._compute_strides(t.shape)
    flat = t.flat_data
    out_vals: list[float] = []
    coord_ranges = [range(1) if i == ax else range(d) for i, d in enumerate(t.shape)]
    for base_coord in itertools.product(*coord_ranges):
        base_idx = builtins.sum(c * s for c, s in zip(base_coord, strides))
        vals = [flat[base_idx + k * strides[ax]] for k in range(dim_len)]
        mu = math.fsum(vals) / float(dim_len)
        v_acc = math.fsum((v - mu) * (v - mu) for v in vals) / float(dim_len)
        out_vals.append(v_acc)
    return Tensor(out_vals, final_shape, dtype=t.dtype)


def ref_matmul(a: TensorOrScalar, b: TensorOrScalar) -> Tensor:
    """Compute matrix multiplication `a @ b` using independent pure-Python nested loops.

    Args:
        a: Left operand of shape `[..., M, K]`.
        b: Right operand of shape `[K, N]`.

    Returns:
        Independent reference matrix product `Tensor` of shape `[..., M, N]`.
    """
    ta = _tensor_ops.asarray(a)
    tb = _tensor_ops.asarray(b)
    k_dim = ta.shape[-1]
    n_dim = tb.shape[-1]
    lead_shape = ta.shape[:-1]
    num_rows = 1
    for d in lead_shape:
        num_rows *= d
    fa = ta.flat_data
    fb = tb.flat_data
    out_vals: list[float] = []
    for r in range(num_rows):
        for c in range(n_dim):
            acc = math.fsum(fa[r * k_dim + k] * fb[k * n_dim + c] for k in range(k_dim))
            out_vals.append(acc)
    return Tensor(out_vals, lead_shape + (n_dim,), dtype=ta.dtype)


def ref_softmax(x: TensorOrScalar, axis: int = -1) -> Tensor:
    """Compute softmax probabilities along the last axis using independent pure-Python math.

    Args:
        x: Input logits tensor.
        axis: Softmax axis (defaults to `-1`).

    Returns:
        Independent reference probability `Tensor`.
    """
    t = _tensor_ops.asarray(x)
    c_dim = t.shape[axis]
    num_rows = t.size // c_dim
    flat = t.flat_data
    out_vals: list[float] = []
    for r in range(num_rows):
        row = [flat[r * c_dim + c] for c in range(c_dim)]
        m = builtins.max(row)
        exps = [math.exp(v - m) for v in row]
        denom = math.fsum(exps)
        out_vals.extend(e / denom for e in exps)
    return Tensor(out_vals, t.shape, dtype=t.dtype)


def ref_log_softmax(x: TensorOrScalar, axis: int = -1) -> Tensor:
    """Compute log-softmax along the last axis using independent pure-Python math.

    Args:
        x: Input logits tensor.
        axis: Softmax axis (defaults to `-1`).

    Returns:
        Independent reference log-probability `Tensor`.
    """
    probs = ref_softmax(x, axis=axis)
    return Tensor(
        [math.log(builtins.max(p, 1e-12)) for p in probs.flat_data], probs.shape, probs.dtype
    )


def ref_layer_norm(
    x: TensorOrScalar,
    scale: TensorOrScalar | None = None,
    bias: TensorOrScalar | None = None,
    eps: float = 1e-5,
    direct_scale: bool = False,
) -> Tensor:
    """Compute LayerNorm along the last axis using independent pure-Python math.

    Args:
        x: Input tensor of shape `[..., D]`.
        scale: Optional scale vector of shape `[D]`.
        bias: Optional bias vector of shape `[D]`.
        eps: Numerical stability epsilon.
        direct_scale: Whether `scale` multiplies directly (`True`) or as `1 + scale` (`False`).

    Returns:
        Independent reference LayerNorm `Tensor`.
    """
    t = _tensor_ops.asarray(x)
    d_dim = t.shape[-1]
    num_rows = t.size // d_dim
    flat = t.flat_data
    s_flat = _tensor_ops.asarray(scale).flat_data if scale is not None else (0.0,) * d_dim
    b_flat = _tensor_ops.asarray(bias).flat_data if bias is not None else (0.0,) * d_dim
    out_vals: list[float] = []
    for r in range(num_rows):
        row = [flat[r * d_dim + c] for c in range(d_dim)]
        mu = math.fsum(row) / float(d_dim)
        var_val = math.fsum((v - mu) * (v - mu) for v in row) / float(d_dim)
        inv_std = 1.0 / math.sqrt(var_val + eps)
        for c in range(d_dim):
            mult = (s_flat[c] if direct_scale else (1.0 + s_flat[c])) if scale is not None else 1.0
            b_val = b_flat[c] if bias is not None else 0.0
            out_vals.append((row[c] - mu) * inv_std * mult + b_val)
    return Tensor(out_vals, t.shape, dtype=t.dtype)


def ref_rms_norm(
    x: TensorOrScalar,
    scale: TensorOrScalar | None = None,
    eps: float = 1e-6,
    direct_scale: bool = True,
) -> Tensor:
    """Compute RMSNorm along the last axis using independent pure-Python math.

    Args:
        x: Input tensor of shape `[..., D]`.
        scale: Optional scale vector of shape `[D]`.
        eps: Numerical stability epsilon.
        direct_scale: Whether `scale` multiplies directly (`True`) or as `1 + scale` (`False`).

    Returns:
        Independent reference RMSNorm `Tensor`.
    """
    t = _tensor_ops.asarray(x)
    d_dim = t.shape[-1]
    num_rows = t.size // d_dim
    flat = t.flat_data
    s_flat = _tensor_ops.asarray(scale).flat_data if scale is not None else (1.0,) * d_dim
    out_vals: list[float] = []
    for r in range(num_rows):
        row = [flat[r * d_dim + c] for c in range(d_dim)]
        ms = math.fsum(v * v for v in row) / float(d_dim)
        inv_rms = 1.0 / math.sqrt(ms + eps)
        for c in range(d_dim):
            mult = (s_flat[c] if direct_scale else (1.0 + s_flat[c])) if scale is not None else 1.0
            out_vals.append(row[c] * inv_rms * mult)
    return Tensor(out_vals, t.shape, dtype=t.dtype)


def ref_conv2d(
    inputs: TensorOrScalar,
    kernel: TensorOrScalar,
    stride: tuple[int, int] = (1, 1),
    padding: str = "SAME",
) -> Tensor:
    """Compute 2D NHWC convolution using independent pure-Python nested loops.

    Args:
        inputs: 4D input tensor `[B, H, W, C_in]`.
        kernel: 4D filter tensor `[KH, KW, C_in, C_out]`.
        stride: Spatial stride `(sh, sw)`.
        padding: Padding mode (`'SAME'` or `'VALID'`).

    Returns:
        Independent reference 4D convolved `Tensor`.
    """
    x = _tensor_ops.asarray(inputs)
    w = _tensor_ops.asarray(kernel)
    b_sz, h_in, w_in, c_in = x.shape
    kh, kw, _, c_out = w.shape
    sh, sw = stride
    if padding.upper() == "SAME":
        out_h = (h_in + sh - 1) // sh
        out_w = (w_in + sw - 1) // sw
        pad_top = builtins.max((out_h - 1) * sh + kh - h_in, 0) // 2
        pad_left = builtins.max((out_w - 1) * sw + kw - w_in, 0) // 2
    else:
        out_h = (h_in - kh) // sh + 1
        out_w = (w_in - kw) // sw + 1
        pad_top = 0
        pad_left = 0
    x_strides = _tensor_ops._compute_strides(x.shape)
    w_strides = _tensor_ops._compute_strides(w.shape)
    xf = x.flat_data
    wf = w.flat_data
    out_vals: list[float] = []
    for b_i in range(b_sz):
        for oh in range(out_h):
            for ow in range(out_w):
                for co in range(c_out):
                    acc = 0.0
                    for fh in range(kh):
                        ih = oh * sh - pad_top + fh
                        if 0 <= ih < h_in:
                            for fw in range(kw):
                                iw = ow * sw - pad_left + fw
                                if 0 <= iw < w_in:
                                    for ci in range(c_in):
                                        x_idx = (
                                            b_i * x_strides[0]
                                            + ih * x_strides[1]
                                            + iw * x_strides[2]
                                            + ci
                                        )
                                        w_idx = (
                                            fh * w_strides[0]
                                            + fw * w_strides[1]
                                            + ci * w_strides[2]
                                            + co
                                        )
                                        acc += xf[x_idx] * wf[w_idx]
                    out_vals.append(acc)
    return Tensor(out_vals, (b_sz, out_h, out_w, c_out), dtype=x.dtype)


def ref_depthwise_conv1d(
    inputs: TensorOrScalar,
    kernel: TensorOrScalar,
    is_causal: bool = False,
) -> Tensor:
    """Compute 1D depthwise convolution using independent pure-Python nested loops.

    Args:
        inputs: 3D input tensor `[B, T, C]`.
        kernel: Depthwise kernel tensor `[K, C]` or `[K, 1, C]`.
        is_causal: Whether causal left-only padding is used.

    Returns:
        Independent reference 3D depthwise convolved `Tensor`.
    """
    x = _tensor_ops.asarray(inputs)
    w = _tensor_ops.asarray(kernel)
    b_sz, t_len, c_dim = x.shape
    k_sz = w.shape[0]
    w_2d = _tensor_ops.reshape(w, (k_sz, c_dim))
    pad_left = (k_sz - 1) if is_causal else (k_sz - 1) // 2
    xf = x.flat_data
    wf = w_2d.flat_data
    out_vals: list[float] = []
    for b_i in range(b_sz):
        for t_i in range(t_len):
            for c_i in range(c_dim):
                acc = 0.0
                for k_i in range(k_sz):
                    src_t = t_i - pad_left + k_i
                    if 0 <= src_t < t_len:
                        acc += xf[(b_i * t_len + src_t) * c_dim + c_i] * wf[k_i * c_dim + c_i]
                out_vals.append(acc)
    return Tensor(out_vals, (b_sz, t_len, c_dim), dtype=x.dtype)


class _TestingNamespace:
    """Namespace providing `assert_allclose` and `assert_not_allclose` for tests."""

    @staticmethod
    def assert_allclose(
        actual: TensorOrScalar,
        desired: TensorOrScalar,
        rtol: float = 1e-6,
        atol: float = 1e-6,
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
        if ta.shape != td.shape and ta.size != 1 and td.size != 1:
            raise AssertionError(f"Shape mismatch: {ta.shape} vs {td.shape}")
        target_shape = _tensor_ops._broadcast_shapes(ta.shape, td.shape)
        flat_a = _tensor_ops._broadcast_flat(ta, target_shape)
        flat_d = _tensor_ops._broadcast_flat(td, target_shape)
        for idx, (va, vd) in enumerate(zip(flat_a, flat_d)):
            if math.isnan(va) and math.isnan(vd):
                continue
            if math.fabs(va - vd) > atol + rtol * math.fabs(vd):
                raise AssertionError(
                    f"Element {idx} mismatch: {va} vs {vd} (atol={atol}, rtol={rtol})"
                )

    @staticmethod
    def assert_not_allclose(
        actual: TensorOrScalar,
        desired: TensorOrScalar,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Assert that `actual` and `desired` differ by more than `(rtol, atol)` at some element.

        Args:
            actual: First tensor or scalar.
            desired: Second tensor or scalar.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Raises:
            AssertionError: If all elements of `actual` and `desired` are within tolerance.
        """
        if allclose(actual, desired, rtol=rtol, atol=atol):
            raise AssertionError("Expected tensors to differ, but they are allclose.")


class _RandomNamespace:
    """Namespace providing deterministic `normal` and `uniform` generation for tests."""

    @staticmethod
    def normal(size: Sequence[int] = (1,), seed: int = 42, scale: float = 1.0) -> Tensor:
        """Generate a deterministic normal `Tensor` of `size` with configurable `seed`.

        Args:
            size: Output tensor shape.
            seed: Deterministic Weyl sequence seed.
            scale: Standard deviation multiplier.

        Returns:
            Normally distributed `Tensor`.
        """
        return WeightInit(method="normal", scale=scale, seed=seed).init_tensor(tuple(size))

    @staticmethod
    def uniform(size: Sequence[int] = (1,), seed: int = 42, scale: float = 1.0) -> Tensor:
        """Generate a deterministic uniform `Tensor` of `size` in `[-scale, scale]`.

        Args:
            size: Output tensor shape.
            seed: Deterministic Weyl sequence seed.
            scale: Half-width bound of the uniform distribution.

        Returns:
            Uniformly distributed `Tensor`.
        """
        return WeightInit(method="uniform", scale=scale, seed=seed).init_tensor(tuple(size))


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
