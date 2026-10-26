"""Linear algebra, einsum, and sliding-window patch extraction operations for zero_pax."""

from __future__ import annotations

import itertools

from zero_pax._tensor_core import (
    Tensor,
    TensorOrScalar,
    _broadcast_shapes,
    _compute_strides,
    _float_sum,
    _int_max,
    _promote_dtypes,
    _record_ir_op,
    _shape_size,
    asarray,
)


def _broadcast_to_internal(t: Tensor, target: tuple[int, ...]) -> Tensor:
    """Broadcast `t` to `target` without recording an intermediate IR op.

    Args:
        t: Source `Tensor`.
        target: Target shape tuple.

    Returns:
        Broadcasted `Tensor`.
    """
    from zero_pax._tensor_core import _broadcast_flat

    return Tensor(_broadcast_flat(t, target), target, dtype=t.dtype)


def matmul(a: TensorOrScalar, b: TensorOrScalar) -> Tensor:
    """Compute matrix product of two tensors (supporting 1D, 2D, and batched N-D tensors).

    Args:
        a: Left-hand tensor operand.
        b: Right-hand tensor operand.

    Returns:
        Matrix product `Tensor`.

    Raises:
        ValueError: If inner contraction dimensions do not match.
    """
    ta = asarray(a)
    tb = asarray(b)
    if ta.ndim == 1 and tb.ndim == 1:
        if ta.shape[0] != tb.shape[0]:
            raise ValueError(f"Inner dimensions do not match for 1D dot: {ta.shape} vs {tb.shape}")
        val = _float_sum(va * vb for va, vb in zip(ta.flat_data, tb.flat_data))
        out_1d = Tensor((val,), (), dtype=ta.dtype)
        return _record_ir_op("MatMul", (ta, tb), out_1d)
    if tb.ndim == 1:
        k_dim = ta.shape[-1]
        if tb.shape[0] != k_dim:
            raise ValueError(f"Shape mismatch in matmul: {ta.shape} @ {tb.shape}")
        leading_shape = ta.shape[:-1]
        num_rows = _shape_size(leading_shape)
        a_flat = ta.flat_data
        b_flat = tb.flat_data
        out_vec = [
            _float_sum(a_flat[r * k_dim + k] * b_flat[k] for k in range(k_dim))
            for r in range(num_rows)
        ]
        out_nd_1d = Tensor(out_vec, leading_shape, dtype=ta.dtype)
        return _record_ir_op("MatMul", (ta, tb), out_nd_1d)
    if ta.ndim == 1:
        k_dim = ta.shape[0]
        if tb.shape[-2] != k_dim:
            raise ValueError(f"Shape mismatch in matmul: {ta.shape} @ {tb.shape}")
        batch_b = tb.shape[:-2]
        n_dim = tb.shape[-1]
        num_batches = _shape_size(batch_b)
        a_flat = ta.flat_data
        b_flat = tb.flat_data
        stride_b = k_dim * n_dim
        out_vec = [
            _float_sum(a_flat[k] * b_flat[b_idx * stride_b + k * n_dim + c] for k in range(k_dim))
            for b_idx in range(num_batches)
            for c in range(n_dim)
        ]
        out_1d_nd = Tensor(out_vec, batch_b + (n_dim,), dtype=ta.dtype)
        return _record_ir_op("MatMul", (ta, tb), out_1d_nd)

    batch_a = ta.shape[:-2]
    m_dim, k_dim = ta.shape[-2], ta.shape[-1]
    batch_b = tb.shape[:-2]
    k_dim_b, n_dim = tb.shape[-2], tb.shape[-1]
    if k_dim != k_dim_b:
        raise ValueError(f"Contraction dimension mismatch in matmul: {ta.shape} vs {tb.shape}")

    batch_shape = _broadcast_shapes(batch_a, batch_b)
    ta_bcast = _broadcast_to_internal(ta, batch_shape + (m_dim, k_dim))
    tb_bcast = _broadcast_to_internal(tb, batch_shape + (k_dim, n_dim))
    num_batches = _shape_size(batch_shape)
    a_flat = ta_bcast.flat_data
    b_flat = tb_bcast.flat_data
    stride_a = m_dim * k_dim
    stride_b = k_dim * n_dim

    out_flat: list[float] = []
    for b_idx in range(num_batches):
        base_a = b_idx * stride_a
        base_b = b_idx * stride_b
        for r in range(m_dim):
            row_offset = base_a + r * k_dim
            for c in range(n_dim):
                acc = 0.0
                for k in range(k_dim):
                    acc += a_flat[row_offset + k] * b_flat[base_b + k * n_dim + c]
                out_flat.append(acc)
    out = Tensor(out_flat, batch_shape + (m_dim, n_dim), dtype=ta.dtype)
    return _record_ir_op("MatMul", (ta, tb), out)


def dot(a: TensorOrScalar, b: TensorOrScalar) -> Tensor:
    """Compute the dot product of two tensors.

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        Dot product `Tensor`.
    """
    return matmul(a, b)


def _expand_ellipsis(spec: str, ndim: int, ellipsis_labels: str) -> str:
    """Replace `'...'` in an einsum subscript `spec` with `ellipsis_labels`.

    Args:
        spec: Single operand or output subscript string.
        ndim: Rank of the corresponding operand (used when `spec` contains `'...'`).
        ellipsis_labels: Assigned label characters for broadcast ellipsis axes.

    Returns:
        Subscript string with `'...'` replaced by concrete axis labels.
    """
    clean = spec.replace(" ", "")
    if "..." not in clean:
        return clean
    explicit_count = len(clean) - 3
    ell_count = _int_max(0, ndim - explicit_count)
    sub_labels = ellipsis_labels[len(ellipsis_labels) - ell_count :] if ell_count > 0 else ""
    return clean.replace("...", sub_labels)


def _allocate_ellipsis_labels(equation: str, count: int) -> str:
    """Allocate `count` subscript characters not already present in `equation`.

    Args:
        equation: Raw einsum subscript equation string.
        count: Number of ellipsis broadcast dimensions to label.

    Returns:
        String of `count` unique label characters disjoint from `equation`.
    """
    used = {ch for ch in equation if ch.isalpha()}
    pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return "".join([ch for ch in pool if ch not in used][:count])


def einsum(equation: str, *operands: TensorOrScalar) -> Tensor:
    """Evaluate the Einstein summation convention on the given operands in pure Python.

    Supports arbitrary explicit and implicit subscript equations, repeated diagonal/trace
    subscripts (e.g. `'ii->'`, `'ii->i'`), as well as `'...'` ellipsis batch dimensions
    even when uppercase letters appear in `equation`.

    Args:
        equation: Einsum subscript specification string (e.g. `'btnh,bsnh->bnts'`, `'...d,df->...f'`).
        *operands: Input tensors for the contraction.

    Returns:
        Contracted output `Tensor`.

    Raises:
        ValueError: If the number of operands or axis dimensions does not match `equation`.
    """
    tensors = [asarray(op) for op in operands]
    if "->" in equation:
        lhs, rhs = equation.split("->", 1)
    else:
        lhs = equation
        rhs = None

    raw_in_specs = [s.strip() for s in lhs.split(",")]
    if len(raw_in_specs) != len(tensors):
        raise ValueError(
            f"Number of einsum subscripts ({len(raw_in_specs)}) does not match operands ({len(tensors)})"
        )

    max_ell_rank = 0
    for spec, t in zip(raw_in_specs, tensors):
        clean = spec.replace(" ", "")
        if "..." in clean:
            max_ell_rank = _int_max(max_ell_rank, t.ndim - (len(clean) - 3))

    ell_labels = _allocate_ellipsis_labels(equation, max_ell_rank)
    in_specs = [
        _expand_ellipsis(spec, t.ndim, ell_labels) for spec, t in zip(raw_in_specs, tensors)
    ]

    if rhs is not None:
        out_spec = rhs.replace(" ", "").replace("...", ell_labels)
    else:
        counts: dict[str, int] = {}
        for sp in in_specs:
            for ch in sp:
                counts[ch] = counts.get(ch, 0) + 1
        out_spec = "".join(sorted(ch for ch, cnt in counts.items() if cnt == 1))

    dim_sizes: dict[str, int] = {}
    for sp, t in zip(in_specs, tensors):
        if len(sp) != t.ndim:
            raise ValueError(f"Einsum subscript {sp!r} does not match operand shape {t.shape}")
        for ch, d in zip(sp, t.shape):
            if ch in dim_sizes and dim_sizes[ch] != d:
                if dim_sizes[ch] == 1:
                    dim_sizes[ch] = d
                elif d != 1:
                    raise ValueError(
                        f"Dimension size mismatch for axis {ch!r}: {dim_sizes[ch]} vs {d}"
                    )
            else:
                dim_sizes[ch] = d

    all_labels: list[str] = list(dict.fromkeys(out_spec))
    sum_labels: list[str] = []
    for sp in in_specs:
        for ch in sp:
            if ch not in all_labels and ch not in sum_labels:
                sum_labels.append(ch)

    out_shape = tuple(dim_sizes[ch] for ch in out_spec)
    out_ranges = [range(dim_sizes[ch]) for ch in all_labels]
    sum_ranges = [range(dim_sizes[ch]) for ch in sum_labels]

    op_strides: list[tuple[int, ...]] = []
    op_flats: list[tuple[float, ...]] = []
    for t in tensors:
        raw_st = _compute_strides(t.shape)
        eff_st = tuple(0 if t.shape[i] == 1 else raw_st[i] for i in range(t.ndim))
        op_strides.append(eff_st)
        op_flats.append(t.flat_data)

    label_to_pos: dict[str, int] = {ch: idx for idx, ch in enumerate(all_labels + sum_labels)}
    op_pos_strides: list[tuple[tuple[int, int], ...]] = []
    for sp, st in zip(in_specs, op_strides):
        accum_by_pos: dict[int, int] = {}
        for ax, ch in enumerate(sp):
            if st[ax] != 0:
                pos = label_to_pos[ch]
                accum_by_pos[pos] = accum_by_pos.get(pos, 0) + st[ax]
        op_pos_strides.append(tuple(accum_by_pos.items()))

    out_flat: list[float] = []
    num_ops = len(tensors)
    for out_coord in itertools.product(*out_ranges):
        acc = 0.0
        for sum_coord in itertools.product(*sum_ranges):
            full_coord = out_coord + sum_coord
            prod = 1.0
            for op_idx in range(num_ops):
                flat_idx = 0
                for pos, stride_val in op_pos_strides[op_idx]:
                    flat_idx += full_coord[pos] * stride_val
                prod *= op_flats[op_idx][flat_idx]
            acc += prod
        out_flat.append(acc)

    out_dtype = tensors[0].dtype
    for t in tensors[1:]:
        out_dtype = _promote_dtypes(out_dtype, t.dtype)
    out = Tensor(out_flat, out_shape, dtype=out_dtype)
    return _record_ir_op("Einsum", tensors, out, attributes={"equation": equation})


def extract_patches_1d(
    x_pad: TensorOrScalar,
    out_t: int,
    window: int,
    stride: int,
    dilation: int = 1,
) -> Tensor:
    """Extract strided 1D sliding windows from a padded 3D tensor `[B, T_pad, C]`.

    Args:
        x_pad: Padded 3D input tensor of shape `[B, T_pad, C]`.
        out_t: Number of output time steps.
        window: Kernel/window size `K`.
        stride: Temporal stride `S`.
        dilation: Kernel dilation factor `D`.

    Returns:
        4D window tensor of shape `[B, T_out, K, C]`.
    """
    from zero_pax._tensor_ops import concatenate, expand_dims

    t = asarray(x_pad)
    slices: list[Tensor] = []
    for k in range(window):
        start = k * dilation
        end = start + out_t * stride
        window_slice = t[:, start:end:stride, :]
        slices.append(expand_dims(window_slice, axis=2))
    return concatenate(slices, axis=2)


def extract_patches_2d(
    x_pad: TensorOrScalar,
    out_h: int,
    out_w: int,
    window_h: int,
    window_w: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int = 1,
    dilation_w: int = 1,
) -> Tensor:
    """Extract strided 2D sliding windows from a padded 4D tensor `[B, H_pad, W_pad, C]`.

    Args:
        x_pad: Padded 4D input tensor of shape `[B, H_pad, W_pad, C]`.
        out_h: Output spatial height `H_out`.
        out_w: Output spatial width `W_out`.
        window_h: Kernel/window height `K_h`.
        window_w: Kernel/window width `K_w`.
        stride_h: Height stride `S_h`.
        stride_w: Width stride `S_w`.
        dilation_h: Height dilation factor `D_h`.
        dilation_w: Width dilation factor `D_w`.

    Returns:
        6D window tensor of shape `[B, H_out, W_out, K_h, K_w, C]`.
    """
    from zero_pax._tensor_ops import concatenate, expand_dims

    t = asarray(x_pad)
    row_slices: list[Tensor] = []
    for i in range(window_h):
        h_start = i * dilation_h
        h_end = h_start + out_h * stride_h
        col_slices: list[Tensor] = []
        for j in range(window_w):
            w_start = j * dilation_w
            w_end = w_start + out_w * stride_w
            patch = t[:, h_start:h_end:stride_h, w_start:w_end:stride_w, :]
            col_slices.append(expand_dims(expand_dims(patch, axis=3), axis=4))
        row_slices.append(concatenate(col_slices, axis=4))
    return concatenate(row_slices, axis=3)
