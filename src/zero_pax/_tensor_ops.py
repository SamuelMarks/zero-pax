"""Pure-Python typed tensor operations and Tensor engine backed by ml_switcheroo_ir."""

from __future__ import annotations

import array as _py_array
import builtins
import itertools
import math
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import Protocol, TypeVar, Union, runtime_checkable

from ml_switcheroo_ir import AttributeValue
from ml_switcheroo_ir.types import DType

Scalar = Union[int, float, bool]
_R_co = TypeVar("_R_co", covariant=True)


@runtime_checkable
class ToListProtocol(Protocol):
    """Protocol for external array-like objects exposing `.tolist()` and `.shape`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tuple of array dimensions.

        Returns:
            Shape tuple of ints.
        """

    def tolist(self) -> NestedSequence:
        """Convert the array into nested Python lists or a scalar.

        Returns:
            Nested Python list or scalar value.
        """


@runtime_checkable
class VariadicCallableProtocol(Protocol):
    """Protocol for callables accepting arbitrary `TensorOrScalar` arguments."""

    def __call__(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Invoke the callable.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Computed result.
        """


@runtime_checkable
class TypedVariadicCallable(Protocol[_R_co]):
    """Generic protocol for callables accepting arbitrary `TensorOrScalar` arguments and returning `_R_co`."""

    def __call__(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> _R_co:
        """Invoke the callable.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Computed result of type `_R_co`.
        """


if typing.TYPE_CHECKING:
    NestedSequence = Union[Scalar, Sequence["NestedSequence"]]
    TensorOrScalar = Union[
        "Tensor",
        Scalar,
        str,
        None,
        DType,
        ToListProtocol,
        FunctionType,
        BuiltinFunctionType,
        MethodType,
        VariadicCallableProtocol,
        TypedVariadicCallable["TensorOrScalar"],
        Sequence["TensorOrScalar"],
        Mapping[str, "TensorOrScalar"],
    ]
    SliceAtom = slice[int | None, int | None, int | None]
    IndexAtom = Union[int, SliceAtom, "builtins.ellipsis", None, "Tensor", Sequence[int]]
    _TensorBase = float
else:
    NestedSequence = AttributeValue
    TensorOrScalar = AttributeValue
    SliceAtom = slice
    IndexAtom = Union[int, slice, "builtins.ellipsis", None, "Tensor", Sequence[int]]
    _TensorBase = ToListProtocol

IndexSpec = Union[IndexAtom, tuple[IndexAtom, ...]]

_EAGER_MODE: bool = True
pi: float = math.pi
inf: float = math.inf
nan: float = math.nan

_DTYPE_BY_NAME: Mapping[str, DType] = {member.value: member for member in DType}
_DTYPE_ALIASES: Mapping[str, DType] = {
    "float": DType.float32,
    "fp32": DType.float32,
    "fp16": DType.float16,
    "bf16": DType.bfloat16,
    "double": DType.float64,
    "int": DType.int32,
    "bool_": DType.bool,
}


def resolve_dtype(dtype: DType | str | None) -> DType | None:
    """Resolve an optional `DType` or string name into an `ml_switcheroo_ir` `DType`.

    Args:
        dtype: Optional `DType` enum or string name (e.g. `'float32'`, `'bfloat16'`).

    Returns:
        Resolved `DType` enum member, or `None` if `dtype` is `None`.

    Raises:
        ValueError: If `dtype` is a string that does not match any supported `DType`.
    """
    if dtype is None:
        return None
    if isinstance(dtype, DType):
        return dtype
    normalized = dtype.strip().lower()
    if normalized in _DTYPE_BY_NAME:
        return _DTYPE_BY_NAME[normalized]
    if normalized in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[normalized]
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def set_eager_mode(enabled: bool = True) -> bool:
    """Configure eager execution mode for tensor operations.

    Args:
        enabled: Whether eager execution should be active.

    Returns:
        Updated boolean eager mode state.
    """
    global _EAGER_MODE
    _EAGER_MODE = enabled
    return _EAGER_MODE


def _int_max(a: int, b: int) -> int:
    """Return the larger of two integers `a` and `b`.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Larger integer.
    """
    return a if a >= b else b


def _float_max(a: float, b: float) -> float:
    """Return the larger of two floats `a` and `b`.

    Args:
        a: First float.
        b: Second float.

    Returns:
        Larger float.
    """
    return a if a >= b else b


def _float_min(a: float, b: float) -> float:
    """Return the smaller of two floats `a` and `b`.

    Args:
        a: First float.
        b: Second float.

    Returns:
        Smaller float.
    """
    return a if a <= b else b


def _float_sum(vals: Iterable[float]) -> float:
    """Sum an iterable of floats into a `float`.

    Args:
        vals: Iterable of float numbers.

    Returns:
        Summed float value.
    """
    acc = 0.0
    for v in vals:
        acc += v
    return acc


def _int_sum(vals: Iterable[int]) -> int:
    """Sum an iterable of ints into an `int`.

    Args:
        vals: Iterable of integers.

    Returns:
        Summed integer value.
    """
    acc = 0
    for v in vals:
        acc += v
    return acc


def _shape_size(shape: tuple[int, ...]) -> int:
    """Compute the total number of elements for `shape`.

    Args:
        shape: Tuple of non-negative dimension sizes.

    Returns:
        Product of all dimensions in `shape` (`1` for `()`).
    """
    total = 1
    for dim in shape:
        total *= dim
    return total


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute row-major C-contiguous strides for `shape`.

    Args:
        shape: Tuple of dimension sizes.

    Returns:
        Tuple of element strides for each axis.
    """
    ndim = len(shape)
    if ndim == 0:
        return ()
    strides: list[int] = [1] * ndim
    running = 1
    for idx in range(ndim - 1, -1, -1):
        strides[idx] = running
        running *= _int_max(1, shape[idx])
    return tuple(strides)


def _broadcast_shapes(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the broadcasted shape of `shape_a` and `shape_b` under NumPy rules.

    Args:
        shape_a: First shape tuple.
        shape_b: Second shape tuple.

    Returns:
        Broadcasted shape tuple.

    Raises:
        ValueError: If `shape_a` and `shape_b` cannot be broadcast together.
    """
    len_a = len(shape_a)
    len_b = len(shape_b)
    max_len = _int_max(len_a, len_b)
    padded_a = (1,) * (max_len - len_a) + shape_a
    padded_b = (1,) * (max_len - len_b) + shape_b
    result: list[int] = []
    for dim_a, dim_b in zip(padded_a, padded_b):
        if dim_a == dim_b:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
        elif dim_b == 1:
            result.append(dim_a)
        else:
            raise ValueError(
                f"Operands could not be broadcast together with shapes {shape_a} {shape_b}"
            )
    return tuple(result)


def _broadcast_flat(tensor: Tensor, target_shape: tuple[int, ...]) -> tuple[float, ...]:
    """Broadcast `tensor`'s flat buffer to `target_shape`.

    Args:
        tensor: Source `Tensor`.
        target_shape: Target shape to broadcast into.

    Returns:
        Flat tuple of floats of length `_shape_size(target_shape)`.

    Raises:
        ValueError: If `tensor.shape` cannot be broadcast to `target_shape`.
    """
    src_shape = tensor.shape
    if src_shape == target_shape:
        return tensor.flat_data
    target_size = _shape_size(target_shape)
    if target_size == 0:
        return ()
    if src_shape == ():
        val = tensor.flat_data[0]
        return (val,) * target_size

    ndim_target = len(target_shape)
    ndim_src = len(src_shape)
    if ndim_src > ndim_target:
        raise ValueError(f"Cannot broadcast shape {src_shape} to smaller rank {target_shape}")

    padded_src = (1,) * (ndim_target - ndim_src) + src_shape
    for s_dim, t_dim in zip(padded_src, target_shape):
        if s_dim != t_dim and s_dim != 1:
            raise ValueError(f"Cannot broadcast shape {src_shape} to {target_shape}")

    raw_strides = _compute_strides(padded_src)
    eff_strides = tuple(0 if padded_src[i] == 1 else raw_strides[i] for i in range(ndim_target))
    src_flat = tensor.flat_data
    out: list[float] = []
    for coords in itertools.product(*(range(d) for d in target_shape)):
        flat_idx = 0
        for c, st in zip(coords, eff_strides):
            flat_idx += c * st
        out.append(src_flat[flat_idx])
    return tuple(out)


def _flatten_nested(val: TensorOrScalar) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Flatten a scalar, nested sequence, `ToListProtocol`, or `Tensor` into `(flat_tuple, shape)`.

    Args:
        val: Input tensor, scalar, external array, or nested sequence.

    Returns:
        Pair `(flat_floats, shape_tuple)`.
    """
    if isinstance(val, Tensor):
        return val.flat_data, val.shape
    if isinstance(val, (int, float, bool)):
        return (float(val),), ()
    if isinstance(val, ToListProtocol):
        nested = val.tolist()
        flat_vals, _ = _flatten_nested(nested)
        return flat_vals, tuple(int(d) for d in val.shape)
    if val is None or isinstance(val, (str, DType, Mapping)) or callable(val):
        return (0.0,), ()

    seq_items = list(val)
    if not seq_items:
        return (), (0,)
    sub_flats: list[tuple[float, ...]] = []
    first_sub_shape: tuple[int, ...] | None = None
    for item in seq_items:
        sub_flat, sub_shape = _flatten_nested(item)
        if first_sub_shape is None:
            first_sub_shape = sub_shape
        sub_flats.append(sub_flat)
    combined: list[float] = []
    for sf in sub_flats:
        combined.extend(sf)
    resolved_sub = first_sub_shape if first_sub_shape is not None else ()
    return tuple(combined), (len(seq_items),) + resolved_sub


def _unflatten_nested(flat: tuple[float, ...], shape: tuple[int, ...]) -> NestedSequence:
    """Reconstruct nested Python lists (or a scalar float for 0D) from `flat` and `shape`.

    Args:
        flat: Flat tuple of float elements.
        shape: Shape tuple.

    Returns:
        Scalar float if `shape == ()`, otherwise nested list of floats.
    """
    if not shape:
        return flat[0] if flat else 0.0
    if len(shape) == 1:
        return list(flat[: shape[0]])
    step = _shape_size(shape[1:])
    return [_unflatten_nested(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


class Tensor(_TensorBase):
    """Pure-Python multi-dimensional tensor backed by flat float storage and `ml_switcheroo_ir.DType`.

    Attributes:
        _flat: Immutable tuple of float64 element values in row-major order.
        _shape: Tuple of non-negative integer dimensions.
        _dtype: `ml_switcheroo_ir.DType` descriptor.
        _buf: Contiguous C-double `array.array` buffer providing zero-copy `__array_interface__` in tests.
    """

    def __init__(
        self,
        flat: Sequence[float],
        shape: tuple[int, ...],
        dtype: DType | None = None,
    ) -> None:
        """Initialize a `Tensor` from flat row-major data, `shape`, and optional `dtype`.

        Args:
            flat: Sequence of numeric values of length `_shape_size(shape)`.
            shape: Tuple of integer dimension sizes.
            dtype: Optional `ml_switcheroo_ir.DType` (defaults to `DType.float32`).
        """
        self._shape: tuple[int, ...] = tuple(int(d) for d in shape)
        self._dtype: DType = dtype if dtype is not None else DType.float32
        raw_floats = tuple(float(v) for v in flat)
        self._buf: _py_array.array[float] = (
            _py_array.array("f", raw_floats) if raw_floats else _py_array.array("f", [0.0])
        )
        self._flat: tuple[float, ...] = tuple(self._buf) if raw_floats else ()

    @property
    def flat_data(self) -> tuple[float, ...]:
        """Return the flat row-major tuple of float values.

        Returns:
            Tuple of floats.
        """
        return self._flat

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor shape tuple.

        Returns:
            Tuple of dimension sizes.
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """Return the number of tensor dimensions (rank).

        Returns:
            Integer rank `len(self._shape)`.
        """
        return len(self._shape)

    @property
    def size(self) -> int:
        """Return the total number of scalar elements in the tensor.

        Returns:
            Total element count.
        """
        return len(self._flat)

    @property
    def dtype(self) -> DType:
        """Return the `ml_switcheroo_ir` data type of the tensor.

        Returns:
            `DType` enum value.
        """
        return self._dtype

    @property
    def __array_interface__(
        self,
    ) -> dict[str, tuple[int, ...] | str | tuple[int, bool] | int]:
        """Provide the standard Python buffer/array interface for test assertions without importing `numpy`.

        Returns:
            Array interface dictionary describing shape, `<f4` typestr, and buffer pointer.
        """
        return {
            "shape": self._shape,
            "typestr": "<f4",
            "data": (self._buf.buffer_info()[0], False),
            "version": 3,
        }

    def tolist(self) -> NestedSequence:
        """Convert the tensor to nested Python lists (or a scalar float if 0D).

        Returns:
            Nested list of floats or scalar float.
        """
        return _unflatten_nested(self._flat, self._shape)

    def astype(self, dtype: DType | str | None) -> Tensor:
        """Return a copy of the tensor cast to `dtype`.

        Args:
            dtype: Target `DType` or string dtype name.

        Returns:
            New `Tensor` with resolved `dtype`.
        """
        return Tensor(self._flat, self._shape, dtype=resolve_dtype(dtype) or self._dtype)

    def reshape(self, *newshape: int | Sequence[int]) -> Tensor:
        """Return a tensor with the same data reshaped to `newshape`.

        Args:
            *newshape: Target shape as positional integers or a single sequence of integers.

        Returns:
            Reshaped `Tensor`.
        """
        if len(newshape) == 1 and isinstance(newshape[0], Sequence):
            dims: Sequence[int] = newshape[0]
        else:
            dims = [int(d) for d in newshape if isinstance(d, int)]
        return reshape(self, dims)

    def __float__(self) -> float:
        """Convert a 0D or single-element tensor to a Python `float`.

        Returns:
            Scalar float value.
        """
        return self._flat[0] if self._flat else 0.0

    def __int__(self) -> int:
        """Convert a 0D or single-element tensor to a Python `int`.

        Returns:
            Scalar integer value.
        """
        return int(self._flat[0]) if self._flat else 0

    def __bool__(self) -> bool:
        """Return True if any element in the tensor is non-zero.

        Returns:
            Boolean truth value.
        """
        return any(v != 0.0 for v in self._flat)

    def __len__(self) -> int:
        """Return the size of the leading dimension.

        Returns:
            Leading dimension `self._shape[0]`.

        Raises:
            TypeError: If the tensor is 0-dimensional.
        """
        if not self._shape:
            raise TypeError("len() of unsized 0D tensor")
        return self._shape[0]

    def __iter__(self) -> Iterator[Tensor]:
        """Iterate over slices along the leading dimension (axis 0).

        Yields:
            Sub-tensors along axis 0.

        Raises:
            TypeError: If the tensor is 0-dimensional.
        """
        if not self._shape:
            raise TypeError("iteration over a 0-d tensor")
        for idx in range(self._shape[0]):
            yield self[idx]

    def __repr__(self) -> str:
        """Return a readable string representation of the tensor.

        Returns:
            Formatted `Tensor(...)` string.
        """
        return f"Tensor(shape={self._shape}, dtype={self._dtype.value})"

    def __getitem__(self, key: IndexSpec) -> Tensor:
        """Index or slice the tensor using standard Python/NumPy indexing rules.

        Supports integers, slices, `Ellipsis` (`...`), `None` (new axis), and 1D/N-D
        integer index `Tensor`s or sequences.

        Args:
            key: Single index item or tuple of index items.

        Returns:
            Indexed or sliced `Tensor`.

        Raises:
            IndexError: If an integer index is out of bounds for its axis.
        """
        items: list[IndexAtom] = list(key) if isinstance(key, tuple) else [key]

        # Handle integer Tensor / Sequence fancy indexing on axis 0 (e.g. table[ids])
        if len(items) == 1 and (
            isinstance(items[0], Tensor)
            or (isinstance(items[0], Sequence) and not isinstance(items[0], str))
        ):
            idx_tensor = asarray(items[0])
            axis0_len = self._shape[0] if self._shape else 1
            sub_shape = self._shape[1:]
            sub_size = _shape_size(sub_shape)
            gathered: list[float] = []
            for raw_idx in idx_tensor.flat_data:
                i_val = int(raw_idx)
                if i_val < 0:
                    i_val += axis0_len
                if i_val < 0 or i_val >= axis0_len:
                    raise IndexError(
                        f"Index {i_val} out of bounds for axis 0 with size {axis0_len}"
                    )
                start = i_val * sub_size
                gathered.extend(self._flat[start : start + sub_size])
            return Tensor(gathered, idx_tensor.shape + sub_shape, dtype=self._dtype)

        # Expand Ellipsis
        non_none_count = _int_sum(1 for it in items if it is not None and it is not Ellipsis)
        expanded: list[IndexAtom] = []
        for it in items:
            if it is Ellipsis:
                missing = _int_max(0, self.ndim - non_none_count)
                expanded.extend([slice(None)] * missing)
                non_none_count += missing
            else:
                expanded.append(it)

        while sum_non_none(expanded) < self.ndim:
            expanded.append(slice(None))

        src_strides = _compute_strides(self._shape)
        axis_ranges: list[Sequence[int]] = []
        out_shape: list[int] = []
        src_axis_for_range: list[int] = []
        cur_axis = 0

        for it in expanded:
            if it is None:
                out_shape.append(1)
            elif isinstance(it, int):
                dim_len = self._shape[cur_axis]
                norm_i = it + dim_len if it < 0 else it
                if norm_i < 0 or norm_i >= dim_len:
                    raise IndexError(
                        f"Index {it} is out of bounds for axis {cur_axis} with size {dim_len}"
                    )
                axis_ranges.append((norm_i,))
                src_axis_for_range.append(cur_axis)
                cur_axis += 1
            elif isinstance(it, slice):
                dim_len = self._shape[cur_axis]
                rng = range(*it.indices(dim_len))
                axis_ranges.append(rng)
                src_axis_for_range.append(cur_axis)
                out_shape.append(len(rng))
                cur_axis += 1
            else:
                assert isinstance(it, (Tensor, Sequence))
                idx_seq = [int(v) for v in asarray(it).flat_data]
                dim_len = self._shape[cur_axis]
                norm_seq = [v + dim_len if v < 0 else v for v in idx_seq]
                axis_ranges.append(norm_seq)
                src_axis_for_range.append(cur_axis)
                out_shape.append(len(norm_seq))
                cur_axis += 1

        if not axis_ranges:
            return Tensor(self._flat, tuple(out_shape), dtype=self._dtype)

        result_flat: list[float] = []
        for coord in itertools.product(*axis_ranges):
            flat_idx = 0
            for c_val, s_ax in zip(coord, src_axis_for_range):
                flat_idx += c_val * src_strides[s_ax]
            result_flat.append(self._flat[flat_idx])

        return Tensor(result_flat, tuple(out_shape), dtype=self._dtype)

    def __neg__(self) -> Tensor:
        """Compute elementwise negation `-self`.

        Returns:
            Negated `Tensor`.
        """
        return _unary_elementwise(self, lambda a: -a)

    def __pos__(self) -> Tensor:
        """Compute elementwise unary positive `+self`.

        Returns:
            Copy of `self`.
        """
        return Tensor(self._flat, self._shape, dtype=self._dtype)

    def __abs__(self) -> Tensor:
        """Compute elementwise absolute value `abs(self)`.

        Returns:
            Absolute value `Tensor`.
        """
        return _unary_elementwise(self, math.fabs)

    def __add__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise addition `self + other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Sum `Tensor`.
        """
        return _binary_elementwise(self, other, lambda a, b: a + b)

    def __radd__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise addition `other + self`.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Sum `Tensor`.
        """
        return _binary_elementwise(other, self, lambda a, b: a + b)

    def __sub__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise subtraction `self - other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Difference `Tensor`.
        """
        return _binary_elementwise(self, other, lambda a, b: a - b)

    def __rsub__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise subtraction `other - self`.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Difference `Tensor`.
        """
        return _binary_elementwise(other, self, lambda a, b: a - b)

    def __mul__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise multiplication `self * other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Product `Tensor`.
        """
        return _binary_elementwise(self, other, lambda a, b: a * b)

    def __rmul__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise multiplication `other * self`.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Product `Tensor`.
        """
        return _binary_elementwise(other, self, lambda a, b: a * b)

    def __truediv__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise true division `self / other`.

        Args:
            other: Right-hand tensor or scalar divisor.

        Returns:
            Quotient `Tensor`.
        """
        return _binary_elementwise(self, other, _safe_div)

    def __rtruediv__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise true division `other / self`.

        Args:
            other: Left-hand tensor or scalar dividend.

        Returns:
            Quotient `Tensor`.
        """
        return _binary_elementwise(other, self, _safe_div)

    def __floordiv__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise floor division `self // other`.

        Args:
            other: Right-hand tensor or scalar divisor.

        Returns:
            Floor quotient `Tensor`.
        """
        return _binary_elementwise(self, other, lambda a, b: math.floor(_safe_div(a, b)))

    def __rfloordiv__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise floor division `other // self`.

        Args:
            other: Left-hand tensor or scalar dividend.

        Returns:
            Floor quotient `Tensor`.
        """
        return _binary_elementwise(other, self, lambda a, b: math.floor(_safe_div(a, b)))

    def __pow__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise exponentiation `self ** other`.

        Args:
            other: Right-hand exponent tensor or scalar.

        Returns:
            Exponentiated `Tensor`.
        """
        return _binary_elementwise(self, other, _safe_pow)

    def __rpow__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise exponentiation `other ** self`.

        Args:
            other: Left-hand base tensor or scalar.

        Returns:
            Exponentiated `Tensor`.
        """
        return _binary_elementwise(other, self, _safe_pow)

    def __matmul__(self, other: TensorOrScalar) -> Tensor:
        """Compute matrix multiplication `self @ other`.

        Args:
            other: Right-hand tensor operand.

        Returns:
            Matrix product `Tensor`.
        """
        return matmul(self, other)

    def __lt__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise less-than comparison `self < other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self, other, lambda a, b: 1.0 if a < b else 0.0, dtype=DType.bool
        )

    def __le__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise less-than-or-equal comparison `self <= other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self, other, lambda a, b: 1.0 if a <= b else 0.0, dtype=DType.bool
        )

    def __gt__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise greater-than comparison `self > other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self, other, lambda a, b: 1.0 if a > b else 0.0, dtype=DType.bool
        )

    def __ge__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise greater-than-or-equal comparison `self >= other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self, other, lambda a, b: 1.0 if a >= b else 0.0, dtype=DType.bool
        )

    def __eq__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise equality comparison `self == other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self, other, lambda a, b: 1.0 if a == b else 0.0, dtype=DType.bool
        )

    def __ne__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise inequality comparison `self != other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self, other, lambda a, b: 1.0 if a != b else 0.0, dtype=DType.bool
        )

    def __or__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise logical OR `self | other`.

        Args:
            other: Right-hand mask tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where either operand is non-zero).
        """
        return _binary_elementwise(
            self,
            other,
            lambda a, b: 1.0 if (a != 0.0 or b != 0.0) else 0.0,
            dtype=DType.bool,
        )

    def __and__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise logical AND `self & other`.

        Args:
            other: Right-hand mask tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where both operands are non-zero).
        """
        return _binary_elementwise(
            self,
            other,
            lambda a, b: 1.0 if (a != 0.0 and b != 0.0) else 0.0,
            dtype=DType.bool,
        )


def sum_non_none(items: Sequence[IndexAtom]) -> int:
    """Count how many index atoms in `items` are not `None`.

    Args:
        items: Sequence of index specifications.

    Returns:
        Count of non-`None` index items.
    """
    return _int_sum(1 for it in items if it is not None)


def _safe_div(a: float, b: float) -> float:
    """Divide `a` by `b`, handling zero division gracefully under IEEE-754 conventions.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        Quotient `a / b`.
    """
    if b == 0.0:
        if a == 0.0:
            return 0.0
        return math.copysign(math.inf, a)
    return a / b


def _safe_pow(a: float, b: float) -> float:
    """Raise `a` to the power `b` safely for fractional exponents.

    Args:
        a: Base value.
        b: Exponent value.

    Returns:
        Result `a ** b`.
    """
    if a < 0.0 and b != math.floor(b):
        return 0.0
    return float(a**b)


def _unary_elementwise(
    x: TensorOrScalar,
    fn: Callable[[float], float],
    dtype: DType | None = None,
) -> Tensor:
    """Apply unary scalar function `fn` elementwise to `x`.

    Args:
        x: Input tensor or scalar.
        fn: Unary float-to-float transformation.
        dtype: Optional output `DType`.

    Returns:
        Transformed `Tensor`.
    """
    t = asarray(x)
    return Tensor([fn(v) for v in t.flat_data], t.shape, dtype=dtype or t.dtype)


def _binary_elementwise(
    a: TensorOrScalar,
    b: TensorOrScalar,
    fn: Callable[[float, float], float],
    dtype: DType | None = None,
) -> Tensor:
    """Apply binary scalar function `fn` elementwise with broadcasting between `a` and `b`.

    Args:
        a: Left tensor or scalar operand.
        b: Right tensor or scalar operand.
        fn: Binary float-to-float function.
        dtype: Optional output `DType`.

    Returns:
        Broadcasted result `Tensor`.
    """
    ta = asarray(a)
    tb = asarray(b)
    target_shape = _broadcast_shapes(ta.shape, tb.shape)
    flat_a = _broadcast_flat(ta, target_shape)
    flat_b = _broadcast_flat(tb, target_shape)
    out_dtype = dtype if dtype is not None else ta.dtype
    return Tensor(
        [fn(va, vb) for va, vb in zip(flat_a, flat_b)],
        target_shape,
        dtype=out_dtype,
    )


def _normalize_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    """Normalize an integer or sequence of integers into a shape tuple.

    Args:
        shape: Single dimension size or sequence of dimension sizes.

    Returns:
        Tuple of ints.
    """
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(d) for d in shape)


# ============================================================================
# Creation Operations
# ============================================================================


def asarray(x: TensorOrScalar, dtype: DType | str | None = None) -> Tensor:
    """Convert `x` into a pure-Python `Tensor`.

    Args:
        x: Input tensor, scalar, sequence, or array-like object.
        dtype: Optional target `DType` or string name.

    Returns:
        `Tensor` representation of `x`.
    """
    resolved = resolve_dtype(dtype)
    if isinstance(x, Tensor):
        if resolved is None or x.dtype == resolved:
            return x
        return Tensor(x.flat_data, x.shape, dtype=resolved)
    flat, shape = _flatten_nested(x)
    return Tensor(flat, shape, dtype=resolved)


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


# ============================================================================
# Shape & Manipulation Operations
# ============================================================================


def reshape(x: TensorOrScalar, newshape: Sequence[int]) -> Tensor:
    """Give a new shape to a tensor without changing its data.

    Args:
        x: Tensor to be reshaped.
        newshape: Target shape tuple or sequence (may contain at most one `-1`).

    Returns:
        Reshaped `Tensor`.

    Raises:
        ValueError: If `newshape` is incompatible with `x.size`.
    """
    t = asarray(x)
    raw_dims = [int(d) for d in newshape]
    neg_indices = [i for i, d in enumerate(raw_dims) if d < 0]
    if len(neg_indices) == 1:
        known_prod = 1
        for d in raw_dims:
            if d >= 0:
                known_prod *= _int_max(1, d)
        inferred = t.size // _int_max(1, known_prod)
        raw_dims[neg_indices[0]] = inferred
    target_shape = tuple(raw_dims)
    if _shape_size(target_shape) != t.size:
        raise ValueError(f"Cannot reshape tensor of size {t.size} into shape {target_shape}")
    return Tensor(t.flat_data, target_shape, dtype=t.dtype)


def transpose(x: TensorOrScalar, axes: Sequence[int] | None = None) -> Tensor:
    """Permute the dimensions of a tensor.

    Args:
        x: Input tensor.
        axes: Optional permutation of `[0, 1, ..., ndim - 1]`.

    Returns:
        Transposed `Tensor`.
    """
    t = asarray(x)
    ndim = t.ndim
    if ndim <= 1 and axes is None:
        return t
    perm = (
        tuple(range(ndim - 1, -1, -1))
        if axes is None
        else tuple(ax + ndim if ax < 0 else ax for ax in axes)
    )
    out_shape = tuple(t.shape[p] for p in perm)
    src_strides = _compute_strides(t.shape)
    src_flat = t.flat_data
    out_flat: list[float] = []
    for out_coords in itertools.product(*(range(d) for d in out_shape)):
        flat_idx = 0
        for out_ax, p_ax in enumerate(perm):
            flat_idx += out_coords[out_ax] * src_strides[p_ax]
        out_flat.append(src_flat[flat_idx])
    return Tensor(out_flat, out_shape, dtype=t.dtype)


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
    return Tensor(_broadcast_flat(t, target), target, dtype=t.dtype)


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
    return Tensor(t.flat_data, new_shape, dtype=t.dtype)


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
    return Tensor(out_flat, out_shape, dtype=first.dtype)


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
        parts.append(t[slicer])
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
    """
    t = asarray(x)
    pads = [(int(b), int(a)) for b, a in pad_width]
    out_shape = tuple(t.shape[i] + pads[i][0] + pads[i][1] for i in range(t.ndim))
    src_strides = _compute_strides(t.shape)
    src_flat = t.flat_data
    is_edge = mode == "edge"
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
    return Tensor(out_flat, out_shape, dtype=t.dtype)


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
        return t
    src_strides = _compute_strides(t.shape)
    src_flat = t.flat_data
    out_flat: list[float] = []
    for out_coords in itertools.product(*(range(d) for d in t.shape)):
        flat_idx = 0
        for ax, c in enumerate(out_coords):
            src_c = (c - shift) % dim_len if ax == norm_ax else c
            flat_idx += src_c * src_strides[ax]
        out_flat.append(src_flat[flat_idx])
    return Tensor(out_flat, t.shape, dtype=t.dtype)


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
    return Tensor(out_flat, target_shape, dtype=tx.dtype)


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


# ============================================================================
# Linear Algebra & Einsum Operations
# ============================================================================


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
        return Tensor((val,), (), dtype=ta.dtype)
    if ta.ndim == 2 and tb.ndim == 1:
        m_dim, k_dim = ta.shape
        if tb.shape[0] != k_dim:
            raise ValueError(f"Shape mismatch in matmul: {ta.shape} @ {tb.shape}")
        a_flat = ta.flat_data
        b_flat = tb.flat_data
        out_vec = [
            _float_sum(a_flat[r * k_dim + k] * b_flat[k] for k in range(k_dim))
            for r in range(m_dim)
        ]
        return Tensor(out_vec, (m_dim,), dtype=ta.dtype)
    if ta.ndim == 1 and tb.ndim == 2:
        k_dim = ta.shape[0]
        if tb.shape[0] != k_dim:
            raise ValueError(f"Shape mismatch in matmul: {ta.shape} @ {tb.shape}")
        n_dim = tb.shape[1]
        a_flat = ta.flat_data
        b_flat = tb.flat_data
        out_vec = [
            _float_sum(a_flat[k] * b_flat[k * n_dim + c] for k in range(k_dim))
            for c in range(n_dim)
        ]
        return Tensor(out_vec, (n_dim,), dtype=ta.dtype)

    batch_a = ta.shape[:-2]
    m_dim, k_dim = ta.shape[-2], ta.shape[-1]
    batch_b = tb.shape[:-2]
    k_dim_b, n_dim = tb.shape[-2], tb.shape[-1]
    if k_dim != k_dim_b:
        raise ValueError(f"Contraction dimension mismatch in matmul: {ta.shape} vs {tb.shape}")

    batch_shape = _broadcast_shapes(batch_a, batch_b)
    ta_bcast = broadcast_to(ta, batch_shape + (m_dim, k_dim))
    tb_bcast = broadcast_to(tb, batch_shape + (k_dim, n_dim))
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
    return Tensor(out_flat, batch_shape + (m_dim, n_dim), dtype=ta.dtype)


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
        ellipsis_labels: Assigned uppercase label characters for broadcast ellipsis axes.

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


def einsum(equation: str, *operands: TensorOrScalar) -> Tensor:
    """Evaluate the Einstein summation convention on the given operands in pure Python.

    Supports arbitrary explicit and implicit subscript equations as well as `'...'`
    ellipsis batch dimensions.

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

    ell_labels = "".join(chr(ord("A") + i) for i in range(max_ell_rank))
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
        pairs = tuple((label_to_pos[ch], st[ax]) for ax, ch in enumerate(sp) if st[ax] != 0)
        op_pos_strides.append(pairs)

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

    return Tensor(out_flat, out_shape, dtype=tensors[0].dtype)


# ============================================================================
# Elementwise Mathematical Operations
# ============================================================================


def abs(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise absolute value.

    Args:
        x: Input tensor or scalar.

    Returns:
        Absolute value `Tensor`.
    """
    return _unary_elementwise(x, math.fabs)


def exp(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise exponential.

    Args:
        x: Input tensor or scalar.

    Returns:
        Exponential `Tensor`.
    """
    return _unary_elementwise(x, lambda v: math.exp(_float_min(88.0, _float_max(-88.0, v))))


def log(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise natural logarithm.

    Args:
        x: Input tensor or scalar.

    Returns:
        Natural logarithm `Tensor`.
    """
    return _unary_elementwise(x, lambda v: math.log(v) if v > 0.0 else -math.inf)


def log1p(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise natural logarithm of `1 + x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Elementwise `log(1 + x)` `Tensor`.
    """
    return _unary_elementwise(x, lambda v: math.log1p(v) if v > -1.0 else -math.inf)


def sqrt(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise non-negative square root.

    Args:
        x: Input tensor or scalar.

    Returns:
        Square root `Tensor`.
    """
    return _unary_elementwise(x, lambda v: math.sqrt(_float_max(0.0, v)))


def square(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise square `x * x`.

    Args:
        x: Input tensor or scalar.

    Returns:
        Squared `Tensor`.
    """
    return _unary_elementwise(x, lambda v: v * v)


def sin(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise trigonometric sine.

    Args:
        x: Input tensor in radians.

    Returns:
        Sine `Tensor`.
    """
    return _unary_elementwise(x, math.sin)


def cos(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise trigonometric cosine.

    Args:
        x: Input tensor in radians.

    Returns:
        Cosine `Tensor`.
    """
    return _unary_elementwise(x, math.cos)


def tanh(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise hyperbolic tangent.

    Args:
        x: Input tensor or scalar.

    Returns:
        Hyperbolic tangent `Tensor`.
    """
    return _unary_elementwise(x, math.tanh)


def erf(x: TensorOrScalar) -> Tensor:
    """Compute the elementwise Gauss error function.

    Args:
        x: Input tensor or scalar.

    Returns:
        Error function `Tensor`.
    """
    return _unary_elementwise(x, math.erf)


def floor(x: TensorOrScalar) -> Tensor:
    """Return the elementwise floor of the input.

    Args:
        x: Input tensor or scalar.

    Returns:
        Floored `Tensor`.
    """
    return _unary_elementwise(x, lambda v: float(math.floor(v)))


def maximum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Compute the elementwise maximum of two tensors.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.

    Returns:
        Elementwise maximum `Tensor`.
    """
    return _binary_elementwise(x, y, _float_max)


def minimum(x: TensorOrScalar, y: TensorOrScalar) -> Tensor:
    """Compute the elementwise minimum of two tensors.

    Args:
        x: First input tensor or scalar.
        y: Second input tensor or scalar.

    Returns:
        Elementwise minimum `Tensor`.
    """
    return _binary_elementwise(x, y, _float_min)


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


# ============================================================================
# Reduction & Scan Operations
# ============================================================================


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
) -> Tensor:
    """Apply `reducer` over the specified `axis` (or axes) of `x`.

    Args:
        x: Input tensor or scalar.
        axis: Axis or axes to reduce over (`None` reduces all axes).
        keepdims: Whether reduced axes are retained with size `1`.
        reducer: Function reducing a non-empty sequence of floats to a single float.

    Returns:
        Reduced `Tensor`.
    """
    t = asarray(x)
    red_axes = _resolve_reduction_axes(t.ndim, axis)
    if not red_axes:
        return t

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

    return Tensor(out_flat, out_shape, dtype=t.dtype)


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
    return _reduce_axes(x, axis, keepdims, _float_sum)


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
    return _reduce_axes(x, axis, keepdims, lambda vals: _float_sum(vals) / len(vals))


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

    return _reduce_axes(x, axis, keepdims, _var_fn)


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
    return _reduce_axes(x, axis, keepdims, _seq_max)


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
    return _reduce_axes(x, axis, keepdims, _seq_min)


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
        return Tensor((float(best_i),), out_shape, dtype=DType.int32)

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

    res = _reduce_axes(t, axis, keepdims, _argmin_fn)
    return Tensor(res.flat_data, res.shape, dtype=DType.int32)


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
        return Tensor((float(best_i),), out_shape, dtype=DType.int32)

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

    res = _reduce_axes(t, axis, keepdims, _argmax_fn)
    return Tensor(res.flat_data, res.shape, dtype=DType.int32)


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
    return Tensor(out_flat, t.shape, dtype=t.dtype)


__all__ = [
    "DType",
    "IndexAtom",
    "IndexSpec",
    "NestedSequence",
    "Scalar",
    "Tensor",
    "TensorOrScalar",
    "ToListProtocol",
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
