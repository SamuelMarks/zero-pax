"""Core pure-Python Tensor engine, dtypes, indexing, and ml_switcheroo_ir graph tracing."""

from __future__ import annotations

import array as _py_array
import builtins
import inspect
import itertools
import math
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import Protocol, TypeVar, Union, runtime_checkable

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode, TensorSpec
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
_ACTIVE_GRAPH: LogicalGraph = LogicalGraph(name="ZeroPaxGraph")
_NODE_COUNTER: int = 0
pi: float = math.pi
inf: float = math.inf
nan: float = math.nan

_INTEGER_DTYPES: frozenset[DType] = frozenset(
    {
        DType.int64,
        DType.int32,
        DType.int16,
        DType.int8,
        DType.uint64,
        DType.uint32,
        DType.uint16,
        DType.uint8,
        DType.int4,
        DType.uint4,
        DType.int2,
        DType.qint8,
        DType.quint8,
        DType.qint4,
    }
)

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

    When `enabled` is `False`, tensor operations also record `LogicalNode` entries
    into the active `ml_switcheroo_ir.LogicalGraph` trace.

    Args:
        enabled: Whether eager execution should be active.

    Returns:
        Updated boolean eager mode state.
    """
    global _EAGER_MODE
    _EAGER_MODE = enabled
    return _EAGER_MODE


def get_eager_mode() -> bool:
    """Return whether eager execution mode is currently active.

    Returns:
        `True` if eager mode is active, `False` if graph tracing is active.
    """
    return _EAGER_MODE


def get_active_graph() -> LogicalGraph:
    """Return the active `ml_switcheroo_ir.LogicalGraph` recorded during non-eager execution.

    Returns:
        Active `LogicalGraph` instance.
    """
    return _ACTIVE_GRAPH


def reset_active_graph(name: str = "ZeroPaxGraph") -> LogicalGraph:
    """Reset the active `ml_switcheroo_ir.LogicalGraph` trace and node counter.

    Args:
        name: Name for the newly initialized `LogicalGraph`.

    Returns:
        Freshly initialized `LogicalGraph`.
    """
    global _ACTIVE_GRAPH, _NODE_COUNTER
    _ACTIVE_GRAPH = LogicalGraph(name=name)
    _NODE_COUNTER = 0
    return _ACTIVE_GRAPH


def _capture_source_ast_ref() -> str:
    """Capture the caller file path and line number outside `_tensor_*` modules for IR provenance.

    Returns:
        Formatted `'filepath:lineno'` source AST reference string.
    """
    frames = inspect.stack()
    selected = next(
        (
            f
            for f in frames
            if "_tensor_core.py" not in f.filename
            and "_tensor_ops.py" not in f.filename
            and "_tensor_linalg.py" not in f.filename
            and "_tensor_reductions.py" not in f.filename
        ),
        frames[0],
    )
    return f"{selected.filename}:{selected.lineno}"


def register_graph_input(
    tensor: Tensor,
    name: str | None = None,
    metadata: Mapping[str, AttributeValue] | None = None,
) -> Tensor:
    """Register `tensor` as a `GraphInput` node in the active `LogicalGraph` when tracing is active.

    Args:
        tensor: Parameter or input `Tensor` to register.
        name: Optional symbolic name for the parameter/input node.
        metadata: Optional sharding or layer metadata attributes to attach to the node.

    Returns:
        `tensor` with its `node_id` bound to the registered `LogicalNode` when non-eager.
    """
    global _NODE_COUNTER
    if _EAGER_MODE:
        return tensor
    if tensor.node_id is not None and tensor.node_id in _ACTIVE_GRAPH.nodes:
        if metadata:
            existing_node = _ACTIVE_GRAPH.nodes[tensor.node_id]
            for k, v in metadata.items():
                existing_node.attributes[k] = v
        return tensor
    _NODE_COUNTER += 1
    clean_name = name.replace("/", "_") if name else "param"
    in_nid = f"input_{clean_name}_{_NODE_COUNTER}"
    in_spec = TensorSpec(shape=tensor.shape, dtype=tensor.dtype)
    _ACTIVE_GRAPH.inputs.append(in_nid)
    _ACTIVE_GRAPH.input_specs[in_nid] = in_spec
    in_node = LogicalNode(
        id=in_nid,
        op_type="Input",
        attributes=metadata,
        outputs=[in_nid],
        shape_metadata=tensor.shape,
        source_ast_ref=_capture_source_ast_ref(),
        dtype=tensor.dtype,
        output_specs=[in_spec],
    )
    _ACTIVE_GRAPH.add_node(in_node)
    tensor._node_id = in_nid
    return tensor


def _coerce_flat_for_dtype(flat: Sequence[float], dtype: DType) -> tuple[float, ...]:
    """Coerce numeric elements according to target `dtype` (truncating integers and booleans).

    Args:
        flat: Sequence of numeric values.
        dtype: Target `DType`.

    Returns:
        Tuple of coerced float values.
    """
    if dtype == DType.bool:
        return tuple(1.0 if float(v) != 0.0 else 0.0 for v in flat)
    if dtype in _INTEGER_DTYPES:
        return tuple(float(int(float(v))) if math.isfinite(float(v)) else 0.0 for v in flat)
    return tuple(float(v) for v in flat)


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


def _unflatten_nested(
    flat: tuple[float, ...],
    shape: tuple[int, ...],
    dtype: DType = DType.float32,
) -> NestedSequence:
    """Reconstruct nested Python lists (or a scalar for 0D) from `flat`, `shape`, and `dtype`.

    Args:
        flat: Flat tuple of float elements.
        shape: Shape tuple.
        dtype: Data type controlling scalar representation (`bool`, `int`, or `float`).

    Returns:
        Scalar if `shape == ()`, otherwise nested list of elements matching `dtype`.
    """
    if not shape:
        raw_val = flat[0] if flat else 0.0
        if dtype == DType.bool:
            return bool(raw_val != 0.0)
        if dtype in _INTEGER_DTYPES:
            return int(raw_val)
        return raw_val
    if len(shape) == 1:
        if dtype == DType.bool:
            return [bool(v != 0.0) for v in flat[: shape[0]]]
        if dtype in _INTEGER_DTYPES:
            return [int(v) for v in flat[: shape[0]]]
        return list(flat[: shape[0]])
    step = _shape_size(shape[1:])
    return [
        _unflatten_nested(flat[i * step : (i + 1) * step], shape[1:], dtype=dtype)
        for i in range(shape[0])
    ]


class Tensor(_TensorBase):
    """Pure-Python multi-dimensional tensor backed by flat float storage and `ml_switcheroo_ir.DType`.

    Attributes:
        _flat: Immutable tuple of numeric element values in row-major order.
        _shape: Tuple of non-negative integer dimensions.
        _dtype: `ml_switcheroo_ir.DType` descriptor.
        _buf: Contiguous C-float `array.array` buffer providing zero-copy `__array_interface__` in tests.
        _node_id: Optional `ml_switcheroo_ir.LogicalNode` identifier when traced in non-eager mode.
    """

    def __init__(
        self,
        flat: Sequence[float],
        shape: tuple[int, ...],
        dtype: DType | None = None,
        node_id: str | None = None,
    ) -> None:
        """Initialize a `Tensor` from flat row-major data, `shape`, and optional `dtype`.

        Args:
            flat: Sequence of numeric values of length `_shape_size(shape)`.
            shape: Tuple of integer dimension sizes.
            dtype: Optional `ml_switcheroo_ir.DType` (defaults to `DType.float32`).
            node_id: Optional `LogicalNode` identifier in the active `LogicalGraph`.
        """
        self._shape: tuple[int, ...] = tuple(int(d) for d in shape)
        self._dtype: DType = dtype if dtype is not None else DType.float32
        raw_floats = _coerce_flat_for_dtype(flat, self._dtype)
        self._buf: _py_array.array[float] = (
            _py_array.array("f", raw_floats) if raw_floats else _py_array.array("f", [0.0])
        )
        self._flat: tuple[float, ...] = tuple(self._buf) if raw_floats else ()
        self._node_id: str | None = node_id

    @property
    def node_id(self) -> str | None:
        """Return the `LogicalNode` identifier associated with this tensor in non-eager mode.

        Returns:
            Node identifier string or `None`.
        """
        return self._node_id

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
        """Convert the tensor to nested Python lists (or a scalar if 0D).

        Returns:
            Nested list of values or scalar matching `self.dtype`.
        """
        return _unflatten_nested(self._flat, self._shape, dtype=self._dtype)

    def astype(self, dtype: DType | str | None) -> Tensor:
        """Return a copy of the tensor cast to `dtype`.

        Args:
            dtype: Target `DType` or string dtype name.

        Returns:
            New `Tensor` with resolved `dtype` and coerced elements.
        """
        target_dtype = resolve_dtype(dtype) or self._dtype
        out = Tensor(self._flat, self._shape, dtype=target_dtype)
        return _record_ir_op("Cast", (self,), out, attributes={"to": target_dtype.value})

    def copy(self) -> Tensor:
        """Return a deep copy of the tensor.

        Returns:
            New `Tensor` with identical flat data, shape, and dtype.
        """
        return Tensor(self._flat, self._shape, dtype=self._dtype)

    @property
    def T(self) -> Tensor:
        """Return the reversed-axis transpose of the tensor.

        Returns:
            Transposed `Tensor`.
        """
        return transpose(self)

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
            out_gather = Tensor(gathered, idx_tensor.shape + sub_shape, dtype=self._dtype)
            return _record_ir_op(
                "Gather",
                (self, idx_tensor),
                out_gather,
                attributes={"axis": 0},
            )

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
            out_empty = Tensor(self._flat, tuple(out_shape), dtype=self._dtype)
            return _record_ir_op(
                "Slice",
                (self,),
                out_empty,
                attributes={"index": repr(key)},
            )

        result_flat: list[float] = []
        for coord in itertools.product(*axis_ranges):
            flat_idx = 0
            for c_val, s_ax in zip(coord, src_axis_for_range):
                flat_idx += c_val * src_strides[s_ax]
            result_flat.append(self._flat[flat_idx])

        out_slice = Tensor(result_flat, tuple(out_shape), dtype=self._dtype)
        return _record_ir_op(
            "Slice",
            (self,),
            out_slice,
            attributes={"index": repr(key)},
        )

    def __neg__(self) -> Tensor:
        """Compute elementwise negation `-self`.

        Returns:
            Negated `Tensor`.
        """
        return _unary_elementwise(self, lambda a: -a, op_type="Neg")

    def __pos__(self) -> Tensor:
        """Compute elementwise unary positive `+self`.

        Returns:
            Copy of `self`.
        """
        return Tensor(self._flat, self._shape, dtype=self._dtype, node_id=self._node_id)

    def __abs__(self) -> Tensor:
        """Compute elementwise absolute value `abs(self)`.

        Returns:
            Absolute value `Tensor`.
        """
        return _unary_elementwise(self, math.fabs, op_type="Abs")

    def __add__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise addition `self + other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Sum `Tensor`.
        """
        return _binary_elementwise(self, other, lambda a, b: a + b, op_type="Add")

    def __radd__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise addition `other + self`.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Sum `Tensor`.
        """
        return _binary_elementwise(other, self, lambda a, b: a + b, op_type="Add")

    def __sub__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise subtraction `self - other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Difference `Tensor`.
        """
        return _binary_elementwise(self, other, lambda a, b: a - b, op_type="Sub")

    def __rsub__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise subtraction `other - self`.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Difference `Tensor`.
        """
        return _binary_elementwise(other, self, lambda a, b: a - b, op_type="Sub")

    def __mul__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise multiplication `self * other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Product `Tensor`.
        """
        return _binary_elementwise(self, other, lambda a, b: a * b, op_type="Mul")

    def __rmul__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise multiplication `other * self`.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            Product `Tensor`.
        """
        return _binary_elementwise(other, self, lambda a, b: a * b, op_type="Mul")

    def __truediv__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise true division `self / other`.

        Args:
            other: Right-hand tensor or scalar divisor.

        Returns:
            Quotient `Tensor`.
        """
        return _binary_elementwise(self, other, _safe_div, op_type="Div")

    def __rtruediv__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise true division `other / self`.

        Args:
            other: Left-hand tensor or scalar dividend.

        Returns:
            Quotient `Tensor`.
        """
        return _binary_elementwise(other, self, _safe_div, op_type="Div")

    def __floordiv__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise floor division `self // other`.

        Args:
            other: Right-hand tensor or scalar divisor.

        Returns:
            Floor quotient `Tensor`.
        """
        return _binary_elementwise(
            self, other, lambda a, b: math.floor(_safe_div(a, b)), op_type="FloorDiv"
        )

    def __rfloordiv__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise floor division `other // self`.

        Args:
            other: Left-hand tensor or scalar dividend.

        Returns:
            Floor quotient `Tensor`.
        """
        return _binary_elementwise(
            other, self, lambda a, b: math.floor(_safe_div(a, b)), op_type="FloorDiv"
        )

    def __pow__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise exponentiation `self ** other`.

        Args:
            other: Right-hand exponent tensor or scalar.

        Returns:
            Exponentiated `Tensor`.
        """
        return _binary_elementwise(self, other, _safe_pow, op_type="Pow")

    def __rpow__(self, other: TensorOrScalar) -> Tensor:
        """Compute reflected elementwise exponentiation `other ** self`.

        Args:
            other: Left-hand base tensor or scalar.

        Returns:
            Exponentiated `Tensor`.
        """
        return _binary_elementwise(other, self, _safe_pow, op_type="Pow")

    def __matmul__(self, other: TensorOrScalar) -> Tensor:
        """Compute matrix multiplication `self @ other`.

        Args:
            other: Right-hand tensor operand.

        Returns:
            Matrix product `Tensor`.
        """
        from zero_pax._tensor_linalg import matmul

        return matmul(self, other)

    def __lt__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise less-than comparison `self < other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self,
            other,
            lambda a, b: 1.0 if a < b else 0.0,
            dtype=DType.bool,
            op_type="Less",
        )

    def __le__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise less-than-or-equal comparison `self <= other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self,
            other,
            lambda a, b: 1.0 if a <= b else 0.0,
            dtype=DType.bool,
            op_type="LessOrEqual",
        )

    def __gt__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise greater-than comparison `self > other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self,
            other,
            lambda a, b: 1.0 if a > b else 0.0,
            dtype=DType.bool,
            op_type="Greater",
        )

    def __ge__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise greater-than-or-equal comparison `self >= other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self,
            other,
            lambda a, b: 1.0 if a >= b else 0.0,
            dtype=DType.bool,
            op_type="GreaterOrEqual",
        )

    def __eq__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise equality comparison `self == other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self,
            other,
            lambda a, b: 1.0 if a == b else 0.0,
            dtype=DType.bool,
            op_type="Equal",
        )

    def __ne__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise inequality comparison `self != other`.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            Boolean mask `Tensor` (`1.0` where True, `0.0` where False).
        """
        return _binary_elementwise(
            self,
            other,
            lambda a, b: 1.0 if a != b else 0.0,
            dtype=DType.bool,
            op_type="NotEqual",
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
            op_type="Or",
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
            op_type="And",
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


def _record_ir_op(
    op_type: str,
    inputs: Sequence[Tensor],
    output: Tensor,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> Tensor:
    """Record a `LogicalNode` in `_ACTIVE_GRAPH` when `_EAGER_MODE` is `False`.

    Args:
        op_type: Operator type name (e.g. `'Add'`, `'MatMul'`, `'Einsum'`).
        inputs: Sequence of input `Tensor` operands.
        output: Computed output `Tensor`.
        attributes: Optional operator attributes dictionary.

    Returns:
        `output` `Tensor` with `_node_id` attached when graph tracing is active.
    """
    global _NODE_COUNTER
    if _EAGER_MODE:
        return output
    ast_ref = _capture_source_ast_ref()
    in_ids: list[str] = []
    for inp in inputs:
        if inp.node_id is not None:
            in_ids.append(inp.node_id)
        else:
            _NODE_COUNTER += 1
            in_nid = f"input_{_NODE_COUNTER}"
            in_spec = TensorSpec(shape=inp.shape, dtype=inp.dtype)
            _ACTIVE_GRAPH.inputs.append(in_nid)
            _ACTIVE_GRAPH.input_specs[in_nid] = in_spec
            in_node = LogicalNode(
                id=in_nid,
                op_type="Input",
                outputs=[in_nid],
                shape_metadata=inp.shape,
                source_ast_ref=ast_ref,
                dtype=inp.dtype,
                output_specs=[in_spec],
            )
            _ACTIVE_GRAPH.add_node(in_node)
            inp._node_id = in_nid
            in_ids.append(in_nid)
    _NODE_COUNTER += 1
    nid = f"{op_type.lower()}_{_NODE_COUNTER}"
    out_spec = TensorSpec(shape=output.shape, dtype=output.dtype)
    node = LogicalNode(
        id=nid,
        op_type=op_type,
        attributes=attributes,
        inputs=in_ids,
        outputs=[nid],
        shape_metadata=output.shape,
        source_ast_ref=ast_ref,
        dtype=output.dtype,
        output_specs=[out_spec],
    )
    _ACTIVE_GRAPH.add_node(node)
    _ACTIVE_GRAPH.outputs = [nid]
    output._node_id = nid
    return output


def _unary_elementwise(
    x: TensorOrScalar,
    fn: Callable[[float], float],
    dtype: DType | None = None,
    op_type: str = "UnaryOp",
) -> Tensor:
    """Apply unary scalar function `fn` elementwise to `x`.

    Args:
        x: Input tensor or scalar.
        fn: Unary float-to-float transformation.
        dtype: Optional output `DType`.
        op_type: IR operation name recorded when non-eager tracing is active.

    Returns:
        Transformed `Tensor`.
    """
    t = asarray(x)
    out = Tensor([fn(v) for v in t.flat_data], t.shape, dtype=dtype or t.dtype)
    return _record_ir_op(op_type, (t,), out)


def _promote_dtypes(dtype_a: DType, dtype_b: DType) -> DType:
    """Promote two `DType` values under standard numeric promotion rules.

    Args:
        dtype_a: Left operand `DType`.
        dtype_b: Right operand `DType`.

    Returns:
        Promoted `DType`.
    """
    if dtype_a == dtype_b:
        return dtype_a
    if dtype_a == DType.float64 or dtype_b == DType.float64:
        return DType.float64
    if dtype_a in (DType.float32, DType.float16, DType.bfloat16):
        return dtype_a
    if dtype_b in (DType.float32, DType.float16, DType.bfloat16):
        return dtype_b
    if dtype_a == DType.bool:
        return dtype_b
    if dtype_b == DType.bool:
        return dtype_a
    return DType.float32


def _binary_elementwise(
    a: TensorOrScalar,
    b: TensorOrScalar,
    fn: Callable[[float, float], float],
    dtype: DType | None = None,
    op_type: str = "BinaryOp",
) -> Tensor:
    """Apply binary scalar function `fn` elementwise with broadcasting between `a` and `b`.

    Args:
        a: Left tensor or scalar operand.
        b: Right tensor or scalar operand.
        fn: Binary float-to-float function.
        dtype: Optional output `DType`.
        op_type: IR operation name recorded when non-eager tracing is active.

    Returns:
        Broadcasted result `Tensor`.
    """
    ta = asarray(a)
    tb = asarray(b)
    target_shape = _broadcast_shapes(ta.shape, tb.shape)
    flat_a = _broadcast_flat(ta, target_shape)
    flat_b = _broadcast_flat(tb, target_shape)
    out_dtype = dtype if dtype is not None else _promote_dtypes(ta.dtype, tb.dtype)
    out = Tensor(
        [fn(va, vb) for va, vb in zip(flat_a, flat_b)],
        target_shape,
        dtype=out_dtype,
    )
    return _record_ir_op(op_type, (ta, tb), out)


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
        return x.astype(resolved)
    flat, shape = _flatten_nested(x)
    return Tensor(flat, shape, dtype=resolved)


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
    out = Tensor(t.flat_data, target_shape, dtype=t.dtype)
    return _record_ir_op("Reshape", (t,), out, attributes={"shape": list(target_shape)})


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
    perm = (
        tuple(range(ndim - 1, -1, -1))
        if axes is None
        else tuple(ax + ndim if ax < 0 else ax for ax in axes)
    )
    if ndim <= 1 and axes is None:
        out_1d = Tensor(t.flat_data, t.shape, dtype=t.dtype)
        return _record_ir_op("Transpose", (t,), out_1d, attributes={"perm": list(perm)})
    out_shape = tuple(t.shape[p] for p in perm)
    src_strides = _compute_strides(t.shape)
    src_flat = t.flat_data
    out_flat: list[float] = []
    for out_coords in itertools.product(*(range(d) for d in out_shape)):
        flat_idx = 0
        for out_ax, p_ax in enumerate(perm):
            flat_idx += out_coords[out_ax] * src_strides[p_ax]
        out_flat.append(src_flat[flat_idx])
    out = Tensor(out_flat, out_shape, dtype=t.dtype)
    return _record_ir_op("Transpose", (t,), out, attributes={"perm": list(perm)})
