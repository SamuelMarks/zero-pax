"""Explicit domain types, configuration containers, and protocols for zero_pax."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Generic, Literal, Optional, Protocol, TypeVar, Union, runtime_checkable

from ml_switcheroo_ir.types import DType as CompilerDType
from pydantic import BaseModel, ConfigDict, Field

from zero_pax import _tensor_ops as _ops
from zero_pax._tensor_ops import Tensor, TensorOrScalar

T_co = TypeVar("T_co", covariant=True)

DType = Union[str, CompilerDType, None]
AxisType = Union[int, tuple[int, ...], None]
Shape = Union[tuple[int, ...], Sequence[int]]
PaddingTuple = tuple[tuple[int, int], ...]
RNNState = Union[Tensor, tuple[Tensor, Tensor]]
QuantizerOutput = tuple[Tensor, Tensor, dict[str, Union[Tensor, float]]]
PoolingOutput = tuple[Tensor, Optional[Tensor]]
LayerOutput = Union[Tensor, PoolingOutput, RNNState, QuantizerOutput]
SplitDimsMapping = Optional[Sequence[Union[str, Sequence[str], None]]]
ActivationType = Union[
    Literal["NONE", "RELU", "LEAKY_RELU", "SIGMOID", "TANH", "GELU", "SWISH", "SILU"], str
]
ConfigParamValue = Union[
    TensorOrScalar,
    BaseModel,
    str,
    int,
    float,
    bool,
    Sequence[int],
    Sequence[str],
    None,
]


@runtime_checkable
class CallableLayerProtocol(Protocol):
    """Protocol representing any callable layer accepting tensor inputs."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        /,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> TensorOrScalar:
        """Apply the layer transformation to `inputs`.

        Args:
            inputs: Input tensor or array.
            *args: Additional positional tensor or scalar arguments.
            **kwargs: Additional keyword tensor or scalar arguments.

        Returns:
            Transformed output `Tensor` or tensor-like value.
        """


@runtime_checkable
class StrictCallableLayerProtocol(Protocol):
    """Protocol representing a resolved layer returning a `Tensor`."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        /,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the layer transformation to `inputs` and return a `Tensor`.

        Args:
            inputs: Input tensor or array.
            *args: Additional positional tensor or scalar arguments.
            **kwargs: Additional keyword tensor or scalar arguments.

        Returns:
            Transformed output `Tensor`.
        """


CallableLayer = Union[
    CallableLayerProtocol,
    Callable[[TensorOrScalar], TensorOrScalar],
    Callable[[TensorOrScalar, TensorOrScalar], TensorOrScalar],
]


class BaseParameter:
    """Container holding a trainable or non-trainable parameter value.

    Attributes:
        value: Numeric or tensor parameter value.
    """

    def __init__(self, value: TensorOrScalar = 0.0) -> None:
        """Initialize the `BaseParameter` with `value`.

        Args:
            value: Initial parameter value.
        """
        self.value: Tensor = _ops.asarray(value)

    def __call__(self) -> Tensor:
        """Return the underlying parameter value.

        Returns:
            Stored parameter `Tensor`.
        """
        return self.value


class WeightInit(BaseModel):
    """Configuration model for parameter weight initialization strategies.

    Attributes:
        method: Initialization distribution or scheme ('xavier', 'constant', 'zeros', 'ones',
            'uniform', 'normal', 'gaussian', 'glorot_uniform', 'glorot_normal',
            'kaiming_uniform', 'kaiming_normal').
        scale: Scaling factor applied to the initialized weights.
        seed: Optional deterministic seed offset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: str = "xavier"
    scale: float = 1.0
    seed: int | None = None

    def init_tensor(self, shape: Sequence[int], dtype: DType = None) -> Tensor:
        """Create a deterministic tensor of `shape` according to this initialization strategy.

        Uses low-discrepancy Weyl sequences on `(0, 1)` for uniform distributions and the
        Box-Muller transform for Gaussian distributions so that mean and variance match the
        theoretical initialization distribution.

        Args:
            shape: Target tensor dimensions.
            dtype: Optional compiler data type or string name for the initialized tensor.

        Returns:
            Initialized `Tensor` with shape `tuple(shape)`.
        """
        dims = tuple(int(d) for d in shape)
        compiler_dtype = _ops.resolve_dtype(dtype)
        method_lower = self.method.lower()
        if method_lower == "zeros":
            return _ops.zeros(dims, dtype=compiler_dtype)
        if method_lower == "ones":
            return _ops.ones(dims, dtype=compiler_dtype) * self.scale
        if method_lower == "constant":
            return _ops.full(dims, self.scale, dtype=compiler_dtype)
        if self.scale == 0.0:
            return _ops.zeros(dims, dtype=compiler_dtype)

        total_elems = 1
        for d in dims:
            total_elems *= max(1, d)

        fan_in = float(dims[0]) if len(dims) >= 1 else 1.0
        fan_out = float(dims[-1]) if len(dims) >= 2 else fan_in

        seed_offset = float((self.seed if self.seed is not None else 0) % 997)
        indices = _ops.arange(1, total_elems + 1, dtype=compiler_dtype) + seed_offset
        u1_raw = indices * 0.6180339887498949 + 0.123456789
        u1 = u1_raw - _ops.floor(u1_raw)
        u2_raw = indices * 0.7548776662466927 + 0.567891234
        u2 = u2_raw - _ops.floor(u2_raw)

        if method_lower == "orthogonal":
            u1_safe = _ops.clip(u1, 1e-7, 1.0 - 1e-7)
            z_norm = _ops.sqrt(-2.0 * _ops.log(u1_safe)) * _ops.cos(2.0 * math.pi * u2)
            rows = max(1, dims[0]) if len(dims) >= 1 else 1
            cols = max(1, total_elems // rows)
            raw_vals = [float(z_norm[i]) for i in range(total_elems)]
            transpose_mat = rows < cols
            m_rows = cols if transpose_mat else rows
            m_cols = rows if transpose_mat else cols
            basis: list[list[float]] = []
            for c_idx in range(m_cols):
                vec = [raw_vals[c_idx * m_rows + r_idx] for r_idx in range(m_rows)]
                for prev in basis:
                    dot_val = sum(v * p for v, p in zip(vec, prev))
                    vec = [v - dot_val * p for v, p in zip(vec, prev)]
                norm_val = max(math.sqrt(sum(v * v for v in vec)), 1e-12)
                vec = [v / norm_val for v in vec]
                basis.append(vec)
            ortho_flat: list[float] = [
                (basis[r_idx][c_idx] if transpose_mat else basis[c_idx][r_idx]) * self.scale
                for r_idx in range(rows)
                for c_idx in range(cols)
            ]
            return _ops.reshape(_ops.asarray(ortho_flat, dtype=compiler_dtype), dims)

        if method_lower in ("uniform", "glorot_uniform", "xavier_uniform", "kaiming_uniform"):
            denom = (
                (fan_in + fan_out)
                if "glorot" in method_lower or "xavier" in method_lower
                else (2.0 * fan_in)
            )
            limit = (
                self.scale * math.sqrt(6.0 / max(1.0, denom))
                if method_lower != "uniform"
                else self.scale
            )
            flat = (2.0 * u1 - 1.0) * limit
        elif method_lower in ("normal", "gaussian", "glorot_normal", "kaiming_normal"):
            denom = (
                (fan_in + fan_out)
                if "glorot" in method_lower
                else (fan_in if "kaiming" in method_lower else 2.0)
            )
            std = (
                self.scale * math.sqrt(2.0 / max(1.0, denom))
                if method_lower not in ("normal", "gaussian")
                else self.scale
            )
            u1_safe = _ops.clip(u1, 1e-7, 1.0 - 1e-7)
            z_norm = _ops.sqrt(-2.0 * _ops.log(u1_safe)) * _ops.cos(2.0 * math.pi * u2)
            flat = z_norm * std
        else:
            limit = self.scale * math.sqrt(6.0 / max(1.0, fan_in + fan_out))
            flat = (2.0 * u1 - 1.0) * limit

        return _ops.reshape(flat, dims)


class WeightSharding(BaseModel):
    """Configuration model for SPMD weight sharding annotations.

    Attributes:
        wt: Sharding axis specification for layer weights.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wt: SplitDimsMapping = None


class ActivationSharding(BaseModel):
    """Configuration model for SPMD activation sharding annotations.

    Attributes:
        out: Sharding axis specification for layer output activations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    out: SplitDimsMapping = None


class DecoderHParams(BaseModel):
    """Configuration model for sequence and language model decoding hyperparameters.

    Attributes:
        seqlen: Maximum decoding sequence length.
        min_prefix_len: Minimum prefix length before generation.
        eos_id: End-of-sequence token ID.
        max_decode_steps: Optional maximum number of decode steps.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seqlen: int = 0
    min_prefix_len: int = 0
    eos_id: int = 0
    max_decode_steps: int | None = None


def _deep_clone_param(val: ConfigParamValue) -> ConfigParamValue:
    """Recursively deep-clone a `ConfigParamValue` container or model.

    Args:
        val: Configuration parameter value to clone.

    Returns:
        Deep-cloned configuration parameter value.
    """
    if isinstance(val, PaxConfig):
        return val.clone()
    if isinstance(val, Tensor):
        return val.copy()
    if isinstance(val, BaseModel):
        return val.model_copy(deep=True)
    if isinstance(val, list):
        return list(val)
    if isinstance(val, tuple):
        return tuple(val)
    if isinstance(val, dict):
        return dict(val)
    return val


class PaxConfig(BaseModel, Generic[T_co]):
    """Generic configuration container emulating `pax_fiddle.Config`.

    Attributes:
        cls: Optional target class or callable factory for instantiation.
        params: Optional dictionary of keyword arguments passed to `cls` on instantiation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cls: type[T_co] | None = None
    params: dict[str, ConfigParamValue] = Field(default_factory=dict)

    def set(self, **kwargs: ConfigParamValue) -> PaxConfig[T_co]:
        """Update configuration parameters in-place and return `self`.

        Args:
            **kwargs: Parameter key-value pairs to store in `self.params`.

        Returns:
            Updated `PaxConfig` instance (`self`).
        """
        self.params.update(kwargs)
        return self

    def clone(self) -> PaxConfig[T_co]:
        """Return an isolated deep copy of this `PaxConfig` template.

        Returns:
            Cloned `PaxConfig` instance with recursively copied `params`.
        """
        cloned_params: dict[str, ConfigParamValue] = {
            k: _deep_clone_param(v) for k, v in self.params.items()
        }
        return PaxConfig(cls=self.cls, params=cloned_params)

    def instantiate(self) -> T_co:
        """Instantiate the configured `cls` with `self.params`.

        Returns:
            Constructed instance of `cls`.

        Raises:
            ValueError: If `self.cls` is `None`.
        """
        if self.cls is None:
            raise ValueError("Cannot instantiate PaxConfig when cls is None.")
        return self.cls(**self.params)


__all__ = [
    "ActivationSharding",
    "ActivationType",
    "AxisType",
    "BaseParameter",
    "CallableLayer",
    "ConfigParamValue",
    "DType",
    "DecoderHParams",
    "LayerOutput",
    "PaddingTuple",
    "PaxConfig",
    "PoolingOutput",
    "QuantizerOutput",
    "RNNState",
    "Shape",
    "SplitDimsMapping",
    "Tensor",
    "TensorOrScalar",
    "WeightInit",
    "WeightSharding",
]
