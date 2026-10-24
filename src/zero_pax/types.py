"""Explicit domain types, configuration containers, and protocols for zero_pax."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Generic, Literal, Protocol, TypeVar, Union, runtime_checkable

from ml_switcheroo_compiler import Tensor
from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from pydantic import BaseModel, ConfigDict, Field

from zero_pax import _tensor_ops as np

T_co = TypeVar("T_co", covariant=True)

DType = Union[str, CompilerDType, None]
AxisType = Union[int, tuple[int, ...], None]
Shape = Union[tuple[int, ...], Sequence[int]]
PaddingTuple = tuple[tuple[int, int], ...]
RNNState = Union[Tensor, tuple[Tensor, Tensor]]
QuantizerOutput = tuple[Tensor, Tensor, dict[str, Union[Tensor, float]]]
PoolingOutput = tuple[Tensor, Union[Tensor, None]]
SplitDimsMapping = Union[Sequence[Union[str, Sequence[str], None]], None]
ActivationType = Union[
    Literal["NONE", "RELU", "LEAKY_RELU", "SIGMOID", "TANH", "GELU", "SWISH", "SILU"],
    str,
]
ConfigParamValue = Union[
    TensorOrScalar,
    str,
    int,
    float,
    bool,
    Sequence[int],
    Sequence[str],
    None,
]


@runtime_checkable
class CallableLayer(Protocol):
    """Protocol representing any callable layer accepting tensor inputs."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        /,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the layer transformation to `inputs`.

        Args:
            inputs: Input tensor or array.
            *args: Additional positional tensor or scalar arguments.
            **kwargs: Additional keyword tensor or scalar arguments.

        Returns:
            Transformed output `Tensor`.
        """


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
        self.value: TensorOrScalar = value

    def __call__(self) -> TensorOrScalar:
        """Return the underlying parameter value.

        Returns:
            Stored parameter value.
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

        Args:
            shape: Target tensor dimensions.
            dtype: Optional compiler data type for the initialized tensor.

        Returns:
            Initialized `Tensor` with shape `tuple(shape)`.
        """
        dims = tuple(int(d) for d in shape)
        compiler_dtype: CompilerDType | None = dtype if isinstance(dtype, CompilerDType) else None
        method_lower = self.method.lower()
        if method_lower == "zeros":
            return np.zeros(dims, dtype=compiler_dtype)
        if method_lower == "ones":
            return np.ones(dims, dtype=compiler_dtype) * self.scale
        if method_lower == "constant":
            return np.full(dims, self.scale, dtype=compiler_dtype)
        if self.scale == 0.0:
            return np.zeros(dims, dtype=compiler_dtype)

        total_elems = 1
        for d in dims:
            total_elems *= max(1, d)

        fan_in = float(dims[0]) if len(dims) >= 1 else 1.0
        fan_out = float(dims[-1]) if len(dims) >= 2 else fan_in

        seed_offset = float((self.seed if self.seed is not None else 0) % 997)
        indices = np.arange(1, total_elems + 1, dtype=compiler_dtype) + seed_offset
        golden_phase = indices * 0.6180339887498949
        wave = np.sin(indices * 1.618033988749895 + np.cos(golden_phase))

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
            flat = wave * limit
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
            flat = wave * std
        else:
            limit = self.scale * math.sqrt(6.0 / max(1.0, fan_in + fan_out))
            flat = wave * limit

        return np.reshape(flat, dims)


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


class PaxConfig(BaseModel, Generic[T_co]):
    """Generic configuration container emulating `pax_fiddle.Config`.

    Attributes:
        cls: Optional target class or callable factory for instantiation.
        params: Optional dictionary of keyword arguments passed to `cls` on instantiation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cls: type[T_co] | None = None
    params: Mapping[str, ConfigParamValue] = Field(default_factory=dict)

    def clone(self) -> PaxConfig[T_co]:
        """Return a shallow copy of this `PaxConfig` template.

        Returns:
            Cloned `PaxConfig` instance with copied `params`.
        """
        return PaxConfig(cls=self.cls, params=dict(self.params))

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
