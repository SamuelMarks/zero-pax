"""Base layer and shared configuration definitions for zero_pax.praxis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

from pydantic import BaseModel, ConfigDict, Field

from zero_pax import _tensor_ops as np
from zero_pax.types import (
    ActivationSharding,
    BaseParameter,
    CallableLayer,
    DType,
    PaxConfig,
    Tensor,
    TensorOrScalar,
    WeightInit,
    WeightSharding,
)


class BasePraxisLayer(BaseModel):
    """Base class for all Praxis neural network layers.

    Attributes:
        dtype: Default parameter weight data type.
        fprop_dtype: Forward pass activation data type (defaults to `dtype` when None).
        params_init: Weight initialization configuration.
        skip_lp_regularization: Whether to skip L1/L2 regularization on parameters.
        ici_mesh_shape: Intra-chip interconnect device mesh shape.
        dcn_mesh_shape: Data-center network device mesh shape.
        contiguous_submeshes: Whether submeshes are contiguous.
        mesh_axis_names: Logical names for each device mesh axis.
        shared_weight_layer_id: Identifier for weight sharing across layers.
        weight_split_dims_mapping: Sharding specification for layer parameters.
        activation_split_dims_mapping: Sharding specification for layer activations.
        parent: Optional parent layer in the module hierarchy.
        name: Optional instance name of the layer.
        params: Dictionary of registered `BaseParameter` instances on this layer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dtype: DType = "float32"
    fprop_dtype: DType = None
    params_init: WeightInit | None = Field(default_factory=lambda: WeightInit(method="zeros"))
    skip_lp_regularization: bool | None = None
    ici_mesh_shape: Sequence[int] | None = None
    dcn_mesh_shape: Sequence[int] | None = None
    contiguous_submeshes: bool | None = None
    mesh_axis_names: Sequence[str] | None = None
    shared_weight_layer_id: str | None = None
    weight_split_dims_mapping: WeightSharding | None = Field(default_factory=WeightSharding)
    activation_split_dims_mapping: ActivationSharding | None = Field(
        default_factory=ActivationSharding
    )
    parent: BasePraxisLayer | None = None
    name: str | None = None
    params: dict[str, BaseParameter] = Field(default_factory=dict, exclude=True)

    def register_parameter(self, name: str, value: TensorOrScalar) -> None:
        """Register a named parameter in `self.params`.

        Args:
            name: Parameter identifier string.
            value: Initial parameter value wrapped in a `BaseParameter`.
        """
        self.params[name] = BaseParameter(value)

    def create_variable(
        self,
        name: str,
        shape: Sequence[int],
        init: WeightInit | None = None,
    ) -> Tensor:
        """Create and register a named tensor parameter initialized according to `init` or `params_init`.

        Args:
            name: Parameter identifier string.
            shape: Target tensor dimensions.
            init: Optional explicit `WeightInit` override.

        Returns:
            Initialized `Tensor` stored in `self.params[name]`.
        """
        initializer = init if init is not None else (self.params_init or WeightInit(method="zeros"))
        tensor = initializer.init_tensor(shape, dtype=self.dtype)
        self.register_parameter(name, tensor)
        return tensor

    def get_variable(
        self,
        name: str,
        shape: Sequence[int],
        explicit: TensorOrScalar | None = None,
        init: WeightInit | None = None,
    ) -> Tensor:
        """Resolve a weight tensor from an explicit override, registered parameter, or initializer.

        Args:
            name: Parameter identifier string.
            shape: Target tensor shape if initialization is required.
            explicit: Optional caller-supplied tensor override.
            init: Optional explicit `WeightInit` strategy.

        Returns:
            Resolved `Tensor` parameter.
        """
        if explicit is not None:
            return np.asarray(explicit)
        if name in self.params:
            return np.asarray(self.params[name]())
        return self.create_variable(name, shape, init=init)

    def setup(self) -> None:
        """Initialize sub-layers and stateful parameters prior to forward evaluation."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        /,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Execute the forward pass of the layer on `inputs`.

        Args:
            inputs: Input tensor or array.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` resulting from the forward computation.
        """
        return np.asarray(inputs)

    def fprop(
        self,
        inputs: TensorOrScalar,
        /,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Execute the forward pass (`fprop`) by delegating to `__call__`.

        Args:
            inputs: Input tensor or array.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` resulting from the forward computation.
        """
        return self(inputs, *args, **kwargs)


def as_tensor_if_present(val: TensorOrScalar | None) -> Tensor | None:
    """Convert `val` to a `Tensor` if it is not None.

    Args:
        val: Optional tensor or scalar value.

    Returns:
        Converted `Tensor` or `None`.
    """
    if val is None:
        return None
    return np.asarray(val)


LayerTpl = Union[
    PaxConfig[BasePraxisLayer],
    type[BasePraxisLayer],
    BasePraxisLayer,
    CallableLayer,
    None,
]


def instantiate_layer(tpl: LayerTpl, default: CallableLayer) -> CallableLayer:
    """Resolve a `LayerTpl` into a concrete `CallableLayer` instance.

    Handles `PaxConfig` templates, `BasePraxisLayer` subclasses, already-instantiated
    callable layers, or `None` (returning `default`).

    Args:
        tpl: Layer template (`PaxConfig`, class, callable instance, or `None`).
        default: Default callable layer used when `tpl` is `None`.

    Returns:
        Concrete `CallableLayer` ready for forward invocation.
    """
    if tpl is None:
        return default
    if isinstance(tpl, PaxConfig):
        inst = tpl.instantiate()
        return inst if isinstance(inst, CallableLayer) else default
    if isinstance(tpl, type):
        return tpl() if issubclass(tpl, BasePraxisLayer) else default
    return tpl if isinstance(tpl, CallableLayer) else default


__all__ = [
    "BasePraxisLayer",
    "LayerTpl",
    "as_tensor_if_present",
    "instantiate_layer",
]
