"""Base layer and shared configuration definitions for zero_pax.praxis."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Literal, Protocol, Union, runtime_checkable

from ml_switcheroo_ir import AttributeValue
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from zero_pax import _tensor_ops as _ops
from zero_pax.types import (
    ActivationSharding,
    BaseParameter,
    CallableLayer,
    CallableLayerProtocol,
    DType,
    LayerOutput,
    PaxConfig,
    StrictCallableLayerProtocol,
    Tensor,
    TensorOrScalar,
    WeightInit,
    WeightSharding,
)

_SHARED_WEIGHT_REGISTRY: dict[tuple[str, str, tuple[int, ...]], Tensor] = {}


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
        activation_split_dims_mapping: Activation sharding specification.
        parent: Optional parent layer in the module hierarchy.
        name: Optional instance name of the layer.
        params: Dictionary of registered `BaseParameter` instances on this layer.
        sublayers: Dictionary of instantiated child `CallableLayer` instances keyed by name.
        _is_setup: Private boolean tracking whether `setup()` has been invoked on this instance.
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
    sublayers: dict[str, CallableLayer] = Field(default_factory=dict, exclude=True)
    _is_setup: bool = PrivateAttr(default=False)

    def _ensure_setup(self) -> None:
        """Invoke `self.setup()` once prior to parameter or sub-layer resolution."""
        if not self._is_setup:
            self._is_setup = True
            self.setup()

    def _build_ir_metadata(self) -> dict[str, AttributeValue]:
        """Collect sharding, mesh, and regularization metadata for IR node annotations.

        Returns:
            Dictionary of serializable `AttributeValue` metadata entries.
        """
        meta: dict[str, AttributeValue] = {}
        if self.skip_lp_regularization is not None:
            meta["skip_lp_regularization"] = bool(self.skip_lp_regularization)
        if self.ici_mesh_shape is not None:
            meta["ici_mesh_shape"] = [int(d) for d in self.ici_mesh_shape]
        if self.dcn_mesh_shape is not None:
            meta["dcn_mesh_shape"] = [int(d) for d in self.dcn_mesh_shape]
        if self.contiguous_submeshes is not None:
            meta["contiguous_submeshes"] = bool(self.contiguous_submeshes)
        if self.mesh_axis_names is not None:
            meta["mesh_axis_names"] = [str(ax) for ax in self.mesh_axis_names]
        if self.shared_weight_layer_id is not None:
            meta["shared_weight_layer_id"] = self.shared_weight_layer_id
        if (
            self.weight_split_dims_mapping is not None
            and self.weight_split_dims_mapping.wt is not None
        ):
            meta["weight_sharding"] = str(list(self.weight_split_dims_mapping.wt))
        if (
            self.activation_split_dims_mapping is not None
            and self.activation_split_dims_mapping.out is not None
        ):
            meta["activation_sharding"] = str(list(self.activation_split_dims_mapping.out))
        return meta

    @property
    def effective_fprop_dtype(self) -> DType:
        """Return `fprop_dtype` if set, otherwise fall back to `dtype`.

        Returns:
            Resolved forward-pass `DType`.
        """
        return self.fprop_dtype if self.fprop_dtype is not None else self.dtype

    def _to_tensor(self, x: TensorOrScalar) -> Tensor:
        """Convert `x` to a `Tensor` using `fprop_dtype` when explicitly configured.

        Args:
            x: Input tensor or scalar.

        Returns:
            Converted `Tensor`.
        """
        self._ensure_setup()
        if self.fprop_dtype is not None:
            return _ops.asarray(x, dtype=self.fprop_dtype)
        return _ops.asarray(x)

    def register_parameter(self, name: str, value: TensorOrScalar) -> None:
        """Register a named parameter in `self.params` and propagate to `self.parent` if attached.

        Args:
            name: Parameter identifier string.
            value: Initial parameter value wrapped in a `BaseParameter`.
        """
        self.params[name] = BaseParameter(value)
        if self.parent is not None:
            prefix = f"{self.name}/" if self.name else ""
            self.parent.register_parameter(f"{prefix}{name}", value)

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
        self._ensure_setup()
        target_shape = tuple(int(d) for d in shape)
        initializer = init if init is not None else (self.params_init or WeightInit(method="zeros"))
        tensor = initializer.init_tensor(target_shape, dtype=self.dtype)
        if self.shared_weight_layer_id:
            _SHARED_WEIGHT_REGISTRY[(self.shared_weight_layer_id, name, target_shape)] = tensor
        self.register_parameter(name, tensor)
        return _ops.register_graph_input(tensor, name=name, metadata=self._build_ir_metadata())

    def get_variable(
        self,
        name: str,
        shape: Sequence[int],
        explicit: TensorOrScalar | None = None,
        init: WeightInit | None = None,
    ) -> Tensor:
        """Resolve a weight tensor from an explicit override, registered parameter, parent scope, or initializer.

        Args:
            name: Parameter identifier string.
            shape: Target tensor shape if initialization is required.
            explicit: Optional caller-supplied tensor override.
            init: Optional explicit `WeightInit` strategy.

        Returns:
            Resolved `Tensor` parameter.
        """
        self._ensure_setup()
        target_shape = tuple(int(d) for d in shape)
        if explicit is not None:
            exp_t = _ops.asarray(explicit)
            return _ops.register_graph_input(exp_t, name=name, metadata=self._build_ir_metadata())
        if name in self.params:
            cached = _ops.asarray(self.params[name]())
            if cached.shape == target_shape:
                return _ops.register_graph_input(
                    cached, name=name, metadata=self._build_ir_metadata()
                )
        if self.shared_weight_layer_id:
            shared_key = (self.shared_weight_layer_id, name, target_shape)
            if shared_key in _SHARED_WEIGHT_REGISTRY:
                shared_t = _SHARED_WEIGHT_REGISTRY[shared_key]
                self.params[name] = BaseParameter(shared_t)
                return _ops.register_graph_input(
                    shared_t, name=name, metadata=self._build_ir_metadata()
                )
        if self.parent is not None:
            curr: BasePraxisLayer | None = self
            scoped_name = name
            while curr is not None and curr.parent is not None:
                prefix = f"{curr.name}/" if curr.name else ""
                scoped_name = f"{prefix}{scoped_name}"
                curr = curr.parent
                if scoped_name in curr.params:
                    resolved = _ops.asarray(curr.params[scoped_name]())
                    if resolved.shape == target_shape:
                        self.params[name] = BaseParameter(resolved)
                        return _ops.register_graph_input(
                            resolved, name=name, metadata=self._build_ir_metadata()
                        )
        return self.create_variable(name, shape, init=init)

    def get_sublayer(
        self,
        name: str,
        tpl: LayerTpl,
        default: BasePraxisLayer | StrictCallableLayerProtocol,
    ) -> StrictCallableLayerProtocol:
        """Resolve and parent-link a named sub-layer in `self.sublayers`.

        Args:
            name: Sub-layer scope name.
            tpl: Optional layer template override.
            default: Default callable layer when `tpl` is `None`.

        Returns:
            Resolved and parent-linked `StrictCallableLayerProtocol`.
        """
        self._ensure_setup()
        resolved = instantiate_layer(tpl, default, parent=self, name=name)
        self.sublayers[name] = resolved
        return resolved

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
        return self._to_tensor(inputs)

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
    return _ops.asarray(val)


LayerTpl = Union[
    PaxConfig[BasePraxisLayer],
    type[BasePraxisLayer],
    BasePraxisLayer,
    CallableLayer,
    None,
]
BasePraxisLayer.model_rebuild()


@runtime_checkable
class _UnaryFnProtocol(Protocol):
    """Protocol for single-argument functions."""

    def __call__(self, x: TensorOrScalar, /) -> TensorOrScalar:
        """Call with a single positional argument.

        Args:
            x: Input tensor or scalar.

        Returns:
            Output tensor or scalar.
        """


@runtime_checkable
class _BinaryFnProtocol(Protocol):
    """Protocol for two-argument functions."""

    def __call__(self, a: TensorOrScalar, b: TensorOrScalar, /) -> TensorOrScalar:
        """Call with two positional arguments.

        Args:
            a: First input tensor or scalar.
            b: Second input tensor or scalar.

        Returns:
            Output tensor or scalar.
        """


class _CallableFnAdapter:
    """Adapter wrapping a callable template into a `StrictCallableLayerProtocol`."""

    def __init__(self, fn: CallableLayer) -> None:
        """Initialize the adapter with `fn` and precompute its positional arity mode.

        Args:
            fn: Underlying function or callable layer.
        """
        self._fn = fn
        sig = inspect.signature(fn)
        has_var_pos = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        )
        pos_params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if not has_var_pos and len(pos_params) == 1:
            self._arity_mode: Literal["unary", "binary", "varargs"] = "unary"
        elif not has_var_pos and len(pos_params) == 2:
            self._arity_mode = "binary"
        else:
            self._arity_mode = "varargs"

    def __call__(
        self,
        inputs: TensorOrScalar,
        /,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Invoke the wrapped function and coerce its output to `Tensor`.

        Args:
            inputs: First positional argument.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Result converted to `Tensor`.
        """
        if self._arity_mode == "unary" and isinstance(self._fn, _UnaryFnProtocol):
            return _ops.asarray(self._fn(inputs))
        if (
            self._arity_mode == "binary"
            and len(args) >= 1
            and isinstance(self._fn, _BinaryFnProtocol)
        ):
            return _ops.asarray(self._fn(inputs, args[0]))
        assert isinstance(self._fn, CallableLayerProtocol)
        return _ops.asarray(self._fn(inputs, *args, **kwargs))


def instantiate_layer(
    tpl: LayerTpl,
    default: BasePraxisLayer | StrictCallableLayerProtocol,
    *,
    parent: BasePraxisLayer | None = None,
    name: str | None = None,
) -> StrictCallableLayerProtocol:
    """Resolve a `LayerTpl` into a concrete `StrictCallableLayerProtocol` instance.

    Handles `PaxConfig` templates, `BasePraxisLayer` subclasses, already-instantiated
    callable layers, or `None` (returning `default`). When `parent` or `name` is provided
    and the resolved layer is a `BasePraxisLayer`, links the child layer to `parent` so
    created variables are registered under `parent.params`.

    Args:
        tpl: Layer template (`PaxConfig`, class, callable instance, or `None`).
        default: Default callable layer used when `tpl` is `None`.
        parent: Optional parent `BasePraxisLayer` owning this sub-layer.
        name: Optional scope name for the sub-layer under `parent`.

    Returns:
        Concrete `StrictCallableLayerProtocol` ready for forward invocation.
    """
    if tpl is None:
        raw: BasePraxisLayer | CallableLayer | StrictCallableLayerProtocol = default
    elif isinstance(tpl, PaxConfig):
        raw = tpl.instantiate()
    elif isinstance(tpl, type):
        raw = tpl()
    else:
        raw = tpl

    if isinstance(raw, BasePraxisLayer):
        if parent is not None:
            if raw.parent is None:
                raw.parent = parent
            if raw.dtype == "float32" and parent.dtype != "float32":
                raw.dtype = parent.dtype
            if raw.fprop_dtype is None and parent.fprop_dtype is not None:
                raw.fprop_dtype = parent.fprop_dtype
        if name is not None and raw.name is None:
            raw.name = name
        return raw

    return _CallableFnAdapter(raw)


__all__ = [
    "BasePraxisLayer",
    "LayerOutput",
    "LayerTpl",
    "as_tensor_if_present",
    "instantiate_layer",
]
