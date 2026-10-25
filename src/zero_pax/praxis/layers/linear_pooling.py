"""Linear, feedforward, einsum, and pooling layers for zero_pax.praxis.layers."""

from __future__ import annotations

from collections.abc import Sequence

from zero_pax import _tensor_ops as _ops
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl, instantiate_layer
from zero_pax.praxis.layers.activations import Identity, ReLU
from zero_pax.types import (
    PoolingOutput,
    Tensor,
    TensorOrScalar,
    WeightInit,
)


def _infer_einsum_weight_shape(equation: str, input_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Infer the weight tensor shape from an `equation` and `input_shape` when `w_shape` is unset.

    Args:
        equation: Einstein summation equation string (e.g., `'...d,df->...f'`).
        input_shape: Shape tuple of the first input operand.

    Returns:
        Inferred weight shape tuple.
    """
    in_dim = input_shape[-1] if input_shape else 1
    lhs, _ = equation.split("->", 1)
    w_spec = lhs.split(",", 1)[1].strip().replace("...", "")
    return (in_dim,) if len(w_spec) == 1 else (in_dim, in_dim)


class EinsumOp(BasePraxisLayer):
    """Stateless Einstein summation operation layer.

    Attributes:
        equation: Default Einstein summation equation string.
    """

    equation: str = ""

    def __call__(
        self,
        *args: TensorOrScalar,
        equation: str | None = None,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Evaluate Einstein summation on `*args`.

        Args:
            *args: Leading equation string (optional) followed by input tensor operands.
            equation: Optional explicit equation string overriding `self.equation`.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` from the Einstein summation.
        """
        eq = equation if equation is not None else self.equation
        if args and isinstance(args[0], str):
            eq = args[0]
            operands = args[1:]
        else:
            operands = args

        if not operands:
            return _ops.zeros((1,))
        if not eq:
            return self._to_tensor(operands[0])
        return _ops.einsum(eq, *operands)


class Einsum(BasePraxisLayer):
    """Parameterized Einstein summation layer with optional learned weights and bias.

    Attributes:
        eqn: Einstein summation equation string.
        equation: Alias for `eqn`.
        w_shape: Shape tuple of the learned weight tensor.
        use_bias: Whether to add a learned bias tensor.
        bias_shape: Optional explicit shape for the bias tensor.
        weight_init: Optional weight initialization strategy for `w`.
        bias_init: Optional weight initialization strategy for `b`.
        einsum_op_tpl: Optional template for the underlying `EinsumOp`.
    """

    eqn: str = ""
    equation: str = ""
    w_shape: tuple[int, ...] = ()
    use_bias: bool = False
    bias_shape: tuple[int, ...] = ()
    weight_init: WeightInit | None = None
    bias_init: WeightInit | None = None
    einsum_op_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        t2: TensorOrScalar | None = None,
        w: TensorOrScalar | None = None,
        b: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply Einstein summation to `inputs` and either `t2` or weight `w`.

        Args:
            inputs: Primary input tensor.
            t2: Optional second input tensor for two-operand einsum.
            w: Optional explicit weight tensor.
            b: Optional explicit bias tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Projected `Tensor`.
        """
        x = self._to_tensor(inputs)
        eq = self.eqn if self.eqn else self.equation
        if not eq:
            return x
        op_layer = self.get_sublayer("einsum_op", self.einsum_op_tpl, EinsumOp(equation=eq))
        if t2 is not None:
            return (
                op_layer(eq, x, _ops.asarray(t2))
                if isinstance(op_layer, EinsumOp)
                else op_layer(x, _ops.asarray(t2))
            )

        target_w_shape = self.w_shape if self.w_shape else _infer_einsum_weight_shape(eq, x.shape)
        default_w_init = self.weight_init or (
            self.params_init if self.w_shape else WeightInit(method="ones", scale=1.0)
        )
        w_t = self.get_variable("w", target_w_shape, explicit=w, init=default_w_init)

        out = op_layer(eq, x, w_t) if isinstance(op_layer, EinsumOp) else op_layer(x, w_t)
        if self.use_bias:
            b_shape = self.bias_shape if self.bias_shape else (out.shape[-1],)
            b_t = self.get_variable("b", b_shape, explicit=b, init=self.bias_init)
            out = out + b_t
        return out


class Linear(BasePraxisLayer):
    """Linear projection layer without bias (`y = x @ w`).

    Attributes:
        input_dims: Input feature dimension size.
        output_dims: Output feature dimension size.
        weight_init: Optional weight initialization strategy.
        einsum_tpl: Optional template for the underlying einsum operation.
    """

    input_dims: int = 0
    output_dims: int = 0
    weight_init: WeightInit | None = None
    einsum_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        w: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Project `inputs` from `input_dims` to `output_dims`.

        Args:
            inputs: Input tensor of shape `[..., input_dims]`.
            w: Optional explicit weight matrix of shape `[input_dims, output_dims]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Projected `Tensor` of shape `[..., output_dims]`.
        """
        x = self._to_tensor(inputs)
        in_d = self.input_dims if self.input_dims > 0 else x.shape[-1]
        out_d = self.output_dims if self.output_dims > 0 else in_d
        w_t = self.get_variable("w", (in_d, out_d), explicit=w, init=self.weight_init)
        op_layer = self.get_sublayer("einsum", self.einsum_tpl, EinsumOp(equation="...d,df->...f"))
        return (
            op_layer("...d,df->...f", x, w_t)
            if isinstance(op_layer, EinsumOp)
            else op_layer(x, w_t)
        )


class Bias(BasePraxisLayer):
    """Additive bias layer (`y = x + b`).

    Attributes:
        dims: Feature dimension size of the bias vector.
        bias_init: Optional weight initialization strategy or scalar constant for the bias.
    """

    dims: int = 0
    bias_init: float | WeightInit | None = 0.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        b: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Add bias vector `b` to `inputs`.

        Args:
            inputs: Input tensor of shape `[..., dims]`.
            b: Optional explicit bias tensor of shape `[dims]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Biased `Tensor` of the same shape as `inputs`.
        """
        x = self._to_tensor(inputs)
        feat_dim = self.dims if self.dims > 0 else x.shape[-1]
        eff_init = (
            WeightInit(method="constant", scale=self.bias_init)
            if isinstance(self.bias_init, (int, float))
            else self.bias_init
        )
        b_t = self.get_variable("b", (feat_dim,), explicit=b, init=eff_init)
        return x + b_t


class FeedForward(BasePraxisLayer):
    """Single-layer FeedForward block combining Linear projection, optional Bias, and Activation.

    Attributes:
        input_dims: Input feature dimension size.
        output_dims: Output feature dimension size.
        has_bias: Whether to apply an additive bias after the linear projection.
        linear_tpl: Optional template for the `Linear` sub-layer.
        bias_tpl: Optional template for the `Bias` sub-layer.
        activation_tpl: Optional callable activation layer or template (defaults to `ReLU`).
        dropout_tpl: Optional callable dropout layer or template.
        weight_init: Optional weight initialization configuration.
        bias_init: Optional bias initialization configuration or scalar constant.
        checkpoint_str: Activation checkpointing policy string.
    """

    input_dims: int = 0
    output_dims: int = 0
    has_bias: bool = True
    linear_tpl: LayerTpl = None
    bias_tpl: LayerTpl = None
    activation_tpl: LayerTpl = None
    dropout_tpl: LayerTpl = None
    weight_init: WeightInit | None = None
    bias_init: float | WeightInit | None = 0.0
    checkpoint_str: str | None = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        w: TensorOrScalar | None = None,
        b: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply linear transformation, optional bias, activation, and optional dropout.

        Args:
            inputs: Input tensor of shape `[..., input_dims]`.
            w: Optional explicit weight matrix of shape `[input_dims, output_dims]`.
            b: Optional explicit bias vector of shape `[output_dims]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` of shape `[..., output_dims]`.
        """
        x = self._to_tensor(inputs)
        in_d = self.input_dims if self.input_dims > 0 else x.shape[-1]
        out_d = self.output_dims if self.output_dims > 0 else in_d
        default_lin = Linear(
            input_dims=in_d,
            output_dims=out_d,
            weight_init=self.weight_init,
            params_init=self.params_init,
        )
        linear = self.get_sublayer("linear", self.linear_tpl, default_lin)
        out = linear(x, w=w) if isinstance(linear, Linear) else linear(x)

        if self.has_bias:
            default_bias = Bias(
                dims=out_d,
                bias_init=self.bias_init,
                params_init=self.params_init,
            )
            bias_layer = self.get_sublayer("bias", self.bias_tpl, default_bias)
            out = bias_layer(out, b=b) if isinstance(bias_layer, Bias) else bias_layer(out)

        act_layer = self.get_sublayer("activation", self.activation_tpl, ReLU())
        out = act_layer(out)

        drop_layer = self.get_sublayer("dropout", self.dropout_tpl, Identity())
        out = drop_layer(out)

        return out


class MLPBlock(BasePraxisLayer):
    """Multi-Layer Perceptron (MLP) block stacking multiple `FeedForward` layers.

    Attributes:
        num_layers: Number of `FeedForward` layers in the stack.
        hidden_dims: Hidden feature dimension size.
        has_bias: Whether each `FeedForward` layer includes an additive bias.
        activate_final: Whether to apply the activation function on the final layer.
        ff_tpl: Optional template for each `FeedForward` layer.
        fflayer_tpl: Optional alias template for each `FeedForward` layer.
        activation_tpl: Optional activation layer template applied to intermediate layers.
        dropout_tpl: Optional dropout layer template applied after each layer.
    """

    num_layers: int = 3
    hidden_dims: int = 128
    has_bias: bool = True
    activate_final: bool = True
    ff_tpl: LayerTpl = None
    fflayer_tpl: LayerTpl = None
    activation_tpl: LayerTpl = None
    dropout_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        w: TensorOrScalar | None = None,
        b: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the stacked MLP forward pass to `inputs`.

        Args:
            inputs: Input tensor of shape `[..., D]`.
            w: Optional explicit weight matrix applied across layers.
            b: Optional explicit bias vector applied across layers.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` after `num_layers` feedforward projections.
        """
        out = self._to_tensor(inputs)
        feat_dim = (
            self.hidden_dims
            if (self.hidden_dims > 0 and self.hidden_dims != 128)
            else out.shape[-1]
        )
        tpl_to_use = self.ff_tpl if self.ff_tpl is not None else self.fflayer_tpl
        n_layers = max(1, self.num_layers)
        for idx in range(n_layers):
            is_last = idx == n_layers - 1
            act_for_layer: LayerTpl = (
                Identity() if (is_last and not self.activate_final) else self.activation_tpl
            )
            default_ff = FeedForward(
                input_dims=out.shape[-1],
                output_dims=feat_dim,
                has_bias=self.has_bias,
                activation_tpl=act_for_layer,
                dropout_tpl=self.dropout_tpl,
                params_init=self.params_init,
            )
            ff = instantiate_layer(tpl_to_use, default_ff, parent=self, name=f"ff_{idx}")
            out = ff(out, w=w, b=b) if isinstance(ff, FeedForward) else ff(out)
        return out


class PerDimScale(BasePraxisLayer):
    """Layer that scales each feature dimension by a learned parameter vector.

    Attributes:
        dim: Feature dimension size.
        dims: Alias for `dim`.
    """

    dim: int = 0
    dims: int = 0

    def __call__(
        self,
        inputs: TensorOrScalar,
        scale: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Scale `inputs` elementwise along the last dimension.

        Args:
            inputs: Input tensor of shape `[..., D]`.
            scale: Optional explicit scale vector of shape `[D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Scaled `Tensor` of the same shape as `inputs`.
        """
        x = self._to_tensor(inputs)
        d = self.dim if self.dim > 0 else (self.dims if self.dims > 0 else x.shape[-1])
        s = self.get_variable(
            "scale", (d,), explicit=scale, init=WeightInit(method="ones", scale=1.0)
        )
        return x * s


class Pooling(BasePraxisLayer):
    """2D spatial pooling layer supporting MAX and AVG pooling.

    Attributes:
        window_shape: Spatial pooling window `(height, width)`.
        window_stride: Spatial pooling stride `(stride_h, stride_w)`.
        pooling_type: Pooling reduction type (`'MAX'` or `'AVG'`).
        padding: Padding algorithm (`'SAME'` or `'VALID'`).
    """

    window_shape: tuple[int, int] = (0, 0)
    window_stride: tuple[int, int] = (0, 0)
    pooling_type: str = "MAX"
    padding: str = "SAME"

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor | PoolingOutput:
        """Apply vectorized 2D spatial pooling to `inputs`.

        Args:
            inputs: 4D input tensor of shape `[B, H, W, C]`.
            paddings: Optional temporal/spatial padding mask of shape `[B, H]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Pooled `Tensor`, or `(pooled_tensor, out_paddings)` if `paddings` is provided.

        Raises:
            ValueError: If `window_shape`, `window_stride`, or `pooling_type` is invalid.
        """
        for entry in self.window_shape:
            if entry <= 0:
                raise ValueError("window_shape entries must be positive integers.")
        for entry in self.window_stride:
            if entry <= 0:
                raise ValueError("window_stride entries must be positive integers.")
        pool_mode = self.pooling_type.upper()
        if pool_mode not in ("MAX", "AVG"):
            raise ValueError(f"pooling_type must be MAX or AVG, got {self.pooling_type}")

        x = self._to_tensor(inputs)
        if self.window_shape == (1, 1) and self.window_stride == (1, 1):
            if paddings is not None:
                return x, _ops.asarray(paddings)
            return x

        b_sz, h_in, w_in, _ = x.shape
        wh, ww = self.window_shape
        sh, sw = self.window_stride

        if self.padding == "SAME":
            out_h = (h_in + sh - 1) // sh
            out_w = (w_in + sw - 1) // sw
            pad_h = max((out_h - 1) * sh + wh - h_in, 0)
            pad_w = max((out_w - 1) * sw + ww - w_in, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            out_h = (h_in - wh) // sh + 1
            out_w = (w_in - ww) // sw + 1
            pad_top = pad_bottom = pad_left = pad_right = 0

        fill_val = -_ops.inf if pool_mode == "MAX" else 0.0
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            x_pad = _ops.pad(
                x,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=fill_val,
            )
            mask = _ops.ones((b_sz, h_in, w_in, 1))
            mask_pad = _ops.pad(
                mask,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
        else:
            x_pad = x
            mask_pad = _ops.ones((b_sz, h_in, w_in, 1))

        if paddings is not None:
            pad_t = _ops.asarray(paddings)
            valid_mask = 1.0 - _ops.reshape(pad_t, (b_sz, h_in, 1, 1))
            if pad_top > 0 or pad_bottom > 0:
                valid_mask = _ops.pad(
                    valid_mask,
                    ((0, 0), (pad_top, pad_bottom), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
            mask_pad = mask_pad * valid_mask
            if pool_mode == "MAX":
                x_pad = _ops.where(mask_pad > 0.0, x_pad, -_ops.inf)
            else:
                x_pad = x_pad * mask_pad

        patches = _ops.extract_patches_2d(x_pad, out_h, out_w, wh, ww, sh, sw)
        if pool_mode == "MAX":
            out = _ops.max(patches, axis=(3, 4))
        else:
            mask_patches = _ops.extract_patches_2d(mask_pad, out_h, out_w, wh, ww, sh, sw)
            denom = _ops.maximum(_ops.sum(mask_patches, axis=(3, 4)), 1.0)
            out = _ops.sum(patches, axis=(3, 4)) / denom

        if paddings is not None:
            pad_t = _ops.asarray(paddings)
            if sh > 1 or self.padding == "VALID":
                p_pad = (
                    _ops.pad(
                        pad_t,
                        ((0, 0), (pad_top, pad_bottom)),
                        mode="constant",
                        constant_values=1.0,
                    )
                    if (pad_top > 0 or pad_bottom > 0)
                    else pad_t
                )
                p_patches = _ops.extract_patches_1d(_ops.expand_dims(p_pad, axis=-1), out_h, wh, sh)
                out_paddings = _ops.min(p_patches[:, :, :, 0], axis=2)
            else:
                out_paddings = pad_t
            if pool_mode == "AVG":
                out = out * (1.0 - _ops.reshape(out_paddings, (b_sz, out_h, 1, 1)))
            return out, out_paddings

        return out


class Pooling1D(BasePraxisLayer):
    """1D temporal pooling layer wrapping 2D `Pooling`.

    Attributes:
        stride: Pooling stride along the time dimension.
        window: Pooling window size along the time dimension.
        pooling_type: Pooling reduction type (`'MAX'` or `'AVG'`).
        padding: Padding algorithm (`'SAME'` or `'VALID'`).
    """

    stride: int = 1
    window: int = 0
    pooling_type: str = "AVG"
    padding: str = "SAME"

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor | PoolingOutput:
        """Apply 1D pooling along axis 1 of `inputs`.

        Args:
            inputs: 3D input tensor of shape `[B, T, D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Pooled 3D `Tensor`, or `(pooled_tensor, out_paddings)` if `paddings` is provided.
        """
        x = self._to_tensor(inputs)
        win = self.window if self.window > 0 else 1
        if win == 1 and self.stride == 1:
            if paddings is not None:
                return x, _ops.asarray(paddings)
            return x

        expanded = _ops.expand_dims(x, axis=2)
        pool_2d = Pooling(
            window_shape=(win, 1),
            window_stride=(self.stride, 1),
            pooling_type=self.pooling_type,
            padding=self.padding,
        )
        res = pool_2d(expanded, paddings=paddings)
        if isinstance(res, tuple):
            out_4d, out_pad = res
            return out_4d[:, :, 0, :], out_pad
        return res[:, :, 0, :]


class GlobalPooling(BasePraxisLayer):
    """Global spatial/temporal pooling layer over arbitrary `pooling_dims`.

    Attributes:
        pooling_type: Pooling reduction type (`'AVG'` or `'MAX'`).
        pooling_dims: Sequence of non-negative dimension indices to pool over.
        keepdims: Whether to retain reduced dimensions with length 1.
    """

    pooling_type: str = "AVG"
    pooling_dims: Sequence[int] | None = None
    keepdims: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        compatible_paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply global pooling across `self.pooling_dims`.

        Args:
            inputs: Input tensor.
            compatible_paddings: Optional broadcast-compatible padding mask where `1` indicates padded entries.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Globally pooled `Tensor`.

        Raises:
            ValueError: If `pooling_dims` or `pooling_type` is invalid.
        """
        if self.pooling_dims is None:
            raise ValueError("pooling_dims must be set as a list")
        if not all(d >= 0 for d in self.pooling_dims):
            raise ValueError("pooling_dims must be non-negative integers")

        x = self._to_tensor(inputs)
        if not all(d < x.ndim for d in self.pooling_dims):
            raise ValueError("pooling_dims must be smaller than input rank.")

        pool_mode = self.pooling_type.upper()
        if pool_mode not in ("MAX", "AVG"):
            raise ValueError(f"pooling_type must be MAX or AVG, got {self.pooling_type}")

        axes = tuple(self.pooling_dims)
        if compatible_paddings is not None:
            pad_t = _ops.asarray(compatible_paddings)
            weights = 1.0 - pad_t
            if pool_mode == "AVG":
                num = _ops.sum(x * weights, axis=axes, keepdims=self.keepdims)
                den = _ops.maximum(_ops.sum(weights, axis=axes, keepdims=self.keepdims), 1.0)
                out = num / den
                all_padded = _ops.sum(weights, axis=axes, keepdims=self.keepdims) == 0.0
                out = _ops.where(all_padded, 0.0, out)
            else:
                masked_x = _ops.where(pad_t > 0.0, -_ops.inf, x)
                out = _ops.max(masked_x, axis=axes, keepdims=self.keepdims)
        else:
            if pool_mode == "AVG":
                out = _ops.mean(x, axis=axes, keepdims=self.keepdims)
            else:
                out = _ops.max(x, axis=axes, keepdims=self.keepdims)

        return out


__all__ = [
    "Bias",
    "Einsum",
    "EinsumOp",
    "FeedForward",
    "GlobalPooling",
    "Linear",
    "MLPBlock",
    "PerDimScale",
    "Pooling",
    "Pooling1D",
]
