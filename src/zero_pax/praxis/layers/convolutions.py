"""Convolution layers for zero_pax.praxis.layers."""

from __future__ import annotations

from zero_pax import _tensor_ops as np
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl, instantiate_layer
from zero_pax.praxis.layers.activations import Identity, ReLU, Swish
from zero_pax.praxis.layers.linear_pooling import Linear
from zero_pax.praxis.layers.normalizations import BatchNorm, LayerNorm
from zero_pax.types import (
    PoolingOutput,
    Tensor,
    TensorOrScalar,
    WeightInit,
)


class Conv2D(BasePraxisLayer):
    """2D Convolution layer supporting stride, dilation, groups, and causal padding.

    Attributes:
        filter_shape: 4D filter dimensions `(kernel_h, kernel_w, in_channels_per_group, out_channels)`.
        filter_stride: 2D spatial stride `(stride_h, stride_w)`.
        dilations: 2D filter dilation factors `(dilation_h, dilation_w)`.
        bias: Whether to add a learned bias vector to the output channels.
        bias_init: Optional weight initialization strategy for the bias.
        kernel_init: Optional weight initialization strategy for the convolution kernel.
        padding: Padding algorithm (`'SAME'` or `'VALID'`).
        tf_equivalent_padding: Whether to use TensorFlow-compatible asymmetric SAME padding.
        is_causal: Whether to apply causal (left-only) padding along the height/time dimension.
        weight_norm_tpl: Optional weight normalization template.
        mask_t: Optional mask tensor applied to the convolution kernel.
        einsum_tpl: Optional einsum template.
    """

    filter_shape: tuple[int, int, int, int] = (0, 0, 0, 0)
    filter_stride: tuple[int, int] = (1, 1)
    dilations: tuple[int, int] = (1, 1)
    bias: bool = False
    bias_init: WeightInit | None = None
    kernel_init: WeightInit | None = None
    padding: str = "SAME"
    tf_equivalent_padding: bool = False
    is_causal: bool = False
    weight_norm_tpl: LayerTpl = None
    mask_t: TensorOrScalar | None = None
    einsum_tpl: LayerTpl = None

    def _compute_temporal_output_paddings(self, paddings: Tensor, h_in: int) -> Tensor:
        """Compute output temporal paddings after strided or VALID 2D convolution.

        Args:
            paddings: Input temporal padding tensor of shape `[B, H_in]`.
            h_in: Input height/time dimension length.

        Returns:
            Downsampled padding `Tensor` of shape `[B, H_out]`.
        """
        fh = self.filter_shape[0] if self.filter_shape[0] > 0 else 3
        sh = self.filter_stride[0] if self.filter_stride[0] > 0 else 1
        dh = self.dilations[0] if self.dilations[0] > 0 else 1
        eff_fh = (fh - 1) * dh + 1
        if self.padding == "SAME":
            out_h = (h_in + sh - 1) // sh
            if sh == 1:
                return paddings
            return paddings[:, ::sh][:, :out_h]
        out_h = (h_in - eff_fh) // sh + 1
        b_sz = paddings.shape[0]
        out_pad = np.zeros((b_sz, out_h))
        for i in range(out_h):
            hs = i * sh
            out_pad[:, i] = np.max(paddings[:, hs : hs + eff_fh], axis=1)
        return out_pad

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        w: TensorOrScalar | None = None,
        bias: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply 2D convolution to `inputs`.

        Args:
            inputs: 4D input tensor of shape `[B, H, W, C_in]`.
            paddings: Optional temporal padding mask of shape `[B, H]`.
            w: Optional explicit filter tensor of shape `self.filter_shape`.
            bias: Optional explicit bias tensor of shape `[C_out]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Convolved 4D `Tensor` of shape `[B, H_out, W_out, C_out]`.
        """
        x = np.asarray(inputs)
        b_sz, h_in, w_in, c_in = x.shape

        if paddings is not None:
            pad_in = np.asarray(paddings)
            x = x * (1.0 - np.reshape(pad_in, (b_sz, h_in, 1, 1)))

        if self.filter_shape == (0, 0, 0, 0):
            fh, fw, f_in, f_out = (3, 3, c_in, c_in)
        else:
            fh, fw, f_in, f_out = self.filter_shape

        w_t = self.get_variable("w", (fh, fw, f_in, f_out), explicit=w, init=self.kernel_init)
        if self.weight_norm_tpl is not None:
            wn_layer = instantiate_layer(self.weight_norm_tpl, Identity())
            w_t = wn_layer(w_t)
        if self.mask_t is not None:
            w_t = w_t * np.asarray(self.mask_t)

        sh = self.filter_stride[0] if self.filter_stride[0] > 0 else 1
        sw = self.filter_stride[1] if self.filter_stride[1] > 0 else 1
        dh = self.dilations[0] if self.dilations[0] > 0 else 1
        dw = self.dilations[1] if self.dilations[1] > 0 else 1

        eff_fh = (fh - 1) * dh + 1
        eff_fw = (fw - 1) * dw + 1

        if self.padding == "SAME":
            out_h = (h_in + sh - 1) // sh
            out_w = (w_in + sw - 1) // sw
            pad_h = max((out_h - 1) * sh + eff_fh - h_in, 0)
            pad_w = max((out_w - 1) * sw + eff_fw - w_in, 0)
            if self.is_causal:
                pad_top = pad_h
                pad_bottom = 0
            else:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            out_h = (h_in - eff_fh) // sh + 1
            out_w = (w_in - eff_fw) // sw + 1
            pad_top = pad_bottom = pad_left = pad_right = 0

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            x_pad = np.pad(
                x,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
        else:
            x_pad = x

        groups = max(1, c_in // max(1, f_in))
        out_c_per_group = f_out // groups

        out = np.zeros((b_sz, out_h, out_w, f_out))
        for i in range(out_h):
            for j in range(out_w):
                hs = i * sh
                ws = j * sw
                window = x_pad[:, hs : hs + eff_fh : dh, ws : ws + eff_fw : dw, :]
                if groups == 1:
                    out[:, i, j, :] = np.einsum("bhwc,hwco->bo", window, w_t)
                else:
                    win_g = np.reshape(window, (b_sz, fh, fw, groups, f_in))
                    w_g = np.reshape(w_t, (fh, fw, f_in, groups, out_c_per_group))
                    res_g = np.einsum("bhwgc,hwcgo->bgo", win_g, w_g)
                    out[:, i, j, :] = np.reshape(res_g, (b_sz, f_out))

        if self.bias:
            b_t = self.get_variable("b", (f_out,), explicit=bias, init=self.bias_init)
            out = out + b_t

        return out


class ConvBNAct(Conv2D):
    """Convolution layer followed by Batch Normalization and an Activation function.

    Attributes:
        batch_norm_tpl: Optional callable `BatchNorm` layer or template.
        activation_tpl: Optional callable activation layer or template.
    """

    batch_norm_tpl: LayerTpl = None
    activation_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        w: TensorOrScalar | None = None,
        bias: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply Conv2D -> BatchNorm -> Activation to `inputs`.

        Args:
            inputs: 4D input tensor of shape `[B, H, W, C_in]`.
            paddings: Optional padding mask of shape `[B, H]`.
            w: Optional explicit filter tensor.
            bias: Optional explicit bias tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Convolved, batch-normalized, and activated 4D `Tensor`.
        """
        x = np.asarray(inputs)
        out = super().__call__(x, paddings=paddings, w=w, bias=bias)
        out_paddings = (
            self._compute_temporal_output_paddings(np.asarray(paddings), x.shape[1])
            if paddings is not None
            else None
        )
        bn_layer = instantiate_layer(self.batch_norm_tpl, BatchNorm())
        if isinstance(bn_layer, BatchNorm):
            out = bn_layer(out, paddings=out_paddings)
        else:
            out = bn_layer(out)

        act_layer = instantiate_layer(self.activation_tpl, ReLU())
        return act_layer(out)


class ConvBNActWithPadding(ConvBNAct):
    """ConvBNAct layer that also computes and returns updated temporal paddings.

    Attributes:
        compat_with_lingvo: Whether to use Lingvo-compatible padding propagation.
    """

    compat_with_lingvo: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        w: TensorOrScalar | None = None,
        bias: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor | PoolingOutput:
        """Apply ConvBNAct and return `(output, out_paddings)` when `paddings` is provided.

        Args:
            inputs: 4D input tensor of shape `[B, H, W, C_in]`.
            paddings: Optional padding mask of shape `[B, H]`.
            w: Optional explicit filter tensor.
            bias: Optional explicit bias tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            `(output, out_paddings)` tuple if `paddings` is not None, otherwise `output`.
        """
        x = np.asarray(inputs)
        out = super().__call__(x, paddings=paddings, w=w, bias=bias)
        if paddings is not None:
            out_pad = self._compute_temporal_output_paddings(np.asarray(paddings), x.shape[1])
            return out, out_pad
        return out


class DepthwiseConv1D(BasePraxisLayer):
    """1D Depthwise separable convolution layer.

    Attributes:
        filter_shape: 3D filter dimensions `(kernel_size, in_channels, channel_multiplier)` or `(kernel_size, 1, in_channels)`.
        filter_stride: 1D temporal stride `(stride,)`.
        bias: Whether to add a learned bias vector.
        bias_init: Optional weight initialization strategy for the bias.
        is_causal: Whether to apply causal left-padding.
        use_2d_conv_weight_shape: Whether weights are stored in 2D conv format.
        rhs_dilation_rate: Dilation factor along the sequence axis.
        einsum_tpl: Optional einsum template.
    """

    filter_shape: tuple[int, int, int] = (0, 0, 0)
    filter_stride: tuple[int] = (1,)
    bias: bool = False
    bias_init: WeightInit | None = None
    is_causal: bool = False
    use_2d_conv_weight_shape: bool = False
    rhs_dilation_rate: int = 1
    einsum_tpl: LayerTpl = None

    def _resolve_kernel_size(self) -> int:
        """Resolve the 1D convolution kernel size from `filter_shape`.

        Returns:
            Positive integer kernel size.
        """
        return self.filter_shape[0] if self.filter_shape[0] > 0 else 3

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        w: TensorOrScalar | None = None,
        bias: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply 1D depthwise convolution to `inputs`.

        Args:
            inputs: 3D input tensor of shape `[B, T, C]`.
            paddings: Optional padding mask of shape `[B, T]`.
            w: Optional explicit depthwise kernel tensor.
            bias: Optional explicit bias tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Convolved 3D `Tensor` of shape `[B, T_out, C]`.
        """
        x = np.asarray(inputs)
        if paddings is not None:
            pad_t = np.asarray(paddings)
            x = x * (1.0 - np.expand_dims(pad_t, -1))

        b_sz, t_len, c_in = x.shape
        k_sz = self._resolve_kernel_size()
        dil = self.rhs_dilation_rate if self.rhs_dilation_rate > 0 else 1
        stride = self.filter_stride[0] if (self.filter_stride and self.filter_stride[0] > 0) else 1

        w_t: Tensor
        if w is None:
            w_t = self.get_variable("w", (k_sz, 1, c_in))
        else:
            w_raw = np.asarray(w)
            w_t = np.reshape(w_raw, (k_sz, 1, c_in)) if w_raw.shape != (k_sz, 1, c_in) else w_raw

        eff_k = (k_sz - 1) * dil + 1
        out_t = (t_len + stride - 1) // stride
        pad_total = max((out_t - 1) * stride + eff_k - t_len, 0)
        if self.is_causal:
            pad_left = pad_total
            pad_right = 0
        else:
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left

        x_pad = (
            np.pad(
                x,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
            if (pad_left > 0 or pad_right > 0)
            else x
        )

        out = np.zeros((b_sz, out_t, c_in))
        w_kernel = w_t[:, 0, :]
        for i in range(out_t):
            t_start = i * stride
            window = x_pad[:, t_start : t_start + eff_k : dil, :]
            out[:, i, :] = np.einsum("btc,tc->bc", window, w_kernel)

        if self.bias:
            b_t = self.get_variable("b", (c_in,), explicit=bias, init=self.bias_init)
            out = out + b_t

        if paddings is not None:
            pad_t = np.asarray(paddings)
            out_pad = pad_t[:, ::stride][:, :out_t] if stride > 1 else pad_t
            out = out * (1.0 - np.expand_dims(out_pad, -1))

        return out


class CausalDepthwiseConv1D(DepthwiseConv1D):
    """Causal 1D Depthwise Convolution layer that only attends to past and current frames.

    Attributes:
        kernel_size: 1D convolution kernel size (overrides `filter_shape[0]` when > 0).
        hidden_dims: Channel dimension size (overrides `filter_shape[1]` when > 0).
    """

    kernel_size: int = 0
    hidden_dims: int = 0
    is_causal: bool = True

    def _resolve_kernel_size(self) -> int:
        """Resolve the 1D convolution kernel size from `kernel_size` or `filter_shape`.

        Returns:
            Positive integer kernel size.
        """
        if self.kernel_size > 0:
            return self.kernel_size
        return super()._resolve_kernel_size()


class LightConv1D(BasePraxisLayer):
    """Lightweight Convolution block used in Conformer architectures.

    Executes: `LayerNorm` -> gated `Linear` (GLU) -> `DepthwiseConv1D` -> `BatchNorm` -> `Swish` -> `Linear`.

    Attributes:
        input_dims: Input and output feature dimension size.
        kernel_size: Kernel size of the depthwise 1D convolution.
        conv_activation_tpl: Activation layer applied after convolution normalization.
        dropout_prob: Dropout probability applied at the end of the block.
        ln_tpl: Layer normalization template.
        linear_start_tpl: Pointwise expansion linear projection template.
        depthwise_conv_tpl: Depthwise 1D convolution template.
        conv_norm_layer_tpl: Convolution normalization layer template.
        linear_end_tpl: Pointwise output linear projection template.
        dropout_tpl: Dropout layer template.
        is_causal: Whether the depthwise convolution is causal.
        use_2d_conv_norm: Whether to apply 2D-style normalization after convolution.
    """

    input_dims: int = 0
    kernel_size: int = 3
    conv_activation_tpl: LayerTpl = None
    dropout_prob: float = 0.0
    ln_tpl: LayerTpl = None
    linear_start_tpl: LayerTpl = None
    depthwise_conv_tpl: LayerTpl = None
    conv_norm_layer_tpl: LayerTpl = None
    linear_end_tpl: LayerTpl = None
    dropout_tpl: LayerTpl = None
    is_causal: bool = False
    use_2d_conv_norm: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        ln_scale: TensorOrScalar | None = None,
        ln_bias: TensorOrScalar | None = None,
        w_start: TensorOrScalar | None = None,
        w_conv: TensorOrScalar | None = None,
        bn_gamma: TensorOrScalar | None = None,
        bn_beta: TensorOrScalar | None = None,
        w_end: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the Conformer Lightweight Convolution block to `inputs`.

        Args:
            inputs: 3D input tensor of shape `[B, T, D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            ln_scale: Optional pre-LayerNorm scale parameter.
            ln_bias: Optional pre-LayerNorm bias parameter.
            w_start: Optional GLU input projection weight matrix of shape `[D, 2 * D]`.
            w_conv: Optional depthwise convolution kernel of shape `[K, 1, D]`.
            bn_gamma: Optional BatchNorm gamma parameter.
            bn_beta: Optional BatchNorm beta parameter.
            w_end: Optional output projection weight matrix of shape `[D, D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Convolved 3D `Tensor` of shape `[B, T, D]`.
        """
        x = np.asarray(inputs)
        d_model = self.input_dims if self.input_dims > 0 else x.shape[-1]
        k_sz = self.kernel_size if self.kernel_size > 0 else 3

        ln = instantiate_layer(self.ln_tpl, LayerNorm(dim=d_model))
        normed = ln(x, scale=ln_scale, bias=ln_bias) if isinstance(ln, LayerNorm) else ln(x)

        lin_start = instantiate_layer(
            self.linear_start_tpl,
            Linear(input_dims=d_model, output_dims=2 * d_model, params_init=self.params_init),
        )
        proj = lin_start(normed, w=w_start) if isinstance(lin_start, Linear) else lin_start(normed)
        act_inputs, gate = np.split(proj, 2, axis=-1)
        gated = act_inputs * (1.0 / (1.0 + np.exp(-gate)))

        default_dconv: DepthwiseConv1D = (
            CausalDepthwiseConv1D(filter_shape=(k_sz, 1, d_model), params_init=self.params_init)
            if self.is_causal
            else DepthwiseConv1D(filter_shape=(k_sz, 1, d_model), params_init=self.params_init)
        )
        dconv = instantiate_layer(self.depthwise_conv_tpl, default_dconv)
        conv_out = (
            dconv(gated, paddings=paddings, w=w_conv)
            if isinstance(dconv, DepthwiseConv1D)
            else dconv(gated)
        )

        bn = instantiate_layer(self.conv_norm_layer_tpl, BatchNorm(dim=d_model))
        norm_conv = (
            bn(conv_out, paddings=paddings, gamma=bn_gamma, beta=bn_beta)
            if isinstance(bn, BatchNorm)
            else bn(conv_out)
        )

        act_layer = instantiate_layer(self.conv_activation_tpl, Swish())
        act_out = act_layer(norm_conv)

        lin_end = instantiate_layer(
            self.linear_end_tpl,
            Linear(input_dims=d_model, output_dims=d_model, params_init=self.params_init),
        )
        out = lin_end(act_out, w=w_end) if isinstance(lin_end, Linear) else lin_end(act_out)

        drop_layer = instantiate_layer(self.dropout_tpl, Identity())
        out = drop_layer(out)

        if paddings is not None:
            pad_t = np.asarray(paddings)
            out = out * (1.0 - np.expand_dims(pad_t, -1))

        return out


__all__ = [
    "CausalDepthwiseConv1D",
    "Conv2D",
    "ConvBNAct",
    "ConvBNActWithPadding",
    "DepthwiseConv1D",
    "LightConv1D",
]
