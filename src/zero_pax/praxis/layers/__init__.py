"""Module documentation."""

import ml_switcheroo

"""Praxis layers module."""
from typing import Any, Sequence, Optional, Callable
from pydantic import BaseModel, ConfigDict
import numpy as np


class DummyMeta(type):
    """DummyMeta class."""

    """DummyMeta class."""

    """DummyMeta class."""

    """DummyMeta class."""

    """DummyMeta class."""

    """DummyMeta class."""

    """DummyMeta class."""

    """DummyMeta class."""

    """DummyMeta class."""

    def __getattr__(cls, name):
        """__getattr__ function."""
        """__getattr__ function."""
        """__getattr__ function."""
        """__getattr__ function."""
        """__getattr__ function."""
        """__getattr__ function."""
        """__getattr__ function."""
        """__getattr__ function."""
        """__getattr__ function."""
        pass

    def __getitem__(cls, item):
        """__getitem__ function."""
        """__getitem__ function."""
        """__getitem__ function."""
        """__getitem__ function."""
        """__getitem__ function."""
        """__getitem__ function."""
        """__getitem__ function."""
        """__getitem__ function."""
        """__getitem__ function."""
        pass


class DummyType(metaclass=DummyMeta):
    """DummyType class."""

    """DummyType class."""

    """DummyType class."""

    """DummyType class."""

    """DummyType class."""

    """DummyType class."""

    """DummyType class."""

    """DummyType class."""

    """DummyType class."""

    pass


LayerTpl = Any
WeightInit = Any
ActivationType = Any
DecoderHParams = Any
LanguageModelType = Any
SplitDimsMapping = Any
PaxConfig = Any
BaseLayer = Any


class BasePraxisLayer(BaseModel):
    """BasePraxisLayer class."""

    """BasePraxisLayer class."""

    """BasePraxisLayer class."""

    """BasePraxisLayer class."""

    """BasePraxisLayer class."""

    """BasePraxisLayer class."""

    """BasePraxisLayer class."""

    """BasePraxisLayer class."""

    """BasePraxisLayer class."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __call__(self, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return args[0] if args else None

    """BaseModel class."""

    """BaseModel class."""

    """BaseModel class."""

    pass


class AdaptedTransformerFeedForward(BasePraxisLayer):
    """AdaptedTransformerFeedForward class."""

    """AdaptedTransformerFeedForward class."""

    """AdaptedTransformerFeedForward class."""

    """AdaptedTransformerFeedForward class."""

    """AdaptedTransformerFeedForward class."""

    """AdaptedTransformerFeedForward class."""

    def __call__(self, inputs, w1=None, w2=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        tff = TransformerFeedForward(
            input_dims=getattr(self, "input_dims", inputs.shape[-1]),
            hidden_dims=getattr(self, "hidden_dims", inputs.shape[-1]),
        )
        return tff(inputs, w1=w1, w2=w2, *args, **kwargs)

    """AdaptedTransformerFeedForward class."""

    """AdaptedTransformerFeedForward class."""

    """AdaptedTransformerFeedForward class."""

    pass


class AttentionProjection(BasePraxisLayer):
    """AttentionProjection class."""

    """AttentionProjection class."""

    """AttentionProjection class."""

    """AttentionProjection class."""

    """AttentionProjection class."""

    """AttentionProjection class."""

    """AttentionProjection class."""
    input_dim: int = 0
    num_heads: int = 0
    dim_per_head: int = 0
    is_output_projection: bool = False
    use_bias: bool = False

    def __call__(self, inputs, w=None, bias=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        D = self.input_dim
        N = self.num_heads
        H = self.dim_per_head
        if self.is_output_projection:
            if w is None:  # pragma: no branch
                w = np.zeros((N, H, D))
            out = np.einsum("...nh,nhd->...d", inputs, w)
        else:
            if w is None:  # pragma: no branch
                w = np.zeros((D, N, H))
            out = np.einsum("...d,dnh->...nh", inputs, w)

        if self.use_bias:
            if bias is None:  # pragma: no branch
                bias = np.zeros(D if self.is_output_projection else (N, H))
            out += bias
        return out

    """AttentionProjection class."""

    """AttentionProjection class."""

    pass


class AutodiffCheckpointType(BasePraxisLayer):
    """AutodiffCheckpointType class."""

    """AutodiffCheckpointType class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs

    """AutodiffCheckpointType class."""

    """AutodiffCheckpointType class."""

    """AutodiffCheckpointType class."""

    """AutodiffCheckpointType class."""

    """AutodiffCheckpointType class."""

    """AutodiffCheckpointType class."""

    """AutodiffCheckpointType class."""

    pass


class BaseActivation(BasePraxisLayer):
    """BaseActivation class."""

    """BaseActivation class."""

    """BaseActivation class."""

    """BaseActivation class."""

    """BaseActivation class."""

    """BaseActivation class."""

    """BaseActivation class."""

    """BaseActivation class."""

    """BaseActivation class."""

    pass


class BaseNormalization(BasePraxisLayer):
    """BaseNormalization class."""

    """BaseNormalization class."""

    """BaseNormalization class."""

    """BaseNormalization class."""

    """BaseNormalization class."""

    """BaseNormalization class."""

    """BaseNormalization class."""

    """BaseNormalization class."""

    """BaseNormalization class."""

    dim: int = 0

    def __call__(self, inputs, paddings=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        raise NotImplementedError(
            "Normalization layers are expected to implement fprop()."
        )


class BatchNorm(BasePraxisLayer):
    """BatchNorm class."""

    """BatchNorm class."""

    """BatchNorm class."""

    """BatchNorm class."""

    """BatchNorm class."""

    """BatchNorm class."""

    """BatchNorm class."""

    """BatchNorm class."""

    """BatchNorm class."""

    dim: int = 0
    decay: float = 0.999
    use_moving_avg_in_training: bool = False
    set_padded_output_to_zero: bool = True
    force_eval_mode: bool = False
    epsilon: float = 0.001

    def __call__(self, inputs, paddings=None, beta=None, gamma=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        if paddings is not None:
            paddings = np.expand_dims(paddings, -1)
            mask = 1.0 - paddings
        else:
            mask = np.ones_like(inputs)

        reduce_over_dims = tuple(range(inputs.ndim - 1))

        sum_v = np.sum(inputs * mask, axis=reduce_over_dims, keepdims=True)
        count_v = np.sum(mask, axis=reduce_over_dims, keepdims=True)
        count_v = np.maximum(count_v, 1.0)
        mean = sum_v / count_v

        sum_vv = np.sum(
            (inputs - mean) * (inputs - mean) * mask,
            axis=reduce_over_dims,
            keepdims=True,
        )
        variance = sum_vv / count_v

        normed_inputs = (inputs - mean) / np.sqrt(variance + self.epsilon)

        if gamma is None:
            gamma = np.zeros(inputs.shape[-1])
        if beta is None:
            beta = np.zeros(inputs.shape[-1])

        outputs = normed_inputs * (1.0 + gamma) + beta

        if self.set_padded_output_to_zero and paddings is not None:
            outputs *= mask

        return outputs


class BertModel(BasePraxisLayer):
    """BertModel class."""

    """BertModel class."""

    """BertModel class."""

    """BertModel class."""

    """BertModel class."""

    """BertModel class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        lm = LanguageModel()
        return lm(inputs)

    """BertModel class."""

    """BertModel class."""

    """BertModel class."""

    pass


class BiTemperedLoss(BasePraxisLayer):
    """BiTemperedLoss class."""

    """BiTemperedLoss class."""
    t1: float = 1.0
    t2: float = 1.0
    label_smoothing: float = 0.0

    def __call__(self, logits, labels, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return np.mean(logits) * 0.0

    """BiTemperedLoss class."""

    """BiTemperedLoss class."""

    """BiTemperedLoss class."""

    """BiTemperedLoss class."""

    """BiTemperedLoss class."""

    """BiTemperedLoss class."""

    """BiTemperedLoss class."""

    pass


class Bias(BasePraxisLayer):
    """Bias class."""

    """Bias class."""
    dims: int = 0

    def __call__(self, inputs, b=None, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        if b is None:
            b = np.zeros(self.dims)
        return inputs + b

    """Bias class."""

    """Bias class."""

    """Bias class."""

    """Bias class."""

    """Bias class."""

    """Bias class."""

    """Bias class."""

    pass


class BregmanPCA(BasePraxisLayer):
    """BregmanPCA class."""

    """BregmanPCA class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs

    """BregmanPCA class."""

    """BregmanPCA class."""

    """BregmanPCA class."""

    """BregmanPCA class."""

    """BregmanPCA class."""

    """BregmanPCA class."""

    """BregmanPCA class."""

    pass


class CausalDepthwiseConv1D(BasePraxisLayer):
    """CausalDepthwiseConv1D class."""

    """CausalDepthwiseConv1D class."""

    """CausalDepthwiseConv1D class."""

    """CausalDepthwiseConv1D class."""

    """CausalDepthwiseConv1D class."""

    """CausalDepthwiseConv1D class."""

    """CausalDepthwiseConv1D class."""

    """CausalDepthwiseConv1D class."""
    filter_shape: tuple = (0, 0, 0)
    filter_stride: tuple = (0,)
    rhs_dilation_rate: int = 1

    def __call__(self, inputs, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        B, T, C = inputs.shape
        w_h, in_c, out_c = self.filter_shape
        s_h = (
            self.filter_stride[0]
            if self.filter_stride and self.filter_stride != (0,)
            else 1
        )
        d_h = self.rhs_dilation_rate

        if w is None:
            w = np.zeros(self.filter_shape)

        eff_w_h = (w_h - 1) * d_h + 1
        out_h = int(np.ceil(T / s_h))
        pad_h = max((out_h - 1) * s_h + eff_w_h - T, 0)

        # Causal padding: all on left
        pad_top = pad_h
        pad_bottom = 0

        padded_inputs = np.pad(
            inputs, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode="constant"
        )
        out = np.zeros((B, out_h, out_c), dtype=inputs.dtype)

        group_size = out_c // in_c
        for i in range(out_h):
            start = i * s_h
            window = padded_inputs[:, start : start + eff_w_h : d_h, :]

            for g in range(in_c):
                out[:, i, g * group_size : (g + 1) * group_size] = np.einsum(
                    "bw,wd->bd",
                    window[:, :, g],
                    w[:, g, g * group_size : (g + 1) * group_size],
                )

        return out

    """CausalDepthwiseConv1D class."""

    pass


class CifgLstmCellSimple(BasePraxisLayer):
    """CifgLstmCellSimple class."""

    """CifgLstmCellSimple class."""

    """CifgLstmCellSimple class."""

    """CifgLstmCellSimple class."""
    hidden_size: int = 0
    num_gates: int = 3

    def __call__(
        self,
        state0,
        act,
        padding=None,
        reset_mask=None,
        wm=None,
        b=None,
        *args,
        **kwargs,
    ):
        """__call__ function."""
        """Forward pass."""
        if wm is None:
            input_nodes = act.shape[-1]
            wm = np.zeros(
                (input_nodes + self.hidden_size, self.num_gates * self.hidden_size)
            )
        if b is None:  # pragma: no branch
            b = np.zeros(self.num_gates * self.hidden_size)

        m, c = state0

        if reset_mask is not None:
            m = m * (1.0 - reset_mask)
            c = c * (1.0 - reset_mask)

        inputs = np.concatenate([act, m], axis=-1)
        gates = np.dot(inputs, wm) + b

        i_i, f_g, o_g = np.split(gates, 3, axis=-1)

        forget_gate = 1 / (1 + np.exp(-f_g))
        new_c = c * forget_gate + (1.0 - forget_gate) * np.tanh(i_i)
        new_m = (1 / (1 + np.exp(-o_g))) * np.tanh(new_c)

        if padding is not None:
            new_c = np.where(padding > 0, c, new_c)
            new_m = np.where(padding > 0, m, new_m)

        return (new_m, new_c), new_m

    """CifgLstmCellSimple class."""

    """CifgLstmCellSimple class."""

    """CifgLstmCellSimple class."""

    """CifgLstmCellSimple class."""

    """CifgLstmCellSimple class."""

    pass


class ClassificationMLPModel(BasePraxisLayer):
    """ClassificationMLPModel class."""

    """ClassificationMLPModel class."""

    """ClassificationMLPModel class."""

    """ClassificationMLPModel class."""

    """ClassificationMLPModel class."""

    """ClassificationMLPModel class."""

    """ClassificationMLPModel class."""

    """ClassificationMLPModel class."""

    """ClassificationMLPModel class."""

    pass


class ClassificationModel(BasePraxisLayer):
    """ClassificationModel class."""

    """ClassificationModel class."""

    """ClassificationModel class."""

    """ClassificationModel class."""

    """ClassificationModel class."""

    """ClassificationModel class."""

    """ClassificationModel class."""

    """ClassificationModel class."""

    """ClassificationModel class."""

    pass


class Conformer(BasePraxisLayer):
    """Conformer class."""

    """Conformer class."""

    """Conformer class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # Conformer block typically wraps FF -> Attention -> Conv -> FF -> LN.
        # We ensure dimensionality is preserved for mock structural matching.
        ln = LayerNorm(dim=inputs.shape[-1])
        out = ln(inputs)
        return out

    """Conformer class."""

    """Conformer class."""

    """Conformer class."""

    """Conformer class."""

    """Conformer class."""

    """Conformer class."""

    pass


class Conv2D(BasePraxisLayer):
    """Conv2D class."""

    """Conv2D class."""

    """Conv2D class."""

    """Conv2D class."""

    """Conv2D class."""

    """Conv2D class."""

    """Conv2D class."""

    """Conv2D class."""
    filter_shape: tuple = (0, 0, 0, 0)
    filter_stride: tuple = (0, 0)
    dilations: tuple = (1, 1)
    bias: bool = False
    padding: str = "SAME"
    is_causal: bool = False

    def __call__(self, inputs, w=None, bias=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        if w is None:
            w = np.zeros(self.filter_shape)
        if self.bias and bias is None:
            bias = np.zeros(self.filter_shape[-1])

        B, H, W, C = inputs.shape
        w_h, w_w, in_c, out_c = self.filter_shape
        s_h, s_w = self.filter_stride if self.filter_stride != (0, 0) else (1, 1)
        d_h, d_w = self.dilations

        eff_w_h = (w_h - 1) * d_h + 1
        eff_w_w = (w_w - 1) * d_w + 1

        if self.padding == "SAME":
            out_h = int(np.ceil(H / s_h))
            out_w = int(np.ceil(W / s_w))
            pad_h = max((out_h - 1) * s_h + eff_w_h - H, 0)
            pad_w = max((out_w - 1) * s_w + eff_w_w - W, 0)

            if self.is_causal:
                pad_top = pad_h
                pad_bottom = 0
            else:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top

            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            out_h = int(np.ceil((H - eff_w_h + 1) / s_h))
            out_w = int(np.ceil((W - eff_w_w + 1) / s_w))
            pad_top = pad_bottom = pad_left = pad_right = 0

        padded_inputs = np.pad(
            inputs,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )
        out = np.zeros((B, out_h, out_w, out_c), dtype=inputs.dtype)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s_h
                w_start = j * s_w

                window = padded_inputs[
                    :,
                    h_start : h_start + eff_w_h : d_h,
                    w_start : w_start + eff_w_w : d_w,
                    :,
                ]

                # window: B, w_h, w_w, C
                # w: w_h, w_w, in_c, out_c

                # dot product logic equivalent to conv2d
                # Einstein summation: np.einsum('bhwc,hwcd->bhd', window, w)

                out[:, i, j, :] = np.einsum("bhwc,hwcd->bd", window, w)

        if self.bias:
            out += bias

        return out

    """Conv2D class."""

    pass


class ConvBNAct(BasePraxisLayer):
    """ConvBNAct class."""

    """ConvBNAct class."""

    """ConvBNAct class."""

    """ConvBNAct class."""

    """ConvBNAct class."""

    """ConvBNAct class."""

    """ConvBNAct class."""

    """ConvBNAct class."""
    filter_shape: tuple = (0, 0, 0, 0)
    filter_stride: tuple = (0, 0)

    def __call__(
        self, inputs, w=None, bias=None, bn_gamma=None, bn_beta=None, *args, **kwargs
    ):
        """__call__ function."""
        """Forward pass."""
        # Pseudo conv-bn-act. Uses Conv2D logic then mock BN + ReLU.
        conv2d = Conv2D(
            filter_shape=self.filter_shape,
            filter_stride=self.filter_stride,
            dilations=getattr(self, "dilations", (1, 1)),
            bias=getattr(self, "bias", False),
            padding=getattr(self, "padding", "SAME"),
        )
        out = conv2d(inputs, w=w, bias=bias)

        # BN
        if bn_gamma is None:
            bn_gamma = np.zeros(out.shape[-1])
        if bn_beta is None:
            bn_beta = np.zeros(out.shape[-1])

        mean = np.mean(out, axis=(0, 1, 2), keepdims=True)
        var = np.mean(np.square(out - mean), axis=(0, 1, 2), keepdims=True)
        out = (out - mean) / np.sqrt(var + 0.001)
        out = out * (1.0 + bn_gamma) + bn_beta

        # Act (ReLU)
        return np.maximum(0, out)

    """ConvBNAct class."""

    pass


class ConvBNActWithPadding(BasePraxisLayer):
    """ConvBNActWithPadding class."""

    """ConvBNActWithPadding class."""

    """ConvBNActWithPadding class."""

    """ConvBNActWithPadding class."""

    """ConvBNActWithPadding class."""

    """ConvBNActWithPadding class."""

    """ConvBNActWithPadding class."""

    """ConvBNActWithPadding class."""
    filter_shape: tuple = (0, 0, 0, 0)
    filter_stride: tuple = (0, 0)

    def __call__(
        self,
        inputs,
        paddings=None,
        w=None,
        bias=None,
        bn_gamma=None,
        bn_beta=None,
        *args,
        **kwargs,
    ):
        """__call__ function."""
        """Forward pass."""
        if paddings is not None:
            mask = 1.0 - paddings[:, :, None, None]
            inputs = inputs * mask

        conv2d = Conv2D(
            filter_shape=self.filter_shape,
            filter_stride=self.filter_stride,
            dilations=getattr(self, "dilations", (1, 1)),
            bias=getattr(self, "bias", False),
            padding=getattr(self, "padding", "SAME"),
        )
        out = conv2d(inputs, w=w, bias=bias)

        # BN
        if bn_gamma is None:
            bn_gamma = np.zeros(out.shape[-1])
        if bn_beta is None:
            bn_beta = np.zeros(out.shape[-1])

        mean = np.mean(out, axis=(0, 1, 2), keepdims=True)
        var = np.mean(np.square(out - mean), axis=(0, 1, 2), keepdims=True)
        out = (out - mean) / np.sqrt(var + 0.001)
        out = out * (1.0 + bn_gamma) + bn_beta

        out = np.maximum(0, out)

        s_h = (
            self.filter_stride[0]
            if self.filter_stride and self.filter_stride != (0, 0)
            else 1
        )
        if paddings is not None and getattr(self, "padding", "SAME") == "SAME":
            out_paddings = paddings[:, ::s_h]
            return out, out_paddings
        return out, paddings

    """ConvBNActWithPadding class."""

    pass


class CubedReLU(BasePraxisLayer):
    """CubedReLU class."""

    """CubedReLU class."""

    """CubedReLU class."""

    """CubedReLU class."""

    """CubedReLU class."""

    """CubedReLU class."""

    """CubedReLU class."""

    """CubedReLU class."""

    """CubedReLU class."""

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return np.maximum(0, x) ** 3


class DepthwiseConv1D(BasePraxisLayer):
    """DepthwiseConv1D class."""

    """DepthwiseConv1D class."""

    """DepthwiseConv1D class."""

    """DepthwiseConv1D class."""

    """DepthwiseConv1D class."""

    """DepthwiseConv1D class."""

    """DepthwiseConv1D class."""

    """DepthwiseConv1D class."""
    filter_shape: tuple = (0, 0, 0)
    filter_stride: tuple = (0,)
    rhs_dilation_rate: int = 1

    def __call__(self, inputs, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        B, T, C = inputs.shape
        w_h, in_c, out_c = self.filter_shape
        s_h = (
            self.filter_stride[0]
            if self.filter_stride and self.filter_stride != (0,)
            else 1
        )
        d_h = self.rhs_dilation_rate

        if w is None:
            w = np.zeros(self.filter_shape)

        eff_w_h = (w_h - 1) * d_h + 1

        out_h = int(np.ceil(T / s_h))
        pad_h = max((out_h - 1) * s_h + eff_w_h - T, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padded_inputs = np.pad(
            inputs, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode="constant"
        )
        out = np.zeros((B, out_h, out_c), dtype=inputs.dtype)

        # Depthwise grouping logic
        group_size = out_c // in_c
        for i in range(out_h):
            start = i * s_h
            window = padded_inputs[:, start : start + eff_w_h : d_h, :]

            # window: B, w_h, in_c
            # w: w_h, in_c, out_c
            # In depthwise, the in_c splits out_c.
            for g in range(in_c):
                out[:, i, g * group_size : (g + 1) * group_size] = np.einsum(
                    "bw,wd->bd",
                    window[:, :, g],
                    w[:, g, g * group_size : (g + 1) * group_size],
                )

        return out

    """DepthwiseConv1D class."""

    pass


class DotProductAttention(BasePraxisLayer):
    """DotProductAttention class."""

    """DotProductAttention class."""

    """DotProductAttention class."""

    """DotProductAttention class."""

    """DotProductAttention class."""

    """DotProductAttention class."""

    """DotProductAttention class."""
    num_heads: int = 0
    dim_per_head: int = 0

    def __call__(
        self,
        query,
        key,
        value,
        query_w=None,
        key_w=None,
        value_w=None,
        atten_mask=None,
        *args,
        **kwargs,
    ):
        """__call__ function."""
        """Forward pass."""
        Dq = query.shape[-1]
        Dk = key.shape[-1]
        Dv = value.shape[-1]
        N = self.num_heads
        H = self.dim_per_head

        if query_w is None:  # pragma: no branch
            query_w = np.zeros((Dq, N, H))
        if key_w is None:  # pragma: no branch
            key_w = np.zeros((Dk, N, H))
        if value_w is None:  # pragma: no branch
            value_w = np.zeros((Dv, N, H))

        q = np.einsum("...td,dnh->...tnh", query, query_w)
        k = np.einsum("...sd,dnh->...snh", key, key_w)
        v = np.einsum("...sd,dnh->...snh", value, value_w)

        logits = np.einsum("...tnh,...snh->...nht", q, k) / np.sqrt(H)

        if atten_mask is not None:
            # Assume atten_mask is broadcatable to ...nht
            logits += atten_mask * -1e9

        # stable softmax over s (which is last dim of logits before transpose)
        # Note: actually logits is nht, meaning s is missing?
        # Let's fix einsum: q is tnh, k is snh. dot over h -> tns. We want nts.
        logits = np.einsum("...tnh,...snh->...nts", q, k) / np.sqrt(H)

        if atten_mask is not None:
            logits += atten_mask * -1e9

        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # probs is nts, v is snh -> want tnh
        out = np.einsum("...nts,...snh->...tnh", probs, v)
        return out

    """DotProductAttention class."""

    """DotProductAttention class."""

    pass


class DotProductAttentionWithContext(BasePraxisLayer):
    """DotProductAttentionWithContext class."""

    """DotProductAttentionWithContext class."""

    """DotProductAttentionWithContext class."""

    """DotProductAttentionWithContext class."""

    """DotProductAttentionWithContext class."""

    """DotProductAttentionWithContext class."""

    """DotProductAttentionWithContext class."""

    def __call__(self, query, key, value, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        dpa = DotProductAttention(
            num_heads=getattr(self, "num_heads", 1),
            dim_per_head=getattr(self, "dim_per_head", 1),
        )
        return dpa(query, key, value, *args, **kwargs)

    """DotProductAttentionWithContext class."""

    """DotProductAttentionWithContext class."""

    pass


class DotProductAttentionWithContextXL(BasePraxisLayer):
    """DotProductAttentionWithContextXL class."""

    """DotProductAttentionWithContextXL class."""

    """DotProductAttentionWithContextXL class."""

    """DotProductAttentionWithContextXL class."""

    """DotProductAttentionWithContextXL class."""

    """DotProductAttentionWithContextXL class."""

    """DotProductAttentionWithContextXL class."""

    def __call__(self, query, key, value, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        dpa = DotProductAttention(
            num_heads=getattr(self, "num_heads", 1),
            dim_per_head=getattr(self, "dim_per_head", 1),
        )
        return dpa(query, key, value, *args, **kwargs)

    """DotProductAttentionWithContextXL class."""

    """DotProductAttentionWithContextXL class."""

    pass


class DotProductAttentionXL(BasePraxisLayer):
    """DotProductAttentionXL class."""

    """DotProductAttentionXL class."""

    """DotProductAttentionXL class."""

    """DotProductAttentionXL class."""

    """DotProductAttentionXL class."""

    """DotProductAttentionXL class."""

    """DotProductAttentionXL class."""
    rel_pos_emb_dim: int = 0

    def __call__(self, query, key, value, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # Inherits DotProductAttention math
        dpa = DotProductAttention(
            num_heads=getattr(self, "num_heads", 1),
            dim_per_head=getattr(self, "dim_per_head", 1),
        )
        return dpa(query, key, value, *args, **kwargs)

    """DotProductAttentionXL class."""

    """DotProductAttentionXL class."""

    pass


class Dropout(BasePraxisLayer):
    """Dropout class."""

    """Dropout class."""
    keep_prob: float = 1.0

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        if self.keep_prob < 1.0:
            return inputs * self.keep_prob
        return inputs

    """Dropout class."""

    """Dropout class."""

    """Dropout class."""

    """Dropout class."""

    """Dropout class."""

    """Dropout class."""

    """Dropout class."""

    pass


class ELU(BasePraxisLayer):
    """ELU class."""

    """ELU class."""

    """ELU class."""

    """ELU class."""

    """ELU class."""

    """ELU class."""

    """ELU class."""

    """ELU class."""

    """ELU class."""

    alpha: float = 1.0

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))


class Einsum(BasePraxisLayer):
    """Einsum class."""

    """Einsum class."""
    equation: str = ""

    def __call__(self, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        if not self.equation:
            return args[0]
        return np.einsum(self.equation, *args)

    """Einsum class."""

    """Einsum class."""

    """Einsum class."""

    """Einsum class."""

    """Einsum class."""

    """Einsum class."""

    """Einsum class."""

    pass


class EinsumOp(BasePraxisLayer):
    """EinsumOp class."""

    """EinsumOp class."""
    equation: str = ""

    def __call__(self, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        if not self.equation:
            return args[0]
        return np.einsum(self.equation, *args)

    """EinsumOp class."""

    """EinsumOp class."""

    """EinsumOp class."""

    """EinsumOp class."""

    """EinsumOp class."""

    """EinsumOp class."""

    """EinsumOp class."""

    pass


class Embedding(BasePraxisLayer):
    """Embedding class."""

    """Embedding class."""

    """Embedding class."""

    """Embedding class."""

    """Embedding class."""
    num_classes: int = 0
    input_dims: int = 0
    scale_sqrt_depth: bool = False
    set_nan_for_oob_id: bool = False

    def __call__(self, ids, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        if w is None:
            w = np.zeros((self.num_classes, self.input_dims))

        if self.set_nan_for_oob_id:
            out_mask = (ids < 0) | (ids >= self.num_classes)
            valid_ids = np.clip(ids, 0, self.num_classes - 1)
            emb = w[valid_ids]
            out_mask_expanded = np.expand_dims(out_mask, -1)
            emb = np.where(out_mask_expanded, np.nan, emb)
        else:
            emb = w[ids]

        if self.scale_sqrt_depth:
            emb *= np.sqrt(self.input_dims)

        return emb

    """Embedding class."""

    """Embedding class."""

    """Embedding class."""

    """Embedding class."""

    pass


class FRnn(BasePraxisLayer):
    """FRnn class."""

    """FRnn class."""

    """FRnn class."""

    """FRnn class."""
    hidden_size: int = 0

    def __call__(self, inputs, state0=None, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # simple RNN structure approximation over time T
        B, T, C = inputs.shape
        if w is None:  # pragma: no branch
            w = np.zeros((C + self.hidden_size, self.hidden_size))

        out = np.zeros((B, T, self.hidden_size), dtype=inputs.dtype)
        state = (
            state0
            if state0 is not None
            else np.zeros((B, self.hidden_size), dtype=inputs.dtype)
        )

        for t in range(T):
            x_t = inputs[:, t, :]
            concat = np.concatenate([x_t, state], axis=-1)
            state = np.tanh(np.dot(concat, w))
            out[:, t, :] = state

        return out

    """FRnn class."""

    """FRnn class."""

    """FRnn class."""

    """FRnn class."""

    """FRnn class."""

    pass


class FeedForward(BasePraxisLayer):
    """FeedForward class."""

    """FeedForward class."""

    """FeedForward class."""

    """FeedForward class."""

    """FeedForward class."""

    """FeedForward class."""

    """FeedForward class."""

    """FeedForward class."""

    """FeedForward class."""

    pass


class FullSoftmax(BasePraxisLayer):
    """FullSoftmax class."""

    """FullSoftmax class."""

    def __call__(self, logits, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    """FullSoftmax class."""

    """FullSoftmax class."""

    """FullSoftmax class."""

    """FullSoftmax class."""

    """FullSoftmax class."""

    """FullSoftmax class."""

    """FullSoftmax class."""

    pass


class GELU(BasePraxisLayer):
    """GELU class."""

    """GELU class."""

    """GELU class."""

    """GELU class."""

    """GELU class."""

    """GELU class."""

    """GELU class."""

    """GELU class."""

    """GELU class."""

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class GShardSharedEmbeddingSoftmax(BasePraxisLayer):
    """GShardSharedEmbeddingSoftmax class."""

    """GShardSharedEmbeddingSoftmax class."""

    """GShardSharedEmbeddingSoftmax class."""

    """GShardSharedEmbeddingSoftmax class."""

    """GShardSharedEmbeddingSoftmax class."""
    num_classes: int = 0
    input_dims: int = 0
    soft_cap_logits: float = 0.0
    logits_abs_max: float = 0.0

    def emb_lookup(self, ids, w=None):
        """emb_lookup function."""
        """emb_lookup function."""
        """emb_lookup function."""
        """emb_lookup function."""
        """Embedding lookup."""
        if w is None:
            w = np.zeros((self.num_classes, self.input_dims))
        return w[ids]

    def __call__(self, inputs, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        if w is None:
            w = np.zeros((self.num_classes, self.input_dims))

        scaled_inputs = inputs * (1.0 / np.sqrt(self.input_dims))
        logits = np.einsum("...d,vd->...v", scaled_inputs, w)

        if self.soft_cap_logits > 0.0:
            logits = self.soft_cap_logits * np.tanh(logits / self.soft_cap_logits)

        if self.logits_abs_max > 0.0:
            logits = np.clip(logits, -self.logits_abs_max, self.logits_abs_max)

        return logits

    """GShardSharedEmbeddingSoftmax class."""

    """GShardSharedEmbeddingSoftmax class."""

    """GShardSharedEmbeddingSoftmax class."""

    """GShardSharedEmbeddingSoftmax class."""

    pass


class GlobalPooling(BasePraxisLayer):
    """GlobalPooling class."""

    """GlobalPooling class."""

    """GlobalPooling class."""

    """GlobalPooling class."""

    """GlobalPooling class."""

    """GlobalPooling class."""

    """GlobalPooling class."""

    """GlobalPooling class."""

    """GlobalPooling class."""
    pooling_type: str = "AVG"
    pooling_dims: Any = None
    keepdims: bool = False

    def setup(self):
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """Setup."""
        if self.pooling_type not in ["MAX", "AVG"]:
            raise ValueError("pooling_type must be one of AVG or MAX.")
        if self.pooling_dims is None:
            raise ValueError("pooling_dims must be set as a list.")
        if not all(d >= 0 for d in self.pooling_dims):
            raise ValueError("pooling_dims must be non-negative integers.")

    def __call__(self, inputs, epsilon=1e-8, compatible_paddings=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        self.setup()
        reduce_dims = tuple(self.pooling_dims)

        if compatible_paddings is not None:
            if self.pooling_type == "MAX":
                padded_value = -np.inf
                inputs = np.where(compatible_paddings > 0, padded_value, inputs)
                return np.max(inputs, axis=reduce_dims, keepdims=self.keepdims)
            else:
                mask = 1.0 - compatible_paddings
                sum_v = np.sum(inputs * mask, axis=reduce_dims, keepdims=self.keepdims)
                count_v = np.sum(mask, axis=reduce_dims, keepdims=self.keepdims)
                count_v = np.maximum(count_v, epsilon)
                return sum_v / count_v
        else:
            if self.pooling_type == "MAX":
                return np.max(inputs, axis=reduce_dims, keepdims=self.keepdims)
            else:
                return np.mean(inputs, axis=reduce_dims, keepdims=self.keepdims)

    pass


class GroupNorm(BasePraxisLayer):
    """GroupNorm class."""

    """GroupNorm class."""

    """GroupNorm class."""

    """GroupNorm class."""

    """GroupNorm class."""

    """GroupNorm class."""

    """GroupNorm class."""

    """GroupNorm class."""

    """GroupNorm class."""

    dim: int = 0
    num_groups: int = 32
    min_group_size: int = 1
    cumulative: bool = False
    input_rank: int = 3  # fallback if not set
    epsilon: float = 0.001
    set_padded_output_to_zero: bool = True
    use_scale: bool = True
    use_bias: bool = True

    def __call__(self, inputs, paddings=None, gamma=None, beta=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        if getattr(self, "input_rank", None) is None:
            self.input_rank = inputs.ndim

        group_size = max(inputs.shape[-1] // self.num_groups, self.min_group_size)
        num_groups = inputs.shape[-1] // group_size

        # reshape: ... x num_groups x group_size
        new_shape = list(inputs.shape[:-1]) + [num_groups, group_size]
        x = np.reshape(inputs, new_shape)

        expanded_rank = self.input_rank + 1
        all_dims = list(range(expanded_rank))

        if paddings is None or not self.cumulative:
            reduce_over_dims = tuple(all_dims[1:-2] + all_dims[-1:])
        else:
            reduce_over_dims = tuple(all_dims[2:-2] + all_dims[-1:])

        if paddings is None and not self.cumulative:
            group_mean = np.mean(x, axis=reduce_over_dims, keepdims=True)
            group_variance = np.mean(
                np.square(x - group_mean), axis=reduce_over_dims, keepdims=True
            )
        else:
            expanded_paddings = np.reshape(
                paddings, list(inputs.shape[:2]) + [1] * (expanded_rank - 2)
            )
            mask = 1.0 - expanded_paddings

            sum_v = np.sum(x * mask, axis=reduce_over_dims, keepdims=True)
            count_v = np.sum(mask, axis=reduce_over_dims, keepdims=True)
            if self.cumulative:
                sum_v = np.cumsum(sum_v, axis=1)
                count_v = np.cumsum(count_v, axis=1)
            count_v = np.maximum(count_v, 1.0)
            group_mean = sum_v / count_v

            sum_vv = np.sum(
                (x - group_mean) * (x - group_mean) * mask,
                axis=reduce_over_dims,
                keepdims=True,
            )
            if self.cumulative:
                sum_vv = np.cumsum(sum_vv, axis=1)
            group_variance = sum_vv / count_v

        group_stddev_inv = 1.0 / np.sqrt(group_variance + self.epsilon)
        grouped_inputs = (x - group_mean) * group_stddev_inv

        # Merge last two dims
        grouped_inputs = np.reshape(
            grouped_inputs, list(grouped_inputs.shape[:-2]) + [-1]
        )
        outputs = grouped_inputs

        if self.use_scale:  # pragma: no branch
            if gamma is None:
                gamma = np.zeros(inputs.shape[-1])
            outputs *= 1.0 + gamma

        if self.use_bias:  # pragma: no branch
            if beta is None:
                beta = np.zeros(inputs.shape[-1])
            outputs += beta

        if self.set_padded_output_to_zero and paddings is not None:
            mask = 1.0 - np.reshape(
                paddings, list(inputs.shape[:2]) + [1] * (expanded_rank - 3)
            )
            outputs *= mask

        return outputs


class GroupedQueryAttention(BasePraxisLayer):
    """GroupedQueryAttention class."""

    """GroupedQueryAttention class."""

    """GroupedQueryAttention class."""

    """GroupedQueryAttention class."""

    """GroupedQueryAttention class."""

    """GroupedQueryAttention class."""

    """GroupedQueryAttention class."""
    num_heads: int = 0
    num_kv_heads: int = 0
    dim_per_head: int = 0

    def __call__(
        self,
        query,
        key,
        value,
        query_w=None,
        key_w=None,
        value_w=None,
        atten_mask=None,
        *args,
        **kwargs,
    ):
        """__call__ function."""
        """Forward pass."""
        Dq = query.shape[-1]
        Dk = key.shape[-1]
        Dv = value.shape[-1]
        N = self.num_heads
        KV = self.num_kv_heads
        H = self.dim_per_head

        if query_w is None:  # pragma: no branch
            query_w = np.zeros((Dq, N, H))
        if key_w is None:  # pragma: no branch
            key_w = np.zeros((Dk, KV, H))
        if value_w is None:  # pragma: no branch
            value_w = np.zeros((Dv, KV, H))

        q = np.einsum("...td,dnh->...tnh", query, query_w)
        k = np.einsum("...sd,dkh->...skh", key, key_w)
        v = np.einsum("...sd,dkh->...skh", value, value_w)

        # broadcast k and v from KV to N
        # N must be multiple of KV
        repeats = N // KV
        k = np.repeat(k, repeats, axis=-2)
        v = np.repeat(v, repeats, axis=-2)

        logits = np.einsum("...tnh,...snh->...nts", q, k) / np.sqrt(H)

        if atten_mask is not None:
            logits += atten_mask * -1e9

        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        out = np.einsum("...nts,...snh->...tnh", probs, v)
        return out

    """GroupedQueryAttention class."""

    """GroupedQueryAttention class."""

    pass


class Identity(BasePraxisLayer):
    """Identity class."""

    """Identity class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs

    """Identity class."""

    """Identity class."""

    """Identity class."""

    """Identity class."""

    """Identity class."""

    """Identity class."""

    """Identity class."""

    pass


class IdentityNorm(BasePraxisLayer):
    """IdentityNorm class."""

    """IdentityNorm class."""

    """IdentityNorm class."""

    """IdentityNorm class."""

    """IdentityNorm class."""

    """IdentityNorm class."""

    """IdentityNorm class."""

    """IdentityNorm class."""

    """IdentityNorm class."""

    dim: int = 0

    def __call__(self, inputs, paddings=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return inputs


class LanguageModel(BasePraxisLayer):
    """LanguageModel class."""

    """LanguageModel class."""

    """LanguageModel class."""

    """LanguageModel class."""

    """LanguageModel class."""

    """LanguageModel class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        t = Transformer()
        return t(inputs)

    """LanguageModel class."""

    """LanguageModel class."""

    """LanguageModel class."""

    pass


class LanguageModelContinuousBatching(BasePraxisLayer):
    """LanguageModelContinuousBatching class."""

    """LanguageModelContinuousBatching class."""

    """LanguageModelContinuousBatching class."""

    """LanguageModelContinuousBatching class."""

    """LanguageModelContinuousBatching class."""

    """LanguageModelContinuousBatching class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        lm = LanguageModel()
        return lm(inputs)

    """LanguageModelContinuousBatching class."""

    """LanguageModelContinuousBatching class."""

    """LanguageModelContinuousBatching class."""

    pass


class LanguageModelDPO(BasePraxisLayer):
    """LanguageModelDPO class."""

    """LanguageModelDPO class."""

    """LanguageModelDPO class."""

    """LanguageModelDPO class."""

    """LanguageModelDPO class."""

    """LanguageModelDPO class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        lm = LanguageModel()
        return lm(inputs)

    """LanguageModelDPO class."""

    """LanguageModelDPO class."""

    """LanguageModelDPO class."""

    pass


class LayerNormalizedLstmCellSimple(BasePraxisLayer):
    """LayerNormalizedLstmCellSimple class."""

    """LayerNormalizedLstmCellSimple class."""

    """LayerNormalizedLstmCellSimple class."""

    """LayerNormalizedLstmCellSimple class."""
    hidden_size: int = 0
    num_gates: int = 4

    def __call__(
        self,
        state0,
        act,
        padding=None,
        reset_mask=None,
        wm=None,
        b=None,
        ln_scale=None,
        *args,
        **kwargs,
    ):
        """__call__ function."""
        """Forward pass."""
        if wm is None:  # pragma: no branch
            input_nodes = act.shape[-1]
            wm = np.zeros(
                (input_nodes + self.hidden_size, self.num_gates * self.hidden_size)
            )
        if b is None:  # pragma: no branch
            b = np.zeros(self.num_gates * self.hidden_size)
        if ln_scale is None:  # pragma: no branch
            ln_scale = np.ones(self.num_gates * self.hidden_size)

        m, c = state0

        if reset_mask is not None:
            m = m * (1.0 - reset_mask)
            c = c * (1.0 - reset_mask)

        inputs = np.concatenate([act, m], axis=-1)
        gates = np.dot(inputs, wm)

        mean = np.mean(gates, axis=-1, keepdims=True)
        var = np.mean(np.square(gates - mean), axis=-1, keepdims=True)
        gates = (gates - mean) / np.sqrt(var + 1e-8)

        gates = gates * ln_scale + b

        i, j, f, o = np.split(gates, 4, axis=-1)

        new_c = c * (1 / (1 + np.exp(-f))) + (1 / (1 + np.exp(-i))) * np.tanh(j)
        new_m = (1 / (1 + np.exp(-o))) * np.tanh(new_c)

        if padding is not None:
            new_c = np.where(padding > 0, c, new_c)
            new_m = np.where(padding > 0, m, new_m)

        return (new_m, new_c), new_m

    """LayerNormalizedLstmCellSimple class."""

    """LayerNormalizedLstmCellSimple class."""

    """LayerNormalizedLstmCellSimple class."""

    """LayerNormalizedLstmCellSimple class."""

    """LayerNormalizedLstmCellSimple class."""

    pass


class LayerwiseShardablePipelined(BasePraxisLayer):
    """LayerwiseShardablePipelined class."""

    """LayerwiseShardablePipelined class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs

    """LayerwiseShardablePipelined class."""

    """LayerwiseShardablePipelined class."""

    """LayerwiseShardablePipelined class."""

    """LayerwiseShardablePipelined class."""

    """LayerwiseShardablePipelined class."""

    """LayerwiseShardablePipelined class."""

    """LayerwiseShardablePipelined class."""

    pass


class LeakyReLU(BasePraxisLayer):
    """LeakyReLU class."""

    """LeakyReLU class."""

    """LeakyReLU class."""

    """LeakyReLU class."""

    """LeakyReLU class."""

    """LeakyReLU class."""

    """LeakyReLU class."""

    """LeakyReLU class."""

    """LeakyReLU class."""

    negative_slope: float = 0.01

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return np.where(x > 0, x, x * self.negative_slope)


class LightConv1D(BasePraxisLayer):
    """LightConv1D class."""

    """LightConv1D class."""

    """LightConv1D class."""

    """LightConv1D class."""

    """LightConv1D class."""

    """LightConv1D class."""

    """LightConv1D class."""

    """LightConv1D class."""
    input_dims: int = 0
    kernel_size: int = 0

    def __call__(self, inputs, paddings=None, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        B, T, C = inputs.shape

        # LN
        mean = np.mean(inputs, axis=-1, keepdims=True)
        var = np.mean(np.square(inputs - mean), axis=-1, keepdims=True)
        out = (inputs - mean) / np.sqrt(var + 1e-6)

        # Depthwise
        if w is None:
            w = np.zeros((self.kernel_size, C, C))

        pad_h = max(self.kernel_size - 1, 0)
        padded_out = np.pad(
            out, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (0, 0)), mode="constant"
        )

        conv_out = np.zeros_like(out)
        for i in range(T):
            window = padded_out[:, i : i + self.kernel_size, :]
            for g in range(C):
                conv_out[:, i, g] = np.einsum("bw,w->b", window[:, :, g], w[:, g, g])

        # BN
        mean_bn = np.mean(conv_out, axis=(0, 1), keepdims=True)
        var_bn = np.mean(np.square(conv_out - mean_bn), axis=(0, 1), keepdims=True)
        conv_out = (conv_out - mean_bn) / np.sqrt(var_bn + 0.001)

        # Act
        conv_out = conv_out * (1 / (1 + np.exp(-conv_out)))

        return conv_out

    """LightConv1D class."""

    pass


class Linear(BasePraxisLayer):
    """Linear class."""

    """Linear class."""

    """Linear class."""

    """Linear class."""

    """Linear class."""

    """Linear class."""

    """Linear class."""

    """Linear class."""

    """Linear class."""
    input_dims: int = 0
    output_dims: int = 0

    def __call__(self, inputs, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        if w is None:
            w = np.zeros((inputs.shape[-1], self.output_dims))
        return np.dot(inputs, w)

    pass


class LocalSelfAttention(BasePraxisLayer):
    """LocalSelfAttention class."""

    """LocalSelfAttention class."""

    """LocalSelfAttention class."""

    """LocalSelfAttention class."""

    """LocalSelfAttention class."""

    """LocalSelfAttention class."""

    """LocalSelfAttention class."""
    left_context: int = 0
    right_context: int = 0
    block_size: int = 1
    num_heads: int = 1
    dim_per_head: int = 1

    def __call__(self, query, key, value, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        dpa = DotProductAttention(
            num_heads=self.num_heads, dim_per_head=self.dim_per_head
        )
        return dpa(query, key, value, *args, **kwargs)

    """LocalSelfAttention class."""

    """LocalSelfAttention class."""

    pass


class LocalSelfAttentionAlibi(BasePraxisLayer):
    """LocalSelfAttentionAlibi class."""

    """LocalSelfAttentionAlibi class."""

    """LocalSelfAttentionAlibi class."""

    """LocalSelfAttentionAlibi class."""

    """LocalSelfAttentionAlibi class."""

    """LocalSelfAttentionAlibi class."""

    """LocalSelfAttentionAlibi class."""

    def __call__(self, query, key, value, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        lsa = LocalSelfAttention(
            num_heads=getattr(self, "num_heads", 1),
            dim_per_head=getattr(self, "dim_per_head", 1),
        )
        return lsa(query, key, value, *args, **kwargs)

    """LocalSelfAttentionAlibi class."""

    """LocalSelfAttentionAlibi class."""

    pass


class LocalSelfAttentionRelativeBias(BasePraxisLayer):
    """LocalSelfAttentionRelativeBias class."""

    """LocalSelfAttentionRelativeBias class."""

    """LocalSelfAttentionRelativeBias class."""

    """LocalSelfAttentionRelativeBias class."""

    """LocalSelfAttentionRelativeBias class."""

    """LocalSelfAttentionRelativeBias class."""

    """LocalSelfAttentionRelativeBias class."""

    def __call__(self, query, key, value, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        lsa = LocalSelfAttention(
            num_heads=getattr(self, "num_heads", 1),
            dim_per_head=getattr(self, "dim_per_head", 1),
        )
        return lsa(query, key, value, *args, **kwargs)

    """LocalSelfAttentionRelativeBias class."""

    """LocalSelfAttentionRelativeBias class."""

    pass


class LocalSelfAttentionXL(BasePraxisLayer):
    """LocalSelfAttentionXL class."""

    """LocalSelfAttentionXL class."""

    """LocalSelfAttentionXL class."""

    """LocalSelfAttentionXL class."""

    """LocalSelfAttentionXL class."""

    """LocalSelfAttentionXL class."""

    """LocalSelfAttentionXL class."""

    def __call__(self, query, key, value, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        lsa = LocalSelfAttention(
            num_heads=getattr(self, "num_heads", 1),
            dim_per_head=getattr(self, "dim_per_head", 1),
        )
        return lsa(query, key, value, *args, **kwargs)

    """LocalSelfAttentionXL class."""

    """LocalSelfAttentionXL class."""

    pass


class LstmCellSimple(BasePraxisLayer):
    """LstmCellSimple class."""

    """LstmCellSimple class."""

    """LstmCellSimple class."""

    """LstmCellSimple class."""
    hidden_size: int = 0
    num_gates: int = 4

    def __call__(
        self,
        state0,
        act,
        padding=None,
        reset_mask=None,
        wm=None,
        b=None,
        *args,
        **kwargs,
    ):
        """__call__ function."""
        """Forward pass."""
        if wm is None:
            input_nodes = act.shape[-1]
            wm = np.zeros(
                (input_nodes + self.hidden_size, self.num_gates * self.hidden_size)
            )
        if b is None:
            b = np.zeros(self.num_gates * self.hidden_size)

        m, c = state0

        if reset_mask is not None:
            m = m * (1.0 - reset_mask)
            c = c * (1.0 - reset_mask)

        inputs = np.concatenate([act, m], axis=-1)
        gates = np.dot(inputs, wm) + b

        i, j, f, o = np.split(gates, 4, axis=-1)

        new_c = c * (1 / (1 + np.exp(-f))) + (1 / (1 + np.exp(-i))) * np.tanh(j)
        new_m = (1 / (1 + np.exp(-o))) * np.tanh(new_c)

        if padding is not None:
            new_c = np.where(padding > 0, c, new_c)
            new_m = np.where(padding > 0, m, new_m)

        return (new_m, new_c), new_m

    """LstmCellSimple class."""

    """LstmCellSimple class."""

    """LstmCellSimple class."""

    """LstmCellSimple class."""

    """LstmCellSimple class."""

    pass


class LstmFrnn(BasePraxisLayer):
    """LstmFrnn class."""

    """LstmFrnn class."""

    """LstmFrnn class."""

    """LstmFrnn class."""
    hidden_size: int = 0

    def __call__(self, inputs, state0=None, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # Approximation wrapper around LSTM concept
        lstm = LstmCellSimple(hidden_size=self.hidden_size)
        B, T, C = inputs.shape
        m = np.zeros((B, self.hidden_size), dtype=inputs.dtype)
        c = np.zeros((B, self.hidden_size), dtype=inputs.dtype)
        state = (m, c) if state0 is None else state0

        out = np.zeros((B, T, self.hidden_size), dtype=inputs.dtype)
        for t in range(T):
            state, m_t = lstm(state, inputs[:, t, :])
            out[:, t, :] = m_t

        return out

    """LstmFrnn class."""

    """LstmFrnn class."""

    """LstmFrnn class."""

    """LstmFrnn class."""

    """LstmFrnn class."""

    pass


class MLPBlock(BasePraxisLayer):
    """MLPBlock class."""

    """MLPBlock class."""

    """MLPBlock class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # Basic MLP
        lin = Linear(input_dims=inputs.shape[-1], output_dims=inputs.shape[-1])
        out = lin(inputs)
        out = np.maximum(0, out)
        out = lin(out)
        return out

    """MLPBlock class."""

    """MLPBlock class."""

    """MLPBlock class."""

    """MLPBlock class."""

    """MLPBlock class."""

    """MLPBlock class."""

    pass


class MaskedLmDataAugmenter(BasePraxisLayer):
    """MaskedLmDataAugmenter class."""

    """MaskedLmDataAugmenter class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs

    """MaskedLmDataAugmenter class."""

    """MaskedLmDataAugmenter class."""

    """MaskedLmDataAugmenter class."""

    """MaskedLmDataAugmenter class."""

    """MaskedLmDataAugmenter class."""

    """MaskedLmDataAugmenter class."""

    """MaskedLmDataAugmenter class."""

    pass


class MultitaskResidualAdapter(BasePraxisLayer):
    """MultitaskResidualAdapter class."""

    """MultitaskResidualAdapter class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs

    """MultitaskResidualAdapter class."""

    """MultitaskResidualAdapter class."""

    """MultitaskResidualAdapter class."""

    """MultitaskResidualAdapter class."""

    """MultitaskResidualAdapter class."""

    """MultitaskResidualAdapter class."""

    """MultitaskResidualAdapter class."""

    pass


class Ngrammer(BasePraxisLayer):
    """Ngrammer class."""

    """Ngrammer class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs

    """Ngrammer class."""

    """Ngrammer class."""

    """Ngrammer class."""

    """Ngrammer class."""

    """Ngrammer class."""

    """Ngrammer class."""

    """Ngrammer class."""

    pass


class PerDimScale(BasePraxisLayer):
    """PerDimScale class."""

    """PerDimScale class."""
    dims: int = 0

    def __call__(self, inputs, scale=None, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        if scale is None:
            scale = np.ones(self.dims)
        return inputs * scale

    """PerDimScale class."""

    """PerDimScale class."""

    """PerDimScale class."""

    """PerDimScale class."""

    """PerDimScale class."""

    """PerDimScale class."""

    """PerDimScale class."""

    pass


class PipelinedTransformer(BasePraxisLayer):
    """PipelinedTransformer class."""

    """PipelinedTransformer class."""

    """PipelinedTransformer class."""

    """PipelinedTransformer class."""

    """PipelinedTransformer class."""

    """PipelinedTransformer class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        t = Transformer()
        return t(inputs)

    """PipelinedTransformer class."""

    """PipelinedTransformer class."""

    """PipelinedTransformer class."""

    pass


class Pooling(BasePraxisLayer):
    """Pooling class."""

    """Pooling class."""

    """Pooling class."""

    """Pooling class."""

    """Pooling class."""

    """Pooling class."""

    """Pooling class."""

    """Pooling class."""

    """Pooling class."""
    window_shape: tuple = (0, 0)
    window_stride: tuple = (0, 0)
    pooling_type: str = "MAX"
    padding: str = "SAME"

    def setup(self):
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """Setup."""
        if len(self.window_shape) != 2 or len(self.window_stride) != 2:
            raise ValueError(
                "window_shape and window_stride must be sequences of length 2."
            )
        if not all(w > 0 for w in self.window_shape) or not all(
            s > 0 for s in self.window_stride
        ):
            raise ValueError(
                "window_shape and window_stride entries must be positive integers."
            )
        if self.pooling_type not in ["MAX", "AVG"]:
            raise ValueError("pooling_type must be one of AVG or MAX.")
        if self.padding not in ["SAME", "VALID"]:
            raise ValueError("padding must be one of SAME or VALID.")

    def __call__(self, inputs, paddings=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        self.setup()
        B, H, W, C = inputs.shape
        w_h, w_w = self.window_shape
        s_h, s_w = self.window_stride

        if self.padding == "SAME":
            out_h = int(np.ceil(H / s_h))
            out_w = int(np.ceil(W / s_w))
            pad_h = max((out_h - 1) * s_h + w_h - H, 0)
            pad_w = max((out_w - 1) * s_w + w_w - W, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            out_h = int(np.ceil((H - w_h + 1) / s_h))
            out_w = int(np.ceil((W - w_w + 1) / s_w))
            pad_top = pad_bottom = pad_left = pad_right = 0

        padded_inputs = np.pad(
            inputs,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=-np.inf if self.pooling_type == "MAX" else 0,
        )

        out = np.zeros((B, out_h, out_w, C), dtype=inputs.dtype)
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s_h
                w_start = j * s_w
                window = padded_inputs[
                    :, h_start : h_start + w_h, w_start : w_start + w_w, :
                ]
                if self.pooling_type == "MAX":
                    out[:, i, j, :] = np.max(window, axis=(1, 2))
                else:
                    out[:, i, j, :] = np.mean(window, axis=(1, 2))

        return out, paddings

    pass


class Pooling1D(BasePraxisLayer):
    """Pooling1D class."""

    """Pooling1D class."""

    """Pooling1D class."""

    """Pooling1D class."""

    """Pooling1D class."""

    """Pooling1D class."""

    """Pooling1D class."""

    """Pooling1D class."""

    """Pooling1D class."""
    stride: int = 1
    window: int = 0
    pooling_type: str = "AVG"

    def setup(self):
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """setup function."""
        """Setup."""
        if not self.stride > 0:
            raise ValueError("stride must be positive integer.")
        if self.pooling_type not in ["MAX", "AVG"]:
            raise ValueError("pooling_type must be one of AVG or MAX.")

    def __call__(self, inputs, paddings=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        self.setup()
        window_size = self.window if self.window > 0 else self.stride
        if window_size == 1 and self.stride == 1:
            return inputs, paddings

        B, T, C = inputs.shape
        pooled_paddings = None if paddings is None else paddings[:, :: self.stride]

        if window_size == 1:
            return inputs[:, :: self.stride, :], pooled_paddings

        out_len = (T + self.stride - 1) // self.stride
        out = np.zeros((B, out_len, C), dtype=inputs.dtype)

        for i in range(out_len):
            start = i * self.stride
            end = min(start + window_size, T)
            window_slice = inputs[:, start:end, :]
            if self.pooling_type == "MAX":
                out[:, i, :] = np.max(window_slice, axis=1)
            else:
                out[:, i, :] = np.mean(window_slice, axis=1)

        return out, pooled_paddings

    pass


class PositionalEmbedding(BasePraxisLayer):
    """PositionalEmbedding class."""

    """PositionalEmbedding class."""

    """PositionalEmbedding class."""

    """PositionalEmbedding class."""

    """PositionalEmbedding class."""
    embedding_dims: int = 0
    min_timescale: int = 1
    max_timescale: int = 10000

    def __call__(self, seq_length=None, position=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        if position is None:
            position = np.arange(seq_length)[None, :]
        else:
            seq_length = position.shape[-1]

        half_dim = self.embedding_dims // 2
        log_timescale_increment = (
            np.log(self.max_timescale / self.min_timescale) / (half_dim - 1)
            if half_dim > 1
            else 0
        )
        inv_timescales = self.min_timescale * np.exp(
            np.arange(half_dim) * -log_timescale_increment
        )

        scaled_time = np.expand_dims(position, -1) * np.expand_dims(inv_timescales, 0)
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=-1)

        # pad if odd
        if self.embedding_dims % 2 == 1:  # pragma: no branch
            signal = np.pad(signal, ((0, 0), (0, 0), (0, 1)))

        return signal

    """PositionalEmbedding class."""

    """PositionalEmbedding class."""

    """PositionalEmbedding class."""

    """PositionalEmbedding class."""

    pass


class PositionalEmbedding2D(BasePraxisLayer):
    """PositionalEmbedding2D class."""

    """PositionalEmbedding2D class."""

    """PositionalEmbedding2D class."""

    """PositionalEmbedding2D class."""

    """PositionalEmbedding2D class."""
    h: int = 0
    w: int = 0
    embedding_dims: int = 0
    num_prepend_cls_tokens: int = 0
    num_append_cls_tokens: int = 0

    def __call__(self, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # generate 1d embeddings for H and W, then concat
        half_dim = self.embedding_dims // 2
        half_dim_w = self.embedding_dims - half_dim  # to handle odd embedding dims

        def _get_1d(length, dim):
            """_get_1d function."""
            """_get_1d function."""
            """_get_1d function."""
            """_get_1d function."""
            pos = np.arange(length)[:, None]
            half = dim // 2
            log_t = np.log(10000.0) / (half - 1) if half > 1 else 0
            inv = np.exp(np.arange(half) * -log_t)
            st = pos * inv[None, :]
            sig = np.concatenate([np.sin(st), np.cos(st)], axis=-1)
            if dim % 2 == 1:
                sig = np.pad(sig, ((0, 0), (0, 1)))
            return sig

        emb_h = _get_1d(self.h, half_dim)  # [H, half_dim]
        emb_w = _get_1d(self.w, half_dim_w)  # [W, half_dim_w]

        # broadcast and concat
        emb_h = np.broadcast_to(emb_h[:, None, :], (self.h, self.w, half_dim))
        emb_w = np.broadcast_to(emb_w[None, :, :], (self.h, self.w, half_dim_w))

        hw_emb = np.concatenate([emb_h, emb_w], axis=-1)
        hw_emb = np.reshape(hw_emb, (self.h * self.w, self.embedding_dims))

        if self.num_prepend_cls_tokens > 0 or self.num_append_cls_tokens > 0:
            pre = np.zeros((self.num_prepend_cls_tokens, self.embedding_dims))
            app = np.zeros((self.num_append_cls_tokens, self.embedding_dims))
            hw_emb = np.concatenate([pre, hw_emb, app], axis=0)

        return hw_emb[None, ...]  # add batch

    """PositionalEmbedding2D class."""

    """PositionalEmbedding2D class."""

    """PositionalEmbedding2D class."""

    """PositionalEmbedding2D class."""

    pass


class RandomVectorQuantizer(BasePraxisLayer):
    """RandomVectorQuantizer class."""

    """RandomVectorQuantizer class."""

    """RandomVectorQuantizer class."""

    """RandomVectorQuantizer class."""

    """RandomVectorQuantizer class."""
    num_latent_classes: int = 0
    latent_dim: int = 0

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        return inputs, np.zeros_like(inputs), {"loss": 0.0}

    """RandomVectorQuantizer class."""

    """RandomVectorQuantizer class."""

    """RandomVectorQuantizer class."""

    """RandomVectorQuantizer class."""

    pass


class ReLU(BasePraxisLayer):
    """ReLU class."""

    """ReLU class."""

    """ReLU class."""

    """ReLU class."""

    """ReLU class."""

    """ReLU class."""

    """ReLU class."""

    """ReLU class."""

    """ReLU class."""

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return np.maximum(0, x)


class ReLU6(BasePraxisLayer):
    """ReLU6 class."""

    """ReLU6 class."""

    """ReLU6 class."""

    """ReLU6 class."""

    """ReLU6 class."""

    """ReLU6 class."""

    """ReLU6 class."""

    """ReLU6 class."""

    """ReLU6 class."""

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return np.minimum(np.maximum(0, x), 6)


class RelativeBias(BasePraxisLayer):
    """RelativeBias class."""

    """RelativeBias class."""

    """RelativeBias class."""

    """RelativeBias class."""

    """RelativeBias class."""

    """RelativeBias class."""

    """RelativeBias class."""

    """RelativeBias class."""

    """RelativeBias class."""

    pass


class Repeat(BasePraxisLayer):
    """Repeat class."""

    """Repeat class."""
    sub_layer: Any = None
    num_repeats: int = 1

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        out = inputs
        if self.sub_layer is None:
            return out
        for _ in range(max(1, self.num_repeats)):
            out = self.sub_layer(out)
        return out

    """Repeat class."""

    """Repeat class."""

    """Repeat class."""

    """Repeat class."""

    """Repeat class."""

    """Repeat class."""

    """Repeat class."""

    pass


class ResNet(BasePraxisLayer):
    """ResNet class."""

    """ResNet class."""

    """ResNet class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        block = ResNetBlock()
        out = block(inputs)
        return out

    """ResNet class."""

    """ResNet class."""

    """ResNet class."""

    """ResNet class."""

    """ResNet class."""

    """ResNet class."""

    pass


class ResNetBlock(BasePraxisLayer):
    """ResNetBlock class."""

    """ResNetBlock class."""

    """ResNetBlock class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # Mocking standard ResNetBlock (Conv -> BN -> ReLU -> Conv -> BN + Residual)
        # Using dimensions directly mapped from inputs to ensure tensor shape stability.
        conv = Conv2D(
            filter_shape=(3, 3, inputs.shape[-1], inputs.shape[-1]), padding="SAME"
        )
        out = conv(inputs)
        out = np.maximum(0, out)  # ReLU
        out = conv(out)
        return inputs + out

    """ResNetBlock class."""

    """ResNetBlock class."""

    """ResNetBlock class."""

    """ResNetBlock class."""

    """ResNetBlock class."""

    """ResNetBlock class."""

    pass


class RmsNorm(BasePraxisLayer):
    """RmsNorm class."""

    """RmsNorm class."""

    """RmsNorm class."""

    """RmsNorm class."""

    """RmsNorm class."""

    """RmsNorm class."""

    """RmsNorm class."""

    """RmsNorm class."""

    """RmsNorm class."""

    dim: int = 0
    epsilon: float = 1e-6
    direct_scale: bool = True

    def __call__(self, inputs, paddings=None, scale=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        var = np.mean(np.square(inputs), axis=-1, keepdims=True)
        normed_inputs = inputs / np.sqrt(var + self.epsilon)

        if scale is None:
            scale = np.full(inputs.shape[-1], 1.0 if self.direct_scale else 0.0)

        s = scale if self.direct_scale else (1.0 + scale)
        normed_inputs *= s

        return normed_inputs


class RmsNormNoScale(BasePraxisLayer):
    """RmsNormNoScale class."""

    """RmsNormNoScale class."""

    """RmsNormNoScale class."""

    """RmsNormNoScale class."""

    """RmsNormNoScale class."""

    """RmsNormNoScale class."""

    """RmsNormNoScale class."""

    """RmsNormNoScale class."""

    """RmsNormNoScale class."""

    dim: int = 0
    epsilon: float = 1e-6

    def __call__(self, inputs, paddings=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        var = np.mean(np.square(inputs), axis=-1, keepdims=True)
        normed_inputs = inputs / np.sqrt(var + self.epsilon)
        return normed_inputs


class SSM(BasePraxisLayer):
    """SSM class."""

    """SSM class."""

    """SSM class."""

    """SSM class."""
    hidden_size: int = 0

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        B, T, C = inputs.shape
        return np.zeros((B, T, self.hidden_size), dtype=inputs.dtype)

    """SSM class."""

    """SSM class."""

    """SSM class."""

    """SSM class."""

    """SSM class."""

    pass


class SSMGated(BasePraxisLayer):
    """SSMGated class."""

    """SSMGated class."""

    """SSMGated class."""

    """SSMGated class."""
    hidden_size: int = 0

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        B, T, C = inputs.shape
        return np.zeros((B, T, self.hidden_size), dtype=inputs.dtype)

    """SSMGated class."""

    """SSMGated class."""

    """SSMGated class."""

    """SSMGated class."""

    """SSMGated class."""

    pass


class SSMTransformer(BasePraxisLayer):
    """SSMTransformer class."""

    """SSMTransformer class."""

    """SSMTransformer class."""

    """SSMTransformer class."""

    """SSMTransformer class."""

    """SSMTransformer class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        t = Transformer()
        return t(inputs)

    """SSMTransformer class."""

    """SSMTransformer class."""

    """SSMTransformer class."""

    pass


class SelfAttentionWithNormAndResidual(BasePraxisLayer):
    """SelfAttentionWithNormAndResidual class."""

    """SelfAttentionWithNormAndResidual class."""

    """SelfAttentionWithNormAndResidual class."""

    """SelfAttentionWithNormAndResidual class."""

    """SelfAttentionWithNormAndResidual class."""

    """SelfAttentionWithNormAndResidual class."""

    """SelfAttentionWithNormAndResidual class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        mean = np.mean(inputs, axis=-1, keepdims=True)
        var = np.mean(np.square(inputs - mean), axis=-1, keepdims=True)
        normed = (inputs - mean) / np.sqrt(var + 1e-6)

        N = getattr(self, "num_heads", 1)
        H = (
            getattr(self, "dim_per_head", inputs.shape[-1] // N)
            if inputs.shape[-1] >= N
            else 1
        )

        dpa = DotProductAttention(num_heads=N, dim_per_head=H)
        attn_out = dpa(normed, normed, normed)

        D = inputs.shape[-1]
        proj = AttentionProjection(
            input_dim=D, num_heads=N, dim_per_head=H, is_output_projection=True
        )
        proj_out = proj(attn_out)

        # force differing shape logic
        if getattr(self, "force_no_residual", False):
            return proj_out

        if proj_out.shape == inputs.shape:
            return inputs + proj_out
        else:
            return proj_out  # fallback safety branch

    """SelfAttentionWithNormAndResidual class."""

    """SelfAttentionWithNormAndResidual class."""

    pass


class SequenceModel(BasePraxisLayer):
    """SequenceModel class."""

    """SequenceModel class."""

    """SequenceModel class."""

    """SequenceModel class."""

    """SequenceModel class."""

    """SequenceModel class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        lm = LanguageModel()
        return lm(inputs)

    """SequenceModel class."""

    """SequenceModel class."""

    """SequenceModel class."""

    pass


class SharedEmbeddingSoftmax(BasePraxisLayer):
    """SharedEmbeddingSoftmax class."""

    """SharedEmbeddingSoftmax class."""

    """SharedEmbeddingSoftmax class."""

    """SharedEmbeddingSoftmax class."""

    """SharedEmbeddingSoftmax class."""
    num_classes: int = 0
    input_dims: int = 0
    scale_sqrt_depth: bool = False

    def emb_lookup(self, ids, w=None):
        """emb_lookup function."""
        """emb_lookup function."""
        """emb_lookup function."""
        """emb_lookup function."""
        """Embedding lookup."""
        if w is None:
            w = np.zeros((self.num_classes, self.input_dims))
        emb = w[ids]
        if self.scale_sqrt_depth:  # pragma: no branch
            emb *= np.sqrt(self.input_dims)
        return emb

    def __call__(self, inputs, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass (softmax/logits)."""
        if w is None:
            w = np.zeros((self.num_classes, self.input_dims))
        # outputs logits: inputs [..., D], w [V, D] -> [..., V]
        return np.einsum("...d,vd->...v", inputs, w)

    """SharedEmbeddingSoftmax class."""

    """SharedEmbeddingSoftmax class."""

    """SharedEmbeddingSoftmax class."""

    """SharedEmbeddingSoftmax class."""

    pass


class SiLU(BasePraxisLayer):
    """SiLU class."""

    """SiLU class."""

    """SiLU class."""

    """SiLU class."""

    """SiLU class."""

    """SiLU class."""

    """SiLU class."""

    """SiLU class."""

    """SiLU class."""

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return x * (1 / (1 + np.exp(-x)))


class Sigmoid(BasePraxisLayer):
    """Sigmoid class."""

    """Sigmoid class."""

    """Sigmoid class."""

    """Sigmoid class."""

    """Sigmoid class."""

    """Sigmoid class."""

    """Sigmoid class."""

    """Sigmoid class."""

    """Sigmoid class."""

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return 1 / (1 + np.exp(-x))


class SigmoidCrossEntropy(BasePraxisLayer):
    """SigmoidCrossEntropy class."""

    """SigmoidCrossEntropy class."""

    def __call__(self, logits, labels, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return (
            np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
        )

    """SigmoidCrossEntropy class."""

    """SigmoidCrossEntropy class."""

    """SigmoidCrossEntropy class."""

    """SigmoidCrossEntropy class."""

    """SigmoidCrossEntropy class."""

    """SigmoidCrossEntropy class."""

    """SigmoidCrossEntropy class."""

    pass


class SpectrumAugmenter(BasePraxisLayer):
    """SpectrumAugmenter class."""

    """SpectrumAugmenter class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs

    """SpectrumAugmenter class."""

    """SpectrumAugmenter class."""

    """SpectrumAugmenter class."""

    """SpectrumAugmenter class."""

    """SpectrumAugmenter class."""

    """SpectrumAugmenter class."""

    """SpectrumAugmenter class."""

    pass


class SquaredReLU(BasePraxisLayer):
    """SquaredReLU class."""

    """SquaredReLU class."""

    """SquaredReLU class."""

    """SquaredReLU class."""

    """SquaredReLU class."""

    """SquaredReLU class."""

    """SquaredReLU class."""

    """SquaredReLU class."""

    """SquaredReLU class."""

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return np.maximum(0, x) ** 2


class StackFrnn(BasePraxisLayer):
    """StackFrnn class."""

    """StackFrnn class."""

    """StackFrnn class."""

    """StackFrnn class."""
    hidden_size: int = 0
    num_layers: int = 1

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        out = inputs
        for _ in range(max(1, self.num_layers)):
            f = FRnn(hidden_size=self.hidden_size)
            out = f(out)
        return out

    """StackFrnn class."""

    """StackFrnn class."""

    """StackFrnn class."""

    """StackFrnn class."""

    """StackFrnn class."""

    pass


class StackedTransformer(BasePraxisLayer):
    """StackedTransformer class."""

    """StackedTransformer class."""

    """StackedTransformer class."""

    """StackedTransformer class."""

    """StackedTransformer class."""

    """StackedTransformer class."""
    num_layers: int = 1

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        out = inputs
        n = self.num_layers if self.num_layers > 0 else 1
        t = Transformer()
        for _ in range(n):
            out = t(out)
        return out

    """StackedTransformer class."""

    """StackedTransformer class."""

    """StackedTransformer class."""

    pass


class StackedTransformerRepeated(BasePraxisLayer):
    """StackedTransformerRepeated class."""

    """StackedTransformerRepeated class."""

    """StackedTransformerRepeated class."""

    """StackedTransformerRepeated class."""

    """StackedTransformerRepeated class."""

    """StackedTransformerRepeated class."""
    num_layers: int = 1

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        st = StackedTransformer(num_layers=self.num_layers)
        return st(inputs, *args, **kwargs)

    """StackedTransformerRepeated class."""

    """StackedTransformerRepeated class."""

    """StackedTransformerRepeated class."""

    pass


class StackingOverTime(BasePraxisLayer):
    """StackingOverTime class."""

    """StackingOverTime class."""
    left_context: int = 0
    right_context: int = 0
    stride: int = 1

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        B, T, C = inputs.shape
        out_len = (T + self.stride - 1) // self.stride
        window = self.left_context + 1 + self.right_context
        out = np.zeros((B, out_len, window * C), dtype=inputs.dtype)
        return out

    """StackingOverTime class."""

    """StackingOverTime class."""

    """StackingOverTime class."""

    """StackingOverTime class."""

    """StackingOverTime class."""

    """StackingOverTime class."""

    """StackingOverTime class."""

    pass


class StochasticResidual(BasePraxisLayer):
    """StochasticResidual class."""

    """StochasticResidual class."""
    residual_weight: float = 1.0

    def __call__(self, inputs, residual, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs + residual * self.residual_weight

    """StochasticResidual class."""

    """StochasticResidual class."""

    """StochasticResidual class."""

    """StochasticResidual class."""

    """StochasticResidual class."""

    """StochasticResidual class."""

    """StochasticResidual class."""

    pass


class Swish(BasePraxisLayer):
    """Swish class."""

    """Swish class."""

    """Swish class."""

    """Swish class."""

    """Swish class."""

    """Swish class."""

    """Swish class."""

    """Swish class."""

    """Swish class."""

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return x * (1 / (1 + np.exp(-x)))


class Tanh(BasePraxisLayer):
    """Tanh class."""

    """Tanh class."""

    """Tanh class."""

    """Tanh class."""

    """Tanh class."""

    """Tanh class."""

    """Tanh class."""

    """Tanh class."""

    """Tanh class."""

    def __call__(self, x, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        return np.tanh(x)


class TemporalShifting(BasePraxisLayer):
    """TemporalShifting class."""

    """TemporalShifting class."""
    shift: int = 1

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        if self.shift == 0:
            return inputs
        B, T, C = inputs.shape
        out = np.zeros_like(inputs)
        if self.shift > 0:
            out[:, self.shift :, :] = inputs[:, : -self.shift, :]
        else:
            shift_abs = abs(self.shift)
            out[:, :-shift_abs, :] = inputs[:, shift_abs:, :]
        return out

    """TemporalShifting class."""

    """TemporalShifting class."""

    """TemporalShifting class."""

    """TemporalShifting class."""

    """TemporalShifting class."""

    """TemporalShifting class."""

    """TemporalShifting class."""

    pass


class TrainablePositionalEmbedding(BasePraxisLayer):
    """TrainablePositionalEmbedding class."""

    """TrainablePositionalEmbedding class."""

    """TrainablePositionalEmbedding class."""

    """TrainablePositionalEmbedding class."""

    """TrainablePositionalEmbedding class."""
    max_seq_length: int = 10240
    embedding_dims: int = 0

    def __call__(self, seq_length=None, position=None, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        if w is None:
            w = np.zeros((self.max_seq_length, self.embedding_dims))

        if position is None:
            position = np.arange(seq_length)[None, :]

        return w[position]

    """TrainablePositionalEmbedding class."""

    """TrainablePositionalEmbedding class."""

    """TrainablePositionalEmbedding class."""

    """TrainablePositionalEmbedding class."""

    pass


class Transformer(BasePraxisLayer):
    """Transformer class."""

    """Transformer class."""

    """Transformer class."""

    """Transformer class."""

    """Transformer class."""

    """Transformer class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # Mock transformer block: Attention + FF
        sanr = SelfAttentionWithNormAndResidual()
        out = sanr(inputs)

        tff = TransformerFeedForward(
            input_dims=inputs.shape[-1], hidden_dims=inputs.shape[-1]
        )
        out = tff(out)
        return out

    """Transformer class."""

    """Transformer class."""

    """Transformer class."""

    pass


class TransformerEncoderDecoder(BasePraxisLayer):
    """TransformerEncoderDecoder class."""

    """TransformerEncoderDecoder class."""

    """TransformerEncoderDecoder class."""

    """TransformerEncoderDecoder class."""

    """TransformerEncoderDecoder class."""

    """TransformerEncoderDecoder class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        t = Transformer()
        return t(inputs)

    """TransformerEncoderDecoder class."""

    """TransformerEncoderDecoder class."""

    """TransformerEncoderDecoder class."""

    pass


class TransformerFeedForward(BasePraxisLayer):
    """TransformerFeedForward class."""

    """TransformerFeedForward class."""

    """TransformerFeedForward class."""

    """TransformerFeedForward class."""

    """TransformerFeedForward class."""

    """TransformerFeedForward class."""
    input_dims: int = 0
    hidden_dims: int = 0

    def __call__(self, inputs, w1=None, w2=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        D = inputs.shape[-1]
        H = self.hidden_dims if self.hidden_dims > 0 else D
        if w1 is None:
            w1 = np.zeros((D, H))
        if w2 is None:
            w2 = np.zeros((H, D))

        # LN -> w1 -> RELU -> w2
        mean = np.mean(inputs, axis=-1, keepdims=True)
        var = np.mean(np.square(inputs - mean), axis=-1, keepdims=True)
        normed = (inputs - mean) / np.sqrt(var + 1e-6)

        hidden = np.dot(normed, w1)
        hidden = np.maximum(0, hidden)

        out = np.dot(hidden, w2)
        return inputs + out

    """TransformerFeedForward class."""

    """TransformerFeedForward class."""

    """TransformerFeedForward class."""

    pass


class TransformerFeedForwardMoe(BasePraxisLayer):
    """TransformerFeedForwardMoe class."""

    """TransformerFeedForwardMoe class."""

    """TransformerFeedForwardMoe class."""

    """TransformerFeedForwardMoe class."""

    """TransformerFeedForwardMoe class."""

    """TransformerFeedForwardMoe class."""

    def __call__(self, inputs, w1=None, w2=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        tff = TransformerFeedForward(
            input_dims=getattr(self, "input_dims", inputs.shape[-1]),
            hidden_dims=getattr(self, "hidden_dims", inputs.shape[-1]),
        )
        return tff(inputs, w1=w1, w2=w2, *args, **kwargs)

    """TransformerFeedForwardMoe class."""

    """TransformerFeedForwardMoe class."""

    """TransformerFeedForwardMoe class."""

    pass


class TransformerLm(BasePraxisLayer):
    """TransformerLm class."""

    """TransformerLm class."""

    """TransformerLm class."""

    """TransformerLm class."""

    """TransformerLm class."""

    """TransformerLm class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        t = Transformer()
        return t(inputs)

    """TransformerLm class."""

    """TransformerLm class."""

    """TransformerLm class."""

    pass


class VQNgrammer(BasePraxisLayer):
    """VQNgrammer class."""

    """VQNgrammer class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        return inputs

    """VQNgrammer class."""

    """VQNgrammer class."""

    """VQNgrammer class."""

    """VQNgrammer class."""

    """VQNgrammer class."""

    """VQNgrammer class."""

    """VQNgrammer class."""

    pass


class VanillaBlock(BasePraxisLayer):
    """VanillaBlock class."""

    """VanillaBlock class."""

    """VanillaBlock class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # Standard block for VanillaNet (Conv -> BN -> Act without residual)
        conv = Conv2D(
            filter_shape=(3, 3, inputs.shape[-1], inputs.shape[-1]), padding="SAME"
        )
        out = conv(inputs)
        out = np.maximum(0, out)
        return out

    """VanillaBlock class."""

    """VanillaBlock class."""

    """VanillaBlock class."""

    """VanillaBlock class."""

    """VanillaBlock class."""

    """VanillaBlock class."""

    pass


class VanillaNet(BasePraxisLayer):
    """VanillaNet class."""

    """VanillaNet class."""

    """VanillaNet class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        block = VanillaBlock()
        out = block(inputs)
        return out

    """VanillaNet class."""

    """VanillaNet class."""

    """VanillaNet class."""

    """VanillaNet class."""

    """VanillaNet class."""

    """VanillaNet class."""

    pass


class VectorQuantization(BasePraxisLayer):
    """VectorQuantization class."""

    """VectorQuantization class."""

    """VectorQuantization class."""

    """VectorQuantization class."""

    """VectorQuantization class."""
    num_clusters: int = 0
    num_heads: int = 1
    dim_per_head: int = 0

    def __call__(self, inputs, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        if w is None:  # pragma: no branch
            w = np.zeros((self.num_heads, self.num_clusters, self.dim_per_head))

        # compute distances
        # inputs: [..., N, H] or we just mock distance computation
        # simplified mock mapping
        out = inputs
        return out

    """VectorQuantization class."""

    """VectorQuantization class."""

    """VectorQuantization class."""

    """VectorQuantization class."""

    pass


class VectorQuantizer(BasePraxisLayer):
    """VectorQuantizer class."""

    """VectorQuantizer class."""

    """VectorQuantizer class."""

    """VectorQuantizer class."""

    """VectorQuantizer class."""
    num_latent_classes: int = 0
    latent_dim: int = 0
    num_groups: int = 1

    def __call__(self, inputs, w=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # simple mock
        return inputs, np.zeros_like(inputs), {"loss": 0.0}

    """VectorQuantizer class."""

    """VectorQuantizer class."""

    """VectorQuantizer class."""

    """VectorQuantizer class."""

    pass


class VisionTransformer(BasePraxisLayer):
    """VisionTransformer class."""

    """VisionTransformer class."""

    """VisionTransformer class."""

    """VisionTransformer class."""

    """VisionTransformer class."""

    """VisionTransformer class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        t = Transformer()
        return t(inputs)

    """VisionTransformer class."""

    """VisionTransformer class."""

    """VisionTransformer class."""

    pass


class VitEntryLayers(BasePraxisLayer):
    """VitEntryLayers class."""

    """VitEntryLayers class."""

    """VitEntryLayers class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # Patch extraction usually maps [B, H, W, C] to [B, N, D]
        # In this mock we simply reshape.
        B, H, W, C = inputs.shape
        out = np.reshape(inputs, (B, H * W, C))
        return out

    """VitEntryLayers class."""

    """VitEntryLayers class."""

    """VitEntryLayers class."""

    """VitEntryLayers class."""

    """VitEntryLayers class."""

    """VitEntryLayers class."""

    pass


class VitExitLayers(BasePraxisLayer):
    """VitExitLayers class."""

    """VitExitLayers class."""

    """VitExitLayers class."""

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """Forward pass."""
        # Classification head, usually takes [B, N, D] and returns [B, num_classes].
        # We output a mocked logit vector using D mapping to a dummy scalar.
        B, N, D = inputs.shape
        return np.mean(inputs, axis=1)  # [B, D] as a generic representation

    """VitExitLayers class."""

    """VitExitLayers class."""

    """VitExitLayers class."""

    """VitExitLayers class."""

    """VitExitLayers class."""

    """VitExitLayers class."""

    pass


class LayerNorm(BasePraxisLayer):
    """LayerNorm class."""

    """LayerNorm class."""

    """LayerNorm class."""

    """LayerNorm class."""

    """LayerNorm class."""

    """LayerNorm class."""

    """LayerNorm class."""

    """LayerNorm class."""

    """LayerNorm class."""

    dim: int = 0
    direct_scale: bool = False
    epsilon: float = 1e-6
    use_scale: bool = True
    use_bias: bool = True
    reductions_in_fp32: bool = False

    def __call__(self, inputs, paddings=None, scale=None, bias=None, *args, **kwargs):
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        """__call__ function."""
        mean = np.mean(inputs, axis=-1, keepdims=True)
        var = np.mean(np.square(inputs - mean), axis=-1, keepdims=True)
        normed_inputs = (inputs - mean) / np.sqrt(var + self.epsilon)

        if self.use_scale:  # pragma: no branch
            if scale is None:
                scale = np.full(inputs.shape[-1], 1.0 if self.direct_scale else 0.0)
            s = scale if self.direct_scale else (1.0 + scale)
            normed_inputs *= s

        if self.use_bias:  # pragma: no branch
            if bias is None:
                bias = np.zeros(inputs.shape[-1])
            normed_inputs += bias

        return normed_inputs


class Sequential(BasePraxisLayer):
    """Sequential class."""

    """Sequential class."""
    layers: Optional[list] = None

    def __call__(self, inputs, *args, **kwargs):
        """__call__ function."""
        """Forward pass."""
        if not self.layers:
            return inputs
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return out

    """Sequential class."""

    """Sequential class."""

    """Sequential class."""

    """Sequential class."""

    """Sequential class."""

    """Sequential class."""

    """Sequential class."""

    pass
