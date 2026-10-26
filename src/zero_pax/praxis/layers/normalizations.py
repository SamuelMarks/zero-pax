"""Normalization layers for zero_pax.praxis.layers."""

from __future__ import annotations

from zero_pax import _tensor_ops as _ops
from zero_pax.praxis.base import BasePraxisLayer
from zero_pax.types import DType, Tensor, TensorOrScalar, WeightInit


def _expand_paddings_to_rank(paddings: Tensor, target_ndim: int) -> Tensor:
    """Expand `paddings` with trailing singleton dimensions until its rank reaches `target_ndim`.

    Args:
        paddings: Input padding mask tensor.
        target_ndim: Target tensor rank (`ndim`).

    Returns:
        Expanded padding `Tensor` with `ndim >= target_ndim`.
    """
    expanded = paddings
    while expanded.ndim < target_ndim:
        expanded = _ops.expand_dims(expanded, -1)
    return expanded


class BaseNormalization(BasePraxisLayer):
    """Base class for all normalization layers.

    Attributes:
        dim: Feature dimension size to normalize.
    """

    dim: int = 0

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Raise NotImplementedError unless overridden by concrete normalization subclasses.

        Args:
            inputs: Input tensor to normalize.
            paddings: Optional padding mask tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Normalized `Tensor`.

        Raises:
            NotImplementedError: Always raised on the abstract base class.
        """
        raise NotImplementedError("Normalization layers are expected to implement fprop().")


class IdentityNorm(BaseNormalization):
    """Identity normalization layer returning inputs without modification."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Return `inputs` unchanged as a `Tensor`.

        Args:
            inputs: Input tensor.
            paddings: Optional padding mask tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Unchanged `Tensor`.
        """
        return self._to_tensor(inputs)


class BatchNorm(BaseNormalization):
    """Batch Normalization layer normalizing across batch and spatial dimensions.

    Attributes:
        decay: Exponential moving average decay factor for running statistics.
        use_moving_avg_in_training: Whether to use running stats during training.
        set_padded_output_to_zero: Whether to zero out padded positions in the output.
        force_eval_mode: Whether to force evaluation mode statistics.
        gamma_init: Optional weight initialization for scale parameter `gamma`.
        beta_init: Optional weight initialization for shift parameter `beta`.
        epsilon: Small constant added to variance for numerical stability.
    """

    decay: float = 0.999
    use_moving_avg_in_training: bool = False
    set_padded_output_to_zero: bool = True
    force_eval_mode: bool = False
    gamma_init: WeightInit | None = None
    beta_init: WeightInit | None = None
    epsilon: float = 0.001

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        gamma: TensorOrScalar | None = None,
        beta: TensorOrScalar | None = None,
        moving_mean: TensorOrScalar | None = None,
        moving_variance: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Normalize `inputs` across batch and spatial dimensions and update running statistics.

        Args:
            inputs: Input tensor of shape `[..., C]`.
            paddings: Optional padding mask tensor where `1` indicates padded entries.
            gamma: Optional learned scale tensor of shape `[C]`.
            beta: Optional learned bias tensor of shape `[C]`.
            moving_mean: Optional precomputed moving mean tensor of shape `[C]`.
            moving_variance: Optional precomputed moving variance tensor of shape `[C]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Batch-normalized `Tensor` of the same shape as `inputs`.
        """
        x = self._to_tensor(inputs)
        feat_dim = self.dim if self.dim > 0 else x.shape[-1]
        gamma_t = self.get_variable("gamma", (feat_dim,), explicit=gamma, init=self.gamma_init)
        beta_t = self.get_variable("beta", (feat_dim,), explicit=beta, init=self.beta_init)
        stored_mean = self.get_variable(
            "moving_mean",
            (feat_dim,),
            explicit=moving_mean,
            init=WeightInit(method="zeros"),
        )
        stored_var = self.get_variable(
            "moving_variance",
            (feat_dim,),
            explicit=moving_variance,
            init=WeightInit(method="ones", scale=1.0),
        )

        reduce_axes = tuple(range(x.ndim - 1))
        use_moving = self.use_moving_avg_in_training or self.force_eval_mode

        if use_moving:
            mean_t = stored_mean
            var_t = stored_var
        else:
            if paddings is None:
                mean_t = _ops.mean(x, axis=reduce_axes, keepdims=True)
                var_t = _ops.var(x, axis=reduce_axes, keepdims=True)
            else:
                pad_t = _ops.asarray(paddings)
                expanded_paddings = _expand_paddings_to_rank(pad_t, x.ndim)
                weights = 1.0 - expanded_paddings
                unpadded_spatial_prod = 1
                for ax in range(pad_t.ndim, x.ndim - 1):
                    unpadded_spatial_prod *= x.shape[ax]
                sum_weights = _ops.maximum(
                    _ops.sum(weights, axis=reduce_axes, keepdims=True)
                    * float(unpadded_spatial_prod),
                    1.0,
                )
                mean_t = _ops.sum(x * weights, axis=reduce_axes, keepdims=True) / sum_weights
                var_t = (
                    _ops.sum(_ops.square(x - mean_t) * weights, axis=reduce_axes, keepdims=True)
                    / sum_weights
                )
            flat_mean = _ops.reshape(mean_t, (feat_dim,))
            flat_var = _ops.reshape(var_t, (feat_dim,))
            self.register_parameter(
                "moving_mean", self.decay * stored_mean + (1.0 - self.decay) * flat_mean
            )
            self.register_parameter(
                "moving_variance", self.decay * stored_var + (1.0 - self.decay) * flat_var
            )

        inv = (1.0 + gamma_t) / _ops.sqrt(var_t + self.epsilon)
        out = (x - mean_t) * inv + beta_t

        if paddings is not None and self.set_padded_output_to_zero:
            pad_t = _ops.asarray(paddings)
            expanded_paddings = _expand_paddings_to_rank(pad_t, x.ndim)
            out = out * (1.0 - expanded_paddings)

        return out


class GroupNorm(BaseNormalization):
    """Group Normalization layer dividing channels into groups for normalization.

    Attributes:
        num_groups: Number of channel groups.
        min_group_size: Minimum number of channels per group.
        cumulative: Whether to compute cumulative statistics along the time axis.
        input_rank: Expected rank of the input tensor.
        epsilon: Small constant added to variance for numerical stability.
        set_padded_output_to_zero: Whether to zero out padded positions in the output.
        use_scale: Whether to apply a learned scale parameter.
        use_bias: Whether to apply a learned bias parameter.
    """

    num_groups: int = 32
    min_group_size: int = 1
    cumulative: bool = False
    input_rank: int | None = None
    epsilon: float = 0.001
    set_padded_output_to_zero: bool = True
    use_scale: bool = True
    use_bias: bool = True

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        scale: TensorOrScalar | None = None,
        bias: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply Group Normalization to `inputs`.

        Args:
            inputs: Input tensor of shape `[B, ..., C]`.
            paddings: Optional padding mask tensor.
            scale: Optional scale parameter tensor of shape `[C]`.
            bias: Optional bias parameter tensor of shape `[C]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Normalized `Tensor` of the same shape as `inputs`.

        Raises:
            ValueError: If `self.dim > 0` and does not match the input channel dimension, or if `self.input_rank` is set and does not match `inputs.ndim`.
        """
        x = self._to_tensor(inputs)
        orig_shape = x.shape
        if self.input_rank is not None and len(orig_shape) != self.input_rank:
            raise ValueError(f"Expected input rank {self.input_rank}, got {len(orig_shape)}.")
        channels = orig_shape[-1]
        if self.dim > 0 and channels != self.dim:
            raise ValueError(f"Expected input channel dimension {self.dim}, got {channels}")

        rank = len(orig_shape)
        real_num_groups = min(self.num_groups, max(1, channels // max(1, self.min_group_size)))
        if channels % real_num_groups != 0:
            real_num_groups = 1

        group_size = channels // real_num_groups
        grouped_shape = list(orig_shape[:-1]) + [real_num_groups, group_size]
        x_grouped = _ops.reshape(x, grouped_shape)

        if not self.cumulative:
            norm_axes = tuple(range(1, rank - 1)) + (rank,)
            if paddings is None:
                mean_t = _ops.mean(x_grouped, axis=norm_axes, keepdims=True)
                var_t = _ops.var(x_grouped, axis=norm_axes, keepdims=True)
            else:
                pad_t = _ops.asarray(paddings)
                exp_pad = _expand_paddings_to_rank(pad_t, len(grouped_shape))
                weights = 1.0 - exp_pad
                unpadded_spatial = 1
                for ax in norm_axes[:-1]:
                    if ax >= pad_t.ndim:
                        unpadded_spatial *= grouped_shape[ax]
                denom = _ops.maximum(
                    _ops.sum(weights, axis=norm_axes[:-1], keepdims=True)
                    * float(unpadded_spatial * group_size),
                    1.0,
                )
                mean_t = _ops.sum(x_grouped * weights, axis=norm_axes, keepdims=True) / denom
                var_t = (
                    _ops.sum(
                        _ops.square(x_grouped - mean_t) * weights,
                        axis=norm_axes,
                        keepdims=True,
                    )
                    / denom
                )
        else:
            spatial_axes = tuple(range(2, rank - 1)) + (rank,)
            frame_mean = _ops.mean(x_grouped, axis=spatial_axes, keepdims=True)
            frame_sq_mean = _ops.mean(_ops.square(x_grouped), axis=spatial_axes, keepdims=True)
            if paddings is None:
                steps = _ops.arange(1, orig_shape[1] + 1)
                steps_shape = [1, orig_shape[1]] + [1] * (len(grouped_shape) - 2)
                count = _ops.reshape(steps, steps_shape)
                mean_t = _ops.cumsum(frame_mean, axis=1) / count
                sq_mean = _ops.cumsum(frame_sq_mean, axis=1) / count
                var_t = _ops.maximum(sq_mean - _ops.square(mean_t), 0.0)
            else:
                pad_t = _ops.asarray(paddings)
                exp_pad = _expand_paddings_to_rank(pad_t, len(grouped_shape))
                weights = 1.0 - exp_pad
                cum_weights = _ops.maximum(_ops.cumsum(weights, axis=1), 1.0)
                mean_t = _ops.cumsum(frame_mean * weights, axis=1) / cum_weights
                sq_mean = _ops.cumsum(frame_sq_mean * weights, axis=1) / cum_weights
                var_t = _ops.maximum(sq_mean - _ops.square(mean_t), 0.0)

        normed = (x_grouped - mean_t) / _ops.sqrt(var_t + self.epsilon)
        out = _ops.reshape(normed, orig_shape)

        if self.use_scale:
            scale_t = self.get_variable("scale", (channels,), explicit=scale)
            out = out * (1.0 + scale_t)

        if self.use_bias:
            bias_t = self.get_variable("bias", (channels,), explicit=bias)
            out = out + bias_t

        if paddings is not None and self.set_padded_output_to_zero:
            pad_t = _ops.asarray(paddings)
            exp_pad = _expand_paddings_to_rank(pad_t, len(orig_shape))
            out = out * (1.0 - exp_pad)

        return out


class LayerNorm(BaseNormalization):
    """Layer Normalization layer normalizing across the last feature dimension.

    Attributes:
        direct_scale: Whether to multiply by `scale` directly (`True`) or `1 + scale` (`False`).
        epsilon: Small constant added to variance for numerical stability.
        use_scale: Whether to apply a learned scale parameter.
        use_bias: Whether to apply a learned bias parameter.
        reductions_in_fp32: Whether to compute mean and variance in float32 precision.
    """

    direct_scale: bool = False
    epsilon: float = 1e-6
    use_scale: bool = True
    use_bias: bool = True
    reductions_in_fp32: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        scale: TensorOrScalar | None = None,
        bias: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply Layer Normalization along the last dimension of `inputs`.

        Args:
            inputs: Input tensor of shape `[..., D]`.
            paddings: Optional padding mask tensor.
            scale: Optional learned scale tensor of shape `[D]`.
            bias: Optional learned bias tensor of shape `[D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Layer-normalized `Tensor` of the same shape as `inputs`.
        """
        x = (
            _ops.asarray(inputs, dtype="float32")
            if self.reductions_in_fp32
            else self._to_tensor(inputs)
        )
        feat_dim = self.dim if self.dim > 0 else x.shape[-1]
        mean_t = _ops.mean(x, axis=-1, keepdims=True)
        var_t = _ops.var(x, axis=-1, keepdims=True)
        normed = (x - mean_t) / _ops.sqrt(var_t + self.epsilon)

        if self.use_scale:
            default_scale_init = WeightInit(
                method="ones" if self.direct_scale else "zeros", scale=1.0
            )
            scale_t = self.get_variable(
                "scale", (feat_dim,), explicit=scale, init=default_scale_init
            )
            if not self.direct_scale:
                scale_t = 1.0 + scale_t
            normed = normed * scale_t

        if self.use_bias:
            bias_t = self.get_variable("bias", (feat_dim,), explicit=bias)
            normed = normed + bias_t

        return normed.astype(self.effective_fprop_dtype)


class RmsNorm(BaseNormalization):
    """Root Mean Square Layer Normalization (RMSNorm).

    Attributes:
        epsilon: Small constant added to mean square for numerical stability.
        direct_scale: Whether to multiply by `scale` directly (`True`) or `1 + scale` (`False`).
        intermediate_dtype: Optional intermediate data type for RMS reduction.
    """

    epsilon: float = 1e-6
    direct_scale: bool = True
    intermediate_dtype: DType = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        scale: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply Root Mean Square normalization to `inputs`.

        Args:
            inputs: Input tensor of shape `[..., D]`.
            paddings: Optional padding mask tensor.
            scale: Optional scale parameter tensor of shape `[D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            RMS-normalized `Tensor`.
        """
        x = (
            _ops.asarray(inputs, dtype=self.intermediate_dtype)
            if self.intermediate_dtype is not None
            else self._to_tensor(inputs)
        )
        feat_dim = self.dim if self.dim > 0 else x.shape[-1]
        default_scale_init = WeightInit(method="ones" if self.direct_scale else "zeros", scale=1.0)
        scale_t = self.get_variable("scale", (feat_dim,), explicit=scale, init=default_scale_init)

        var_t = _ops.mean(_ops.square(x), axis=-1, keepdims=True)
        normed = x / _ops.sqrt(var_t + self.epsilon)
        mult = scale_t if self.direct_scale else 1.0 + scale_t
        return (normed * mult).astype(self.effective_fprop_dtype)


class RmsNormNoScale(BaseNormalization):
    """Parameter-free Root Mean Square Layer Normalization.

    Attributes:
        epsilon: Small constant added to mean square for numerical stability.
    """

    epsilon: float = 1e-6

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply parameter-free RMS normalization to `inputs`.

        Args:
            inputs: Input tensor of shape `[..., D]`.
            paddings: Optional padding mask tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            RMS-normalized `Tensor` without learned scaling.
        """
        x = self._to_tensor(inputs)
        var_t = _ops.mean(_ops.square(x), axis=-1, keepdims=True)
        return x / _ops.sqrt(var_t + self.epsilon)


__all__ = [
    "BaseNormalization",
    "BatchNorm",
    "GroupNorm",
    "IdentityNorm",
    "LayerNorm",
    "RmsNorm",
    "RmsNormNoScale",
]
