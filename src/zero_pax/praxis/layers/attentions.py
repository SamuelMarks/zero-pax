"""Attention mechanisms and projections for zero_pax.praxis.layers."""

from __future__ import annotations

import math

from zero_pax import _tensor_ops as np
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl, instantiate_layer
from zero_pax.praxis.layers.activations import Identity
from zero_pax.praxis.layers.convolutions import CausalDepthwiseConv1D
from zero_pax.praxis.layers.linear_pooling import EinsumOp
from zero_pax.praxis.layers.normalizations import LayerNorm
from zero_pax.types import Tensor, TensorOrScalar, WeightInit


def _apply_rotary_position_emb(
    inputs: Tensor,
    min_timescale: int = 1,
    max_timescale: int = 10000,
) -> Tensor:
    """Apply Rotary Position Embeddings (RoPE) to a 4D query/key tensor `[B, T, N, H]`.

    Args:
        inputs: 4D query or key tensor of shape `[B, T, N, H]`.
        min_timescale: Minimum geometric wavelength timescale.
        max_timescale: Maximum geometric wavelength timescale.

    Returns:
        RoPE-rotated `Tensor` of shape `[B, T, N, H]`.
    """
    _, seq_len, _, head_dim = inputs.shape
    half_dim = head_dim // 2
    if half_dim == 0:
        return inputs
    fraction = 2.0 * np.arange(0, half_dim) / float(head_dim)
    timescale = float(min_timescale) * ((float(max_timescale) / float(min_timescale)) ** fraction)
    positions = np.reshape(np.arange(0, seq_len), (1, seq_len, 1, 1))
    sinusoid_inp = positions / np.reshape(timescale, (1, 1, 1, half_dim))
    sin = np.sin(sinusoid_inp)
    cos = np.cos(sinusoid_inp)

    first_half = inputs[:, :, :, :half_dim]
    second_half = inputs[:, :, :, half_dim : 2 * half_dim]
    rot_first = first_half * cos - second_half * sin
    rot_second = second_half * cos + first_half * sin
    rotated = np.concatenate([rot_first, rot_second], axis=-1)
    if head_dim % 2 == 1:
        rotated = np.concatenate([rotated, inputs[:, :, :, 2 * half_dim :]], axis=-1)
    return rotated


def _normalize_attention_mask(atten_mask: TensorOrScalar) -> Tensor:
    """Broadcast-normalize a 2D, 3D, or 4D attention mask into 4D `[B_or_1, N_or_1, T_q, T_k]`.

    Args:
        atten_mask: Raw additive attention mask tensor.

    Returns:
        4D broadcast-compatible attention mask `Tensor`.
    """
    mask_t = np.asarray(atten_mask)
    if mask_t.ndim == 2:
        return np.reshape(mask_t, (1, 1, mask_t.shape[0], mask_t.shape[1]))
    if mask_t.ndim == 3:
        return np.expand_dims(mask_t, 1)
    return mask_t


class AttentionProjection(BasePraxisLayer):
    """Multi-head attention linear projection layer (`[B, T, D] <-> [B, T, N, H]`).

    Attributes:
        input_dim: Input/output model dimension `D`.
        num_heads: Number of attention heads `N`.
        dim_per_head: Per-head feature dimension `H`.
        is_output_projection: Whether projecting `[..., N, H] -> [..., D]` (`True`) or `[..., D] -> [..., N, H]` (`False`).
        use_bias: Whether to add a learned bias tensor.
        attention_combine_dims: Whether `N` and `H` are flattened into `N * H`.
        use_nhd_shape: Whether output projection weights use `(N, H, D)` layout.
        explicit_fan_in_fan_out_axes: Whether fan-in and fan-out axes are explicitly annotated.
        einsum_tpl: Optional template for the underlying einsum operation.
    """

    input_dim: int = 0
    num_heads: int = 1
    dim_per_head: int = 0
    is_output_projection: bool = False
    use_bias: bool = True
    attention_combine_dims: bool = False
    use_nhd_shape: bool = False
    explicit_fan_in_fan_out_axes: bool = False
    einsum_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        w: TensorOrScalar | None = None,
        b: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Project `inputs` between model space `D` and multi-head space `(N, H)`.

        Args:
            inputs: Input tensor of shape `[..., N, H]` (if `is_output_projection`) or `[..., D]`.
            w: Optional explicit weight tensor.
            b: Optional explicit bias tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Projected `Tensor`.
        """
        x = np.asarray(inputs)
        n_heads = self.num_heads if self.num_heads > 0 else 1

        if self.is_output_projection:
            if n_heads == 1 and x.ndim >= 2:
                n_heads = x.shape[-2]
            h_dim = self.dim_per_head if self.dim_per_head > 0 else x.shape[-1]
            d_dim = self.input_dim if self.input_dim > 0 else n_heads * h_dim
            if w is None:
                w_t = self.get_variable("w", (n_heads, h_dim, d_dim))
            else:
                w_raw = np.asarray(w)
                w_t = (
                    np.reshape(w_raw, (n_heads, h_dim, d_dim))
                    if w_raw.shape != (n_heads, h_dim, d_dim)
                    else w_raw
                )
            op_out = instantiate_layer(self.einsum_tpl, EinsumOp(equation="...nh,nhd->...d"))
            out = op_out("...nh,nhd->...d", x, w_t)
            if self.use_bias:
                b_t = self.get_variable("b", (d_dim,), explicit=b)
                out = out + b_t
            return out

        d_dim = self.input_dim if self.input_dim > 0 else x.shape[-1]
        h_dim = self.dim_per_head if self.dim_per_head > 0 else max(1, d_dim // n_heads)
        if w is None:
            w_t = self.get_variable("w", (d_dim, n_heads, h_dim))
        else:
            w_raw = np.asarray(w)
            w_t = (
                np.reshape(w_raw, (d_dim, n_heads, h_dim))
                if w_raw.shape != (d_dim, n_heads, h_dim)
                else w_raw
            )
        op_in = instantiate_layer(self.einsum_tpl, EinsumOp(equation="...d,dnh->...nh"))
        out = op_in("...d,dnh->...nh", x, w_t)
        if self.use_bias:
            if b is None:
                b_t = self.get_variable("b", (n_heads, h_dim))
            else:
                b_raw = np.asarray(b)
                b_t = (
                    np.reshape(b_raw, (n_heads, h_dim))
                    if b_raw.shape != (n_heads, h_dim)
                    else b_raw
                )
            out = out + b_t
        if self.attention_combine_dims:
            out = np.reshape(out, list(out.shape[:-2]) + [n_heads * h_dim])
        return out


class RelativeBias(BasePraxisLayer):
    """T5-style Relative Position Bias layer computing bucketed attention logit biases.

    Attributes:
        num_heads: Number of attention heads `N`.
        use_length_as_position: Whether to derive positions from `0..T-1` sequence lengths.
        relative_attention_num_buckets: Number of discrete relative distance buckets.
        relative_attention_max_distance: Maximum distance before logarithmic bucket saturation.
        bidirectional: Whether attention is bidirectional (`True`) or causal (`False`).
        use_xavier_init: Whether to use Xavier initialization for bucket embeddings.
        rb_stddev: Optional standard deviation for bucket weight initialization.
    """

    num_heads: int = 1
    use_length_as_position: bool = True
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    bidirectional: bool = True
    use_xavier_init: bool = False
    rb_stddev: float | None = None

    def _relative_position_bucket(self, relative_position: Tensor) -> Tensor:
        """Map relative positions `k_pos - q_pos` into integer bucket indices.

        Args:
            relative_position: Integer/float tensor of relative offsets `key_pos - query_pos`.

        Returns:
            Bucket index `Tensor` with values in `[0, relative_attention_num_buckets - 1]`.
        """
        num_buckets = max(2, self.relative_attention_num_buckets)
        max_distance = max(1, self.relative_attention_max_distance)
        ret = np.zeros_like(relative_position)
        n = -relative_position

        if self.bidirectional:
            num_buckets = num_buckets // 2
            ret = ret + np.where(n < 0.0, float(num_buckets), 0.0)
            n = np.abs(n)
        else:
            n = np.maximum(n, 0.0)

        max_exact = max(1, num_buckets // 2)
        is_small = n < float(max_exact)
        n_safe = np.maximum(n, 1.0)
        denom = math.log(max(float(max_distance) / float(max_exact), 1.0001))
        val_if_large = float(max_exact) + (
            np.log(n_safe / float(max_exact)) / denom * float(num_buckets - max_exact)
        )
        val_if_large = np.minimum(val_if_large, float(num_buckets - 1))
        return ret + np.where(is_small, n, val_if_large)

    def __call__(
        self,
        query_segment_pos: TensorOrScalar | int,
        key_segment_pos: TensorOrScalar | int | None = None,
        w: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute relative attention bias tensor of shape `[B_or_1, num_heads, Q, K]`.

        Args:
            query_segment_pos: Query position tensor `[B, Q]` or `[Q]`, or integer length `Q`.
            key_segment_pos: Optional key position tensor `[B, K]` or `[K]`, or integer length `K`.
            w: Optional explicit bucket weight table of shape `[num_heads, num_buckets]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Relative bias `Tensor` of shape `[1, num_heads, Q, K]` or `[B, num_heads, Q, K]`.
        """
        q_pos: Tensor
        if isinstance(query_segment_pos, int):
            q_pos = np.expand_dims(np.arange(query_segment_pos), 0)
        else:
            q_arr = np.asarray(query_segment_pos)
            q_pos = np.expand_dims(q_arr, 0) if q_arr.ndim == 1 else q_arr

        k_pos: Tensor
        if key_segment_pos is None:
            k_pos = q_pos
        elif isinstance(key_segment_pos, int):
            k_pos = np.expand_dims(np.arange(key_segment_pos), 0)
        else:
            k_arr = np.asarray(key_segment_pos)
            k_pos = np.expand_dims(k_arr, 0) if k_arr.ndim == 1 else k_arr

        rel_pos = np.expand_dims(k_pos, 1) - np.expand_dims(q_pos, 2)
        buckets = self._relative_position_bucket(rel_pos)

        n_heads = max(1, self.num_heads)
        n_buckets = max(2, self.relative_attention_num_buckets)
        rb_init = (
            WeightInit(method="xavier", scale=self.rb_stddev or 1.0)
            if self.use_xavier_init
            else None
        )
        w_t = self.get_variable("w", (n_heads, n_buckets), explicit=w, init=rb_init)

        bucket_idx = np.clip(buckets, 0, n_buckets - 1)
        one_hot = np.reshape(bucket_idx, list(bucket_idx.shape) + [1]) == np.arange(n_buckets)
        return np.einsum("bqkc,nc->bnqk", one_hot, w_t)


class DotProductAttention(BasePraxisLayer):
    """Multi-Head Scaled Dot-Product Attention layer.

    Attributes:
        input_dim: Input/output model feature dimension `D`.
        hidden_dim: Total hidden dimension across all heads (`N * H`).
        num_heads: Number of attention heads `N`.
        dim_per_head: Feature dimension per head `H`.
        dropout_tpl: Optional attention dropout layer template.
        atten_dropout_prob: Attention probability dropout rate.
        proj_tpl: Optional `AttentionProjection` template.
        dconv_qkv: Whether to apply depthwise convolution to Q, K, and V.
        dconv_kernel_size: Kernel size for depthwise QKV convolution.
        internal_gshard_gaussian_init: Whether to use GShard Gaussian initialization.
        combine_qkv: Whether Q, K, and V projections are combined into a single tensor.
        combined_qkv_proj_tpl: Optional combined QKV projection template.
        use_bias: Whether Q, K, V, and output projections include bias.
        output_proj_use_nhd_shape: Whether the output projection weight uses `(N, H, D)`.
        internal_enable_query_scale: Whether to scale query vectors before dot product.
        internal_enable_per_dim_scale: Whether to apply learned `PerDimScale` to queries.
        scale_query_by_dim_per_head: Whether to scale queries by `dim_per_head ** -0.5`.
        scale_query_by_head_dims: Alias for `scale_query_by_dim_per_head`.
        scale_logits_by_head_dims: Whether to scale attention logits by `dim_per_head ** -0.5`.
        atten_logit_cap: Tanh soft-capping limit for attention logits (`0.0` disables capping).
        use_rotary_position_emb: Whether to apply Rotary Position Embeddings (RoPE).
        consolidate_rope_key_state: Whether to consolidate RoPE key state in decode cache.
        rotary_position_emb_tpl: Optional RoPE layer template.
        cast_rotary_position_emb: Whether to cast RoPE outputs to `fprop_dtype`.
        relative_bias_tpl: Optional `RelativeBias` layer or template.
        attention_extra_logit: Optional extra constant logit appended to softmax denominator.
        ngrammer_tpl: Optional Ngrammer layer template.
        decode_cache: Whether to maintain KV cache during autoregressive decoding.
        attention_mask_summary: Whether to record attention mask summaries.
        zero_fully_masked: Whether to zero out outputs for tokens whose keys are all masked.
        qk_einsum_tpl: Optional QK einsum template.
        pv_einsum_tpl: Optional PV einsum template.
        per_dim_scale_tpl: Optional `PerDimScale` template.
        causal_depthwise_conv1d_tpl: Optional causal depthwise conv1d template.
        ln_tpl: Optional layer normalization template.
        decoding_maybe_shard_projections: Whether to shard projections during decoding.
    """

    input_dim: int = 0
    hidden_dim: int = 0
    num_heads: int = 1
    dim_per_head: int = 0
    dropout_tpl: LayerTpl = None
    atten_dropout_prob: float = 0.0
    proj_tpl: LayerTpl = None
    dconv_qkv: bool = False
    dconv_kernel_size: int = 3
    internal_gshard_gaussian_init: bool = False
    combine_qkv: bool = False
    combined_qkv_proj_tpl: LayerTpl = None
    use_bias: bool = True
    output_proj_use_nhd_shape: bool = False
    internal_enable_query_scale: bool = True
    internal_enable_per_dim_scale: bool = True
    scale_query_by_dim_per_head: bool = False
    scale_query_by_head_dims: bool = False
    scale_logits_by_head_dims: bool = False
    atten_logit_cap: float = 0.0
    use_rotary_position_emb: bool = False
    consolidate_rope_key_state: bool = False
    rotary_position_emb_tpl: LayerTpl = None
    cast_rotary_position_emb: bool = True
    relative_bias_tpl: LayerTpl = None
    attention_extra_logit: float | None = None
    ngrammer_tpl: LayerTpl = None
    decode_cache: bool = True
    attention_mask_summary: bool = False
    zero_fully_masked: bool = False
    qk_einsum_tpl: LayerTpl = None
    pv_einsum_tpl: LayerTpl = None
    per_dim_scale_tpl: LayerTpl = None
    causal_depthwise_conv1d_tpl: LayerTpl = None
    ln_tpl: LayerTpl = None
    decoding_maybe_shard_projections: bool = False

    def _compute_context_mask(self, seq_len: int) -> Tensor | None:
        """Return an optional `(1, 1, T, T)` additive mask (`0.0` allowed, `-1e9` masked).

        Args:
            seq_len: Sequence length `T`.

        Returns:
            Optional additive mask `Tensor` or `None`.
        """
        return None

    def _compute_xl_bias(self, q_len: int, k_len: int, n_heads: int) -> Tensor | None:
        """Return an optional Transformer-XL sinusoidal relative positional bias `(1, N, Q, K)`.

        Args:
            q_len: Query sequence length.
            k_len: Key sequence length.
            n_heads: Number of attention heads.

        Returns:
            Optional additive relative bias `Tensor` or `None`.
        """
        return None

    def __call__(
        self,
        query_vec: TensorOrScalar,
        key_vec: TensorOrScalar | None = None,
        value_vec: TensorOrScalar | None = None,
        atten_mask: TensorOrScalar | None = None,
        query_w: TensorOrScalar | None = None,
        query_b: TensorOrScalar | None = None,
        key_w: TensorOrScalar | None = None,
        key_b: TensorOrScalar | None = None,
        value_w: TensorOrScalar | None = None,
        value_b: TensorOrScalar | None = None,
        post_w: TensorOrScalar | None = None,
        post_b: TensorOrScalar | None = None,
        per_dim_scale: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute multi-head dot-product attention over `query_vec`, `key_vec`, and `value_vec`.

        Args:
            query_vec: Query input tensor of shape `[B, T_q, D]`.
            key_vec: Optional key input tensor of shape `[B, T_k, D]` (defaults to `query_vec`).
            value_vec: Optional value input tensor of shape `[B, T_k, D]` (defaults to `query_vec`).
            atten_mask: Optional additive attention mask broadcastable to `[B, N, T_q, T_k]`.
            query_w: Optional query projection weight tensor.
            query_b: Optional query projection bias tensor.
            key_w: Optional key projection weight tensor.
            key_b: Optional key projection bias tensor.
            value_w: Optional value projection weight tensor.
            value_b: Optional value projection bias tensor.
            post_w: Optional output projection weight tensor.
            post_b: Optional output projection bias tensor.
            per_dim_scale: Optional per-dimension query scale vector of shape `[H]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Attended output `Tensor` of shape `[B, T_q, D]`.
        """
        q_in = np.asarray(query_vec)
        k_in = q_in if key_vec is None else np.asarray(key_vec)
        v_in = q_in if value_vec is None else np.asarray(value_vec)

        if self.dconv_qkv:
            k_sz = max(1, self.dconv_kernel_size)
            dconv_layer = instantiate_layer(
                self.causal_depthwise_conv1d_tpl,
                CausalDepthwiseConv1D(kernel_size=k_sz, params_init=self.params_init),
            )
            q_in = dconv_layer(q_in)
            k_in = dconv_layer(k_in)
            v_in = dconv_layer(v_in)

        d_dim = self.input_dim if self.input_dim > 0 else q_in.shape[-1]
        n_heads = self.num_heads if self.num_heads > 0 else 1
        h_dim = (
            self.dim_per_head
            if self.dim_per_head > 0
            else (self.hidden_dim // n_heads if self.hidden_dim > 0 else max(1, d_dim // n_heads))
        )

        proj_q = AttentionProjection(
            input_dim=d_dim,
            num_heads=n_heads,
            dim_per_head=h_dim,
            is_output_projection=False,
            use_bias=self.use_bias,
            params_init=self.params_init,
        )
        proj_k = AttentionProjection(
            input_dim=d_dim,
            num_heads=n_heads,
            dim_per_head=h_dim,
            is_output_projection=False,
            use_bias=self.use_bias,
            params_init=self.params_init,
        )
        proj_v = AttentionProjection(
            input_dim=d_dim,
            num_heads=n_heads,
            dim_per_head=h_dim,
            is_output_projection=False,
            use_bias=self.use_bias,
            params_init=self.params_init,
        )
        proj_post = AttentionProjection(
            input_dim=d_dim,
            num_heads=n_heads,
            dim_per_head=h_dim,
            is_output_projection=True,
            use_bias=self.use_bias,
            use_nhd_shape=self.output_proj_use_nhd_shape,
            params_init=self.params_init,
        )

        q = proj_q(q_in, w=query_w, b=query_b)
        k = proj_k(k_in, w=key_w, b=key_b)
        v = proj_v(v_in, w=value_w, b=value_b)

        if self.use_rotary_position_emb:
            q = _apply_rotary_position_emb(q)
            k = _apply_rotary_position_emb(k)

        if (
            self.internal_enable_query_scale
            or self.scale_query_by_dim_per_head
            or self.scale_query_by_head_dims
        ):
            scale_vec = (
                np.ones((h_dim,))
                if (per_dim_scale is None or not self.internal_enable_per_dim_scale)
                else np.asarray(per_dim_scale)
            )
            q = q * (scale_vec / np.sqrt(float(h_dim)))

        logits = np.einsum("btnh,bsnh->bnts", q, k)
        if self.scale_logits_by_head_dims:
            logits = logits * (float(h_dim) ** -0.5)

        if self.atten_logit_cap > 0.0:
            logits = self.atten_logit_cap * np.tanh(logits / self.atten_logit_cap)

        if isinstance(self.relative_bias_tpl, RelativeBias):
            logits = logits + self.relative_bias_tpl(q_in.shape[1], k_in.shape[1])

        xl_bias = self._compute_xl_bias(q_in.shape[1], k_in.shape[1], n_heads)
        if xl_bias is not None:
            logits = logits + xl_bias

        ctx_mask = self._compute_context_mask(q_in.shape[1])
        if ctx_mask is not None and q_in.shape[1] == k_in.shape[1]:
            logits = logits + ctx_mask

        norm_mask: Tensor | None = None
        if atten_mask is not None:
            norm_mask = _normalize_attention_mask(atten_mask)
            logits = logits + norm_mask

        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        denom = np.sum(exp_logits, axis=-1, keepdims=True)
        if self.attention_extra_logit is not None:
            denom = denom + np.exp(float(self.attention_extra_logit) - max_logits)
        probs = exp_logits / np.maximum(denom, 1e-12)

        if self.zero_fully_masked and norm_mask is not None:
            fully_masked = np.max(norm_mask, axis=-1, keepdims=True) < -1e5
            probs = np.where(fully_masked, 0.0, probs)

        drop_layer = instantiate_layer(self.dropout_tpl, Identity())
        probs = drop_layer(probs)

        encoded = np.einsum("bnts,bsnh->btnh", probs, v)
        return proj_post(encoded, w=post_w, b=post_b)


class DotProductAttentionXL(DotProductAttention):
    """Transformer-XL style Dot-Product Attention with relative sinusoidal positional bias.

    Attributes:
        rel_pos_emb_dim: Dimension of the relative positional embedding.
    """

    rel_pos_emb_dim: int = 0

    def _compute_xl_bias(self, q_len: int, k_len: int, n_heads: int) -> Tensor | None:
        """Compute Transformer-XL sinusoidal relative distance bias `(1, N, Q, K)`.

        Args:
            q_len: Query sequence length.
            k_len: Key sequence length.
            n_heads: Number of attention heads.

        Returns:
            Sinusoidal relative positional bias `Tensor` of shape `(1, N, Q, K)` or `None`.
        """
        if self.rel_pos_emb_dim <= 0:
            return None
        q_pos = np.expand_dims(np.arange(q_len), 1)
        k_pos = np.expand_dims(np.arange(k_len), 0)
        rel_dist = k_pos - q_pos
        head_slopes = 1.0 / (10000.0 ** (np.arange(n_heads) / max(1.0, float(n_heads))))
        phase = np.reshape(head_slopes, (1, n_heads, 1, 1)) * np.reshape(
            rel_dist, (1, 1, q_len, k_len)
        )
        return np.cos(phase) * (float(self.rel_pos_emb_dim) ** -0.5)


class DotProductAttentionWithContext(DotProductAttention):
    """Dot-Product Attention restricted to a local `[left_context, right_context]` band.

    Attributes:
        left_context: Number of past frames (including current frame) allowed to attend.
        right_context: Number of future frames allowed to attend.
    """

    left_context: int | None = None
    right_context: int | None = None

    def _compute_context_mask(self, seq_len: int) -> Tensor | None:
        """Build banded context window mask for `[left_context, right_context]`.

        Args:
            seq_len: Sequence length `T`.

        Returns:
            Banded additive mask `Tensor` of shape `(1, 1, T, T)` or `None`.
        """
        if self.left_context is None and self.right_context is None:
            return None
        pos = np.arange(seq_len)
        diff = np.expand_dims(pos, 0) - np.expand_dims(pos, 1)
        masked = np.zeros((seq_len, seq_len)) > 1.0
        if self.left_context is not None:
            masked = masked | (diff < -(self.left_context - 1))
        if self.right_context is not None:
            masked = masked | (diff > self.right_context)
        mask_2d = np.where(masked, -1e9, 0.0)
        return np.reshape(mask_2d, (1, 1, seq_len, seq_len))


class DotProductAttentionWithContextXL(DotProductAttentionWithContext, DotProductAttentionXL):
    """Dot-Product Attention combining local `[left_context, right_context]` masking and XL relative bias.

    Attributes:
        rel_pos_emb_dim: Dimension of the relative positional embedding.
    """

    rel_pos_emb_dim: int = 0


class LocalSelfAttention(DotProductAttentionWithContext):
    """Blockwise / sliding-window Local Self-Attention layer.

    Attributes:
        block_size: Local attention block length.
        simulated: Whether local block attention is simulated via full attention masking.
    """

    block_size: int = 0
    simulated: bool = False

    def _compute_context_mask(self, seq_len: int) -> Tensor | None:
        """Compute banded context mask or block-diagonal mask when `block_size > 0`.

        Args:
            seq_len: Sequence length `T`.

        Returns:
            Additive local attention mask `Tensor` of shape `(1, 1, T, T)` or `None`.
        """
        base_mask = super()._compute_context_mask(seq_len)
        if base_mask is not None or self.block_size <= 0:
            return base_mask
        pos = np.arange(seq_len)
        diff = np.abs(np.expand_dims(pos, 0) - np.expand_dims(pos, 1))
        mask_2d = np.where(diff > self.block_size, -1e9, 0.0)
        return np.reshape(mask_2d, (1, 1, seq_len, seq_len))


class LocalSelfAttentionXL(LocalSelfAttention, DotProductAttentionXL):
    """Local Self-Attention layer with Transformer-XL relative positional encoding.

    Attributes:
        rel_pos_emb_dim: Dimension of the relative positional embedding.
    """

    rel_pos_emb_dim: int = 0


class LocalSelfAttentionRelativeBias(LocalSelfAttention):
    """Local Self-Attention layer with T5-style bucketed relative positional bias."""

    def _compute_xl_bias(self, q_len: int, k_len: int, n_heads: int) -> Tensor | None:
        """Compute default `RelativeBias` when `relative_bias_tpl` is not explicitly set.

        Args:
            q_len: Query sequence length.
            k_len: Key sequence length.
            n_heads: Number of attention heads.

        Returns:
            Relative position bias `Tensor` or `None`.
        """
        if self.relative_bias_tpl is not None:
            return None
        rb = RelativeBias(num_heads=n_heads, params_init=self.params_init)
        return rb(q_len, k_len)


class LocalSelfAttentionAlibi(LocalSelfAttention):
    """Local Self-Attention layer with ALiBi (Attention with Linear Biases) distance penalties."""

    def _compute_context_mask(self, seq_len: int) -> Tensor | None:
        """Compute combined local window mask and ALiBi linear distance penalty.

        Args:
            seq_len: Sequence length `T`.

        Returns:
            Additive ALiBi + local context bias `Tensor` of shape `(1, N, T, T)`.
        """
        base_mask = super()._compute_context_mask(seq_len)
        pos = np.arange(seq_len)
        dist = -np.abs(np.expand_dims(pos, 0) - np.expand_dims(pos, 1))
        n_heads = max(1, self.num_heads)
        slopes = 0.5 ** (np.arange(1, n_heads + 1) * (8.0 / float(n_heads)))
        alibi = np.reshape(slopes, (1, n_heads, 1, 1)) * np.reshape(dist, (1, 1, seq_len, seq_len))
        if base_mask is not None:
            return base_mask + alibi
        return alibi


class GroupedQueryAttention(BasePraxisLayer):
    """Grouped-Query Attention (GQA) layer sharing key/value heads across query head groups.

    Attributes:
        input_dim: Input/output model feature dimension `D`.
        hidden_dim: Hidden projection dimension.
        num_heads: Number of query heads `N`.
        num_kv_heads: Number of key/value heads `KV`.
        dim_per_head: Feature dimension per head `H`.
        dropout_tpl: Optional dropout layer template.
        atten_dropout_prob: Attention probability dropout rate.
        atten_temp: Optional attention temperature scaling factor.
        proj_tpl: Optional projection layer template.
        use_bias: Whether projections include bias.
        atten_logit_cap: Optional tanh soft-cap threshold for attention logits.
        rope_min_max_timescales: Optional `(min_timescale, max_timescale)` tuple for RoPE.
    """

    input_dim: int = 0
    hidden_dim: int = 0
    num_heads: int = 1
    num_kv_heads: int = 1
    dim_per_head: int = 1
    dropout_tpl: LayerTpl = None
    atten_dropout_prob: float = 0.0
    atten_temp: float = 1.0
    proj_tpl: LayerTpl = None
    use_bias: bool = True
    atten_logit_cap: float = 0.0
    rope_min_max_timescales: tuple[int, int] | None = None

    def __call__(
        self,
        query_vec: TensorOrScalar,
        key_vec: TensorOrScalar | None = None,
        value_vec: TensorOrScalar | None = None,
        atten_mask: TensorOrScalar | None = None,
        w_q: TensorOrScalar | None = None,
        w_k: TensorOrScalar | None = None,
        w_v: TensorOrScalar | None = None,
        w_o: TensorOrScalar | None = None,
        b_q: TensorOrScalar | None = None,
        b_k: TensorOrScalar | None = None,
        b_v: TensorOrScalar | None = None,
        b_o: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute grouped-query attention over `query_vec`, `key_vec`, and `value_vec`.

        Args:
            query_vec: Query input tensor of shape `[B, T_q, D]`.
            key_vec: Optional key input tensor of shape `[B, T_k, D]`.
            value_vec: Optional value input tensor of shape `[B, T_k, D]`.
            atten_mask: Optional additive attention mask.
            w_q: Optional query weight tensor of shape `[D, N, H]`.
            w_k: Optional key weight tensor of shape `[D, KV, H]`.
            w_v: Optional value weight tensor of shape `[D, KV, H]`.
            w_o: Optional output weight tensor of shape `[N, H, D]`.
            b_q: Optional query bias tensor of shape `[N, H]`.
            b_k: Optional key bias tensor of shape `[KV, H]`.
            b_v: Optional value bias tensor of shape `[KV, H]`.
            b_o: Optional output bias tensor of shape `[D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` of shape `[B, T_q, D]`.
        """
        q_in = np.asarray(query_vec)
        k_in = q_in if key_vec is None else np.asarray(key_vec)
        v_in = q_in if value_vec is None else np.asarray(value_vec)

        d_dim = self.input_dim if self.input_dim > 0 else q_in.shape[-1]
        n_heads = max(1, self.num_heads)
        kv_heads = max(1, min(self.num_kv_heads, n_heads))
        if n_heads % kv_heads != 0:
            kv_heads = 1
        h_dim = self.dim_per_head if self.dim_per_head > 1 else max(1, d_dim // n_heads)

        wq_t = self.get_variable("w_q", (d_dim, n_heads, h_dim), explicit=w_q)
        wk_t = self.get_variable("w_k", (d_dim, kv_heads, h_dim), explicit=w_k)
        wv_t = self.get_variable("w_v", (d_dim, kv_heads, h_dim), explicit=w_v)
        wo_t = self.get_variable("w_o", (n_heads, h_dim, d_dim), explicit=w_o)

        q = np.einsum("btd,dnh->btnh", q_in, wq_t)
        k = np.einsum("bsd,dkh->bskh", k_in, wk_t)
        v = np.einsum("bsd,dkh->bskh", v_in, wv_t)

        if self.use_bias:
            if b_q is not None:
                q = q + np.asarray(b_q)
            if b_k is not None:
                k = k + np.asarray(b_k)
            if b_v is not None:
                v = v + np.asarray(b_v)

        if self.rope_min_max_timescales is not None:
            min_ts, max_ts = self.rope_min_max_timescales
            q = _apply_rotary_position_emb(q, min_timescale=min_ts, max_timescale=max_ts)
            k = _apply_rotary_position_emb(k, min_timescale=min_ts, max_timescale=max_ts)

        if n_heads != kv_heads:
            repeats = n_heads // kv_heads
            b_sz, s_len, _, _ = k.shape
            k_exp = np.broadcast_to(np.expand_dims(k, 3), (b_sz, s_len, kv_heads, repeats, h_dim))
            v_exp = np.broadcast_to(np.expand_dims(v, 3), (b_sz, s_len, kv_heads, repeats, h_dim))
            k = np.reshape(k_exp, (b_sz, s_len, n_heads, h_dim))
            v = np.reshape(v_exp, (b_sz, s_len, n_heads, h_dim))

        temp_scale = self.atten_temp if self.atten_temp > 0.0 else 1.0
        q = q / (np.sqrt(float(h_dim)) * temp_scale)
        logits = np.einsum("btnh,bsnh->bnts", q, k)
        if self.atten_logit_cap > 0.0:
            logits = self.atten_logit_cap * np.tanh(logits / self.atten_logit_cap)
        if atten_mask is not None:
            logits = logits + _normalize_attention_mask(atten_mask)

        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.maximum(np.sum(exp_logits, axis=-1, keepdims=True), 1e-12)

        drop_layer = instantiate_layer(self.dropout_tpl, Identity())
        probs = drop_layer(probs)

        encoded = np.einsum("bnts,bsnh->btnh", probs, v)
        out = np.einsum("btnh,nhd->btd", encoded, wo_t)
        if self.use_bias and b_o is not None:
            out = out + np.asarray(b_o)
        return out


class SelfAttentionWithNormAndResidual(BasePraxisLayer):
    """Self-Attention block wrapped with Layer Normalization and weighted residual connection.

    Attributes:
        residual_weight: Multiplier applied to the attention branch output.
        input_weight: Multiplier applied to the residual skip connection input.
        self_atten_tpl: Optional self-attention layer or template.
        norm_tpl: Optional normalization layer or template.
        pre_layer_norm: Whether normalization is applied before (`True`) or after (`False`) attention.
        residual_dropout_prob: Dropout rate on the residual branch.
        residual_dropout_tpl: Optional dropout layer template.
        norm_policy: Normalization placement policy (`'pre'` or `'post'`).
        atten_dropout_prob: Attention probability dropout rate.
        num_heads: Number of attention heads.
        dim_per_head: Optional dimension per attention head.
        force_no_residual: If True, omits the skip connection and returns only the projected branch.
    """

    residual_weight: float = 1.0
    input_weight: float = 1.0
    self_atten_tpl: LayerTpl = None
    norm_tpl: LayerTpl = None
    pre_layer_norm: bool = True
    residual_dropout_prob: float = 0.0
    residual_dropout_tpl: LayerTpl = None
    norm_policy: str = "pre"
    atten_dropout_prob: float = 0.0
    num_heads: int = 1
    dim_per_head: int | None = None
    force_no_residual: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        atten_mask: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply pre/post-normalized self-attention with residual connection to `inputs`.

        Args:
            inputs: Input tensor of shape `[B, T, D]`.
            atten_mask: Optional additive attention mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` of shape `[B, T, D]`.
        """
        x = np.asarray(inputs)
        d_dim = x.shape[-1]
        n_heads = max(1, self.num_heads)
        h_dim = (
            self.dim_per_head
            if (self.dim_per_head is not None and self.dim_per_head > 0)
            else max(1, d_dim // n_heads)
        )

        ln = instantiate_layer(self.norm_tpl, LayerNorm(dim=d_dim))
        default_attn = DotProductAttention(
            input_dim=d_dim,
            num_heads=n_heads,
            dim_per_head=h_dim,
            params_init=self.params_init,
        )
        attn = instantiate_layer(self.self_atten_tpl, default_attn)

        use_pre = self.pre_layer_norm and self.norm_policy != "post"
        normed = ln(x) if use_pre else x
        atten_out = (
            attn(normed, atten_mask=atten_mask)
            if isinstance(attn, DotProductAttention)
            else attn(normed)
        )
        drop_layer = instantiate_layer(self.residual_dropout_tpl, Identity())
        atten_out = drop_layer(atten_out)

        if self.force_no_residual:
            out = atten_out
        else:
            out = self.input_weight * x + self.residual_weight * atten_out

        if not use_pre:
            out = ln(out)

        return out


__all__ = [
    "AttentionProjection",
    "DotProductAttention",
    "DotProductAttentionWithContext",
    "DotProductAttentionWithContextXL",
    "DotProductAttentionXL",
    "GroupedQueryAttention",
    "LocalSelfAttention",
    "LocalSelfAttentionAlibi",
    "LocalSelfAttentionRelativeBias",
    "LocalSelfAttentionXL",
    "RelativeBias",
    "SelfAttentionWithNormAndResidual",
]
