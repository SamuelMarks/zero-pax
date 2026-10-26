"""Attention mechanisms and projections for zero_pax.praxis.layers."""

from __future__ import annotations

import math

from zero_pax import _tensor_ops as _ops
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl
from zero_pax.praxis.layers.activations import Identity
from zero_pax.praxis.layers.convolutions import CausalDepthwiseConv1D
from zero_pax.praxis.layers.linear_pooling import EinsumOp, PerDimScale
from zero_pax.praxis.layers.normalizations import LayerNorm
from zero_pax.praxis.layers.quantization_utils import Dropout
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
    fraction = 2.0 * _ops.arange(0, half_dim) / float(head_dim)
    timescale = float(min_timescale) * ((float(max_timescale) / float(min_timescale)) ** fraction)
    positions = _ops.reshape(_ops.arange(0, seq_len), (1, seq_len, 1, 1))
    sinusoid_inp = positions / _ops.reshape(timescale, (1, 1, 1, half_dim))
    sin = _ops.sin(sinusoid_inp)
    cos = _ops.cos(sinusoid_inp)

    first_half = inputs[:, :, :, :half_dim]
    second_half = inputs[:, :, :, half_dim : 2 * half_dim]
    rot_first = first_half * cos - second_half * sin
    rot_second = second_half * cos + first_half * sin
    rotated = _ops.concatenate([rot_first, rot_second], axis=-1)
    if head_dim % 2 == 1:
        rotated = _ops.concatenate([rotated, inputs[:, :, :, 2 * half_dim :]], axis=-1)
    return rotated


def _normalize_attention_mask(atten_mask: TensorOrScalar) -> Tensor:
    """Broadcast-normalize a 2D, 3D, or 4D attention mask into 4D `[B_or_1, N_or_1, T_q, T_k]`.

    Args:
        atten_mask: Raw additive attention mask tensor.

    Returns:
        4D broadcast-compatible attention mask `Tensor`.
    """
    mask_t = _ops.asarray(atten_mask)
    if mask_t.ndim == 2:
        return _ops.reshape(mask_t, (1, 1, mask_t.shape[0], mask_t.shape[1]))
    if mask_t.ndim == 3:
        return _ops.expand_dims(mask_t, 1)
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
    num_heads: int = 0
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
        x = self._to_tensor(inputs)
        n_heads = self.num_heads if self.num_heads > 0 else 1

        if self.is_output_projection:
            if n_heads == 1 and x.ndim >= 2:
                n_heads = x.shape[-2]
            h_dim = self.dim_per_head if self.dim_per_head > 0 else x.shape[-1]
            d_dim = self.input_dim if self.input_dim > 0 else n_heads * h_dim
            if w is None:
                w_shape = (n_heads, h_dim, d_dim) if self.use_nhd_shape else (d_dim, n_heads, h_dim)
                w_init = self.params_init
                if (
                    self.explicit_fan_in_fan_out_axes
                    and w_init is not None
                    and w_init.method != "zeros"
                ):
                    fan_in = float(n_heads * h_dim)
                    fan_out = float(d_dim)
                    eff_scale = w_init.scale * math.sqrt(
                        (fan_in + float(h_dim)) / max(1.0, fan_in + fan_out)
                    )
                    w_init = WeightInit(method=w_init.method, scale=eff_scale, seed=w_init.seed)
                w_raw = self.get_variable("w", w_shape, init=w_init)
            else:
                w_raw = _ops.asarray(w)
            if w_raw.shape == (n_heads, h_dim, d_dim):
                w_t = w_raw
            elif w_raw.shape == (d_dim, n_heads, h_dim):
                w_t = _ops.einsum("dnh->nhd", w_raw)
            else:
                w_t = _ops.reshape(w_raw, (n_heads, h_dim, d_dim))
            op_out = self.get_sublayer(
                "einsum_out", self.einsum_tpl, EinsumOp(equation="...nh,nhd->...d")
            )
            out = (
                op_out("...nh,nhd->...d", x, w_t)
                if isinstance(op_out, EinsumOp)
                else op_out(x, w_t)
            )
            if self.use_bias:
                b_t = self.get_variable("b", (d_dim,), explicit=b)
                out = out + b_t
            return out

        d_dim = self.input_dim if self.input_dim > 0 else x.shape[-1]
        h_dim = self.dim_per_head if self.dim_per_head > 0 else max(1, d_dim // n_heads)
        if w is None:
            w_init_in = self.params_init
            if (
                self.explicit_fan_in_fan_out_axes
                and w_init_in is not None
                and w_init_in.method != "zeros"
            ):
                fan_in = float(d_dim)
                fan_out = float(n_heads * h_dim)
                eff_scale = w_init_in.scale * math.sqrt(
                    (fan_in + float(h_dim)) / max(1.0, fan_in + fan_out)
                )
                w_init_in = WeightInit(
                    method=w_init_in.method, scale=eff_scale, seed=w_init_in.seed
                )
            w_t = self.get_variable("w", (d_dim, n_heads, h_dim), init=w_init_in)
        else:
            w_raw = _ops.asarray(w)
            w_t = (
                _ops.reshape(w_raw, (d_dim, n_heads, h_dim))
                if w_raw.shape != (d_dim, n_heads, h_dim)
                else w_raw
            )
        op_in = self.get_sublayer(
            "einsum_in", self.einsum_tpl, EinsumOp(equation="...d,dnh->...nh")
        )
        out = op_in("...d,dnh->...nh", x, w_t) if isinstance(op_in, EinsumOp) else op_in(x, w_t)
        if self.use_bias:
            if b is None:
                b_t = self.get_variable("b", (n_heads, h_dim))
            else:
                b_raw = _ops.asarray(b)
                b_t = (
                    _ops.reshape(b_raw, (n_heads, h_dim))
                    if b_raw.shape != (n_heads, h_dim)
                    else b_raw
                )
            out = out + b_t
        if self.attention_combine_dims:
            out = _ops.reshape(out, list(out.shape[:-2]) + [n_heads * h_dim])
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
    bidirectional: bool = False
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
        ret = _ops.zeros_like(relative_position)
        n = -relative_position

        if self.bidirectional:
            num_buckets = num_buckets // 2
            ret = ret + _ops.where(n < 0.0, float(num_buckets), 0.0)
            n = _ops.abs(n)
        else:
            n = _ops.maximum(n, 0.0)

        max_exact = max(1, num_buckets // 2)
        is_small = n < float(max_exact)
        n_safe = _ops.maximum(n, 1.0)
        denom = math.log(max(float(max_distance) / float(max_exact), 1.0001))
        val_if_large = float(max_exact) + (
            _ops.log(n_safe / float(max_exact)) / denom * float(num_buckets - max_exact)
        )
        val_if_large = _ops.minimum(val_if_large, float(num_buckets - 1))
        return ret + _ops.where(is_small, n, val_if_large)

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
            q_pos = _ops.expand_dims(_ops.arange(query_segment_pos), 0)
        else:
            q_arr = _ops.asarray(query_segment_pos)
            if self.use_length_as_position and q_arr.ndim == 0:
                q_pos = _ops.expand_dims(_ops.arange(max(1, int(q_arr))), 0)
            else:
                q_pos = _ops.expand_dims(q_arr, 0) if q_arr.ndim == 1 else q_arr

        k_pos: Tensor
        if key_segment_pos is None:
            k_pos = q_pos
        elif isinstance(key_segment_pos, int):
            k_pos = _ops.expand_dims(_ops.arange(key_segment_pos), 0)
        else:
            k_arr = _ops.asarray(key_segment_pos)
            if self.use_length_as_position and k_arr.ndim == 0:
                k_pos = _ops.expand_dims(_ops.arange(max(1, int(k_arr))), 0)
            else:
                k_pos = _ops.expand_dims(k_arr, 0) if k_arr.ndim == 1 else k_arr

        rel_pos = _ops.expand_dims(k_pos, 1) - _ops.expand_dims(q_pos, 2)
        buckets = self._relative_position_bucket(rel_pos)

        n_heads = max(1, self.num_heads)
        n_buckets = max(2, self.relative_attention_num_buckets)
        rb_init: WeightInit | None = None
        if self.use_xavier_init:
            rb_init = WeightInit(method="xavier", scale=self.rb_stddev or 1.0)
        elif self.rb_stddev is not None:
            rb_init = WeightInit(method="normal", scale=self.rb_stddev)
        w_t = self.get_variable("w", (n_heads, n_buckets), explicit=w, init=rb_init)

        bucket_idx = _ops.clip(buckets, 0, n_buckets - 1)
        one_hot = _ops.reshape(bucket_idx, list(bucket_idx.shape) + [1]) == _ops.arange(n_buckets)
        return _ops.einsum("bqkc,nc->bnqk", one_hot, w_t)


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
    dim_per_head: int | None = None
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
    scale_query_by_dim_per_head: bool | None = False
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
        q_in = self._to_tensor(query_vec)
        k_in = q_in if key_vec is None else self._to_tensor(key_vec)
        v_in = q_in if value_vec is None else self._to_tensor(value_vec)

        if self.dconv_qkv:
            k_sz = max(1, self.dconv_kernel_size)
            dconv_layer = self.get_sublayer(
                "dconv_qkv",
                self.causal_depthwise_conv1d_tpl,
                CausalDepthwiseConv1D(kernel_size=k_sz, params_init=self.params_init),
            )
            q_in = dconv_layer(q_in)
            k_in = dconv_layer(k_in)
            v_in = dconv_layer(v_in)

        if self.ngrammer_tpl is not None:
            ngrammer = self.get_sublayer("ngrammer", self.ngrammer_tpl, Identity())
            v_in = ngrammer(v_in)

        d_dim = self.input_dim if self.input_dim > 0 else q_in.shape[-1]
        n_heads = self.num_heads if self.num_heads > 0 else 1
        h_dim = (
            self.dim_per_head
            if (self.dim_per_head is not None and self.dim_per_head > 0)
            else (self.hidden_dim // n_heads if self.hidden_dim > 0 else max(1, d_dim // n_heads))
        )
        eff_init = (
            WeightInit(method="normal", scale=1.0 / math.sqrt(float(max(1, d_dim))))
            if self.internal_gshard_gaussian_init
            else self.params_init
        )
        shard_flag = self.decoding_maybe_shard_projections

        proj_post = self.get_sublayer(
            "post",
            None,
            AttentionProjection(
                input_dim=d_dim,
                num_heads=n_heads,
                dim_per_head=h_dim,
                is_output_projection=True,
                use_bias=self.use_bias,
                use_nhd_shape=self.output_proj_use_nhd_shape,
                explicit_fan_in_fan_out_axes=shard_flag,
                params_init=eff_init,
            ),
        )

        if (
            (self.combine_qkv or self.combined_qkv_proj_tpl is not None)
            and key_vec is None
            and value_vec is None
            and query_w is None
            and key_w is None
            and value_w is None
        ):
            comb_proj = self.get_sublayer(
                "combined_qkv",
                self.combined_qkv_proj_tpl,
                AttentionProjection(
                    input_dim=d_dim,
                    num_heads=3 * n_heads,
                    dim_per_head=h_dim,
                    is_output_projection=False,
                    use_bias=self.use_bias,
                    explicit_fan_in_fan_out_axes=shard_flag,
                    params_init=eff_init,
                ),
            )
            qkv = comb_proj(q_in)
            q, k, v = _ops.split(qkv, 3, axis=-2)
        else:
            proj_q = self.get_sublayer(
                "query",
                self.proj_tpl,
                AttentionProjection(
                    input_dim=d_dim,
                    num_heads=n_heads,
                    dim_per_head=h_dim,
                    is_output_projection=False,
                    use_bias=self.use_bias,
                    explicit_fan_in_fan_out_axes=shard_flag,
                    params_init=eff_init,
                ),
            )
            proj_k = self.get_sublayer(
                "key",
                self.proj_tpl,
                AttentionProjection(
                    input_dim=d_dim,
                    num_heads=n_heads,
                    dim_per_head=h_dim,
                    is_output_projection=False,
                    use_bias=self.use_bias,
                    explicit_fan_in_fan_out_axes=shard_flag,
                    params_init=eff_init,
                ),
            )
            proj_v = self.get_sublayer(
                "value",
                self.proj_tpl,
                AttentionProjection(
                    input_dim=d_dim,
                    num_heads=n_heads,
                    dim_per_head=h_dim,
                    is_output_projection=False,
                    use_bias=self.use_bias,
                    explicit_fan_in_fan_out_axes=shard_flag,
                    params_init=eff_init,
                ),
            )
            q = (
                proj_q(q_in, w=query_w, b=query_b)
                if isinstance(proj_q, AttentionProjection)
                else proj_q(q_in)
            )
            k = (
                proj_k(k_in, w=key_w, b=key_b)
                if isinstance(proj_k, AttentionProjection)
                else proj_k(k_in)
            )
            v = (
                proj_v(v_in, w=value_w, b=value_b)
                if isinstance(proj_v, AttentionProjection)
                else proj_v(v_in)
            )

        if self.ln_tpl is not None:
            qk_ln = self.get_sublayer("qk_ln", self.ln_tpl, LayerNorm(dim=h_dim))
            q = qk_ln(q)
            k = qk_ln(k)

        if self.use_rotary_position_emb:
            if self.rotary_position_emb_tpl is not None:
                rope_layer = self.get_sublayer("rope", self.rotary_position_emb_tpl, Identity())
                q = rope_layer(q)
                k = rope_layer(k)
            else:
                q = _apply_rotary_position_emb(q)
                k = _apply_rotary_position_emb(k)
            if self.cast_rotary_position_emb and self.fprop_dtype is not None:
                q = q.astype(self.fprop_dtype)
                k = k.astype(self.fprop_dtype)

        if self.decode_cache:
            self.register_parameter(
                "key_state" if not self.consolidate_rope_key_state else "rope_key_state", k
            )
            self.register_parameter("value_state", v)

        if (
            self.internal_enable_query_scale
            or self.scale_query_by_dim_per_head
            or self.scale_query_by_head_dims
        ):
            if self.internal_enable_per_dim_scale and (
                per_dim_scale is not None or self.per_dim_scale_tpl is not None
            ):
                pds_layer = self.get_sublayer(
                    "per_dim_scale",
                    self.per_dim_scale_tpl,
                    PerDimScale(dim=h_dim, params_init=self.params_init),
                )
                q = (
                    pds_layer(q, scale=per_dim_scale)
                    if isinstance(pds_layer, PerDimScale)
                    else pds_layer(q)
                )
            q = q / _ops.sqrt(float(h_dim))

        qk_einsum = self.get_sublayer(
            "qk_einsum", self.qk_einsum_tpl, EinsumOp(equation="btnh,bsnh->bnts")
        )
        logits = (
            qk_einsum("btnh,bsnh->bnts", q, k)
            if isinstance(qk_einsum, EinsumOp)
            else qk_einsum(q, k)
        )
        if self.scale_logits_by_head_dims:
            logits = logits * (float(h_dim) ** -0.5)

        if self.atten_logit_cap > 0.0:
            logits = self.atten_logit_cap * _ops.tanh(logits / self.atten_logit_cap)

        if self.relative_bias_tpl is not None:
            rb_layer = self.get_sublayer(
                "relative_bias",
                self.relative_bias_tpl,
                RelativeBias(num_heads=n_heads, params_init=self.params_init),
            )
            logits = logits + (
                rb_layer(q_in.shape[1], k_in.shape[1])
                if isinstance(rb_layer, RelativeBias)
                else rb_layer(q_in.shape[1])
            )

        xl_bias = self._compute_xl_bias(q_in.shape[1], k_in.shape[1], n_heads)
        if xl_bias is not None:
            logits = logits + xl_bias

        ctx_mask = self._compute_context_mask(q_in.shape[1])
        if ctx_mask is not None and q_in.shape[1] == k_in.shape[1]:
            logits = logits + ctx_mask

        norm_mask: Tensor | None = None
        if atten_mask is not None:
            norm_mask = _normalize_attention_mask(atten_mask)
            if self.attention_mask_summary:
                self.register_parameter("atten_mask_summary", _ops.mean(norm_mask))
            logits = logits + norm_mask

        max_logits = _ops.max(logits, axis=-1, keepdims=True)
        exp_logits = _ops.exp(logits - max_logits)
        denom = _ops.sum(exp_logits, axis=-1, keepdims=True)
        if self.attention_extra_logit is not None:
            denom = denom + _ops.exp(float(self.attention_extra_logit) - max_logits)
        probs = exp_logits / _ops.maximum(denom, 1e-12)

        if self.zero_fully_masked and norm_mask is not None:
            fully_masked = _ops.max(norm_mask, axis=-1, keepdims=True) < -1e5
            probs = _ops.where(fully_masked, 0.0, probs)

        default_drop: BasePraxisLayer = (
            Dropout(keep_prob=max(0.0, 1.0 - self.atten_dropout_prob))
            if self.atten_dropout_prob > 0.0
            else Identity()
        )
        drop_layer = self.get_sublayer("dropout", self.dropout_tpl, default_drop)
        probs = drop_layer(probs)

        pv_einsum = self.get_sublayer(
            "pv_einsum", self.pv_einsum_tpl, EinsumOp(equation="bnts,bsnh->btnh")
        )
        encoded = (
            pv_einsum("bnts,bsnh->btnh", probs, v)
            if isinstance(pv_einsum, EinsumOp)
            else pv_einsum(probs, v)
        )
        return (
            proj_post(encoded, w=post_w, b=post_b)
            if isinstance(proj_post, AttentionProjection)
            else proj_post(encoded)
        )


class DotProductAttentionXL(DotProductAttention):
    """Transformer-XL style Dot-Product Attention with relative sinusoidal positional bias.

    Attributes:
        rel_pos_emb_dim: Dimension of the relative positional embedding.
    """

    rel_pos_emb_dim: int | None = 0

    def _compute_xl_bias(self, q_len: int, k_len: int, n_heads: int) -> Tensor | None:
        """Compute Transformer-XL multi-frequency sinusoidal relative distance bias `(1, N, Q, K)`.

        Args:
            q_len: Query sequence length.
            k_len: Key sequence length.
            n_heads: Number of attention heads.

        Returns:
            Sinusoidal relative positional bias `Tensor` of shape `(1, N, Q, K)` or `None`.
        """
        if self.rel_pos_emb_dim is None or self.rel_pos_emb_dim <= 0:
            return None
        emb_dim = max(2, int(self.rel_pos_emb_dim))
        half_dim = max(1, emb_dim // 2)
        u_bias = self.get_variable("u_bias", (1, n_heads, 1, 1))
        v_bias = self.get_variable("v_bias", (1, n_heads, 1, 1))
        q_pos = _ops.expand_dims(_ops.arange(q_len), 1)
        k_pos = _ops.expand_dims(_ops.arange(k_len), 0)
        rel_dist = k_pos - q_pos
        freq_seq = 1.0 / (10000.0 ** (_ops.arange(half_dim) / max(1.0, float(half_dim))))
        sinusoid_inp = _ops.expand_dims(rel_dist, -1) * _ops.reshape(freq_seq, (1, 1, half_dim))
        pos_emb = _ops.concatenate([_ops.sin(sinusoid_inp), _ops.cos(sinusoid_inp)], axis=-1)
        pos_summary = _ops.mean(pos_emb, axis=-1)
        head_slopes = 1.0 / (10000.0 ** (_ops.arange(n_heads) / max(1.0, float(n_heads))))
        phase = _ops.reshape(head_slopes, (1, n_heads, 1, 1)) * _ops.reshape(
            rel_dist, (1, 1, q_len, k_len)
        )
        sin_term = (_ops.cos(phase) + _ops.reshape(pos_summary, (1, 1, q_len, k_len))) / math.sqrt(
            float(self.rel_pos_emb_dim)
        )
        return u_bias + (1.0 + v_bias) * sin_term


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
        pos = _ops.arange(seq_len)
        diff = _ops.expand_dims(pos, 0) - _ops.expand_dims(pos, 1)
        masked = _ops.zeros((seq_len, seq_len)) > 1.0
        if self.left_context is not None:
            masked = masked | (diff < -(self.left_context - 1))
        if self.right_context is not None:
            masked = masked | (diff > self.right_context)
        mask_2d = _ops.where(masked, -1e9, 0.0)
        return _ops.reshape(mask_2d, (1, 1, seq_len, seq_len))


class DotProductAttentionWithContextXL(DotProductAttentionWithContext, DotProductAttentionXL):
    """Dot-Product Attention combining local `[left_context, right_context]` masking and XL relative bias.

    Attributes:
        rel_pos_emb_dim: Dimension of the relative positional embedding.
    """

    rel_pos_emb_dim: int | None = None

    def _compute_xl_bias(self, q_len: int, k_len: int, n_heads: int) -> Tensor | None:
        """Compute XL relative positional bias when `rel_pos_emb_dim` is positive.

        Args:
            q_len: Query sequence length.
            k_len: Key sequence length.
            n_heads: Number of attention heads.

        Returns:
            Optional XL relative bias `Tensor` or `None`.
        """
        if self.rel_pos_emb_dim is None or self.rel_pos_emb_dim <= 0:
            return None
        return super()._compute_xl_bias(q_len, k_len, n_heads)


class LocalSelfAttention(DotProductAttentionWithContext):
    """Blockwise / sliding-window Local Self-Attention layer.

    Attributes:
        block_size: Local attention block length.
        simulated: Whether local block attention is simulated via full attention masking.
    """

    block_size: int | None = None
    simulated: bool = False

    def _compute_context_mask(self, seq_len: int) -> Tensor | None:
        """Compute banded context mask or block-diagonal mask when `block_size > 0`.

        Args:
            seq_len: Sequence length `T`.

        Returns:
            Additive local attention mask `Tensor` of shape `(1, 1, T, T)` or `None`.
        """
        base_mask = super()._compute_context_mask(seq_len)
        if base_mask is not None or self.block_size is None or self.block_size <= 0:
            return base_mask
        pos = _ops.arange(seq_len)
        if self.simulated:
            blk_idx = _ops.floor(pos / float(self.block_size))
            masked = _ops.expand_dims(blk_idx, 0) != _ops.expand_dims(blk_idx, 1)
        else:
            diff = _ops.abs(_ops.expand_dims(pos, 0) - _ops.expand_dims(pos, 1))
            masked = diff > self.block_size
        mask_2d = _ops.where(masked, -1e9, 0.0)
        return _ops.reshape(mask_2d, (1, 1, seq_len, seq_len))


class LocalSelfAttentionXL(LocalSelfAttention, DotProductAttentionXL):
    """Local Self-Attention layer with Transformer-XL relative positional encoding.

    Attributes:
        rel_pos_emb_dim: Dimension of the relative positional embedding.
    """

    rel_pos_emb_dim: int | None = 0

    def _compute_xl_bias(self, q_len: int, k_len: int, n_heads: int) -> Tensor | None:
        """Compute XL relative positional bias when `rel_pos_emb_dim` is positive.

        Args:
            q_len: Query sequence length.
            k_len: Key sequence length.
            n_heads: Number of attention heads.

        Returns:
            Optional XL relative bias `Tensor` or `None`.
        """
        if self.rel_pos_emb_dim is None or self.rel_pos_emb_dim <= 0:
            return None
        return super()._compute_xl_bias(q_len, k_len, n_heads)


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
        rb = self.get_sublayer(
            "default_relative_bias",
            None,
            RelativeBias(num_heads=n_heads, params_init=self.params_init),
        )
        return rb(q_len, k_len) if isinstance(rb, RelativeBias) else rb(q_len)


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
        pos = _ops.arange(seq_len)
        dist = -_ops.abs(_ops.expand_dims(pos, 0) - _ops.expand_dims(pos, 1))
        n_heads = max(1, self.num_heads)
        slopes = 0.5 ** (_ops.arange(1, n_heads + 1) * (8.0 / float(n_heads)))
        alibi = _ops.reshape(slopes, (1, n_heads, 1, 1)) * _ops.reshape(
            dist, (1, 1, seq_len, seq_len)
        )
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
    dim_per_head: int = 0
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
        q_in = self._to_tensor(query_vec)
        k_in = q_in if key_vec is None else self._to_tensor(key_vec)
        v_in = q_in if value_vec is None else self._to_tensor(value_vec)

        d_dim = self.input_dim if self.input_dim > 0 else q_in.shape[-1]
        n_heads = max(1, self.num_heads)
        kv_heads = max(1, min(self.num_kv_heads if self.num_kv_heads > 0 else n_heads, n_heads))
        if n_heads % kv_heads != 0:
            kv_heads = 1
        h_dim = (
            self.dim_per_head
            if self.dim_per_head > 0
            else (self.hidden_dim // n_heads if self.hidden_dim > 0 else max(1, d_dim // n_heads))
        )

        if (
            self.proj_tpl is not None
            and w_q is None
            and w_k is None
            and w_v is None
            and w_o is None
        ):
            q_proj = self.get_sublayer(
                "q_proj",
                self.proj_tpl,
                AttentionProjection(
                    input_dim=d_dim, num_heads=n_heads, dim_per_head=h_dim, use_bias=self.use_bias
                ),
            )
            k_proj = self.get_sublayer(
                "k_proj",
                self.proj_tpl,
                AttentionProjection(
                    input_dim=d_dim, num_heads=kv_heads, dim_per_head=h_dim, use_bias=self.use_bias
                ),
            )
            v_proj = self.get_sublayer(
                "v_proj",
                self.proj_tpl,
                AttentionProjection(
                    input_dim=d_dim, num_heads=kv_heads, dim_per_head=h_dim, use_bias=self.use_bias
                ),
            )
            q = q_proj(q_in)
            k = k_proj(k_in)
            v = v_proj(v_in)
            wo_t = self.get_variable("w_o", (n_heads, h_dim, d_dim), explicit=w_o)
        else:
            wq_t = self.get_variable("w_q", (d_dim, n_heads, h_dim), explicit=w_q)
            wk_t = self.get_variable("w_k", (d_dim, kv_heads, h_dim), explicit=w_k)
            wv_t = self.get_variable("w_v", (d_dim, kv_heads, h_dim), explicit=w_v)
            wo_t = self.get_variable("w_o", (n_heads, h_dim, d_dim), explicit=w_o)

            q = _ops.einsum("btd,dnh->btnh", q_in, wq_t)
            k = _ops.einsum("bsd,dkh->bskh", k_in, wk_t)
            v = _ops.einsum("bsd,dkh->bskh", v_in, wv_t)

            if self.use_bias:
                bq_t = self.get_variable("b_q", (n_heads, h_dim), explicit=b_q)
                bk_t = self.get_variable("b_k", (kv_heads, h_dim), explicit=b_k)
                bv_t = self.get_variable("b_v", (kv_heads, h_dim), explicit=b_v)
                q = q + bq_t
                k = k + bk_t
                v = v + bv_t

        if self.rope_min_max_timescales is not None:
            min_ts, max_ts = self.rope_min_max_timescales
            q = _apply_rotary_position_emb(q, min_timescale=min_ts, max_timescale=max_ts)
            k = _apply_rotary_position_emb(k, min_timescale=min_ts, max_timescale=max_ts)

        if n_heads != kv_heads:
            repeats = n_heads // kv_heads
            b_sz, s_len, _, _ = k.shape
            k_exp = _ops.broadcast_to(
                _ops.expand_dims(k, 3), (b_sz, s_len, kv_heads, repeats, h_dim)
            )
            v_exp = _ops.broadcast_to(
                _ops.expand_dims(v, 3), (b_sz, s_len, kv_heads, repeats, h_dim)
            )
            k = _ops.reshape(k_exp, (b_sz, s_len, n_heads, h_dim))
            v = _ops.reshape(v_exp, (b_sz, s_len, n_heads, h_dim))

        temp_scale = self.atten_temp if self.atten_temp > 0.0 else 1.0
        q = q / (_ops.sqrt(float(h_dim)) * temp_scale)
        logits = _ops.einsum("btnh,bsnh->bnts", q, k)
        if self.atten_logit_cap > 0.0:
            logits = self.atten_logit_cap * _ops.tanh(logits / self.atten_logit_cap)
        if atten_mask is not None:
            logits = logits + _normalize_attention_mask(atten_mask)

        max_logits = _ops.max(logits, axis=-1, keepdims=True)
        exp_logits = _ops.exp(logits - max_logits)
        probs = exp_logits / _ops.maximum(_ops.sum(exp_logits, axis=-1, keepdims=True), 1e-12)

        default_gqa_drop: BasePraxisLayer = (
            Dropout(keep_prob=max(0.0, 1.0 - self.atten_dropout_prob))
            if self.atten_dropout_prob > 0.0
            else Identity()
        )
        drop_layer = self.get_sublayer("dropout", self.dropout_tpl, default_gqa_drop)
        probs = drop_layer(probs)

        encoded = _ops.einsum("bnts,bsnh->btnh", probs, v)
        out = _ops.einsum("btnh,nhd->btd", encoded, wo_t)
        if self.use_bias:
            bo_t = self.get_variable("b_o", (d_dim,), explicit=b_o)
            out = out + bo_t
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
    pre_layer_norm: bool | None = True
    residual_dropout_prob: float = 0.0
    residual_dropout_tpl: LayerTpl = None
    norm_policy: str | None = None
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
        x = self._to_tensor(inputs)
        d_dim = x.shape[-1]
        n_heads = max(1, self.num_heads)
        h_dim = (
            self.dim_per_head
            if (self.dim_per_head is not None and self.dim_per_head > 0)
            else max(1, d_dim // n_heads)
        )

        ln = self.get_sublayer("norm", self.norm_tpl, LayerNorm(dim=d_dim))
        default_attn = DotProductAttention(
            input_dim=d_dim,
            num_heads=n_heads,
            dim_per_head=h_dim,
            atten_dropout_prob=self.atten_dropout_prob,
            params_init=self.params_init,
        )
        attn = self.get_sublayer("self_atten", self.self_atten_tpl, default_attn)

        use_pre = bool(self.pre_layer_norm) and self.norm_policy != "post"
        normed = ln(x) if use_pre else x
        atten_out = (
            attn(normed, atten_mask=atten_mask)
            if isinstance(attn, DotProductAttention)
            else attn(normed)
        )
        default_res_drop: BasePraxisLayer = (
            Dropout(keep_prob=max(0.0, 1.0 - self.residual_dropout_prob))
            if self.residual_dropout_prob > 0.0
            else Identity()
        )
        drop_layer = self.get_sublayer(
            "residual_dropout", self.residual_dropout_tpl, default_res_drop
        )
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
