"""Transformer, Mixture-of-Experts, and Language Model layers for zero_pax.praxis.layers."""

from __future__ import annotations

from collections.abc import Sequence

from zero_pax import _tensor_ops as np
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl, instantiate_layer
from zero_pax.praxis.layers.activations import Identity, ReLU
from zero_pax.praxis.layers.attentions import DotProductAttention
from zero_pax.praxis.layers.embeddings import (
    FullSoftmax,
    PositionalEmbedding,
    SharedEmbeddingSoftmax,
)
from zero_pax.praxis.layers.linear_pooling import FeedForward
from zero_pax.praxis.layers.normalizations import LayerNorm
from zero_pax.types import (
    DecoderHParams,
    SplitDimsMapping,
    Tensor,
    TensorOrScalar,
)


def _build_causal_mask(seq_len: int) -> Tensor:
    """Construct a lower-triangular additive causal attention mask of shape `(1, 1, T, T)`.

    Args:
        seq_len: Sequence length `T`.

    Returns:
        Additive causal mask `Tensor` (`0.0` for past/present, `-1e9` for future).
    """
    pos = np.arange(seq_len)
    future = np.expand_dims(pos, 0) > np.expand_dims(pos, 1)
    mask_2d = np.where(future, -1e9, 0.0)
    return np.reshape(mask_2d, (1, 1, seq_len, seq_len))


class TransformerFeedForward(BasePraxisLayer):
    """Transformer Position-wise Feed-Forward Network (FFN) with LayerNorm, Gating, and Residual.

    Supports both standard 2-layer FFN (`x -> act(x @ w1 + b1) @ w2 + b2`) and Gated Linear Unit
    FFN (SwiGLU / GeGLU: `(act(x @ w1_gate) * (x @ w1)) @ w2`).

    Attributes:
        input_dims: Input feature dimension `D`.
        output_dims: Output feature dimension (`0` defaults to `input_dims`).
        hidden_dims: Intermediate FFN expansion dimension `F`.
        has_bias: Whether linear projections include bias vectors.
        apply_padding_first: Whether to zero out padded tokens before normalization.
        ln_tpl: Pre/post LayerNorm template.
        fflayer_tpl: Underlying `FeedForward` template.
        activation_tpl: Non-linear activation layer or template (defaults to `ReLU`).
        use_gated_activation: Whether to use a two-branch gated activation (GLU/SwiGLU/GeGLU).
        ffn0_tpl: First linear projection template (`D -> F`).
        ffn1_tpl: Gating linear projection template (`D -> F`) when `use_gated_activation` is True.
        ffn2_tpl: Output linear projection template (`F -> D_out`).
        relu_dropout_tpl: Dropout template applied after the intermediate activation.
        relu_dropout_prob: Dropout probability after intermediate activation.
        residual_dropout_tpl: Dropout template applied before the residual addition.
        residual_dropout_prob: Dropout probability on the residual branch.
        add_skip_connection: Whether to add the residual skip connection (`input_dims == output_dims`).
        residual_weight: Scaling multiplier applied to the FFN branch before residual addition.
        residual_droppath_prob: Stochastic depth drop-path probability.
        norm_policy: Normalization placement policy (`'pre'`, `'primer_hybrid'`, `'post'`, `'none'`).
        internal_gshard_variance_scaling_fan_in_init: Whether to use GShard fan-in variance scaling.
    """

    input_dims: int = 0
    output_dims: int = 0
    hidden_dims: int = 0
    has_bias: bool = True
    apply_padding_first: bool = False
    ln_tpl: LayerTpl = None
    fflayer_tpl: LayerTpl = None
    activation_tpl: LayerTpl = None
    use_gated_activation: bool = False
    ffn0_tpl: LayerTpl = None
    ffn1_tpl: LayerTpl = None
    ffn2_tpl: LayerTpl = None
    relu_dropout_tpl: LayerTpl = None
    relu_dropout_prob: float = 0.0
    residual_dropout_tpl: LayerTpl = None
    residual_dropout_prob: float = 0.0
    add_skip_connection: bool = True
    residual_weight: float = 1.0
    residual_droppath_prob: float = 0.0
    norm_policy: str = "pre"
    internal_gshard_variance_scaling_fan_in_init: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        segment_Mask: TensorOrScalar | None = None,
        w1: TensorOrScalar | None = None,
        b1: TensorOrScalar | None = None,
        w1_gate: TensorOrScalar | None = None,
        b1_gate: TensorOrScalar | None = None,
        w2: TensorOrScalar | None = None,
        b2: TensorOrScalar | None = None,
        ln_scale: TensorOrScalar | None = None,
        ln_bias: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the Transformer Feed-Forward block to `inputs`.

        Args:
            inputs: Input sequence tensor of shape `[..., D]`.
            paddings: Optional padding mask tensor of shape `[B, T]`.
            segment_Mask: Optional segment mask tensor.
            w1: Optional first-layer weight matrix of shape `[D, F]`.
            b1: Optional first-layer bias vector of shape `[F]`.
            w1_gate: Optional gating branch weight matrix of shape `[D, F]`.
            b1_gate: Optional gating branch bias vector of shape `[F]`.
            w2: Optional output projection weight matrix of shape `[F, D_out]`.
            b2: Optional output projection bias vector of shape `[D_out]`.
            ln_scale: Optional LayerNorm scale vector.
            ln_bias: Optional LayerNorm bias vector.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` of shape `[..., D_out]`.
        """
        x = np.asarray(inputs)
        in_d = self.input_dims if self.input_dims > 0 else x.shape[-1]
        out_d = self.output_dims if self.output_dims > 0 else in_d
        hid_d = self.hidden_dims if self.hidden_dims > 0 else 4 * in_d

        if self.apply_padding_first and paddings is not None:
            pad_t = np.asarray(paddings)
            x = x * (1.0 - np.expand_dims(pad_t, -1))

        residual = x
        ln = instantiate_layer(self.ln_tpl, LayerNorm(dim=in_d))

        if self.norm_policy in ("pre", "primer_hybrid"):
            normed = ln(x, scale=ln_scale, bias=ln_bias) if isinstance(ln, LayerNorm) else ln(x)
        else:
            normed = x

        act_fn = instantiate_layer(self.activation_tpl, ReLU())

        if self.use_gated_activation:
            ff_gate = FeedForward(
                input_dims=in_d,
                output_dims=hid_d,
                has_bias=self.has_bias,
                activation_tpl=act_fn,
                params_init=self.params_init,
            )
            ff_lin = FeedForward(
                input_dims=in_d,
                output_dims=hid_d,
                has_bias=self.has_bias,
                activation_tpl=Identity(),
                params_init=self.params_init,
            )
            gate_val = ff_gate(normed, w=w1_gate, b=b1_gate)
            lin_val = ff_lin(normed, w=w1, b=b1)
            act_out = gate_val * lin_val
        else:
            ff1 = FeedForward(
                input_dims=in_d,
                output_dims=hid_d,
                has_bias=self.has_bias,
                activation_tpl=act_fn,
                params_init=self.params_init,
            )
            act_out = ff1(normed, w=w1, b=b1)

        relu_drop = instantiate_layer(self.relu_dropout_tpl, Identity())
        act_out = relu_drop(act_out)

        if self.norm_policy == "primer_hybrid":
            ln_mid = LayerNorm(dim=hid_d)
            act_out = ln_mid(act_out)

        ff2 = FeedForward(
            input_dims=hid_d,
            output_dims=out_d,
            has_bias=self.has_bias,
            activation_tpl=Identity(),
            params_init=self.params_init,
        )
        projected = ff2(act_out, w=w2, b=b2)
        res_drop = instantiate_layer(self.residual_dropout_tpl, Identity())
        projected = res_drop(projected)

        if self.add_skip_connection and in_d == out_d:
            out = residual + projected * self.residual_weight
        else:
            out = projected

        if self.norm_policy == "post":
            out = ln(out, scale=ln_scale, bias=ln_bias) if isinstance(ln, LayerNorm) else ln(out)

        if not self.apply_padding_first and paddings is not None:
            pad_t = np.asarray(paddings)
            out = out * (1.0 - np.expand_dims(pad_t, -1))

        return out


class AdaptedTransformerFeedForward(TransformerFeedForward):
    """Transformer Feed-Forward layer augmented with a residual bottleneck Adapter.

    Attributes:
        adapter_tpl: Optional callable adapter layer or template.
        mode: Adapter integration mode (`'parallel'` or `'sequential'`).
    """

    adapter_tpl: LayerTpl = None
    mode: str = "parallel"

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply `TransformerFeedForward` combined with `adapter_tpl` in `parallel` or `sequential` mode.

        Args:
            inputs: Input tensor of shape `[..., D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Adapted FFN output `Tensor`.
        """
        x = np.asarray(inputs)
        ffn_out = super().__call__(x, paddings, *args, **kwargs)
        if self.adapter_tpl is None:
            return ffn_out
        adapter = instantiate_layer(self.adapter_tpl, Identity())
        if self.mode == "parallel":
            return ffn_out + (adapter(x) - x)
        return adapter(ffn_out)


class TransformerFeedForwardMoe(BasePraxisLayer):
    """Mixture-of-Experts (MoE) Transformer Feed-Forward layer.

    Attributes:
        input_dims: Input and output feature dimension `D`.
        hidden_dims: Per-expert intermediate FFN dimension `F`.
        apply_padding_first: Whether to zero out padded tokens before normalization.
        ln_tpl: Pre-LayerNorm template.
        activation_tpl: Expert activation layer or template.
        relu_dropout_tpl: Expert intermediate dropout template.
        relu_dropout_prob: Expert intermediate dropout probability.
        residual_dropout_tpl: Output residual dropout template.
        residual_dropout_prob: Output residual dropout probability.
        add_skip_connection: Whether to include the residual skip connection.
        residual_weight: Scaling multiplier applied to the MoE branch.
        norm_policy: Normalization placement policy (`'pre'` or `'post'`).
        residual_droppath_prob: Stochastic depth drop-path probability.
        num_experts: Total number of routed experts `E`.
        num_groups: Number of token routing groups.
        min_group_size: Optional minimum routing group size.
        expert_capacity_dim: Per-expert token capacity dimension.
        expert_capacity_factor: Capacity factor multiplier (`1.0` default).
        expert_weight_shards: Number of shards for expert weights.
        second_expert_policy: Routing policy for the second-choice expert (`'all'`, `'random'`, `'none'`).
        internal_gshard_variance_scaling_fan_in_init: Whether to use GShard variance scaling.
        explicit_fan_in_fan_out_axes: Whether fan-in/fan-out axes are explicitly annotated.
        moe_load_balance_loss_weight: Auxiliary load-balancing loss coefficient.
        gating_func: Router gating function (`'top2'`, `'top1'`, `'expert_choice'`).
        gating_logit_cap: Tanh soft-capping limit for router logits.
        moe_gating_embedding_level: Level at which gating is computed (`'token'` or `'sentence'`).
        unadjusted_expert_capacity_factor: Capacity factor multiplier (`2.0` default).
        use_gated_activation: Whether experts use gated activation (SwiGLU/GeGLU).
    """

    input_dims: int = 0
    hidden_dims: int = 0
    apply_padding_first: bool = False
    ln_tpl: LayerTpl = None
    activation_tpl: LayerTpl = None
    relu_dropout_tpl: LayerTpl = None
    relu_dropout_prob: float = 0.0
    residual_dropout_tpl: LayerTpl = None
    residual_dropout_prob: float = 0.0
    add_skip_connection: bool = True
    residual_weight: float = 1.0
    norm_policy: str = "pre"
    residual_droppath_prob: float = 0.0
    num_experts: int = 0
    num_groups: int = 1
    min_group_size: int | None = None
    expert_capacity_dim: int = 0
    expert_capacity_factor: float = 1.0
    expert_weight_shards: int = 1
    second_expert_policy: str = "all"
    internal_gshard_variance_scaling_fan_in_init: bool = True
    explicit_fan_in_fan_out_axes: bool = False
    moe_load_balance_loss_weight: float | None = None
    gating_func: str = "top2"
    gating_logit_cap: float = 0.0
    moe_gating_embedding_level: str = "token"
    unadjusted_expert_capacity_factor: float = 2.0
    use_gated_activation: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        w1: TensorOrScalar | None = None,
        w2: TensorOrScalar | None = None,
        gate_w: TensorOrScalar | None = None,
        wi_0: TensorOrScalar | None = None,
        wo_0: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Route tokens through `num_experts` FFN experts and combine with residual connection.

        Args:
            inputs: Input sequence tensor of shape `[B, T, D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            w1: Optional shared/expert first-layer weight matrix of shape `[D, F]` or `[E, D, F]`.
            w2: Optional shared/expert second-layer weight matrix of shape `[F, D]` or `[E, F, D]`.
            gate_w: Optional router gating weight matrix of shape `[D, E]`.
            wi_0: Optional 3D per-expert input weight tensor of shape `[E, D, F]`.
            wo_0: Optional 3D per-expert output weight tensor of shape `[E, F, D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            MoE FFN output `Tensor` of shape `[B, T, D]`.
        """
        x = np.asarray(inputs)
        in_d = self.input_dims if self.input_dims > 0 else x.shape[-1]
        hid_d = self.hidden_dims if self.hidden_dims > 0 else 4 * in_d
        n_exp = max(1, self.num_experts)

        if (
            gate_w is None
            and wi_0 is None
            and wo_0 is None
            and (w1 is None or np.asarray(w1).ndim == 2)
        ):
            ffn = TransformerFeedForward(
                input_dims=self.input_dims,
                hidden_dims=self.hidden_dims,
                ln_tpl=self.ln_tpl,
                activation_tpl=self.activation_tpl,
                add_skip_connection=self.add_skip_connection,
                norm_policy=self.norm_policy,
                use_gated_activation=self.use_gated_activation,
                params_init=self.params_init,
            )
            return ffn(x, paddings=paddings, w1=w1, w2=w2)

        ln = instantiate_layer(self.ln_tpl, LayerNorm(dim=in_d))
        normed = ln(x) if self.norm_policy == "pre" else x

        gw_t = self.get_variable("gate_w", (in_d, n_exp), explicit=gate_w)
        wi_raw = wi_0 if wi_0 is not None else w1
        wo_raw = wo_0 if wo_0 is not None else w2
        wi_t = self.get_variable("wi_0", (n_exp, in_d, hid_d), explicit=wi_raw)
        wo_t = self.get_variable("wo_0", (n_exp, hid_d, in_d), explicit=wo_raw)

        router_logits = np.einsum("...d,de->...e", normed, gw_t)
        if self.gating_logit_cap > 0.0:
            router_logits = self.gating_logit_cap * np.tanh(router_logits / self.gating_logit_cap)

        max_r = np.max(router_logits, axis=-1, keepdims=True)
        exp_r = np.exp(router_logits - max_r)
        gates = exp_r / np.maximum(np.sum(exp_r, axis=-1, keepdims=True), 1e-12)

        if self.gating_func.lower() == "top1" and n_exp > 1:
            top1_idx = np.argmax(gates, axis=-1, keepdims=True)
            top1_mask = (top1_idx == np.arange(n_exp)) * 1.0
            gates = gates * top1_mask

        act_fn = instantiate_layer(self.activation_tpl, ReLU())
        expert_hidden = act_fn(np.einsum("...d,edf->...ef", normed, wi_t))
        expert_out = np.einsum("...ef,efd->...ed", expert_hidden, wo_t)
        combined = np.einsum("...e,...ed->...d", gates, expert_out) * self.residual_weight

        out = x + combined if self.add_skip_connection else combined
        if self.norm_policy == "post":
            out = ln(out)
        if paddings is not None:
            pad_t = np.asarray(paddings)
            out = out * (1.0 - np.expand_dims(pad_t, -1))
        return out


class Transformer(BasePraxisLayer):
    """Single Transformer block combining Self-Attention, optional Cross-Attention, and FFN.

    Attributes:
        input_dims: Input feature dimension `D` (alias for `model_dims`).
        model_dims: Model hidden feature dimension `D`.
        hidden_dims: FFN intermediate expansion dimension `F`.
        num_heads: Number of attention heads `N`.
        dim_per_head: Optional dimension per attention head `H`.
        dropout_tpl: Optional dropout layer template.
        dropout_prob: Dropout probability applied across attention and FFN sub-layers.
        atten_dropout_prob: Optional override for attention dropout probability.
        relu_dropout_prob: Optional override for FFN activation dropout probability.
        residual_dropout_prob: Optional override for residual branch dropout probability.
        use_cross_attention: Whether to include an encoder-decoder cross-attention sub-layer.
        allow_skip_cross_attention: Whether cross-attention can be skipped when `cross_inputs` is None.
        allow_at_least_one_headed_attention: Whether at least 1 head is enforced.
        mask_self_attention: Whether self-attention uses causal masking.
        packed_input: Whether inputs contain multiple packed sequences.
        cross_atten_tpl: Optional cross-attention layer template.
        ln_tpl: Layer normalization template.
        tr_atten_tpl: Self-attention layer template.
        tr_fflayer_tpl: Feed-forward layer template.
        norm_policy: Normalization placement policy (`'pre'`, `'primer_hybrid'`, `'post'`, `'none'`).
        residual_droppath_prob: Stochastic depth drop-path probability.
        ngrammer_tpl: Optional Ngrammer layer template.
    """

    input_dims: int = 0
    model_dims: int = 0
    hidden_dims: int = 0
    num_heads: int = 1
    dim_per_head: int | None = None
    dropout_tpl: LayerTpl = None
    dropout_prob: float = 0.0
    atten_dropout_prob: float | None = None
    relu_dropout_prob: float | None = None
    residual_dropout_prob: float | None = None
    use_cross_attention: bool = False
    allow_skip_cross_attention: bool = False
    allow_at_least_one_headed_attention: bool = False
    mask_self_attention: bool = False
    packed_input: bool = False
    cross_atten_tpl: LayerTpl = None
    ln_tpl: LayerTpl = None
    tr_atten_tpl: LayerTpl = None
    tr_fflayer_tpl: LayerTpl = None
    norm_policy: str = "pre"
    residual_droppath_prob: float = 0.0
    ngrammer_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        atten_mask: TensorOrScalar | None = None,
        cross_inputs: TensorOrScalar | None = None,
        cross_atten_mask: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply self-attention, optional cross-attention, and position-wise FFN to `inputs`.

        Args:
            inputs: Input sequence tensor of shape `[B, T, D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            atten_mask: Optional self-attention mask.
            cross_inputs: Optional encoder memory sequence tensor of shape `[B, S, D]`.
            cross_atten_mask: Optional cross-attention mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Transformed sequence `Tensor` of shape `[B, T, D]`.
        """
        x = np.asarray(inputs)
        d_model = (
            self.model_dims
            if self.model_dims > 0
            else (self.input_dims if self.input_dims > 0 else x.shape[-1])
        )
        f_hid = self.hidden_dims if self.hidden_dims > 0 else 4 * d_model
        n_heads = max(1, self.num_heads)
        h_dim = (
            self.dim_per_head
            if (self.dim_per_head is not None and self.dim_per_head > 0)
            else max(1, d_model // n_heads)
        )

        ln1 = instantiate_layer(self.ln_tpl, LayerNorm(dim=d_model))
        default_self_attn = DotProductAttention(
            input_dim=d_model,
            num_heads=n_heads,
            dim_per_head=h_dim,
            params_init=self.params_init,
        )
        self_attn = instantiate_layer(self.tr_atten_tpl, default_self_attn)

        eff_mask: TensorOrScalar | None = atten_mask
        if eff_mask is None and self.mask_self_attention and x.ndim >= 2:
            eff_mask = _build_causal_mask(x.shape[1])

        normed_x = ln1(x) if self.norm_policy in ("pre", "primer_hybrid") else x
        if self.ngrammer_tpl is not None:
            ngrammer = instantiate_layer(self.ngrammer_tpl, Identity())
            normed_x = ngrammer(normed_x)

        atten_out = (
            self_attn(normed_x, normed_x, normed_x, atten_mask=eff_mask)
            if isinstance(self_attn, DotProductAttention)
            else self_attn(normed_x)
        )
        x = x + atten_out
        if self.norm_policy == "post":
            x = ln1(x)

        if self.use_cross_attention and cross_inputs is not None:
            ln_cross = instantiate_layer(self.ln_tpl, LayerNorm(dim=d_model))
            cross_mem = np.asarray(cross_inputs)
            normed_q = ln_cross(x) if self.norm_policy in ("pre", "primer_hybrid") else x
            cross_attn = instantiate_layer(self.cross_atten_tpl, default_self_attn)
            cross_out = (
                cross_attn(normed_q, cross_mem, cross_mem, atten_mask=cross_atten_mask)
                if isinstance(cross_attn, DotProductAttention)
                else cross_attn(normed_q)
            )
            x = x + cross_out
            if self.norm_policy == "post":
                x = ln_cross(x)

        default_ffn = TransformerFeedForward(
            input_dims=d_model,
            hidden_dims=f_hid,
            norm_policy=self.norm_policy,
            ln_tpl=self.ln_tpl,
            params_init=self.params_init,
        )
        ffn = instantiate_layer(self.tr_fflayer_tpl, default_ffn)
        out = ffn(x, paddings=paddings) if isinstance(ffn, TransformerFeedForward) else ffn(x)
        return out


class StackedTransformer(BasePraxisLayer):
    """Stack of `num_layers` Transformer (or MoE) blocks.

    Attributes:
        use_cross_attention: Whether layers include cross-attention.
        mask_self_attention: Whether self-attention uses causal masking.
        num_layers: Number of stacked Transformer blocks.
        model_dims: Model feature dimension `D`.
        hidden_dims: FFN hidden expansion dimension `F`.
        num_heads: Number of attention heads `N`.
        dim_per_head: Optional dimension per attention head `H`.
        dropout_prob: Dropout probability across sub-layers.
        atten_dropout_prob: Optional attention dropout override.
        relu_dropout_prob: Optional FFN activation dropout override.
        residual_dropout_prob: Optional residual dropout override.
        input_dropout_prob: Input sequence dropout probability.
        gating_func: Router gating function for MoE layers.
        unadjusted_expert_capacity_factor: Capacity factor multiplier for MoE layers.
        enable_while_loop: Whether to execute the layer stack using a compiled while loop.
        checkpoint_policy: Activation rematerialization policy.
        autoregressive: Whether the stack operates in autoregressive causal mode.
        transformer_layer_params_tpl: Template or sequence of templates for Transformer layers.
        packed_input: Whether inputs are sequence-packed.
        fold_padding_with_segment_mask: Whether padding is folded into the segment mask.
        moe_layer_tpl: Optional template for Mixture-of-Experts layers.
        num_experts: Number of experts in MoE layers.
        num_groups: Number of routing groups in MoE layers.
        min_group_size: Optional minimum routing group size.
        moe_layers: Optional sequence of layer indices that use `moe_layer_tpl`.
        ngrammer_tpls: Optional sequence of Ngrammer templates per layer.
        remat: Whether activation checkpointing is enabled.
        residual_droppath_prob: Stochastic depth drop-path probability.
    """

    use_cross_attention: bool = False
    mask_self_attention: bool = False
    num_layers: int = 0
    model_dims: int = 0
    hidden_dims: int = 0
    num_heads: int = 0
    dim_per_head: int | None = None
    dropout_prob: float = 0.0
    atten_dropout_prob: float | None = None
    relu_dropout_prob: float | None = None
    residual_dropout_prob: float | None = None
    input_dropout_prob: float = 0.0
    gating_func: str = "top2"
    unadjusted_expert_capacity_factor: float = 2.0
    enable_while_loop: bool = True
    checkpoint_policy: str = "save_nothing"
    autoregressive: bool = False
    transformer_layer_params_tpl: LayerTpl | Sequence[LayerTpl] = None
    packed_input: bool = False
    fold_padding_with_segment_mask: bool = False
    moe_layer_tpl: LayerTpl = None
    num_experts: int = 0
    num_groups: int = 1
    min_group_size: int | None = None
    moe_layers: Sequence[int] | None = None
    ngrammer_tpls: Sequence[LayerTpl] | None = None
    remat: bool = False
    residual_droppath_prob: float = 0.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        segment_mask: TensorOrScalar | None = None,
        cross_inputs: TensorOrScalar | None = None,
        cross_paddings: TensorOrScalar | None = None,
        cross_segment_mask: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Pass `inputs` through `num_layers` of `Transformer` blocks.

        Args:
            inputs: Input sequence tensor of shape `[B, T, D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            segment_mask: Optional attention segment mask.
            cross_inputs: Optional encoder memory sequence of shape `[B, S, D]`.
            cross_paddings: Optional encoder memory padding mask.
            cross_segment_mask: Optional cross-attention segment mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output sequence `Tensor` of shape `[B, T, D]`.
        """
        out = np.asarray(inputs)
        n_layers = self.num_layers if self.num_layers > 0 else 1
        d_model = self.model_dims if self.model_dims > 0 else out.shape[-1]
        f_hid = self.hidden_dims if self.hidden_dims > 0 else 4 * d_model
        n_heads = self.num_heads if self.num_heads > 0 else 1
        moe_set = set(self.moe_layers) if self.moe_layers is not None else set()

        for layer_idx in range(n_layers):
            ng_tpl: LayerTpl = (
                self.ngrammer_tpls[layer_idx]
                if (self.ngrammer_tpls is not None and layer_idx < len(self.ngrammer_tpls))
                else None
            )
            ff_tpl: LayerTpl = None
            if layer_idx in moe_set:
                ff_tpl = self.moe_layer_tpl or TransformerFeedForwardMoe(
                    input_dims=d_model,
                    hidden_dims=f_hid,
                    num_experts=max(1, self.num_experts),
                    num_groups=self.num_groups,
                    params_init=self.params_init,
                )
            default_block = Transformer(
                model_dims=d_model,
                hidden_dims=f_hid,
                num_heads=n_heads,
                dim_per_head=self.dim_per_head,
                use_cross_attention=self.use_cross_attention,
                mask_self_attention=self.mask_self_attention or self.autoregressive,
                tr_fflayer_tpl=ff_tpl,
                ngrammer_tpl=ng_tpl,
                params_init=self.params_init,
            )
            if isinstance(self.transformer_layer_params_tpl, Sequence):
                tpl_item = self.transformer_layer_params_tpl[
                    layer_idx % len(self.transformer_layer_params_tpl)
                ]
                block = instantiate_layer(tpl_item, default_block)
            else:
                block = instantiate_layer(self.transformer_layer_params_tpl, default_block)

            out = (
                block(
                    out,
                    paddings=paddings,
                    atten_mask=segment_mask,
                    cross_inputs=cross_inputs,
                    cross_atten_mask=cross_segment_mask,
                )
                if isinstance(block, Transformer)
                else block(out)
            )
        return out


class StackedTransformerRepeated(BasePraxisLayer):
    """Repeated StackedTransformer container sharing or looping `block` `x_times` times.

    Attributes:
        block: Underlying `StackedTransformer` block or template.
        x_times: Number of repeat iterations.
        checkpoint_policy: Activation checkpointing policy string.
        unroll_in_decode: Whether to unroll during decoding.
        repeat_layer_name: Sub-layer name for the repeat container.
        sublayer_name: Optional sublayer scope name.
        repeat_optimizer_dims_mapping: Sharding mapping for optimizer states across repeats.
        nd_prefix_shape: Optional N-D prefix shape.
        return_intermediate_outputs: Whether to return intermediate outputs.
        collect_intermediate_outputs: Whether to collect intermediate outputs.
        sub_tpl: Alias for `block`.
        unpack_summaries: Whether to unpack summaries per iteration.
        enable_per_layer_input: Whether per-layer inputs are enabled.
        num_layers: Number of layers per block.
    """

    block: LayerTpl = None
    x_times: int = 0
    checkpoint_policy: str = "save_nothing"
    unroll_in_decode: bool = True
    repeat_layer_name: str | None = "repeat"
    sublayer_name: str | None = None
    repeat_optimizer_dims_mapping: SplitDimsMapping = None
    nd_prefix_shape: Sequence[int] | None = None
    return_intermediate_outputs: bool = False
    collect_intermediate_outputs: bool = False
    sub_tpl: LayerTpl = None
    unpack_summaries: bool = False
    enable_per_layer_input: bool = False
    num_layers: int = 1

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        segment_mask: TensorOrScalar | None = None,
        cross_inputs: TensorOrScalar | None = None,
        cross_paddings: TensorOrScalar | None = None,
        cross_segment_mask: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply `block` `x_times` times to `inputs`.

        Args:
            inputs: Input sequence tensor of shape `[B, T, D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            segment_mask: Optional attention segment mask.
            cross_inputs: Optional encoder memory sequence.
            cross_paddings: Optional encoder memory padding mask.
            cross_segment_mask: Optional cross-attention segment mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Repeatedly transformed sequence `Tensor`.
        """
        out = np.asarray(inputs)
        iters = self.x_times if self.x_times > 0 else 1
        tpl = self.block if self.block is not None else self.sub_tpl
        blk = instantiate_layer(tpl, StackedTransformer(num_layers=1, params_init=self.params_init))
        for _ in range(iters):
            out = (
                blk(
                    out,
                    paddings=paddings,
                    segment_mask=segment_mask,
                    cross_inputs=cross_inputs,
                    cross_paddings=cross_paddings,
                    cross_segment_mask=cross_segment_mask,
                )
                if isinstance(blk, StackedTransformer)
                else blk(out)
            )
        return out


class PipelinedTransformer(BasePraxisLayer):
    """Pipeline-parallel Transformer layer executing `pipeline_stage` across `num_pipeline_stages`.

    Attributes:
        pipeline_stage: Underlying `StackedTransformer` stage or template.
        num_pipeline_stages: Number of pipeline stages.
        num_pipeline_microbatches: Number of pipeline microbatches.
        pipeline_microbatch_size: Size of each microbatch.
        stream_io: Whether to stream inputs/outputs across stages.
        pipeline_broadcast_inputs: Whether non-sharded inputs are broadcast across stages.
        circular_repeat: Number of circular repeat passes per stage.
        checkpoint_policy: Activation checkpointing policy string.
        enable_async_circular_transfer: Whether asynchronous circular transfers are enabled.
        bf16_accum_in_fp32: Whether bfloat16 gradients are accumulated in float32.
    """

    pipeline_stage: LayerTpl = None
    num_pipeline_stages: int = 1
    num_pipeline_microbatches: int | None = None
    pipeline_microbatch_size: int | None = None
    stream_io: bool = False
    pipeline_broadcast_inputs: bool = False
    circular_repeat: int = 1
    checkpoint_policy: str = "save_iteration_input"
    enable_async_circular_transfer: bool = False
    bf16_accum_in_fp32: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Execute `pipeline_stage` across `num_pipeline_stages * circular_repeat` stages.

        Args:
            inputs: Input sequence tensor of shape `[B, T, D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Pipelined Transformer output `Tensor`.
        """
        out = np.asarray(inputs)
        stage = instantiate_layer(
            self.pipeline_stage,
            StackedTransformer(num_layers=1, params_init=self.params_init),
        )
        total = max(1, self.num_pipeline_stages) * max(1, self.circular_repeat)
        for _ in range(total):
            out = (
                stage(out, paddings=paddings)
                if isinstance(stage, StackedTransformer)
                else stage(out)
            )
        return out


class TransformerLm(BasePraxisLayer):
    """Transformer Language Model combining token embeddings, positional encoding, Transformer stack, and Softmax.

    Attributes:
        position_emb_tpl: Optional positional embedding layer template.
        model_dims: Model hidden feature dimension `D`.
        stacked_transformer_tpl: Template for the `StackedTransformer` backbone.
        softmax_tpl: Template for the `SharedEmbeddingSoftmax` or `FullSoftmax` layer.
        vocab_size: Vocabulary size `V`.
        packed_input: Whether inputs are sequence-packed.
        model_type: Attention masking type (`'causal'`, `'prefix'`, `'bidirectional'`).
        separate_embedding_tpl: Optional separate input embedding layer template.
        final_ln_tpl: Optional final LayerNorm template applied before logit projection.
        ngrammer_tpl: Optional Ngrammer layer template.
        post_attention_ngrammer_tpls: Optional sequence of post-attention Ngrammer templates.
        skip_compute_loss: If True, returns hidden states/logits without computing loss.
        skip_aux_loss: Whether auxiliary MoE load-balancing losses are skipped.
        record_activations_in_xent_output: Whether hidden activations are attached to the loss output.
        entropy_loss_weight: Weight of entropy regularization loss.
    """

    position_emb_tpl: LayerTpl = None
    model_dims: int = 0
    stacked_transformer_tpl: LayerTpl = None
    softmax_tpl: LayerTpl = None
    vocab_size: int = 0
    packed_input: bool = False
    model_type: str = "causal"
    separate_embedding_tpl: LayerTpl = None
    final_ln_tpl: LayerTpl = None
    ngrammer_tpl: LayerTpl = None
    post_attention_ngrammer_tpls: Sequence[LayerTpl] | None = None
    skip_compute_loss: bool = False
    skip_aux_loss: bool = False
    record_activations_in_xent_output: bool = False
    entropy_loss_weight: float = 0.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        labels: TensorOrScalar | None = None,
        segment_ids: TensorOrScalar | None = None,
        segment_pos: TensorOrScalar | None = None,
        causal_attention_mask: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Run the Transformer Language Model forward pass on `inputs`.

        Args:
            inputs: Integer token ID tensor `[B, T]` or pre-embedded float tensor `[B, T, D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            labels: Optional target token IDs.
            segment_ids: Optional packed segment IDs.
            segment_pos: Optional packed segment positions.
            causal_attention_mask: Optional explicit attention mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output logits or hidden representations `Tensor`.
        """
        x = np.asarray(inputs)
        d_model = self.model_dims if self.model_dims > 0 else (x.shape[-1] if x.ndim >= 3 else 8)
        v_sz = self.vocab_size if self.vocab_size > 0 else d_model

        default_softmax = SharedEmbeddingSoftmax(
            input_dims=d_model, num_classes=v_sz, params_init=self.params_init
        )
        softmax_layer = instantiate_layer(self.softmax_tpl, default_softmax)

        if x.ndim == 2:
            if self.separate_embedding_tpl is not None:
                sep_emb = instantiate_layer(self.separate_embedding_tpl, Identity())
                emb = sep_emb(x)
            elif isinstance(softmax_layer, SharedEmbeddingSoftmax):
                emb = softmax_layer.emb_lookup(x)
            else:
                emb = default_softmax.emb_lookup(x)
        else:
            emb = x

        pos_emb = instantiate_layer(
            self.position_emb_tpl, PositionalEmbedding(embedding_dims=d_model)
        )
        pe = (
            pos_emb(seq_length=emb.shape[1], position=segment_pos)
            if isinstance(pos_emb, PositionalEmbedding)
            else pos_emb(emb)
        )
        hidden = emb + pe

        if self.ngrammer_tpl is not None:
            ngrammer = instantiate_layer(self.ngrammer_tpl, Identity())
            hidden = ngrammer(hidden)

        is_causal = self.model_type.lower() == "causal"
        default_stack = StackedTransformer(
            num_layers=1,
            model_dims=d_model,
            mask_self_attention=is_causal,
            params_init=self.params_init,
        )
        stack = instantiate_layer(self.stacked_transformer_tpl, default_stack)
        hidden = (
            stack(hidden, paddings=paddings, segment_mask=causal_attention_mask)
            if isinstance(stack, StackedTransformer)
            else stack(hidden)
        )

        final_ln = instantiate_layer(self.final_ln_tpl, LayerNorm(dim=d_model))
        hidden = final_ln(hidden)

        if self.skip_compute_loss:
            return hidden

        if self.vocab_size > 0 or self.softmax_tpl is not None:
            if isinstance(softmax_layer, (SharedEmbeddingSoftmax, FullSoftmax)):
                return softmax_layer(hidden, class_ids=labels)
            return softmax_layer(hidden)
        return hidden


class TransformerEncoderDecoder(BasePraxisLayer):
    """Sequence-to-Sequence Transformer Encoder-Decoder model.

    Attributes:
        model_dims: Model feature dimension `D`.
        encoder_stacked_transformer_tpl: Template for the encoder `StackedTransformer`.
        decoder_stacked_transformer_tpl: Template for the decoder `StackedTransformer` (with cross-attention).
        encoder_embedding_tpl: Optional encoder token embedding template.
        decoder_embedding_tpl: Optional decoder token embedding template.
        encoder_ln_tpl: Encoder final LayerNorm template.
        decoder_ln_tpl: Decoder final LayerNorm template.
        position_emb_tpl: Shared positional embedding template.
        encoder_position_emb_tpl: Optional encoder-specific positional embedding template.
        decoder_position_emb_tpl: Optional decoder-specific positional embedding template.
        softmax_tpl: Output softmax projection template.
        packed_input: Whether inputs are sequence-packed.
        encoder_ngrammer_tpl: Optional encoder Ngrammer template.
        encoder_post_attention_ngrammer_tpls: Optional encoder post-attention Ngrammer templates.
        decoder_ngrammer_tpl: Optional decoder Ngrammer template.
        decoder_post_attention_ngrammer_tpls: Optional decoder post-attention Ngrammer templates.
    """

    model_dims: int = 0
    encoder_stacked_transformer_tpl: LayerTpl = None
    decoder_stacked_transformer_tpl: LayerTpl = None
    encoder_embedding_tpl: LayerTpl = None
    decoder_embedding_tpl: LayerTpl = None
    encoder_ln_tpl: LayerTpl = None
    decoder_ln_tpl: LayerTpl = None
    position_emb_tpl: LayerTpl = None
    encoder_position_emb_tpl: LayerTpl = None
    decoder_position_emb_tpl: LayerTpl = None
    softmax_tpl: LayerTpl = None
    packed_input: bool = False
    encoder_ngrammer_tpl: LayerTpl = None
    encoder_post_attention_ngrammer_tpls: Sequence[LayerTpl] | None = None
    decoder_ngrammer_tpl: LayerTpl = None
    decoder_post_attention_ngrammer_tpls: Sequence[LayerTpl] | None = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        input_paddings: TensorOrScalar | None = None,
        targets: TensorOrScalar | None = None,
        target_paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Encode `inputs` and decode `targets` (or `inputs` if `targets` is None) with cross-attention.

        Args:
            inputs: Encoder input tensor of shape `[B, S, D]`.
            input_paddings: Optional encoder padding mask of shape `[B, S]`.
            targets: Optional decoder input tensor of shape `[B, T, D]`.
            target_paddings: Optional decoder padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Decoded sequence `Tensor` of shape `[B, T, D]`.
        """
        enc_in = np.asarray(inputs)
        dec_in = enc_in if targets is None else np.asarray(targets)
        d_model = self.model_dims if self.model_dims > 0 else enc_in.shape[-1]

        enc_pe = instantiate_layer(
            self.encoder_position_emb_tpl or self.position_emb_tpl,
            PositionalEmbedding(embedding_dims=d_model),
        )
        dec_pe = instantiate_layer(
            self.decoder_position_emb_tpl or self.position_emb_tpl,
            PositionalEmbedding(embedding_dims=d_model),
        )
        enc_pos = (
            enc_pe(seq_length=enc_in.shape[1])
            if isinstance(enc_pe, PositionalEmbedding)
            else enc_pe(enc_in)
        )
        dec_pos = (
            dec_pe(seq_length=dec_in.shape[1])
            if isinstance(dec_pe, PositionalEmbedding)
            else dec_pe(dec_in)
        )

        enc_stack = instantiate_layer(
            self.encoder_stacked_transformer_tpl,
            StackedTransformer(
                num_layers=1,
                model_dims=d_model,
                use_cross_attention=False,
                params_init=self.params_init,
            ),
        )
        enc_ln = instantiate_layer(self.encoder_ln_tpl, LayerNorm(dim=d_model))
        enc_out = enc_ln(
            enc_stack(enc_in + enc_pos, paddings=input_paddings)
            if isinstance(enc_stack, StackedTransformer)
            else enc_stack(enc_in + enc_pos)
        )

        dec_stack = instantiate_layer(
            self.decoder_stacked_transformer_tpl,
            StackedTransformer(
                num_layers=1,
                model_dims=d_model,
                use_cross_attention=True,
                mask_self_attention=True,
                params_init=self.params_init,
            ),
        )
        dec_ln = instantiate_layer(self.decoder_ln_tpl, LayerNorm(dim=d_model))
        dec_out = dec_ln(
            dec_stack(
                dec_in + dec_pos,
                paddings=target_paddings,
                cross_inputs=enc_out,
                cross_paddings=input_paddings,
            )
            if isinstance(dec_stack, StackedTransformer)
            else dec_stack(dec_in + dec_pos)
        )
        if self.softmax_tpl is not None:
            sm = instantiate_layer(self.softmax_tpl, Identity())
            return sm(dec_out)
        return dec_out


class LanguageModel(BasePraxisLayer):
    """Top-level Language Model task wrapper around `TransformerLm`.

    Attributes:
        lm_tpl: Underlying `TransformerLm` layer or template.
        return_predictions: Whether to include token predictions in the output.
        decoder_tpl: Decoding hyperparameters configuration.
        model_type: Language model attention type (`'causal'`, `'prefix'`, `'bidirectional'`).
        count_tokens: Whether to track token count metrics.
        apply_eval_sample_weights: Whether per-example weights are applied during eval.
        report_strict_acc: Whether to report strict accuracy metrics.
        label_smoothing_prob: Cross-entropy label smoothing probability.
        z_loss_weight: Optional auxiliary z-loss weight.
    """

    lm_tpl: LayerTpl = None
    return_predictions: bool = False
    decoder_tpl: DecoderHParams | None = None
    model_type: str = "causal"
    count_tokens: bool = True
    apply_eval_sample_weights: bool = False
    report_strict_acc: bool = False
    label_smoothing_prob: float = 0.0
    z_loss_weight: float | None = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Evaluate the wrapped `lm_tpl` on `inputs`.

        Args:
            inputs: Input tensor.
            paddings: Optional padding mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Language model output `Tensor`.
        """
        lm = instantiate_layer(
            self.lm_tpl,
            TransformerLm(model_type=self.model_type, params_init=self.params_init),
        )
        return lm(inputs, paddings=paddings) if isinstance(lm, TransformerLm) else lm(inputs)


class LanguageModelContinuousBatching(LanguageModel):
    """Language Model wrapper supporting continuous batching / paged attention serving."""


class LanguageModelDPO(BasePraxisLayer):
    """Direct Preference Optimization (DPO) Language Model wrapper (Rafailov et al., 2023).

    Attributes:
        mdl: Policy language model layer or template.
        ref_mdl: Reference (frozen) language model layer or template.
        use_ref_model: Whether to evaluate `ref_mdl` to compute log-ratio rewards.
        beta: DPO KL-regularization temperature parameter `beta`.
        token_counter: Optional token counter layer template.
        apply_eval_sample_weights: Whether sample weights are applied during evaluation.
        model_type: Language model masking mode (`'causal'`, `'prefix'`, `'bidirectional'`).
        loss_type: DPO loss formulation (`'sigmoid'`, `'hinge'`, `'ipo'`).
        label_smoothing: Preference label smoothing parameter.
    """

    mdl: LayerTpl = None
    ref_mdl: LayerTpl = None
    use_ref_model: bool = True
    beta: float = 0.1
    token_counter: LayerTpl = None
    apply_eval_sample_weights: bool = False
    model_type: str = "causal"
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        rejected_inputs: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Evaluate DPO policy model (or preference loss when `rejected_inputs` is provided).

        Args:
            inputs: Chosen input sequence tensor.
            rejected_inputs: Optional rejected input sequence tensor for DPO preference loss.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Policy model output `Tensor`, or DPO preference loss `Tensor` if `rejected_inputs` is given.
        """
        policy_fn = instantiate_layer(self.mdl, TransformerLm(params_init=self.params_init))
        chosen_out = policy_fn(inputs)
        if rejected_inputs is None:
            return chosen_out

        rejected_out = policy_fn(rejected_inputs)
        pi_logratios = np.mean(chosen_out, axis=-1) - np.mean(rejected_out, axis=-1)

        if self.use_ref_model and self.ref_mdl is not None:
            ref_fn = instantiate_layer(self.ref_mdl, Identity())
            ref_chosen = ref_fn(inputs)
            ref_rejected = ref_fn(rejected_inputs)
            ref_logratios = np.mean(ref_chosen, axis=-1) - np.mean(ref_rejected, axis=-1)
        else:
            ref_logratios = np.zeros_like(pi_logratios)

        logits = self.beta * (pi_logratios - ref_logratios)
        if self.loss_type.lower() == "ipo":
            return np.square(logits - 1.0 / (2.0 * max(self.beta, 1e-6)))
        if self.loss_type.lower() == "hinge":
            return np.maximum(0.0, 1.0 - logits)

        pos_loss = np.maximum(-logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
        if self.label_smoothing > 0.0:
            neg_loss = np.maximum(logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
            return (1.0 - self.label_smoothing) * pos_loss + self.label_smoothing * neg_loss
        return pos_loss


class SequenceModel(BasePraxisLayer):
    """Encoder-Decoder Sequence-to-Sequence task model wrapper.

    Attributes:
        model_tpl: Underlying `TransformerEncoderDecoder` layer or template.
        return_predictions: Whether to return decoded predictions.
        decoder_tpl: Decoding hyperparameters configuration.
        label_smoothing_prob: Target cross-entropy label smoothing probability.
    """

    model_tpl: LayerTpl = None
    return_predictions: bool = False
    decoder_tpl: DecoderHParams | None = None
    label_smoothing_prob: float = 0.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Evaluate the wrapped sequence-to-sequence `model_tpl` on `inputs`.

        Args:
            inputs: Input sequence tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence model output `Tensor`.
        """
        seq_mdl = instantiate_layer(
            self.model_tpl, TransformerEncoderDecoder(params_init=self.params_init)
        )
        return seq_mdl(inputs)


class BertModel(BasePraxisLayer):
    """Bidirectional Encoder Representations from Transformers (BERT) model wrapper.

    Attributes:
        lm_tpl: Underlying bidirectional `TransformerLm` layer or template.
        label_smoothing_prob: MLM cross-entropy label smoothing probability.
        mask_token_id: Integer token ID used for `[MASK]`.
        force_mask_generation: Whether to force dynamic MLM masking even if masks are provided.
        mlm_augmenter_tpl: Optional `MaskedLmDataAugmenter` template.
        nsp_weight: Weight of the Next Sentence Prediction (NSP) auxiliary loss.
    """

    lm_tpl: LayerTpl = None
    label_smoothing_prob: float = 0.0
    mask_token_id: int = 0
    force_mask_generation: bool = False
    mlm_augmenter_tpl: LayerTpl = None
    nsp_weight: float = 0.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Evaluate the bidirectional BERT encoder on `inputs`.

        Args:
            inputs: Input token or sequence tensor.
            paddings: Optional padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Contextualized bidirectional representations `Tensor`.
        """
        lm = instantiate_layer(
            self.lm_tpl,
            TransformerLm(model_type="bidirectional", params_init=self.params_init),
        )
        return lm(inputs, paddings=paddings) if isinstance(lm, TransformerLm) else lm(inputs)


__all__ = [
    "AdaptedTransformerFeedForward",
    "BertModel",
    "LanguageModel",
    "LanguageModelContinuousBatching",
    "LanguageModelDPO",
    "PipelinedTransformer",
    "SequenceModel",
    "StackedTransformer",
    "StackedTransformerRepeated",
    "Transformer",
    "TransformerEncoderDecoder",
    "TransformerFeedForward",
    "TransformerFeedForwardMoe",
    "TransformerLm",
]
