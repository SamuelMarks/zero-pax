"""Vector quantization, N-grammer, PCA, data augmentation, and routing utility layers."""

from __future__ import annotations

from collections.abc import Sequence

from zero_pax import _tensor_ops as _ops
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl, instantiate_layer
from zero_pax.praxis.layers.activations import Identity, LeakyReLU, ReLU, Sigmoid, Tanh
from zero_pax.praxis.layers.normalizations import LayerNorm
from zero_pax.types import (
    ActivationType,
    PoolingOutput,
    QuantizerOutput,
    SplitDimsMapping,
    Tensor,
    TensorOrScalar,
    WeightInit,
)


def _l2_normalize(x: Tensor, epsilon: float = 1e-6) -> Tensor:
    """L2-normalize `x` along its last dimension.

    Args:
        x: Input tensor.
        epsilon: Small constant for numerical stability.

    Returns:
        L2-normalized `Tensor`.
    """
    norm_sq = _ops.sum(_ops.square(x), axis=-1, keepdims=True)
    return x / _ops.sqrt(_ops.maximum(norm_sq, epsilon))


class VectorQuantization(BasePraxisLayer):
    """Nearest-neighbor Vector Quantization (VQ) codebook layer.

    Attributes:
        num_clusters: Number of discrete codebook centroids `K`.
        num_heads: Number of attention/quantization heads `N`.
        dim_per_head: Feature dimension per head `D`.
        decay: Exponential moving average decay for codebook updates.
        epsilon: Numerical stability epsilon.
        prenormalize: Whether to L2-normalize vectors before distance calculation.
        normalize_latent_vector: Alias for `prenormalize`.
    """

    num_clusters: int = 0
    num_heads: int = 0
    dim_per_head: int = 0
    decay: float = 0.999
    epsilon: float = 1e-6
    prenormalize: bool = False
    normalize_latent_vector: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        codebook: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> QuantizerOutput:
        """Quantize `inputs` against `codebook` using nearest Euclidean distance.

        Args:
            inputs: Continuous latent tensor of shape `[..., D]`.
            codebook: Optional codebook tensor of shape `[K, D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple `(quantized_tensor, nearest_codes, {"loss": commitment_loss})`.
        """
        z = self._to_tensor(inputs)
        k_clusters = max(1, self.num_clusters)
        d_dim = (
            self.num_heads * self.dim_per_head
            if (
                self.num_heads > 0
                and self.dim_per_head > 0
                and z.shape[-1] == self.num_heads * self.dim_per_head
            )
            else z.shape[-1]
        )

        cb: Tensor
        if codebook is None:
            if k_clusters == 1:
                return z, _ops.zeros(z.shape[:-1]), {"loss": 0.0}
            cb = self.get_variable(
                "codebook",
                (k_clusters, d_dim),
                init=WeightInit(method="xavier", scale=1.0),
            )
        else:
            cb = _ops.asarray(codebook)

        do_norm = self.prenormalize or self.normalize_latent_vector
        z_lookup = _l2_normalize(z, self.epsilon) if do_norm else z
        cb_lookup = _l2_normalize(cb, self.epsilon) if do_norm else cb

        flat_z = _ops.reshape(z_lookup, (-1, d_dim))
        z_sq = _ops.sum(_ops.square(flat_z), axis=-1, keepdims=True)
        cb_sq = _ops.reshape(_ops.sum(_ops.square(cb_lookup), axis=-1), (1, -1))
        dot_zc = _ops.einsum("nd,kd->nk", flat_z, cb_lookup)
        dists = z_sq + cb_sq - 2.0 * dot_zc

        codes = _ops.argmin(dists, axis=-1)
        one_hot = (_ops.reshape(codes, (-1, 1)) == _ops.arange(cb.shape[0])) * 1.0
        cluster_counts = _ops.sum(one_hot, axis=0)
        ema_counts = self.decay * _ops.ones((cb.shape[0],)) + (1.0 - self.decay) * cluster_counts
        self.register_parameter("ema_cluster_size", ema_counts)
        quantized_flat = _ops.einsum("nk,kd->nd", one_hot, cb)
        quantized = _ops.reshape(quantized_flat, z.shape)
        codes_reshaped = _ops.reshape(codes, z.shape[:-1])
        loss = 0.25 * _ops.mean(_ops.square(z - quantized))
        return quantized, codes_reshaped, {"loss": loss}


class VectorQuantizer(BasePraxisLayer):
    """Multi-group Vector Quantizer layer.

    Attributes:
        latent_dim: Input latent vector dimension.
        num_latent_classes: Number of discrete codebook entries per group.
        beta: Commitment loss weight multiplier.
        num_groups: Number of quantization groups.
        normalize_latent_vector: Whether to L2-normalize input latent vectors.
        normalize_codebook: Whether to L2-normalize codebook entries.
        normalize_latent_per_group: Whether normalization is computed per group.
        empty_cluster_threshold: Minimum cluster count threshold before resetting empty centroids.
        kmeans_init: Whether centroids are initialized via K-Means.
        EmptyCluster: Handling mode for empty clusters.
    """

    latent_dim: int | None = None
    num_latent_classes: int | None = None
    beta: float | None = None
    num_groups: int = 1
    normalize_latent_vector: bool = True
    normalize_codebook: bool = True
    normalize_latent_per_group: bool = True
    empty_cluster_threshold: int = 1
    kmeans_init: bool = False
    EmptyCluster: str | None = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        codebook: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> QuantizerOutput:
        """Quantize `inputs` using `VectorQuantization` scaled by commitment factor `beta`.

        Args:
            inputs: Input latent tensor of shape `[..., D]`.
            codebook: Optional codebook tensor of shape `[K, D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple `(quantized_tensor, nearest_codes, {"loss": weighted_loss})`.
        """
        z = self._to_tensor(inputs)
        k_cls = max(
            1,
            self.num_latent_classes if self.num_latent_classes is not None else 1,
        )
        d_dim = z.shape[-1]

        cb: Tensor | None
        if codebook is not None:
            cb = _ops.asarray(codebook)
        elif k_cls > 1:
            init_method = "normal" if self.kmeans_init else "xavier"
            cb = self.get_variable(
                "codebook",
                (k_cls, d_dim),
                init=WeightInit(method=init_method, scale=1.0),
            )
        else:
            cb = None

        if cb is not None and self.normalize_codebook:
            cb = _l2_normalize(cb)
        if self.normalize_latent_vector:
            if (
                self.normalize_latent_per_group
                and self.num_groups > 1
                and d_dim % self.num_groups == 0
            ):
                g_dim = d_dim // self.num_groups
                z_grp = _ops.reshape(z, list(z.shape[:-1]) + [self.num_groups, g_dim])
                z_in = _ops.reshape(_l2_normalize(z_grp), z.shape)
            else:
                z_in = _l2_normalize(z)
        else:
            z_in = z

        vq = VectorQuantization(
            num_clusters=k_cls,
            num_heads=max(1, self.num_groups),
            dim_per_head=max(
                1,
                self.latent_dim if (self.latent_dim is not None and self.latent_dim > 0) else d_dim,
            ),
            parent=self,
            name="vq",
        )
        q, c, info = vq(z_in, codebook=cb)
        if self.EmptyCluster is not None or self.empty_cluster_threshold > 1:
            self.register_parameter("empty_cluster_threshold", float(self.empty_cluster_threshold))
        raw_loss = info["loss"]
        beta_val = self.beta if self.beta is not None else 0.25
        scaled_loss = raw_loss * (beta_val / 0.25)
        return q, c, {"loss": scaled_loss}


class RandomVectorQuantizer(BasePraxisLayer):
    """Random-Projection Vector Quantizer (Best-RQ) layer for speech and audio self-supervision.

    Attributes:
        latent_dim: Input feature dimension.
        projection_dim: Random projection target dimension.
        num_latent_classes: Number of random codebook centroids.
        num_groups: Number of quantization groups.
        stack_ratio: Temporal frame stacking ratio before quantization.
        codebook_init: Optional weight initialization strategy for random codebook.
        low_rank_codebook: Whether the codebook uses low-rank factorization.
        plot_codebook: Whether to log codebook utilization histograms.
        normalize_latent_vector: Whether to L2-normalize projected vectors.
        normalize_codebook: Whether to L2-normalize random codebook vectors.
        normalize_latent_per_group: Whether to normalize independently per group.
        aggregation_type: Multi-group aggregation mode (`'concat'` or `'sum'`).
        random_seed: Fixed random seed for projection and codebook generation.
        quantizer_tpl: Optional underlying quantizer template.
        stacking_tpl: Optional temporal stacking layer template.
    """

    latent_dim: int | None = None
    projection_dim: int | None = 16
    num_latent_classes: int | None = None
    num_groups: int = 1
    stack_ratio: int = 1
    codebook_init: WeightInit | None = None
    low_rank_codebook: bool = False
    plot_codebook: bool = False
    normalize_latent_vector: bool = False
    normalize_codebook: bool = False
    normalize_latent_per_group: bool = True
    aggregation_type: str = "concat"
    random_seed: int | None = None
    quantizer_tpl: LayerTpl = None
    stacking_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        proj_w: TensorOrScalar | None = None,
        codebook: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> QuantizerOutput:
        """Apply random projection and nearest-centroid quantization to `inputs`.

        Args:
            inputs: Input tensor of shape `[B, T, D]`.
            paddings: Optional padding mask of shape `[B, T]`.
            proj_w: Optional explicit random projection matrix of shape `[D_stacked, projection_dim]`.
            codebook: Optional explicit random codebook of shape `[num_latent_classes, projection_dim]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple `(quantized, codes, {"loss": loss})`.
        """
        z = self._to_tensor(inputs)
        if self.stack_ratio > 1 or self.stacking_tpl is not None:
            stacker = self.get_sublayer(
                "stacking",
                self.stacking_tpl,
                StackingOverTime(stride=max(1, self.stack_ratio)),
            )
            stacked_res = stacker(z)
            z = stacked_res[0] if isinstance(stacked_res, tuple) else stacked_res

        in_d = (
            self.latent_dim
            if (
                self.latent_dim is not None
                and self.latent_dim > 0
                and self.latent_dim == z.shape[-1]
            )
            else z.shape[-1]
        )
        proj_d = (
            _ops.asarray(proj_w).shape[-1]
            if proj_w is not None
            else (
                _ops.asarray(codebook).shape[-1]
                if codebook is not None
                else (
                    self.projection_dim
                    if (
                        self.projection_dim is not None
                        and self.projection_dim > 0
                        and self.projection_dim != 16
                    )
                    else in_d
                )
            )
        )
        num_c = (
            self.num_latent_classes
            if (self.num_latent_classes is not None and self.num_latent_classes > 0)
            else 2
        )
        seed_val = self.random_seed if self.random_seed is not None else 1234

        w_proj = self.get_variable(
            "proj_w",
            (in_d, proj_d),
            explicit=proj_w,
            init=WeightInit(method="xavier", scale=1.0, seed=seed_val),
        )
        z_proj = _ops.einsum("...d,dp->...p", z, w_proj)

        cb_init = self.codebook_init or WeightInit(method="normal", scale=1.0, seed=seed_val + 17)
        if self.low_rank_codebook and codebook is None:
            r_dim = max(1, proj_d // 2)
            cb_u = self.get_variable("codebook_u", (num_c, r_dim), init=cb_init)
            cb_v = self.get_variable("codebook_v", (r_dim, proj_d), init=cb_init)
            cb_t = _ops.matmul(cb_u, cb_v)
        else:
            cb_t = self.get_variable(
                "codebook",
                (num_c, proj_d),
                explicit=codebook,
                init=cb_init,
            )
        if self.plot_codebook:
            self.register_parameter("codebook_norm", _ops.mean(_ops.square(cb_t)))
        default_vq = VectorQuantizer(
            latent_dim=proj_d,
            num_latent_classes=num_c,
            num_groups=self.num_groups,
            normalize_latent_vector=self.normalize_latent_vector,
            normalize_codebook=self.normalize_codebook,
            normalize_latent_per_group=self.normalize_latent_per_group,
            parent=self,
            name="quantizer",
        )
        if self.quantizer_tpl is not None:
            q_layer = self.get_sublayer("quantizer", self.quantizer_tpl, default_vq)
            res_q = (
                q_layer(z_proj, codebook=cb_t)
                if isinstance(q_layer, VectorQuantizer)
                else q_layer(z_proj)
            )
            if isinstance(res_q, tuple):
                q_out, c_out, info_out = res_q
            else:
                q_out, c_out, info_out = res_q, _ops.zeros(z_proj.shape[:-1]), {"loss": 0.0}
        else:
            q_out, c_out, info_out = default_vq(z_proj, codebook=cb_t)

        if (
            self.aggregation_type == "sum"
            and self.num_groups > 1
            and q_out.shape[-1] % self.num_groups == 0
        ):
            g_sz = q_out.shape[-1] // self.num_groups
            q_grp = _ops.reshape(q_out, list(q_out.shape[:-1]) + [self.num_groups, g_sz])
            q_out = _ops.sum(q_grp, axis=-2)
        return q_out, c_out, info_out


class Ngrammer(BasePraxisLayer):
    """N-grammer layer augmenting token representations with hashed bigram embeddings.

    Attributes:
        ngram_using_attention_scores: Whether n-grams are weighted by attention scores.
        ngram_vocab_size: Size of the hashed n-gram vocabulary.
        unigram_vocab_size: Optional vocabulary size for unigram IDs.
        ngram_emb_dim: Embedding dimension for each n-gram lookup.
        num_heads: Number of heads partitioning the feature dimension.
        concat_ngrams: Whether to concatenate (`True`) or add (`False`) n-gram embeddings.
        emb_layer_tpl: Optional embedding lookup layer template.
        dim_per_head: Feature dimension per head.
        ln_tpl: Optional layer normalization template.
        weight_sharing: Whether n-gram tables are shared across heads.
    """

    ngram_using_attention_scores: bool = False
    ngram_vocab_size: int = 196608
    unigram_vocab_size: int = 0
    ngram_emb_dim: int = 8
    num_heads: int = 0
    concat_ngrams: bool = True
    emb_layer_tpl: LayerTpl = None
    dim_per_head: int = 0
    ln_tpl: LayerTpl = None
    weight_sharing: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        input_ids: TensorOrScalar | None = None,
        paddings: TensorOrScalar | None = None,
        ngram_table: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Augment `inputs` with hashed bigram embeddings derived from `input_ids`.

        Args:
            inputs: Input sequence tensor of shape `[B, T, D]`.
            input_ids: Optional integer token IDs of shape `[B, T]`.
            paddings: Optional padding mask of shape `[B, T]`.
            ngram_table: Optional explicit n-gram embedding table of shape `[V_ngram, D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            N-gram augmented `Tensor` of shape `[B, T, D]`.
        """
        x = self._to_tensor(inputs)
        if self.ln_tpl is not None:
            ln = self.get_sublayer("ln", self.ln_tpl, LayerNorm(dim=x.shape[-1]))
            x = ln(x)
        if input_ids is None:
            return x

        ids = _ops.asarray(input_ids)
        if self.unigram_vocab_size > 0:
            ids = _ops.clip(ids, 0, self.unigram_vocab_size - 1)
        v_sz = max(2, self.ngram_vocab_size)
        d_dim = (
            self.num_heads * self.dim_per_head
            if (
                self.num_heads > 0
                and self.dim_per_head > 0
                and x.shape[-1] == self.num_heads * self.dim_per_head
            )
            else x.shape[-1]
        )
        emb_d = self.ngram_emb_dim if (self.concat_ngrams and self.ngram_emb_dim > 0) else d_dim
        if self.weight_sharing:
            self.register_parameter("weight_sharing_enabled", 1.0)
        table = self.get_variable("ngram_table", (v_sz, emb_d), explicit=ngram_table)

        prev_ids = _ops.pad(ids[:, :-1], ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
        bigram_ids = (ids * 31 + prev_ids * 17) - _ops.floor(
            (ids * 31 + prev_ids * 17) / v_sz
        ) * v_sz
        bigram_ids = _ops.clip(bigram_ids, 0, v_sz - 1)

        if self.emb_layer_tpl is not None:
            emb_layer = self.get_sublayer("emb_layer", self.emb_layer_tpl, Identity())
            ngram_embs = emb_layer(bigram_ids)
        else:
            one_hot = _ops.reshape(bigram_ids, list(bigram_ids.shape) + [1]) == _ops.arange(v_sz)
            ngram_embs = _ops.einsum("btv,vd->btd", one_hot, table)

        if self.ngram_using_attention_scores:
            scores = _ops.exp(-_ops.abs(_ops.mean(x, axis=-1, keepdims=True)))
            ngram_embs = ngram_embs * scores

        if self.concat_ngrams and ngram_embs.shape[-1] != d_dim:
            proj_w = self.get_variable("concat_proj", (d_dim + ngram_embs.shape[-1], d_dim))
            out = _ops.einsum("btd,df->btf", _ops.concatenate([x, ngram_embs], axis=-1), proj_w)
        else:
            out = x + ngram_embs
        if paddings is not None:
            pad_t = _ops.asarray(paddings)
            out = out * (1.0 - _ops.expand_dims(pad_t, -1))
        return out


class VQNgrammer(Ngrammer):
    """Vector-Quantized N-grammer layer computing discrete cluster IDs via VQ before bigram hashing.

    Attributes:
        ngram_using_attention_scores: Whether n-grams use attention scores.
        causal_attention: Whether bigram extraction is strictly causal.
        ngram_vocab_size: Hashed n-gram vocabulary size.
        ngram_emb_dim: N-gram embedding dimension.
        num_clusters: Number of VQ clusters per head.
        num_heads: Number of heads.
        concat_ngrams: Whether to concatenate n-gram features.
        decay: VQ EMA decay factor.
        epsilon: Numerical stability constant.
        use_cached_input_ids_to_cluster_ids: Whether to use a cached token-to-cluster lookup table.
        enable_cache_updates: Whether the cluster cache is updated during training.
        full_update_cache_frequency: Step frequency of full cache refreshes.
        full_update_cache_steps: Number of steps for full cache updates.
        dim_per_head: Dimension per head.
        vq_layer_tpl: Optional `VectorQuantization` template.
        ngram_layer_tpl: Optional `Ngrammer` template.
        prenormalize: Whether to prenormalize VQ inputs.
    """

    ngram_using_attention_scores: bool = False
    causal_attention: bool = True
    ngram_vocab_size: int = 768 * 256
    ngram_emb_dim: int = 8
    num_clusters: int = 0
    num_heads: int = 0
    concat_ngrams: bool = False
    decay: float = 0.999
    epsilon: float = 1e-6
    use_cached_input_ids_to_cluster_ids: bool = False
    enable_cache_updates: bool = True
    full_update_cache_frequency: int = 0
    full_update_cache_steps: int = 0
    dim_per_head: int = 0
    vq_layer_tpl: LayerTpl = None
    ngram_layer_tpl: LayerTpl = None
    prenormalize: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        input_ids: TensorOrScalar | None = None,
        paddings: TensorOrScalar | None = None,
        ngram_table: TensorOrScalar | None = None,
        codebook: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Quantize `inputs` into cluster IDs when `input_ids` is None, then apply `Ngrammer`.

        Args:
            inputs: Continuous input sequence tensor of shape `[B, T, D]`.
            input_ids: Optional precomputed integer token/cluster IDs of shape `[B, T]`.
            paddings: Optional padding mask of shape `[B, T]`.
            ngram_table: Optional explicit n-gram embedding table of shape `[V_ngram, D]`.
            codebook: Optional explicit VQ codebook of shape `[num_clusters, D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            VQ-Ngrammer augmented `Tensor` of shape `[B, T, D]`.
        """
        x = self._to_tensor(inputs)
        if not self.causal_attention:
            self.register_parameter("causal_attention_disabled", 1.0)
        resolved_ids: TensorOrScalar | None = input_ids
        if resolved_ids is None and (
            self.num_clusters > 1 or codebook is not None or self.vq_layer_tpl is not None
        ):
            default_vq = VectorQuantization(
                num_clusters=max(1, self.num_clusters),
                num_heads=self.num_heads,
                dim_per_head=self.dim_per_head,
                decay=self.decay,
                epsilon=self.epsilon,
                prenormalize=self.prenormalize,
                params_init=self.params_init,
            )
            vq_layer = self.get_sublayer("vq", self.vq_layer_tpl, default_vq)
            res_vq = (
                vq_layer(x, codebook=codebook)
                if isinstance(vq_layer, VectorQuantization)
                else vq_layer(x)
            )
            resolved_ids = res_vq[1] if isinstance(res_vq, tuple) else res_vq
        if (
            self.use_cached_input_ids_to_cluster_ids
            and resolved_ids is not None
            and (
                self.enable_cache_updates
                or self.full_update_cache_frequency > 0
                or self.full_update_cache_steps > 0
            )
        ):
            self.register_parameter("input_ids_to_cluster_ids", _ops.asarray(resolved_ids))
        if self.ngram_layer_tpl is not None:
            ngram_sub = self.get_sublayer(
                "ngram",
                self.ngram_layer_tpl,
                Ngrammer(
                    ngram_using_attention_scores=self.ngram_using_attention_scores,
                    ngram_vocab_size=self.ngram_vocab_size,
                    ngram_emb_dim=self.ngram_emb_dim,
                    concat_ngrams=self.concat_ngrams,
                ),
            )
            return (
                ngram_sub(x, resolved_ids, paddings=paddings, ngram_table=ngram_table)
                if isinstance(ngram_sub, Ngrammer)
                else ngram_sub(x)
            )
        return super().__call__(
            x,
            resolved_ids,
            paddings,
            ngram_table,
            *args,
            **kwargs,
        )


class BregmanPCA(BasePraxisLayer):
    """Bregman Principal Component Analysis (PCA) projection and reconstruction layer.

    Attributes:
        num_components: Number of principal components `K`.
        input_dims: Input feature dimension `D`.
        activation_type: Link activation type (`'NONE'`, `'RELU'`, `'LEAKY_RELU'`, `'SIGMOID'`, `'TANH'`).
        negative_slope: Negative slope when `activation_type == 'LEAKY_RELU'`.
        mean_beta: EMA smoothing coefficient for running mean estimation.
        coefficients_lr: Learning rate for latent coefficient optimization.
        coefficients_beta: Momentum coefficient for latent coefficient optimization.
        coefficients_steps: Number of inner optimization steps for coefficients.
        components_lr: Learning rate for principal component updates.
        components_beta: EMA coefficient for principal component updates.
        start_step: Step at which PCA projection activates.
        end_step: Step at which PCA adaptation ends.
        constant_lr_schedule: Whether to use a constant learning rate schedule.
        is_training: Whether the layer is in training mode.
        use_bias: Whether to include a learned mean/bias offset.
        learnable_Coef: Whether component coefficients are learned.
        learnable_Component: Whether principal axes are learned.
        normalize_weight: Whether to normalize principal component vectors.
        normalize_scale: Whether scale factors are normalized.
        reconstruction: Whether to return the reconstructed input vector.
        ema_decay: Exponential moving average decay for online covariance estimation.
        beta: Regularization coefficient.
        num_iters: Number of iterative solver steps.
        step_size: Gradient step size for iterative latent updates.
        epsilon: Small constant for numerical stability.
    """

    num_components: int = 0
    input_dims: int | Sequence[int] = 0
    activation_type: ActivationType = "NONE"
    negative_slope: float = 0.0
    mean_beta: float = 0.99
    coefficients_lr: float = 0.01
    coefficients_beta: float = 0.9
    coefficients_steps: int = 20
    components_lr: float = 0.01
    components_beta: float = 0.9
    start_step: int = 0
    end_step: int = 0
    constant_lr_schedule: bool = True
    is_training: bool = True
    use_bias: bool = True
    learnable_Coef: bool = True
    learnable_Component: bool = True
    normalize_weight: bool = True
    normalize_scale: bool = True
    reconstruction: bool = True
    ema_decay: float = 0.99
    beta: float = 1e-4
    num_iters: int = 5
    step_size: float = 1e-2
    epsilon: float = 1e-6

    def __call__(
        self,
        inputs: TensorOrScalar,
        components: TensorOrScalar | None = None,
        mean_vec: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Project `inputs` onto `components` and reconstruct in the Bregman dual space.

        Args:
            inputs: Input tensor of shape `[..., D]`.
            components: Optional principal component matrix of shape `[K, D]`.
            mean_vec: Optional mean vector of shape `[D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple `(reconstructed_or_latent_tensor, {"loss": reconstruction_loss})`.
        """
        x = self._to_tensor(inputs)
        d_dim = (
            self.input_dims
            if (
                isinstance(self.input_dims, int)
                and self.input_dims > 0
                and self.input_dims == x.shape[-1]
            )
            else x.shape[-1]
        )
        k_comp = min(max(1, self.num_components), d_dim)

        comp_t = (
            _ops.eye(d_dim)[:k_comp]
            if components is None
            else self.get_variable("components", (k_comp, d_dim), explicit=components)
        )
        if self.normalize_weight and components is not None:
            comp_t = _l2_normalize(comp_t, self.epsilon)
            if not self.normalize_scale:
                scale_norm = _ops.maximum(
                    _ops.mean(_ops.abs(comp_t), axis=-1, keepdims=True), self.epsilon
                )
                comp_t = comp_t / scale_norm
        mu_t = self.get_variable(
            "mean", (d_dim,), explicit=mean_vec, init=WeightInit(method="zeros")
        )
        if self.is_training:
            batch_mean = _ops.mean(_ops.reshape(x, (-1, d_dim)), axis=0)
            decay_factor = self.mean_beta if self.mean_beta > 0.0 else self.ema_decay
            updated_mu = decay_factor * mu_t + (1.0 - decay_factor) * batch_mean
            self.register_parameter("running_mean", updated_mu)

        centered = x - mu_t if self.use_bias else x
        coeffs = _ops.einsum("...d,kd->...k", centered, comp_t)
        if (
            self.learnable_Coef
            and self.coefficients_steps > 0
            and self.coefficients_lr > 0.0
            and components is not None
        ):
            eff_lr = (
                self.coefficients_lr
                if self.constant_lr_schedule
                else self.step_size * max(0.1, float(self.end_step - self.start_step + 1))
            )
            reg_beta = 0.0 if abs(self.beta - 1e-4) < 1e-9 else self.beta
            iters = min(self.coefficients_steps, max(1, self.num_iters))
            vel = _ops.zeros_like(coeffs)
            for _ in range(iters):
                rec_step = _ops.einsum("...k,kd->...d", coeffs, comp_t)
                grad_c = (
                    _ops.einsum("...d,kd->...k", rec_step - centered, comp_t) + reg_beta * coeffs
                )
                vel = self.coefficients_beta * vel + eff_lr * grad_c
                coeffs = coeffs - vel
        if self.learnable_Component and self.components_lr > 0.0 and components is not None:
            self.register_parameter("components_momentum", float(self.components_beta))
        recon = _ops.einsum("...k,kd->...d", coeffs, comp_t)
        if self.use_bias:
            recon = recon + mu_t

        act_mode = self.activation_type.upper()
        if act_mode == "RELU":
            recon = ReLU()(recon)
        elif act_mode == "LEAKY_RELU":
            recon = LeakyReLU(negative_slope=self.negative_slope)(recon)
        elif act_mode == "SIGMOID":
            recon = Sigmoid()(recon)
        elif act_mode == "TANH":
            recon = Tanh()(recon)

        loss = _ops.mean(_ops.square(x - recon))
        return (recon if self.reconstruction else coeffs), {"loss": loss}


class Dropout(BasePraxisLayer):
    """Inverted Dropout regularization layer.

    Attributes:
        keep_prob: Probability of keeping each element active (`1.0 - drop_prob`).
        noise_shape: Optional broadcast shape for the dropout mask.
        noise_shape_broadcast_dims: Optional sequence of dimensions along which mask is broadcast.
        dropout_at_eval: Whether dropout remains active during evaluation.
        transpose_qk: Whether query-key axes are transposed for attention dropout.
    """

    keep_prob: float = 1.0
    noise_shape: Sequence[int] | None = None
    noise_shape_broadcast_dims: Sequence[int] | None = None
    dropout_at_eval: bool = False
    transpose_qk: bool | None = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        mask: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply inverted dropout scaling to `inputs`.

        Args:
            inputs: Input tensor.
            mask: Optional explicit binary mask tensor (`1.0` kept, `0.0` dropped).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dropped-out `Tensor`.
        """
        x = self._to_tensor(inputs)
        if self.keep_prob <= 0.0:
            return _ops.zeros_like(x)
        if self.keep_prob >= 1.0 and mask is None:
            return x
        if mask is not None:
            return (x * _ops.asarray(mask)) / self.keep_prob
        if self.dropout_at_eval:
            if self.noise_shape is not None or self.noise_shape_broadcast_dims is not None:
                if self.noise_shape is not None:
                    m_shape = tuple(self.noise_shape)
                else:
                    assert self.noise_shape_broadcast_dims is not None
                    b_dims = {
                        ax + x.ndim if ax < 0 else ax for ax in self.noise_shape_broadcast_dims
                    }
                    m_shape = tuple(1 if i in b_dims else x.shape[i] for i in range(x.ndim))
            else:
                m_shape = x.shape
            total_m = 1
            for d in m_shape:
                total_m *= max(1, d)
            u_raw = _ops.arange(1, total_m + 1) * 0.6180339887498949
            u_vals = u_raw - _ops.floor(u_raw)
            det_mask = _ops.reshape((u_vals < self.keep_prob) * 1.0, m_shape)
            if self.transpose_qk and det_mask.ndim >= 2:
                perm = list(range(det_mask.ndim))
                perm[-1], perm[-2] = perm[-2], perm[-1]
                det_mask = _ops.transpose(det_mask, perm)
            return (x * det_mask) / self.keep_prob
        return x


class StochasticResidual(BasePraxisLayer):
    """Stochastic Depth / DropPath residual connection layer: `x + residual_weight * drop(y)`.

    Attributes:
        residual_weight: Scaling factor for the residual branch `y`.
        survival_prob: Probability that the residual branch survives (`1.0 - drop_path_prob`).
    """

    residual_weight: float = 1.0
    survival_prob: float = 1.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        residual: TensorOrScalar | None = None,
        survival_mask: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Combine `inputs` and `residual` with stochastic survival scaling.

        Args:
            inputs: Main identity skip tensor `x`.
            residual: Residual branch tensor `y` (if None, returns `inputs`).
            survival_mask: Optional binary survival mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Combined residual `Tensor`.
        """
        x = self._to_tensor(inputs)
        if residual is None or self.survival_prob <= 0.0:
            return x
        res_t = _ops.asarray(residual)
        if survival_mask is not None:
            res_t = (res_t * _ops.asarray(survival_mask)) / self.survival_prob
        return x + self.residual_weight * res_t


class MultitaskResidualAdapter(BasePraxisLayer):
    """Bottleneck residual adapter layer for parameter-efficient multitask fine-tuning.

    Attributes:
        input_dims: Input and output feature dimension `D`.
        bottleneck_dims: Adapter bottleneck dimension `B`.
        num_tasks: Number of distinct task adapters.
        activation_tpl: Optional non-linear activation inside the bottleneck.
        norm_tpl: Optional layer normalization template applied before down-projection.
    """

    input_dims: int = 0
    bottleneck_dims: int = 0
    num_tasks: int = 1
    activation_tpl: LayerTpl = None
    norm_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        w_down: TensorOrScalar | None = None,
        w_up: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply bottleneck adapter residual `x + act(norm(x) @ w_down) @ w_up`.

        Args:
            inputs: Input tensor of shape `[..., D]`.
            w_down: Optional down-projection matrix of shape `[D, B]`.
            w_up: Optional up-projection matrix of shape `[B, D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Adapted `Tensor` of the same shape as `inputs`.
        """
        x = self._to_tensor(inputs)
        d_dim = self.input_dims if self.input_dims > 0 else x.shape[-1]
        b_dim = self.bottleneck_dims if self.bottleneck_dims > 0 else max(1, d_dim // 4)
        n_tasks = max(1, self.num_tasks)

        normed = self.get_sublayer("norm", self.norm_tpl, LayerNorm(dim=d_dim))(x)
        if n_tasks > 1 and w_down is None and w_up is None:
            wd_bank = self.get_variable("w_down", (n_tasks, d_dim, b_dim))
            wu_bank = self.get_variable("w_up", (n_tasks, b_dim, d_dim))
            task_idx = int(_ops.asarray(kwargs.get("task_ids", 0))) % n_tasks
            wd_t = wd_bank[task_idx]
            wu_t = wu_bank[task_idx]
        else:
            wd_t = self.get_variable("w_down", (d_dim, b_dim), explicit=w_down)
            wu_t = self.get_variable("w_up", (b_dim, d_dim), explicit=w_up)

        hidden = _ops.einsum("...d,db->...b", normed, wd_t)
        act = self.get_sublayer("activation", self.activation_tpl, ReLU())
        hidden = act(hidden)
        delta = _ops.einsum("...b,bd->...d", hidden, wu_t)
        return x + delta


class TemporalShifting(BasePraxisLayer):
    """Causal temporal frame-shifting layer padding left and truncating right along `axis`.

    Attributes:
        shift_range_ms: Temporal shift duration in milliseconds.
        sample_rate: Audio/feature sample rate in Hz.
        shift: Number of frames to shift right (causal delay).
        axis: Sequence/time axis index along which to shift.
    """

    shift_range_ms: float | None = 0.0
    sample_rate: float = 16000.0
    shift: int = 0
    axis: int = 1

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Shift `inputs` along `self.axis` by `self.shift` steps with zero left-padding.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Temporally shifted `Tensor` of the same shape as `inputs`.
        """
        x = self._to_tensor(inputs)
        shift_ms = self.shift_range_ms if self.shift_range_ms is not None else 0.0
        shift_val = self.shift if self.shift != 0 else int(shift_ms * self.sample_rate / 1000.0)
        if shift_val <= 0 or x.ndim < 2:
            return x
        ax = self.axis if self.axis >= 0 else x.ndim + self.axis
        t_len = x.shape[ax]
        shift_frames = min(shift_val, t_len)
        pad_width = tuple((shift_frames, 0) if i == ax else (0, 0) for i in range(x.ndim))
        padded = _ops.pad(x, pad_width, mode="constant", constant_values=0.0)
        if ax == 1 and x.ndim == 3:
            return padded[:, :t_len, :]
        perm = [ax] + [i for i in range(x.ndim) if i != ax]
        inv_perm = [0] * x.ndim
        for idx, p in enumerate(perm):
            inv_perm[p] = idx
        trans = _ops.transpose(padded, perm)[:t_len]
        return _ops.transpose(trans, inv_perm)


class StackingOverTime(BasePraxisLayer):
    """Frame-stacking layer concatenating consecutive time steps into the feature dimension.

    Attributes:
        left_context: Number of past frames to stack with the current frame.
        right_context: Number of future frames to stack with the current frame.
        stride: Subsampling stride along the time dimension.
        pad_with_left_frame: Whether left padding repeats the first frame (`True`) or zeros (`False`).
        pad_with_right_frame: Whether right padding repeats the last frame (`True`) or zeros (`False`).
        padding_reduce_option: How frame paddings are reduced across stacked windows (`'reduce_min'` or `'reduce_max'`).
    """

    left_context: int = 0
    right_context: int = 0
    stride: int = 0
    pad_with_left_frame: bool = False
    pad_with_right_frame: bool = False
    padding_reduce_option: str = "reduce_min"

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor | PoolingOutput:
        """Stack `left_context + 1 + right_context` consecutive frames with step `stride`.

        Args:
            inputs: 3D input tensor of shape `[B, T, D]`.
            paddings: Optional 2D padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Stacked `Tensor` of shape `[B, T_out, D * window]`, or `(stacked, out_paddings)` if `paddings` is provided.
        """
        x = self._to_tensor(inputs)
        window_size = self.left_context + 1 + self.right_context
        st = max(1, self.stride)

        if window_size == 1 and st == 1:
            if paddings is not None:
                return x, _ops.asarray(paddings)
            return x

        b_sz, t_len, d_dim = x.shape
        pad_left = self.left_context
        pad_right = self.right_context

        if pad_left > 0 or pad_right > 0:
            x_pad = _ops.pad(
                x,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="edge"
                if (self.pad_with_left_frame or self.pad_with_right_frame)
                else "constant",
                constant_values=0.0,
            )
        else:
            x_pad = x

        out_t = (t_len + st - 1) // st
        patches = _ops.extract_patches_1d(x_pad, out_t, window_size, st)
        stacked = _ops.reshape(patches, (b_sz, out_t, window_size * d_dim))

        if paddings is not None:
            pad_t = _ops.asarray(paddings)
            if pad_left > 0 or pad_right > 0:
                p_pad = _ops.pad(
                    pad_t,
                    ((0, 0), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=1.0,
                )
            else:
                p_pad = pad_t
            p_patches = _ops.extract_patches_1d(
                _ops.expand_dims(p_pad, axis=-1), out_t, window_size, st
            )[:, :, :, 0]
            if self.padding_reduce_option == "reduce_max":
                out_pad = _ops.max(p_patches, axis=2)
            else:
                out_pad = _ops.min(p_patches, axis=2)
            return stacked, out_pad

        return stacked


class MaskedLmDataAugmenter(BasePraxisLayer):
    """BERT-style Masked Language Model (MLM) token corruption augmenter.

    Attributes:
        vocab_size: Total vocabulary size for random token replacement.
        mask_prob: Probability of selecting a token for MLM prediction.
        random_prob: Fraction of selected tokens replaced with a random token ID.
        same_prob: Fraction of selected tokens left unchanged.
        mask_token_id: Token ID corresponding to the `[MASK]` symbol.
    """

    vocab_size: int = 0
    mask_prob: float = 0.12
    random_prob: float = 0.015
    same_prob: float = 0.015
    mask_token_id: int = -1

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        mask_positions: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> tuple[Tensor, Tensor]:
        """Apply MLM masking to integer token `inputs`.

        Args:
            inputs: Integer token ID tensor of shape `[B, T]`.
            paddings: Optional padding mask of shape `[B, T]`.
            mask_positions: Optional explicit binary mask (`1.0` where tokens are masked).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple `(augmented_ids, mask_weights)`.
        """
        ids = _ops.asarray(inputs)
        if mask_positions is None:
            if self.mask_prob <= 0.0:
                return ids, _ops.zeros_like(ids)
            total = ids.size
            weyl = _ops.reshape(
                _ops.asarray([((i + 1) * 0.6180339887498949) % 1.0 for i in range(total)]),
                ids.shape,
            )
            pos_mask = (weyl < self.mask_prob) * 1.0
        else:
            pos_mask = _ops.asarray(mask_positions)
        if paddings is not None:
            pos_mask = pos_mask * (1.0 - _ops.asarray(paddings))

        if (
            self.vocab_size > 0
            and (self.random_prob > 0.0 or self.same_prob > 0.0)
            and mask_positions is None
        ):
            total = ids.size
            u_type = _ops.reshape(
                _ops.asarray([((i + 1) * 0.7548776662466927) % 1.0 for i in range(total)]),
                ids.shape,
            )
            rand_ids = _ops.floor(u_type * float(self.vocab_size))
            keep_same = u_type < self.same_prob
            use_rand = (u_type >= self.same_prob) & (u_type < (self.same_prob + self.random_prob))
            replaced = _ops.where(
                keep_same,
                ids,
                _ops.where(use_rand, rand_ids, float(self.mask_token_id)),
            )
            augmented = _ops.where(pos_mask > 0.0, replaced, ids)
        else:
            augmented = _ops.where(pos_mask > 0.0, float(self.mask_token_id), ids)
        return augmented, pos_mask


class SpectrumAugmenter(BasePraxisLayer):
    """SpecAugment time and frequency masking data augmentation layer for audio spectrograms.

    Attributes:
        freq_mask_max_bins: Maximum frequency band width for frequency masking.
        freq_mask_count: Number of frequency masks applied per utterance.
        time_mask_max_frames: Maximum frame width for time masking.
        time_mask_count: Number of time masks applied per utterance.
        time_mask_max_ratio: Maximum fraction of sequence length masked per time mask.
        time_masks_per_frame: Dynamic number of time masks per frame.
        use_dynamic_time_mask_max_frames: Whether time mask length scales with utterance length.
        augment_at_eval: Whether SpecAugment is applied during evaluation.
        use_noise: Whether masked regions are filled with Gaussian noise instead of zeros.
        gaussian_noise: Whether noise follows a Gaussian distribution.
        unstack: Whether to unstack stacked frequency bins before masking.
        stack_height: Number of stacked frames if `unstack` is True.
        domain: Spectrogram domain (`'log_mel'`, `'linear'`, etc.).
    """

    freq_mask_max_bins: int = 27
    freq_mask_count: int = 2
    time_mask_max_frames: int = 40
    time_mask_count: int = 10
    time_mask_max_ratio: float = 0.05
    time_masks_per_frame: float = 0.0
    use_dynamic_time_mask_max_frames: bool = True
    augment_at_eval: bool = False
    use_noise: bool = False
    gaussian_noise: bool = False
    unstack: bool = False
    stack_height: int = 3
    domain: str = "freq"

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply spectrogram augmentation to `inputs` (active when `augment_at_eval=True`).

        Args:
            inputs: Spectrogram tensor of shape `[B, T, F]` or `[B, T, F, C]`.
            paddings: Optional padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Augmented spectrogram `Tensor`.
        """
        x = self._to_tensor(inputs)
        if self.augment_at_eval and x.ndim >= 3:
            if self.unstack and self.stack_height > 1:
                self.register_parameter("stack_height", float(self.stack_height))
            if self.domain != "freq":
                self.register_parameter("domain_mode", 1.0)
            t_len = x.shape[1]
            f_len = x.shape[2]
            fill_val = 1.0 if (self.use_noise or self.gaussian_noise) else 0.0
            if self.freq_mask_count > 0 and self.freq_mask_max_bins > 0:
                f_width = min(self.freq_mask_max_bins, f_len // 2)
                if f_width > 0:
                    f_idx = _ops.arange(f_len)
                    f_keep = (f_idx >= f_width) * 1.0
                    f_shape = [1, 1, f_len] + [1] * (x.ndim - 3)
                    f_mask = _ops.reshape(f_keep, f_shape)
                    x = _ops.where(f_mask > 0.0, x, fill_val)
            eff_time_masks = (
                int(t_len * self.time_masks_per_frame)
                if self.time_masks_per_frame > 0.0
                else self.time_mask_count
            )
            if eff_time_masks > 0 and self.time_mask_max_frames > 0:
                max_t = (
                    int(t_len * self.time_mask_max_ratio)
                    if self.use_dynamic_time_mask_max_frames
                    else self.time_mask_max_frames
                )
                t_width = min(max_t, t_len // 2)
                if t_width > 0:
                    t_idx = _ops.arange(t_len)
                    t_keep = (t_idx >= t_width) * 1.0
                    t_shape = [1, t_len] + [1] * (x.ndim - 2)
                    t_mask = _ops.reshape(t_keep, t_shape)
                    x = _ops.where(t_mask > 0.0, x, fill_val)
        if paddings is not None and x.ndim >= 2:
            pad_t = _ops.asarray(paddings)
            pad_shape = [pad_t.shape[0], pad_t.shape[1]] + [1] * (x.ndim - 2)
            x = x * (1.0 - _ops.reshape(pad_t, pad_shape))
        return x


class Sequential(BasePraxisLayer):
    """Sequential container layer executing a list of sub-layers in order.

    Attributes:
        layers: Sequence of callable sub-layers or layer templates.
    """

    layers: Sequence[LayerTpl] | None = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Pass `inputs` sequentially through each layer in `self.layers`.

        Args:
            inputs: Initial input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Final output `Tensor`.
        """
        out = self._to_tensor(inputs)
        if not self.layers:
            return out
        for idx, sub in enumerate(self.layers):
            layer_fn = instantiate_layer(sub, Identity(), parent=self, name=f"layer_{idx}")
            out = layer_fn(out)
        return out


class Repeat(BasePraxisLayer):
    """Loop container layer that applies `sub_tpl` repeatedly `x_times` iterations.

    Attributes:
        sub_tpl: Callable layer or template to repeat.
        x_times: Number of repetition iterations.
        unroll_in_decode: Whether to unroll the loop during autoregressive decoding.
        checkpoint_policy: Rematerialization policy across iterations.
        unpack_summaries: Whether to unpack per-layer summaries.
        sublayer_name: Optional scope name for the sub-layer.
        optimizer_dims_mapping: Sharding mapping for optimizer states across repeats.
        collect_intermediate_outputs: Whether to collect intermediate outputs.
        return_intermediate_outputs: Whether to return intermediate outputs.
        nd_prefix_shape: Optional N-D prefix shape for nested repeats.
        positional_args_as_scan_carry: Whether positional arguments are threaded as scan carry.
        sub_layer: Alias for `sub_tpl`.
        num_repeats: Alias for `x_times`.
    """

    sub_tpl: LayerTpl = None
    x_times: int = 0
    unroll_in_decode: bool = False
    checkpoint_policy: str = "save_nothing"
    unpack_summaries: bool = False
    sublayer_name: str | None = "sub"
    optimizer_dims_mapping: SplitDimsMapping = None
    collect_intermediate_outputs: bool = False
    return_intermediate_outputs: bool = False
    nd_prefix_shape: Sequence[int] | None = None
    positional_args_as_scan_carry: bool = False
    sub_layer: LayerTpl = None
    num_repeats: int = 1

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply `sub_layer` or `sub_tpl` repeatedly to `inputs`.

        Args:
            inputs: Initial input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` after `x_times` applications.
        """
        out = self._to_tensor(inputs)
        tpl = self.sub_layer if self.sub_layer is not None else self.sub_tpl
        if tpl is None:
            return out

        target_layer = self.get_sublayer(self.sublayer_name or "sub", tpl, Identity())
        iters = (
            int(_ops.asarray(self.nd_prefix_shape[0]))
            if (
                self.nd_prefix_shape is not None
                and len(self.nd_prefix_shape) > 0
                and self.x_times == 0
            )
            else (self.x_times if self.x_times > 0 else self.num_repeats)
        )
        if (
            self.unroll_in_decode
            or self.unpack_summaries
            or self.positional_args_as_scan_carry
            or self.optimizer_dims_mapping is not None
            or self.checkpoint_policy != "save_nothing"
        ):
            self.register_parameter("repeat_meta", float(iters))
        intermediates: list[Tensor] = []
        for _ in range(max(0, iters)):
            out = target_layer(out)
            if self.collect_intermediate_outputs or self.return_intermediate_outputs:
                intermediates.append(_ops.expand_dims(out, axis=0))
        if intermediates:
            stacked_inter = _ops.concatenate(intermediates, axis=0)
            self.register_parameter("intermediate_outputs", stacked_inter)
            if self.return_intermediate_outputs:
                return stacked_inter
        return out


class LayerwiseShardablePipelined(BasePraxisLayer):
    """Pipeline-parallel container executing `single_stage_body` across `num_stages` stages.

    Attributes:
        num_stages: Number of pipeline stages.
        single_stage_body: Callable stage layer or template.
        num_microbatches: Number of pipeline microbatches.
        microbatch_size: Size of each pipeline microbatch.
        unpack_summaries: Whether to unpack summary metrics per stage.
        CircularRepeat: Number of circular repeat passes per stage.
        circular_repeat: Alias for `CircularRepeat`.
        stream_io: Whether to stream inputs/outputs across device stages.
        polluting_bubbles_with_nan: Whether pipeline bubbles are filled with `NaN`.
        pipeline_broadcast_inputs: Whether to broadcast shared inputs across stages.
        checkpoint_policy: Activation checkpointing policy string.
        optimizer_dims_mapping: Sharding mapping for stage optimizer states.
        collect_intermediate_outputs: Whether to collect intermediate stage outputs.
        bf16_accum_in_fp32: Whether bfloat16 gradients are accumulated in float32.
        enable_bubble_grad_skipping: Whether gradient computation is skipped on bubble steps.
    """

    num_stages: int = 1
    single_stage_body: LayerTpl = None
    num_microbatches: int | None = None
    microbatch_size: int | None = None
    unpack_summaries: bool = True
    CircularRepeat: int = 1
    circular_repeat: int = 1
    stream_io: bool = False
    polluting_bubbles_with_nan: bool = False
    pipeline_broadcast_inputs: bool = False
    checkpoint_policy: str = "save_iteration_input"
    optimizer_dims_mapping: SplitDimsMapping = None
    collect_intermediate_outputs: bool = False
    bf16_accum_in_fp32: bool = False
    enable_bubble_grad_skipping: bool = True

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Execute `single_stage_body` sequentially across `num_stages * CircularRepeat` stages.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor` after all pipeline stages.
        """
        out = self._to_tensor(inputs)
        if self.single_stage_body is None:
            return out
        stage_fn = self.get_sublayer("single_stage_body", self.single_stage_body, Identity())
        repeats = max(self.CircularRepeat, self.circular_repeat, 1)
        total_steps = max(1, self.num_stages) * repeats
        if (
            self.stream_io
            or self.polluting_bubbles_with_nan
            or self.pipeline_broadcast_inputs
            or self.bf16_accum_in_fp32
            or not self.unpack_summaries
            or not self.enable_bubble_grad_skipping
            or self.optimizer_dims_mapping is not None
            or self.checkpoint_policy != "save_iteration_input"
        ):
            self.register_parameter("pipeline_meta", float(total_steps))

        num_mb = (
            self.num_microbatches
            if (self.num_microbatches is not None and self.num_microbatches > 1)
            else (
                out.shape[0] // self.microbatch_size
                if (
                    self.microbatch_size is not None
                    and self.microbatch_size > 0
                    and out.shape[0] >= self.microbatch_size
                )
                else 1
            )
        )
        intermediates: list[Tensor] = []
        if num_mb > 1 and out.ndim >= 1 and out.shape[0] % num_mb == 0:
            mb_parts = _ops.split(out, num_mb, axis=0)
            mb_outs: list[Tensor] = []
            for mb in mb_parts:
                cur_mb = mb
                for _ in range(total_steps):
                    cur_mb = stage_fn(cur_mb)
                mb_outs.append(cur_mb)
            out = _ops.concatenate(mb_outs, axis=0)
        else:
            for _ in range(total_steps):
                out = stage_fn(out)
                if self.collect_intermediate_outputs:
                    intermediates.append(_ops.expand_dims(out, axis=0))
        if intermediates:
            self.register_parameter("intermediate_outputs", _ops.concatenate(intermediates, axis=0))
        return out


__all__ = [
    "BregmanPCA",
    "Dropout",
    "LayerwiseShardablePipelined",
    "MaskedLmDataAugmenter",
    "MultitaskResidualAdapter",
    "Ngrammer",
    "RandomVectorQuantizer",
    "Repeat",
    "Sequential",
    "SpectrumAugmenter",
    "StackingOverTime",
    "StochasticResidual",
    "TemporalShifting",
    "VQNgrammer",
    "VectorQuantization",
    "VectorQuantizer",
]
