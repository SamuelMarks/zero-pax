"""Embedding, positional encoding, softmax, and loss layers for zero_pax.praxis.layers."""

from __future__ import annotations

from zero_pax import _tensor_ops as np
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl, instantiate_layer
from zero_pax.praxis.layers.linear_pooling import EinsumOp, FeedForward
from zero_pax.types import Tensor, TensorOrScalar, WeightInit


class Embedding(BasePraxisLayer):
    """Token embedding lookup layer supporting INDEX and MATMUL lookup styles.

    Attributes:
        num_classes: Vocabulary size (number of discrete token IDs).
        input_dims: Embedding vector dimension size.
        lookup_style: Lookup method (`'INDEX'` or `'MATMUL'`).
        scale_sqrt_depth: Whether to multiply embeddings by `sqrt(input_dims)`.
        set_nan_for_oob_id: Whether out-of-bounds token IDs produce `NaN` instead of `0.0`.
        array_lookup: Optional template for array lookup operation.
        einsum: Optional template for einsum operation.
        array_lookup_tpl: Alias for `array_lookup`.
        einsum_tpl: Alias for `einsum`.
    """

    num_classes: int = 0
    input_dims: int = 0
    lookup_style: str = "INDEX"
    scale_sqrt_depth: bool = False
    set_nan_for_oob_id: bool = False
    array_lookup: LayerTpl = None
    einsum: LayerTpl = None
    array_lookup_tpl: LayerTpl = None
    einsum_tpl: LayerTpl = None

    def emb_lookup(self, ids: TensorOrScalar, w: TensorOrScalar | None = None) -> Tensor:
        """Look up embeddings for integer token `ids`.

        Args:
            ids: Integer token ID tensor of shape `[...]`.
            w: Optional explicit embedding table of shape `[num_classes, input_dims]`.

        Returns:
            Embedded `Tensor` of shape `[..., input_dims]`.
        """
        ids_t = np.asarray(ids)
        v_sz = self.num_classes if self.num_classes > 0 else 10
        d_sz = self.input_dims if self.input_dims > 0 else 8
        w_t = self.get_variable("w", (v_sz, d_sz), explicit=w)

        oob_mask = (ids_t < 0) | (ids_t >= v_sz)
        valid_ids = np.where(oob_mask, 0, ids_t)
        one_hot = (np.reshape(valid_ids, list(valid_ids.shape) + [1]) == np.arange(v_sz)) * 1.0

        if self.lookup_style.upper() == "INDEX":
            flat_one_hot = np.reshape(one_hot, (-1, v_sz))
            flat_embs = np.matmul(flat_one_hot, w_t)
            embs = np.reshape(flat_embs, list(valid_ids.shape) + [d_sz])
        else:
            einsum_layer = instantiate_layer(
                self.einsum_tpl or self.einsum,
                EinsumOp(equation="...v,vd->...d"),
            )
            embs = einsum_layer("...v,vd->...d", one_hot, w_t)

        if self.scale_sqrt_depth:
            embs = embs * (d_sz**0.5)

        fill_value = np.nan if self.set_nan_for_oob_id else 0.0
        embs = np.where(
            np.reshape(oob_mask, list(oob_mask.shape) + [1]),
            fill_value,
            embs,
        )
        return embs

    def __call__(
        self,
        ids: TensorOrScalar,
        w: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Look up token embeddings for `ids`.

        Args:
            ids: Integer token ID tensor.
            w: Optional explicit embedding table of shape `[num_classes, input_dims]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Embedded `Tensor` of shape `[..., input_dims]`.
        """
        return self.emb_lookup(ids, w=w)


class SharedEmbeddingSoftmax(BasePraxisLayer):
    """Shared token embedding lookup and output softmax projection layer.

    Attributes:
        lookup_style: Embedding lookup method (`'INDEX'` or `'MATMUL'`).
        scale_sqrt_depth: Whether to scale looked-up embeddings by `sqrt(input_dims)`.
        array_lookup_tpl: Optional array lookup template.
        einsum_tpl: Optional einsum template.
        input_dims: Embedding and hidden feature dimension size.
        num_classes: Vocabulary size (number of classes).
        soft_cap_logits: Optional tanh soft-cap threshold for output logits.
        bi_tempered_loss_tpl: Optional Bi-Tempered loss layer template.
        label_smoothing_prob: Label smoothing probability for cross-entropy.
        label_smoothing_apply_for_eval: Whether to apply label smoothing during evaluation.
        z_loss_weight: Auxiliary z-loss regularization coefficient.
        bias_init: Optional bias weight initialization strategy.
        feed_forward_tpl: Optional feed-forward projection template.
        scale_before_logits: Optional scaling factor before logit computation.
        chunk_size: Optional chunk size for memory-efficient softmax computation.
        ais_temperature: Optional temperature parameter.
        use_bias: Whether to add a class bias vector to the logits.
    """

    lookup_style: str = "INDEX"
    scale_sqrt_depth: bool = False
    array_lookup_tpl: LayerTpl = None
    einsum_tpl: LayerTpl = None
    input_dims: int = 0
    num_classes: int = 0
    soft_cap_logits: float | None = 0.0
    bi_tempered_loss_tpl: LayerTpl = None
    label_smoothing_prob: float = 0.0
    label_smoothing_apply_for_eval: bool = True
    z_loss_weight: float = 0.0
    bias_init: WeightInit | None = None
    feed_forward_tpl: LayerTpl = None
    scale_before_logits: float | None = None
    chunk_size: int = 0
    ais_temperature: float | None = None
    use_bias: bool = True

    def emb_lookup(self, ids: TensorOrScalar, w: TensorOrScalar | None = None) -> Tensor:
        """Look up shared embeddings for integer token `ids`.

        Args:
            ids: Integer token ID tensor.
            w: Optional explicit weight matrix of shape `[num_classes, input_dims]` or `[input_dims, num_classes]`.

        Returns:
            Embedded `Tensor` of shape `[..., input_dims]`.
        """
        v_sz = self.num_classes if self.num_classes > 0 else 10
        d_sz = self.input_dims if self.input_dims > 0 else 8
        emb = Embedding(
            num_classes=v_sz,
            input_dims=d_sz,
            lookup_style=self.lookup_style,
            scale_sqrt_depth=self.scale_sqrt_depth,
            params_init=self.params_init,
        )
        if w is not None:
            w_t = np.asarray(w)
            if w_t.shape == (d_sz, v_sz) and d_sz != v_sz:
                w_t = np.transpose(w_t, (1, 0))
            return emb.emb_lookup(ids, w=w_t)
        return emb.emb_lookup(ids)

    def __call__(
        self,
        inputs: TensorOrScalar,
        class_weights: TensorOrScalar | None = None,
        class_ids: TensorOrScalar | None = None,
        class_probabilities: TensorOrScalar | None = None,
        w: TensorOrScalar | None = None,
        b: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Project `inputs` to vocabulary logits (or compute loss when target labels are provided).

        Args:
            inputs: Hidden state tensor of shape `[..., input_dims]`.
            class_weights: Optional per-example loss weights.
            class_ids: Optional target integer class IDs.
            class_probabilities: Optional target class probability distributions.
            w: Optional explicit projection matrix of shape `[input_dims, num_classes]`.
            b: Optional explicit bias vector of shape `[num_classes]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Logits `Tensor` of shape `[..., num_classes]` or per-example cross-entropy loss `Tensor`.
        """
        x = np.asarray(inputs)
        if self.scale_before_logits is not None:
            x = x * float(self.scale_before_logits)
        d_sz = self.input_dims if self.input_dims > 0 else x.shape[-1]
        v_sz = self.num_classes if self.num_classes > 0 else d_sz

        w_t: Tensor
        if w is None:
            w_t = self.get_variable("w", (d_sz, v_sz))
        else:
            w_raw = np.asarray(w)
            w_t = (
                np.transpose(w_raw, (1, 0))
                if (w_raw.shape == (v_sz, d_sz) and d_sz != v_sz)
                else w_raw
            )

        logits = np.einsum("...d,dv->...v", x, w_t)
        if self.use_bias and b is not None:
            logits = logits + np.asarray(b)
        if self.soft_cap_logits is not None and self.soft_cap_logits > 0.0:
            logits = self.soft_cap_logits * np.tanh(logits / self.soft_cap_logits)

        if class_ids is not None or class_probabilities is not None:
            softmax = FullSoftmax(
                input_dims=d_sz,
                num_classes=v_sz,
                label_smoothing_prob=self.label_smoothing_prob,
                z_loss_weight=self.z_loss_weight,
            )
            return softmax(
                logits,
                class_weights=class_weights,
                class_ids=class_ids,
                class_probabilities=class_probabilities,
            )

        return logits


class GShardSharedEmbeddingSoftmax(BasePraxisLayer):
    """GShard-style shared embedding lookup and logit computation with soft-capping and clipping.

    Attributes:
        input_dims: Embedding and hidden feature dimension size.
        num_classes: Vocabulary size (number of classes).
        use_tgt_labels_size_as_loss_denominator: Whether to normalize loss by target label count.
        soft_cap_logits: Tanh soft-cap threshold for logits (`0.0` disables capping).
        logits_abs_max: Hard symmetric clipping threshold for logits (`0.0` disables clipping).
        z_loss_weight: Auxiliary z-loss coefficient.
        label_smoothing_prob: Label smoothing probability.
    """

    input_dims: int = 0
    num_classes: int = 0
    use_tgt_labels_size_as_loss_denominator: bool = True
    soft_cap_logits: float = 0.0
    logits_abs_max: float = 0.0
    z_loss_weight: float = 0.0
    label_smoothing_prob: float = 0.0

    def emb_lookup(self, ids: TensorOrScalar, w: TensorOrScalar | None = None) -> Tensor:
        """Look up GShard embeddings scaled by `1 / sqrt(input_dims)`.

        Args:
            ids: Integer token ID tensor.
            w: Optional explicit weight matrix of shape `[input_dims, num_classes]` or `[num_classes, input_dims]`.

        Returns:
            Scaled embedded `Tensor` of shape `[..., input_dims]`.
        """
        ids_t = np.asarray(ids)
        v_sz = self.num_classes if self.num_classes > 0 else 10
        d_sz = self.input_dims if self.input_dims > 0 else 8

        w_t: Tensor
        if w is None:
            w_t = self.get_variable("w", (d_sz, v_sz))
        else:
            w_raw = np.asarray(w)
            w_t = (
                np.transpose(w_raw, (1, 0))
                if (w_raw.shape == (v_sz, d_sz) and d_sz != v_sz)
                else w_raw
            )

        valid_ids = np.where((ids_t < 0) | (ids_t >= v_sz), 0, ids_t)
        one_hot = np.reshape(valid_ids, list(valid_ids.shape) + [1]) == np.arange(v_sz)
        ret = np.einsum("...v,dv->...d", one_hot, w_t)
        ret = ret * (d_sz**-0.5)
        return ret

    def __call__(
        self,
        inputs: TensorOrScalar,
        w: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute GShard logits from `inputs` with optional clipping and soft-capping.

        Args:
            inputs: Hidden state tensor of shape `[..., input_dims]`.
            w: Optional explicit weight matrix of shape `[input_dims, num_classes]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Logits `Tensor` of shape `[..., num_classes]`.
        """
        x = np.asarray(inputs)
        d_sz = self.input_dims if self.input_dims > 0 else x.shape[-1]
        v_sz = self.num_classes if self.num_classes > 0 else d_sz
        w_t = self.get_variable("w", (d_sz, v_sz), explicit=w)

        logits = np.einsum("...d,dv->...v", x, w_t)
        if self.logits_abs_max > 0.0:
            logits = np.clip(logits, -self.logits_abs_max, self.logits_abs_max)
        if self.soft_cap_logits > 0.0:
            logits = self.soft_cap_logits * np.tanh(logits / self.soft_cap_logits)
        return logits


class PositionalEmbedding(BasePraxisLayer):
    """1D Sinusoidal Positional Embedding layer (Vaswani et al., 2017).

    Attributes:
        min_timescale: Minimum geometric wavelength timescale.
        max_timescale: Maximum geometric wavelength timescale.
        embedding_dims: Output positional embedding dimension size.
    """

    min_timescale: int = 1
    max_timescale: int = 10000
    embedding_dims: int = 0

    def __call__(
        self,
        seq_length: int | TensorOrScalar | None = None,
        position: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Generate sinusoidal positional embeddings of shape `[B, T, D]`.

        Args:
            seq_length: Integer sequence length `T` or input tensor from which `T` is inferred.
            position: Optional explicit position index tensor of shape `[B, T]` or `[T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Positional embedding `Tensor` of shape `[B, T, embedding_dims]`.
        """
        d_emb = self.embedding_dims if self.embedding_dims > 0 else 8
        pos_t: Tensor
        if position is None:
            if seq_length is None:
                seq_len_int = 1
            elif isinstance(seq_length, int):
                seq_len_int = seq_length
            else:
                seq_arr = np.asarray(seq_length)
                if seq_arr.ndim == 0:
                    seq_len_int = int(seq_arr)
                elif seq_arr.ndim >= 2:
                    seq_len_int = seq_arr.shape[1]
                else:
                    seq_len_int = seq_arr.shape[0]
            pos_t = np.expand_dims(np.arange(seq_len_int), 0)
        else:
            pos_raw = np.asarray(position)
            pos_t = np.expand_dims(pos_raw, 0) if pos_raw.ndim == 1 else pos_raw

        num_timescales = max(1, d_emb // 2)
        log_timescale_increment = np.log(
            float(self.max_timescale) / float(self.min_timescale)
        ) / max(1.0, float(num_timescales) - 1.0)
        inv_timescales = self.min_timescale * np.exp(
            np.arange(num_timescales) * -log_timescale_increment
        )
        scaled_time = np.expand_dims(pos_t, 2) * np.reshape(inv_timescales, (1, 1, num_timescales))
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=2)
        if d_emb % 2 == 1:
            signal = np.pad(
                signal,
                ((0, 0), (0, 0), (0, 1)),
                mode="constant",
                constant_values=0.0,
            )
        return signal[:, :, :d_emb]


class PositionalEmbedding2D(BasePraxisLayer):
    """2D Sinusoidal Positional Embedding layer for image patch grids.

    Attributes:
        h: Grid height (number of vertical patches).
        w: Grid width (number of horizontal patches).
        embedding_dims: Total embedding dimension size.
        pos_transform: Coordinate ordering transformation (`'h_w'` or `'w_h'`).
        num_prepend_cls_tokens: Number of zero-embedded CLS tokens prepended to the sequence.
        num_append_cls_tokens: Number of zero-embedded CLS tokens appended to the sequence.
    """

    h: int = 0
    w: int = 0
    embedding_dims: int = 0
    pos_transform: str = "h_w"
    num_prepend_cls_tokens: int = 0
    num_append_cls_tokens: int = 0

    def __call__(
        self,
        seq_length: int | TensorOrScalar | None = None,
        position: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute 2D sinusoidal positional embeddings of shape `[1, H*W + cls, D]`.

        Args:
            seq_length: Optional sequence length override.
            position: Optional explicit position indices.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            2D positional embedding `Tensor`.
        """
        h_sz = self.h if self.h > 0 else 2
        w_sz = self.w if self.w > 0 else 2
        d_emb = self.embedding_dims if self.embedding_dims > 0 else 8
        half_d = max(2, d_emb // 2)

        pe_1d = PositionalEmbedding(embedding_dims=half_d)
        h_pos = pe_1d(seq_length=h_sz)[0]
        w_pos = pe_1d(seq_length=w_sz)[0]

        h_grid = np.broadcast_to(np.expand_dims(h_pos, 1), (h_sz, w_sz, h_pos.shape[-1]))
        w_grid = np.broadcast_to(np.expand_dims(w_pos, 0), (h_sz, w_sz, w_pos.shape[-1]))
        if self.pos_transform == "w_h":
            emb_2d = np.concatenate([w_grid, h_grid], axis=-1)
        else:
            emb_2d = np.concatenate([h_grid, w_grid], axis=-1)

        emb_flat = np.reshape(emb_2d, (1, h_sz * w_sz, emb_2d.shape[-1]))
        if emb_flat.shape[-1] < d_emb:
            emb_flat = np.pad(
                emb_flat,
                ((0, 0), (0, 0), (0, d_emb - emb_flat.shape[-1])),
                mode="constant",
                constant_values=0.0,
            )
        elif emb_flat.shape[-1] > d_emb:
            emb_flat = emb_flat[:, :, :d_emb]

        if self.num_prepend_cls_tokens > 0:
            prep = np.zeros((1, self.num_prepend_cls_tokens, d_emb))
            emb_flat = np.concatenate([prep, emb_flat], axis=1)
        if self.num_append_cls_tokens > 0:
            app = np.zeros((1, self.num_append_cls_tokens, d_emb))
            emb_flat = np.concatenate([emb_flat, app], axis=1)

        return emb_flat


class TrainablePositionalEmbedding(PositionalEmbedding):
    """Trainable lookup-table Positional Embedding layer.

    Attributes:
        max_seq_length: Maximum supported sequence length in the lookup table.
        lookup_style: Lookup style (`'MATMUL'` or `'INDEX'`).
        array_lookup_tpl: Optional array lookup template.
        einsum_tpl: Optional einsum template.
    """

    max_seq_length: int = 0
    lookup_style: str = "MATMUL"
    array_lookup_tpl: LayerTpl = None
    einsum_tpl: LayerTpl = None

    def __call__(
        self,
        seq_length: int | TensorOrScalar | None = None,
        position: TensorOrScalar | None = None,
        w: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Look up trainable positional embeddings for `seq_length` or `position`.

        Args:
            seq_length: Optional integer sequence length `T`.
            position: Optional explicit position tensor of shape `[B, T]` or `[T]`.
            w: Optional explicit positional embedding table of shape `[max_seq_length, embedding_dims]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Positional embedding `Tensor` of shape `[B, T, embedding_dims]`.
        """
        max_len = self.max_seq_length if self.max_seq_length > 0 else 16
        d_emb = self.embedding_dims if self.embedding_dims > 0 else 8
        w_t = self.get_variable("w", (max_len, d_emb), explicit=w)

        pos_t: Tensor
        if position is None:
            seq_len_int = seq_length if isinstance(seq_length, int) else max_len
            pos_t = np.expand_dims(np.arange(seq_len_int), 0)
        else:
            pos_raw = np.asarray(position)
            pos_t = np.expand_dims(pos_raw, 0) if pos_raw.ndim == 1 else pos_raw

        clipped_pos = np.clip(pos_t, 0, max_len - 1)
        one_hot = (
            np.reshape(clipped_pos, list(clipped_pos.shape) + [1]) == np.arange(max_len)
        ) * 1.0
        if self.lookup_style.upper() == "INDEX":
            flat_one_hot = np.reshape(one_hot, (-1, max_len))
            flat_embs = np.matmul(flat_one_hot, w_t)
            return np.reshape(flat_embs, list(clipped_pos.shape) + [d_emb])
        einsum_layer = instantiate_layer(self.einsum_tpl, EinsumOp(equation="btv,vd->btd"))
        return einsum_layer("btv,vd->btd", one_hot, w_t)


class FullSoftmax(BasePraxisLayer):
    """Full Softmax layer computing normalized probabilities or cross-entropy loss.

    Attributes:
        input_dims: Input hidden feature dimension size.
        num_classes: Number of output classes.
        soft_cap_logits: Optional tanh soft-cap threshold for logits.
        bi_tempered_loss_tpl: Optional Bi-Tempered loss template.
        label_smoothing_prob: Label smoothing probability.
        label_smoothing_apply_for_eval: Whether to apply label smoothing during evaluation.
        z_loss_weight: Auxiliary z-loss weight.
        bias_init: Optional bias weight initialization strategy.
        feed_forward_tpl: Optional feed-forward projection template.
        scale_before_logits: Optional scaling factor before logit computation.
        chunk_size: Optional chunk size for chunked softmax.
        ais_temperature: Optional temperature scaling.
        use_bias: Whether to add a learned bias vector to logits.
    """

    input_dims: int = 0
    num_classes: int = 0
    soft_cap_logits: float | None = 0.0
    bi_tempered_loss_tpl: LayerTpl = None
    label_smoothing_prob: float = 0.0
    label_smoothing_apply_for_eval: bool = True
    z_loss_weight: float = 0.0
    bias_init: WeightInit | None = None
    feed_forward_tpl: LayerTpl = None
    scale_before_logits: float | None = None
    chunk_size: int = 0
    ais_temperature: float | None = None
    use_bias: bool = True

    def __call__(
        self,
        inputs: TensorOrScalar,
        class_weights: TensorOrScalar | None = None,
        class_ids: TensorOrScalar | None = None,
        class_probabilities: TensorOrScalar | None = None,
        w: TensorOrScalar | None = None,
        b: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute softmax probabilities or cross-entropy loss if targets are supplied.

        Args:
            inputs: Logits or hidden state tensor of shape `[..., V]`.
            class_weights: Optional per-example loss weights.
            class_ids: Optional target integer class IDs.
            class_probabilities: Optional target class probability distributions.
            w: Optional explicit projection weight matrix of shape `[input_dims, num_classes]`.
            b: Optional explicit bias vector of shape `[num_classes]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Softmax probability distribution `Tensor`, or per-example cross-entropy loss `Tensor`.
        """
        logits = np.asarray(inputs)
        if self.scale_before_logits is not None:
            logits = logits * float(self.scale_before_logits)
        if w is not None or (self.num_classes > 0 and logits.shape[-1] != self.num_classes):
            in_d = self.input_dims if self.input_dims > 0 else logits.shape[-1]
            out_c = self.num_classes if self.num_classes > 0 else in_d
            if self.feed_forward_tpl is not None:
                ff = instantiate_layer(
                    self.feed_forward_tpl,
                    FeedForward(input_dims=in_d, output_dims=out_c, has_bias=self.use_bias),
                )
                logits = ff(logits)
            else:
                w_t = self.get_variable("w", (in_d, out_c), explicit=w)
                logits = np.einsum("...d,dc->...c", logits, w_t)
                if self.use_bias and b is not None:
                    logits = logits + np.asarray(b)
        elif self.use_bias and b is not None:
            logits = logits + np.asarray(b)

        if self.soft_cap_logits is not None and self.soft_cap_logits > 0.0:
            logits = self.soft_cap_logits * np.tanh(logits / self.soft_cap_logits)

        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_logits
        exp_x = np.exp(shifted)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        probs = exp_x / sum_exp

        if class_ids is not None or class_probabilities is not None:
            log_z = np.log(sum_exp)
            log_probs = shifted - log_z
            num_c = logits.shape[-1]
            if class_probabilities is not None:
                targets = np.asarray(class_probabilities)
            else:
                ids_t = np.asarray(class_ids)
                if ids_t.ndim == logits.ndim:
                    ids_t = ids_t[..., 0]
                targets = (np.reshape(ids_t, list(ids_t.shape) + [1]) == np.arange(num_c)) * 1.0
            if self.label_smoothing_prob > 0.0:
                off_val = self.label_smoothing_prob / float(num_c)
                on_val = 1.0 - self.label_smoothing_prob + off_val
                targets = targets * (on_val - off_val) + off_val
            per_example_loss = -np.sum(targets * log_probs, axis=-1)
            if self.z_loss_weight > 0.0:
                z_val = (max_logits + log_z)[..., 0]
                per_example_loss = per_example_loss + self.z_loss_weight * np.square(z_val)
            if class_weights is not None:
                w_t = np.asarray(class_weights)
                if w_t.ndim > per_example_loss.ndim:
                    w_t = w_t[..., 0]
                per_example_loss = per_example_loss * w_t
            return per_example_loss

        return probs


class SigmoidCrossEntropy(BasePraxisLayer):
    """Multi-label Binary Sigmoid Cross-Entropy loss and probability layer.

    Attributes:
        input_dims: Input hidden feature dimension size.
        num_classes: Number of output classes.
        soft_cap_logits: Optional tanh soft-cap threshold for logits.
        bias_init: Optional bias weight initialization strategy.
        feed_forward_tpl: Optional feed-forward projection template.
    """

    input_dims: int = 0
    num_classes: int = 0
    soft_cap_logits: float | None = 0.0
    bias_init: WeightInit | None = None
    feed_forward_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        labels: TensorOrScalar | None = None,
        class_weights: TensorOrScalar | None = None,
        class_ids: TensorOrScalar | None = None,
        class_probabilities: TensorOrScalar | None = None,
        w: TensorOrScalar | None = None,
        b: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute numerically stable sigmoid cross-entropy loss or sigmoid probabilities.

        Args:
            inputs: Logits tensor of shape `[..., C]`.
            labels: Optional binary target label tensor of shape `[..., C]`.
            class_weights: Optional per-example loss weights.
            class_ids: Optional integer target class IDs.
            class_probabilities: Optional target probability tensor (alias for `labels`).
            w: Optional explicit projection matrix of shape `[input_dims, num_classes]`.
            b: Optional explicit bias vector of shape `[num_classes]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Binary cross-entropy loss `Tensor` if targets are given, otherwise sigmoid probabilities.
        """
        x = np.asarray(inputs)
        if w is not None or (self.num_classes > 0 and x.shape[-1] != self.num_classes):
            in_d = self.input_dims if self.input_dims > 0 else x.shape[-1]
            out_c = self.num_classes if self.num_classes > 0 else in_d
            w_t = self.get_variable("w", (in_d, out_c), explicit=w)
            x = np.einsum("...d,dc->...c", x, w_t)
        if b is not None:
            x = x + np.asarray(b)
        if self.soft_cap_logits is not None and self.soft_cap_logits > 0.0:
            x = self.soft_cap_logits * np.tanh(x / self.soft_cap_logits)

        target = labels if labels is not None else class_probabilities
        if target is None and class_ids is not None:
            ids_t = np.asarray(class_ids)
            target = (np.reshape(ids_t, list(ids_t.shape) + [1]) == np.arange(x.shape[-1])) * 1.0

        if target is None:
            return 1.0 / (1.0 + np.exp(-x))

        y = np.asarray(target)
        loss = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
        if class_weights is not None:
            loss = loss * np.asarray(class_weights)
        return loss


class BiTemperedLoss(BasePraxisLayer):
    """Bi-Tempered Logistic Loss layer (Amid et al., 2019).

    Implements tempered logarithm `log_t` and exponential `exp_t` with Bregman divergence.

    Attributes:
        t1: Temperature 1 (`t1` in `log_t1` controlling boundedness for outliers).
        t2: Temperature 2 (`t2` in `exp_t2` controlling tail heaviness).
        label_smoothing: Label smoothing probability in `[0, 1)`.
        start_step: Training step to start applying bi-tempered temperatures.
        end_step: Training step to finish temperature schedule.
        num_iters: Number of fixed-point iterations for `t2` normalization.
    """

    t1: float = 1.0
    t2: float = 1.0
    label_smoothing: float = 0.0
    start_step: int = 0
    end_step: int = -1
    num_iters: int = 5

    @staticmethod
    def _log_t(u: Tensor, t: float) -> Tensor:
        """Compute the tempered logarithm `log_t(u)` for temperature `t`.

        Args:
            u: Positive input tensor.
            t: Temperature scalar.

        Returns:
            Tempered logarithm `Tensor`.
        """
        u_safe = np.maximum(u, 1e-12)
        if abs(t - 1.0) < 1e-7:
            return np.log(u_safe)
        return (u_safe ** (1.0 - t) - 1.0) / (1.0 - t)

    @staticmethod
    def _exp_t(u: Tensor, t: float) -> Tensor:
        """Compute the tempered exponential `exp_t(u)` for temperature `t`.

        Args:
            u: Input tensor.
            t: Temperature scalar.

        Returns:
            Tempered exponential `Tensor`.
        """
        if abs(t - 1.0) < 1e-7:
            return np.exp(u)
        return np.maximum(0.0, 1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    def _compute_normalization(self, activations: Tensor) -> Tensor:
        """Compute the normalization constant `z_tilde` for tempered softmax with `t2`.

        Args:
            activations: Logit activations tensor of shape `[..., C]`.

        Returns:
            Normalization tensor `z_tilde` of shape `[..., 1]`.
        """
        mu = np.max(activations, axis=-1, keepdims=True)
        if abs(self.t2 - 1.0) < 1e-7:
            return mu + np.log(np.sum(np.exp(activations - mu), axis=-1, keepdims=True))

        normalized_act_step_0 = activations - mu
        normalized_act = normalized_act_step_0
        for _ in range(max(1, self.num_iters)):
            logt_partition = np.sum(self._exp_t(normalized_act, self.t2), axis=-1, keepdims=True)
            normalized_act = normalized_act_step_0 * (logt_partition ** (1.0 - self.t2))
        logt_partition = np.sum(self._exp_t(normalized_act, self.t2), axis=-1, keepdims=True)
        return mu + self._log_t(1.0 / logt_partition, self.t2)

    def __call__(
        self,
        inputs: TensorOrScalar,
        labels: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute Bi-Tempered Logistic Loss between `inputs` (logits) and `labels`.

        Args:
            inputs: Logits tensor of shape `[..., C]`.
            labels: Optional target probability or one-hot tensor of shape `[..., C]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tempered probabilities if `labels` is None, or Bregman divergence loss `Tensor`.
        """
        logits = np.asarray(inputs)
        norm_const = self._compute_normalization(logits)
        probabilities = self._exp_t(logits - norm_const, self.t2)
        probabilities = probabilities / np.maximum(
            np.sum(probabilities, axis=-1, keepdims=True), 1e-12
        )

        if labels is None:
            return probabilities

        y = np.asarray(labels)
        if self.label_smoothing > 0.0:
            num_classes = float(logits.shape[-1])
            y = y * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        temp1 = y * (self._log_t(y, self.t1) - self._log_t(probabilities, self.t1)) - (
            y ** (2.0 - self.t1) - probabilities ** (2.0 - self.t1)
        ) / (2.0 - self.t1)
        return np.sum(temp1, axis=-1)


__all__ = [
    "BiTemperedLoss",
    "Embedding",
    "FullSoftmax",
    "GShardSharedEmbeddingSoftmax",
    "PositionalEmbedding",
    "PositionalEmbedding2D",
    "SharedEmbeddingSoftmax",
    "SigmoidCrossEntropy",
    "TrainablePositionalEmbedding",
]
