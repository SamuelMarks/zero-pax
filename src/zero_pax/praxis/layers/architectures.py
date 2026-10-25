"""Vision, Audio, and Classification architecture layers for zero_pax.praxis.layers."""

from __future__ import annotations

from collections.abc import Sequence

from zero_pax import _tensor_ops as _ops
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl, instantiate_layer
from zero_pax.praxis.layers.activations import Identity, LeakyReLU, ReLU, Swish, Tanh
from zero_pax.praxis.layers.attentions import DotProductAttention
from zero_pax.praxis.layers.convolutions import Conv2D, ConvBNAct, LightConv1D
from zero_pax.praxis.layers.embeddings import FullSoftmax, PositionalEmbedding2D
from zero_pax.praxis.layers.linear_pooling import (
    FeedForward,
    GlobalPooling,
    Linear,
    MLPBlock,
    Pooling,
)
from zero_pax.praxis.layers.normalizations import BatchNorm, LayerNorm
from zero_pax.praxis.layers.transformers import StackedTransformer, TransformerFeedForward
from zero_pax.types import Tensor, TensorOrScalar, WeightInit


class VanillaBlock(BasePraxisLayer):
    """VanillaNet convolutional building block (`ConvBNAct -> ConvBNAct` without skip connection).

    Attributes:
        input_dim: Input channel dimension `C_in`.
        output_dim: Output channel dimension `C_out`.
        conv_params: Optional convolution layer template.
        kernel_size: Spatial kernel size `K`.
        stride: Spatial stride `S`.
        negative_slope: Negative slope for LeakyReLU activation.
        activation_tpl: Activation layer or template.
        body: Optional sequence of sub-layer templates.
    """

    input_dim: int = 0
    output_dim: int = 0
    conv_params: LayerTpl = None
    kernel_size: int = 3
    stride: int = 1
    negative_slope: float = 0.4
    activation_tpl: LayerTpl = None
    body: Sequence[LayerTpl] | None = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the two-layer VanillaNet convolution block to 4D `inputs`.

        Args:
            inputs: 4D input image tensor of shape `[B, H, W, C_in]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            4D output tensor of shape `[B, H_out, W_out, C_out]`.
        """
        x = self._to_tensor(inputs)
        if self.body is not None:
            out = x
            for idx, sub_tpl in enumerate(self.body):
                sub = instantiate_layer(sub_tpl, Identity(), parent=self, name=f"body_{idx}")
                out = sub(out)
            return out

        c_in = self.input_dim if self.input_dim > 0 else x.shape[-1]
        c_out = self.output_dim if self.output_dim > 0 else c_in
        k = max(1, self.kernel_size)
        s = max(1, self.stride)

        eff_act: LayerTpl = (
            self.activation_tpl
            if self.activation_tpl is not None
            else (
                LeakyReLU(negative_slope=self.negative_slope) if self.negative_slope > 0 else ReLU()
            )
        )

        conv1 = self.get_sublayer(
            "conv1",
            self.conv_params,
            ConvBNAct(
                filter_shape=(k, k, c_in, c_out),
                filter_stride=(s, s),
                activation_tpl=eff_act,
                params_init=self.params_init,
            ),
        )
        conv2 = self.get_sublayer(
            "conv2",
            self.conv_params,
            ConvBNAct(
                filter_shape=(1, 1, c_out, c_out),
                filter_stride=(1, 1),
                activation_tpl=Identity(),
                params_init=self.params_init,
            ),
        )
        return conv2(conv1(x))


class VanillaNet(BasePraxisLayer):
    """Multi-stage VanillaNet convolutional network backbone.

    Attributes:
        conv_params: Optional convolution layer template.
        block_params: Optional template for stage blocks.
        strides: Per-stage spatial strides.
        channels: Per-stage channel widths.
        blocks: Number of `VanillaBlock` layers per stage.
        kernels: Per-stage spatial kernel sizes.
        entryflow_conv_params: Optional entry stem convolution template.
        entryflow_conv_kernel: Spatial kernel shape for the entry stem convolution.
        entryflow_conv_stride: Spatial stride for the entry stem convolution.
        output_spatial_pooling_params: Optional global pooling template at the network exit.
        negative_slope: Negative slope for LeakyReLU activation.
    """

    conv_params: LayerTpl = None
    block_params: LayerTpl = None
    strides: Sequence[int] = (1, 2, 2, 2)
    channels: Sequence[int] = (256, 512, 1024, 2048)
    blocks: Sequence[int] = (3, 4, 6, 3)
    kernels: Sequence[int] = (3, 3, 3, 3)
    entryflow_conv_params: LayerTpl = None
    entryflow_conv_kernel: Sequence[int] = (7, 7)
    entryflow_conv_stride: tuple[int, int] = (2, 2)
    output_spatial_pooling_params: LayerTpl = None
    negative_slope: float = 0.4

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply VanillaNet stages to 4D image `inputs`.

        Args:
            inputs: 4D input image tensor of shape `[B, H, W, C]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            4D feature map `Tensor`.
        """
        out = self._to_tensor(inputs)
        if self.entryflow_conv_params is not None:
            stem = self.get_sublayer("entryflow_conv", self.entryflow_conv_params, Identity())
            out = stem(out)

        for s_idx, (stride, ch, n_blk, k_sz) in enumerate(
            zip(self.strides, self.channels, self.blocks, self.kernels)
        ):
            for b_idx in range(max(1, n_blk)):
                blk_stride = stride if b_idx == 0 else 1
                default_blk = VanillaBlock(
                    input_dim=out.shape[-1],
                    output_dim=ch,
                    conv_params=self.conv_params,
                    kernel_size=k_sz,
                    stride=blk_stride,
                    negative_slope=self.negative_slope,
                    params_init=self.params_init,
                )
                stage_blk = instantiate_layer(
                    self.block_params,
                    default_blk,
                    parent=self,
                    name=f"stage_{s_idx}_block_{b_idx}",
                )
                out = stage_blk(out)

        if self.output_spatial_pooling_params is not None:
            pool = self.get_sublayer(
                "output_spatial_pooling",
                self.output_spatial_pooling_params,
                GlobalPooling(pooling_dims=(1, 2)),
            )
            out = pool(out)
        return out


class ResNetBlock(BasePraxisLayer):
    """Residual Network (ResNet) bottleneck block (`1x1 -> 3x3 -> 1x1` with shortcut projection).

    Attributes:
        input_dim: Input channel dimension `C_in`.
        output_dim: Output channel dimension (`4 * C_out_base` in bottleneck blocks).
        conv_params: Optional convolution layer template.
        kernel_size: Spatial kernel size of the middle convolution.
        stride: Spatial stride applied in the block and shortcut.
        activation_tpl: Activation layer or template (defaults to `ReLU`).
        norm_tpl: Normalization layer template (defaults to `BatchNorm`).
        residual_droppath_prob: Stochastic depth drop-path probability.
        zero_init_residual: Whether the final BatchNorm in the residual branch is zero-initialized.
        body: Optional sequence of sub-layer templates.
        shortcut: Optional shortcut projection layer template.
        post_activation: Optional post-addition activation layer template.
    """

    input_dim: int = 0
    output_dim: int = 0
    conv_params: LayerTpl = None
    kernel_size: int = 3
    stride: int = 1
    activation_tpl: LayerTpl = None
    norm_tpl: LayerTpl = None
    residual_droppath_prob: float = 0.0
    zero_init_residual: bool = False
    body: Sequence[LayerTpl] | None = None
    shortcut: LayerTpl = None
    post_activation: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the bottleneck ResNetBlock with residual shortcut to 4D `inputs`.

        Args:
            inputs: 4D input tensor of shape `[B, H, W, C_in]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            4D output `Tensor` of shape `[B, H_out, W_out, 4 * C_out]`.
        """
        x = self._to_tensor(inputs)
        if self.body is not None:
            out = x
            for idx, sub_tpl in enumerate(self.body):
                sub = instantiate_layer(sub_tpl, Identity(), parent=self, name=f"body_{idx}")
                out = sub(out)
            return out

        c_in = self.input_dim if self.input_dim > 0 else x.shape[-1]
        c_mid = self.output_dim if self.output_dim > 0 else max(1, c_in // 4)
        c_out = 4 * c_mid if self.output_dim > 0 else c_in
        k = max(1, self.kernel_size)
        s = max(1, self.stride)

        act = self.get_sublayer("act", self.activation_tpl, ReLU())

        conv1 = self.get_sublayer(
            "conv1",
            self.conv_params,
            Conv2D(
                filter_shape=(1, 1, c_in, c_mid),
                filter_stride=(1, 1),
                params_init=self.params_init,
            ),
        )
        bn1 = self.get_sublayer("bn1", self.norm_tpl, BatchNorm(dim=c_mid))
        h = act(bn1(conv1(x)))

        conv2 = self.get_sublayer(
            "conv2",
            self.conv_params,
            Conv2D(
                filter_shape=(k, k, c_mid, c_mid),
                filter_stride=(s, s),
                params_init=self.params_init,
            ),
        )
        bn2 = self.get_sublayer("bn2", self.norm_tpl, BatchNorm(dim=c_mid))
        h = act(bn2(conv2(h)))

        conv3 = self.get_sublayer(
            "conv3",
            self.conv_params,
            Conv2D(
                filter_shape=(1, 1, c_mid, c_out),
                filter_stride=(1, 1),
                params_init=self.params_init,
            ),
        )
        bn3_gamma_init = (
            WeightInit(method="constant", scale=-1.0) if self.zero_init_residual else None
        )
        bn3 = self.get_sublayer(
            "bn3",
            self.norm_tpl,
            BatchNorm(dim=c_out, gamma_init=bn3_gamma_init),
        )
        h = bn3(conv3(h))

        if c_in != c_out or s != 1:
            proj = self.get_sublayer(
                "shortcut",
                self.shortcut,
                Conv2D(
                    filter_shape=(1, 1, c_in, c_out),
                    filter_stride=(s, s),
                    params_init=self.params_init,
                ),
            )
            bn_sc = self.get_sublayer("bn_sc", self.norm_tpl, BatchNorm(dim=c_out))
            sc = bn_sc(proj(x))
        else:
            sc = x

        post_act = self.get_sublayer("post_act", self.post_activation, act)
        return post_act(h + sc)


class ResNet(BasePraxisLayer):
    """Deep Residual Network (ResNet) image backbone.

    Attributes:
        conv_params: Optional convolution layer template.
        block_params: Optional template for stage `ResNetBlock` layers.
        strides: Per-stage spatial strides.
        channels: Per-stage base bottleneck channel widths.
        blocks: Number of `ResNetBlock` layers per stage.
        kernels: Per-stage spatial kernel sizes.
        entryflow_conv_params: Optional stem convolution template.
        entryflow_conv_kernel: Spatial kernel shape for the stem convolution.
        entryflow_conv_stride: Spatial stride for the stem convolution.
        entry_max_pool: Whether to apply `3x3` max pooling after the entry stem.
        output_spatial_pooling_params: Optional global spatial pooling template at exit.
        return_block_features: Whether to return intermediate stage feature maps.
    """

    conv_params: LayerTpl = None
    block_params: LayerTpl = None
    strides: Sequence[int] = (1, 2, 2, 2)
    channels: Sequence[int] = (256, 512, 1024, 2048)
    blocks: Sequence[int] = (3, 4, 6, 3)
    kernels: Sequence[int] = (3, 3, 3, 3)
    entryflow_conv_params: LayerTpl = None
    entryflow_conv_kernel: Sequence[int] = (7, 7, 3)
    entryflow_conv_stride: tuple[int, int] = (2, 2)
    entry_max_pool: bool = True
    output_spatial_pooling_params: LayerTpl = None
    return_block_features: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the ResNet backbone to 4D image `inputs`.

        Args:
            inputs: 4D input image tensor of shape `[B, H, W, C]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            4D output feature map `Tensor`.
        """
        out = self._to_tensor(inputs)
        if self.entryflow_conv_params is not None:
            stem = self.get_sublayer("entryflow_conv", self.entryflow_conv_params, Identity())
            out = stem(out)
        if self.entry_max_pool and out.shape[1] >= 3 and out.shape[2] >= 3:
            pool_stem = Pooling(window_shape=(3, 3), window_stride=(2, 2), pooling_type="MAX")
            res_pool = pool_stem(out)
            out = res_pool[0] if isinstance(res_pool, tuple) else res_pool

        for s_idx, (stride, ch, n_blk, k_sz) in enumerate(
            zip(self.strides, self.channels, self.blocks, self.kernels)
        ):
            for b_idx in range(max(1, n_blk)):
                blk_stride = stride if b_idx == 0 else 1
                default_blk = ResNetBlock(
                    input_dim=out.shape[-1],
                    output_dim=ch,
                    conv_params=self.conv_params,
                    kernel_size=k_sz,
                    stride=blk_stride,
                    params_init=self.params_init,
                )
                stage_blk = instantiate_layer(
                    self.block_params,
                    default_blk,
                    parent=self,
                    name=f"stage_{s_idx}_block_{b_idx}",
                )
                out = stage_blk(out)

        if self.output_spatial_pooling_params is not None:
            pool_exit = self.get_sublayer(
                "output_spatial_pooling",
                self.output_spatial_pooling_params,
                GlobalPooling(pooling_dims=(1, 2)),
            )
            out = pool_exit(out)
        return out


class Conformer(BasePraxisLayer):
    """Conformer Macaron-style Speech Recognition block (Gulati et al., 2020).

    Executes: `x + 0.5 * FFN_start(x) -> SelfAttn -> LightConv1D -> + 0.5 * FFN_end -> FinalLN`.

    Attributes:
        input_dims: Model feature dimension `D`.
        model_dims: Alias for `input_dims`.
        kernel_size: Depthwise convolution kernel size in `LightConv1D`.
        ff_activation_tpl: Activation template for the Macaron FFN half-steps.
        ff_residual_weight: Residual weight for the Macaron FFN branches (`0.5` default).
        ffn_dim_multiplier: Expansion multiplier for the FFN hidden dimension (`4` default).
        atten_num_heads: Number of attention heads in the self-attention module.
        layer_order: Ordering of attention and convolution modules (`'mhsa_before_conv'`, `'conv_before_mhsa'`, `'mhsa'`, `'conv'`).
        dropout_prob: Dropout probability across sub-layers.
        conv_residual_dropout: Optional override for convolution residual dropout.
        atten_residual_dropout: Optional override for attention residual dropout.
        ffn_residual_dropout: Optional override for FFN residual dropout.
        atten_dropout: Optional override for attention weight dropout.
        ffn_relu_dropout: Optional override for FFN activation dropout.
        relu_dropout: Alias for `ffn_relu_dropout`.
        fflayer_start_tpl: Optional template for the initial half-step FFN.
        trans_atten_tpl: Optional template for the self-attention sub-layer.
        lconv_tpl: Optional template for the `LightConv1D` sub-layer.
        fflayer_end_tpl: Optional template for the final half-step FFN.
        fflayer_weight_sharing: Whether `fflayer_start_tpl` and `fflayer_end_tpl` share weights.
        final_ln_tpl: Optional template for the final `LayerNorm`.
        remat: Whether activation rematerialization is enabled.
    """

    input_dims: int | None = None
    model_dims: int = 512
    kernel_size: int = 32
    ff_activation_tpl: LayerTpl = None
    ff_residual_weight: float = 0.5
    ffn_dim_multiplier: int = 4
    atten_num_heads: int = 8
    layer_order: str = "mhsa_before_conv"
    dropout_prob: float | None = None
    conv_residual_dropout: float | None = None
    atten_residual_dropout: float | None = None
    ffn_residual_dropout: float | None = None
    atten_dropout: float | None = None
    ffn_relu_dropout: float | None = None
    relu_dropout: float | None = None
    fflayer_start_tpl: LayerTpl = None
    trans_atten_tpl: LayerTpl = None
    lconv_tpl: LayerTpl = None
    fflayer_end_tpl: LayerTpl = None
    fflayer_weight_sharing: bool = False
    final_ln_tpl: LayerTpl = None
    remat: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        atten_mask: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the Macaron Conformer block to 3D audio feature sequence `inputs`.

        Args:
            inputs: 3D input sequence tensor of shape `[B, T, D]`.
            paddings: Optional 2D padding mask of shape `[B, T]`.
            atten_mask: Optional additive attention mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            3D Conformer output `Tensor` of shape `[B, T, D]`.

        Raises:
            ValueError: If `self.layer_order` is not one of the supported order strings.
        """
        if self.layer_order not in ("mhsa_before_conv", "conv_before_mhsa", "mhsa", "conv"):
            raise ValueError(f"Unsupported Conformer layer_order: {self.layer_order!r}")

        x = self._to_tensor(inputs)
        d_model = (
            self.input_dims
            if (self.input_dims is not None and self.input_dims > 0)
            else (
                self.model_dims if (self.model_dims > 0 and self.model_dims != 512) else x.shape[-1]
            )
        )
        f_hid = max(1, self.ffn_dim_multiplier) * d_model
        n_heads = max(1, self.atten_num_heads)
        k_sz = max(1, self.kernel_size)

        act_tpl = self.ff_activation_tpl if self.ff_activation_tpl is not None else Swish()

        default_ff_start = TransformerFeedForward(
            input_dims=d_model,
            hidden_dims=f_hid,
            activation_tpl=act_tpl,
            residual_weight=self.ff_residual_weight,
            params_init=self.params_init,
        )
        ff_start = self.get_sublayer("fflayer_start", self.fflayer_start_tpl, default_ff_start)
        x = (
            ff_start(x, paddings=paddings)
            if isinstance(ff_start, TransformerFeedForward)
            else ff_start(x)
        )

        ln_attn = self.get_sublayer("ln_attn", None, LayerNorm(dim=d_model))
        default_attn = DotProductAttention(
            input_dim=d_model,
            num_heads=n_heads,
            dim_per_head=max(1, d_model // n_heads),
            params_init=self.params_init,
        )
        attn = self.get_sublayer("trans_atten", self.trans_atten_tpl, default_attn)
        default_lconv = LightConv1D(
            input_dims=d_model, kernel_size=k_sz, params_init=self.params_init
        )
        lconv = self.get_sublayer("lconv", self.lconv_tpl, default_lconv)

        if self.layer_order == "mhsa_before_conv":
            norm_x = ln_attn(x)
            a_out = (
                attn(norm_x, norm_x, norm_x, atten_mask=atten_mask)
                if isinstance(attn, DotProductAttention)
                else attn(norm_x)
            )
            x = x + a_out
            c_out = lconv(x, paddings=paddings) if isinstance(lconv, LightConv1D) else lconv(x)
            x = x + c_out
        elif self.layer_order == "conv_before_mhsa":
            c_out = lconv(x, paddings=paddings) if isinstance(lconv, LightConv1D) else lconv(x)
            x = x + c_out
            norm_x = ln_attn(x)
            a_out = (
                attn(norm_x, norm_x, norm_x, atten_mask=atten_mask)
                if isinstance(attn, DotProductAttention)
                else attn(norm_x)
            )
            x = x + a_out
        elif self.layer_order == "mhsa":
            norm_x = ln_attn(x)
            a_out = (
                attn(norm_x, norm_x, norm_x, atten_mask=atten_mask)
                if isinstance(attn, DotProductAttention)
                else attn(norm_x)
            )
            x = x + a_out
        else:
            c_out = lconv(x, paddings=paddings) if isinstance(lconv, LightConv1D) else lconv(x)
            x = x + c_out

        default_ff_end = TransformerFeedForward(
            input_dims=d_model,
            hidden_dims=f_hid,
            activation_tpl=act_tpl,
            residual_weight=self.ff_residual_weight,
            params_init=self.params_init,
        )
        ff_end = (
            ff_start
            if self.fflayer_weight_sharing
            else self.get_sublayer("fflayer_end", self.fflayer_end_tpl, default_ff_end)
        )
        x = (
            ff_end(x, paddings=paddings)
            if isinstance(ff_end, TransformerFeedForward)
            else ff_end(x)
        )

        final_ln = self.get_sublayer("final_ln", self.final_ln_tpl, LayerNorm(dim=d_model))
        out = final_ln(x)

        if paddings is not None:
            pad_t = _ops.asarray(paddings)
            out = out * (1.0 - _ops.expand_dims(pad_t, -1))
        return out


class VitEntryLayers(BasePraxisLayer):
    """Vision Transformer (ViT) patchify, linear patch projection, and 2D positional embedding entry block.

    Attributes:
        pos_emb_shapes: 2D grid shape `(grid_h, grid_w)` for positional embeddings.
        patch_size: Spatial patch size `P` (`P x P` pixels per patch).
        input_dims: Flattened patch input dimension (`P * P * C`).
        output_dims: Patch token embedding dimension `D`.
        pos_emb_dropout_prob: Dropout probability applied after adding positional embeddings.
        prepend_cls_tokens: Number of learnable CLS tokens prepended to the patch sequence.
        append_cls_tokens: Number of learnable CLS tokens appended to the patch sequence.
        pos_emb_tpl: Optional 2D positional embedding layer template.
        input_fc_has_bias: Whether the patch projection linear layer includes a bias.
    """

    pos_emb_shapes: tuple[int, int] = (0, 0)
    patch_size: int = 0
    input_dims: int = 0
    output_dims: int = 0
    pos_emb_dropout_prob: float = 0.0
    prepend_cls_tokens: int = 0
    append_cls_tokens: int = 0
    pos_emb_tpl: LayerTpl = None
    input_fc_has_bias: bool = True

    def __call__(
        self,
        inputs: TensorOrScalar,
        w_patch: TensorOrScalar | None = None,
        cls_prepend: TensorOrScalar | None = None,
        cls_append: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Convert 4D images `[B, H, W, C]` (or 3D patch tokens) into embedded tokens `[B, N, D]`.

        Args:
            inputs: 4D image tensor `[B, H, W, C]` or 3D token sequence `[B, N, D_in]`.
            w_patch: Optional explicit patch projection weight matrix.
            cls_prepend: Optional explicit prepended CLS token tensor of shape `[1, prepend_cls_tokens, D]`.
            cls_append: Optional explicit appended CLS token tensor of shape `[1, append_cls_tokens, D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Patch token sequence `Tensor` of shape `[B, N + prepend + append, D]`.
        """
        x = self._to_tensor(inputs)
        if x.ndim == 4:
            b_sz, h_img, w_img, c_img = x.shape
            p_sz = self.patch_size if self.patch_size > 0 else max(1, min(h_img, w_img) // 2)
            gh = max(1, h_img // p_sz)
            gw = max(1, w_img // p_sz)
            cropped = x[:, : gh * p_sz, : gw * p_sz, :]
            reshaped = _ops.reshape(cropped, (b_sz, gh, p_sz, gw, p_sz, c_img))
            transposed = _ops.transpose(reshaped, (0, 1, 3, 2, 4, 5))
            patches = _ops.reshape(transposed, (b_sz, gh * gw, p_sz * p_sz * c_img))
        else:
            patches = x
            b_sz, n_tok, _ = patches.shape
            gh = self.pos_emb_shapes[0] if self.pos_emb_shapes[0] > 0 else 1
            gw = self.pos_emb_shapes[1] if self.pos_emb_shapes[1] > 0 else n_tok

        in_d = patches.shape[-1]
        out_d = self.output_dims if self.output_dims > 0 else in_d
        lin = self.get_sublayer(
            "patch_projection",
            None,
            Linear(
                input_dims=in_d,
                output_dims=out_d,
                weight_init=WeightInit(method="ones", scale=1.0)
                if (self.params_init is None or self.params_init.method == "zeros")
                else self.params_init,
                params_init=self.params_init,
            ),
        )
        tokens = (
            lin(patches, w=w_patch)
            if isinstance(lin, Linear) and w_patch is not None
            else (patches if in_d == out_d and w_patch is None else lin(patches))
        )

        if self.prepend_cls_tokens > 0:
            prep_t = self.get_variable(
                "cls_prepend", (1, self.prepend_cls_tokens, out_d), explicit=cls_prepend
            )
            prep_batch = _ops.broadcast_to(prep_t, (b_sz, self.prepend_cls_tokens, out_d))
            tokens = _ops.concatenate([prep_batch, tokens], axis=1)

        if self.append_cls_tokens > 0:
            app_t = self.get_variable(
                "cls_append", (1, self.append_cls_tokens, out_d), explicit=cls_append
            )
            app_batch = _ops.broadcast_to(app_t, (b_sz, self.append_cls_tokens, out_d))
            tokens = _ops.concatenate([tokens, app_batch], axis=1)

        default_pe = PositionalEmbedding2D(
            h=gh,
            w=gw,
            embedding_dims=out_d,
            num_prepend_cls_tokens=self.prepend_cls_tokens,
            num_append_cls_tokens=self.append_cls_tokens,
        )
        pe_2d = self.get_sublayer("pos_emb", self.pos_emb_tpl, default_pe)
        pos = pe_2d(tokens)
        if pos.shape[1] == tokens.shape[1] and pos.shape[2] == tokens.shape[2]:
            tokens = tokens + pos
        return tokens


class VitExitLayers(BasePraxisLayer):
    """Vision Transformer (ViT) exit block performing LayerNorm, sequence pooling, and output projection.

    Attributes:
        hidden_dim: Input token feature dimension `D`.
        output_dim: Output representation or classification dimension `D_out`.
        pooled: Whether to pool the sequence `[B, N, D]` down to `[B, D_out]`.
        pre_ln: Whether to apply LayerNorm prior to pooling.
        output_fc_tanh: Whether to apply tanh activation after the output projection.
        pooling_tpl: Optional global sequence pooling template.
        ln_tpl: Optional pre-pooling `LayerNorm` template.
        output_dropout_prob: Dropout probability before final projection.
        pre_logits_Dims: Optional intermediate pre-logits MLP dimension.
        pre_logits_activation_tpl: Optional activation for the pre-logits projection.
        output_fc_has_bias: Whether the final output linear projection includes a bias.
    """

    hidden_dim: int = 0
    output_dim: int = 0
    pooled: bool = True
    pre_ln: bool = True
    output_fc_tanh: bool = True
    pooling_tpl: LayerTpl = None
    ln_tpl: LayerTpl = None
    output_dropout_prob: float = 0.0
    pre_logits_Dims: int = 0
    pre_logits_activation_tpl: LayerTpl = None
    output_fc_has_bias: bool = True

    def __call__(
        self,
        inputs: TensorOrScalar,
        w_out: TensorOrScalar | None = None,
        b_out: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply LayerNorm, optional sequence pooling, and output projection to ViT tokens.

        Args:
            inputs: 3D token sequence tensor of shape `[B, N, D]`.
            w_out: Optional explicit output projection matrix of shape `[D, D_out]`.
            b_out: Optional explicit output bias vector of shape `[D_out]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Pooled 2D `Tensor` `[B, D_out]` if `pooled` is True, otherwise 3D `[B, N, D_out]`.
        """
        x = self._to_tensor(inputs)
        d_in = self.hidden_dim if self.hidden_dim > 0 else x.shape[-1]
        d_out = self.output_dim if self.output_dim > 0 else d_in

        ln = self.get_sublayer("ln", self.ln_tpl, LayerNorm(dim=d_in))
        normed = ln(x) if self.pre_ln else x

        if self.pooled and normed.ndim >= 3:
            pool = self.get_sublayer(
                "pooling",
                self.pooling_tpl,
                GlobalPooling(pooling_type="MAX", pooling_dims=[1]),
            )
            feat = pool(normed)
        else:
            feat = normed

        if self.pre_logits_Dims > 0:
            pre_ff = self.get_sublayer(
                "pre_logits",
                None,
                FeedForward(
                    input_dims=feat.shape[-1],
                    output_dims=self.pre_logits_Dims,
                    activation_tpl=self.pre_logits_activation_tpl or Tanh(),
                    params_init=self.params_init,
                ),
            )
            feat = pre_ff(feat)

        if w_out is not None or feat.shape[-1] != d_out:
            lin = self.get_sublayer(
                "output_fc",
                None,
                Linear(
                    input_dims=feat.shape[-1],
                    output_dims=d_out,
                    weight_init=WeightInit(method="ones", scale=1.0)
                    if (self.params_init is None or self.params_init.method == "zeros")
                    else self.params_init,
                    params_init=self.params_init,
                ),
            )
            feat = lin(feat, w=w_out) if isinstance(lin, Linear) else lin(feat)
        if self.output_fc_has_bias and b_out is not None:
            feat = feat + _ops.asarray(b_out)
        if self.output_fc_tanh:
            feat = _ops.tanh(feat)
        return feat


class VisionTransformer(BasePraxisLayer):
    """Full Vision Transformer (ViT) architecture (Dosovitskiy et al., 2020).

    Combines `VitEntryLayers` -> `StackedTransformer` -> `VitExitLayers`.

    Attributes:
        entry_layers_tpl: Optional `VitEntryLayers` template.
        transformer_layers_tpl: Optional `StackedTransformer` template.
        exit_layers_tpl: Optional `VitExitLayers` template.
        full_data_parallel_on_entry_exit: Whether entry and exit layers use full data parallelism.
    """

    entry_layers_tpl: LayerTpl = None
    transformer_layers_tpl: LayerTpl = None
    exit_layers_tpl: LayerTpl = None
    full_data_parallel_on_entry_exit: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Run the full Vision Transformer pipeline on `inputs`.

        Args:
            inputs: 4D image tensor `[B, H, W, C]` or 3D token sequence `[B, N, D]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            ViT output representation `Tensor`.
        """
        entry = self.get_sublayer(
            "entry", self.entry_layers_tpl, VitEntryLayers(params_init=self.params_init)
        )
        tokens = entry(inputs)
        backbone = self.get_sublayer(
            "transformer",
            self.transformer_layers_tpl,
            StackedTransformer(
                num_layers=1, model_dims=tokens.shape[-1], params_init=self.params_init
            ),
        )
        encoded = backbone(tokens)
        exit_layer = self.get_sublayer(
            "exit",
            self.exit_layers_tpl,
            VitExitLayers(
                hidden_dim=encoded.shape[-1],
                output_dim=encoded.shape[-1],
                pooled=(_ops.asarray(inputs).ndim == 4),
                params_init=self.params_init,
            ),
        )
        return exit_layer(encoded)


class ClassificationModel(BasePraxisLayer):
    """Image/Sequence Classification task model combining a feature network and a Softmax head.

    Attributes:
        network_tpl: Backbone feature extraction network template (e.g., `ResNet`, `VisionTransformer`).
        softmax_tpl: Classification head `FullSoftmax` template.
        input_field: Input batch dictionary key for features.
        label_field: Input batch dictionary key for labels.
        label_smoothing_prob: Cross-entropy label smoothing probability.
    """

    network_tpl: LayerTpl = None
    softmax_tpl: LayerTpl = None
    input_field: str = "image"
    label_field: str = "label_probs"
    label_smoothing_prob: float = 0.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        labels: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Extract features via `network_tpl` and compute probabilities/loss via `softmax_tpl`.

        Args:
            inputs: Input feature tensor.
            labels: Optional target class probabilities or one-hot labels.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Class probabilities or per-example loss `Tensor`.
        """
        net = self.get_sublayer("network", self.network_tpl, Identity())
        feats = net(inputs)
        sm = self.get_sublayer(
            "softmax",
            self.softmax_tpl,
            FullSoftmax(
                label_smoothing_prob=self.label_smoothing_prob, params_init=self.params_init
            ),
        )
        if isinstance(sm, FullSoftmax):
            return sm(feats, class_probabilities=labels)
        return sm(feats)


class ClassificationMLPModel(ClassificationModel):
    """Classification task model with an intermediate `MLPBlock` before the Softmax head.

    Attributes:
        mlp_tpl: Intermediate `MLPBlock` or `FeedForward` template.
    """

    mlp_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        labels: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply `network_tpl` -> `mlp_tpl` -> `softmax_tpl` to `inputs`.

        Args:
            inputs: Input feature tensor.
            labels: Optional target class probabilities or one-hot labels.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Class probabilities or per-example loss `Tensor`.
        """
        net = self.get_sublayer("network", self.network_tpl, Identity())
        feats = net(inputs)
        default_mlp = MLPBlock(
            num_layers=1,
            hidden_dims=feats.shape[-1],
            params_init=(
                WeightInit(method="ones", scale=1.0)
                if (self.params_init is None or self.params_init.method == "zeros")
                else self.params_init
            ),
        )
        mlp = self.get_sublayer("mlp", self.mlp_tpl, default_mlp)
        feats = mlp(feats)
        sm = self.get_sublayer(
            "softmax",
            self.softmax_tpl,
            FullSoftmax(
                label_smoothing_prob=self.label_smoothing_prob, params_init=self.params_init
            ),
        )
        if isinstance(sm, FullSoftmax):
            return sm(feats, class_probabilities=labels)
        return sm(feats)


__all__ = [
    "ClassificationMLPModel",
    "ClassificationModel",
    "Conformer",
    "ResNet",
    "ResNetBlock",
    "VanillaBlock",
    "VanillaNet",
    "VisionTransformer",
    "VitEntryLayers",
    "VitExitLayers",
]
