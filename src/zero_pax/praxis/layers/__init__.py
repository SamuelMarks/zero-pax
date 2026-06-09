"""
Praxis layers module.

This module provides the implementation of the praxis layers API.
"""

from typing import Any, Sequence, Optional, Callable
from pydantic import BaseModel, ConfigDict
import numpy as np

numpy = np


# Dummy types and modules for strong typing
class DummyMeta(type):
    """Metaclass for dummy types."""

    def __getattr__(cls, name):
        """Get attribute."""
        if name.startswith("__"):
            raise AttributeError(name)
        return cls

    def __getitem__(cls, item):
        """Get item."""
        return cls


class DummyType(metaclass=DummyMeta):
    """Dummy type class."""

    pass


jnp = DummyType
base_ops = DummyType
pax_fiddle = DummyType
normalizations = DummyType
activations = DummyType
activations_lib = DummyType
transformer_models = DummyType
embedding_softmax = DummyType
transformers = DummyType
convolutions = DummyType
linears = DummyType
resnets = DummyType
stochastics = DummyType
attentions = DummyType
poolings = DummyType
ssm = DummyType
repeats = DummyType
dataclasses = DummyType

LayerTpl = Any
WeightInit = Any
ActivationType = Any
DecoderHParams = Any
LanguageModelType = Any
SplitDimsMapping = Any
PaxConfig = Any
BaseLayer = Any


class AdaptedTransformerFeedForward(BaseModel):
    """This layer is a wrapper designed for MultitaskResidualAdapter.

    Args:
        adapter_tpl (LayerTpl): Description. Default: 'template_field(None)'.
        mode (str): Description. Default: 'sequential'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    adapter_tpl: LayerTpl = "template_field(None)"
    mode: str = "sequential"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for AdaptedTransformerFeedForward.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class AttentionProjection(BaseModel):
    """Layer that computes multi heads projection.

    Args:
        input_dim (int): Description. Default: 0.
        num_heads (int): Description. Default: 0.
        dim_per_head (int): Description. Default: 0.
        is_output_projection (bool): Description. Default: False.
        use_bias (bool): Description. Default: True.
        attention_combine_dims (bool): Description. Default: False.
        use_nhd_shape (bool): Description. Default: False.
        explicit_fan_in_fan_out_axes (bool): Description. Default: False.
        einsum_tpl (LayerTpl): Description. Default: 'template_field(base_ops.EinsumOp)'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dim: int = 0
    num_heads: int = 0
    dim_per_head: int = 0
    is_output_projection: bool = False
    use_bias: bool = True
    attention_combine_dims: bool = False
    use_nhd_shape: bool = False
    explicit_fan_in_fan_out_axes: bool = False
    einsum_tpl: LayerTpl = "template_field(base_ops.EinsumOp)"


class AutodiffCheckpointType(BaseModel):
    """jax.checkpoint policy types.

    Args:
        SAVE_NOTHING (str): Description. Default: 'save_nothing'.
        SAVE_UNET_ALL_CONV (str): Description. Default: 'save_unet_all_conv'.
        SAVE_UNET_CONV (str): Description. Default: 'save_unet_conv'.
        SAVE_EVERYTHING (str): Description. Default: 'save_everything'.
        SAVE_QKV_OUT_PROJ (str): Description. Default: 'save_qkv_out_proj'.
        SAVE_OUT_PROJ (str): Description. Default: 'save_out_proj'.
        SAVE_CONTEXT (str): Description. Default: 'save_context'.
        SAVE_CONTEXT_AND_OUT_PROJ (str): Description. Default: 'save_encoded_and_out_proj'.
        SAVE_DOT_ONLY (str): Description. Default: 'save_dot_only'.
        SAVE_DOT_WITH_NO_BATCH_DIM (str): Description. Default: 'save_dot_with_no_batch_dims'.
        SAVE_DOT_FOR_MLPERF_200B (str): Description. Default: 'save_dot_for_mlperf_200b'.
        SAVE_ITERATION_INPUT (str): Description. Default: 'save_iteration_input'.
        SAVE_TRANSFORMER_LAYER_OUTPUT (str): Description. Default: 'save_transformer_layer_output'.
        SAVE_QUANTIZED (str): Description. Default: 'save_quantized'.
        SAVE_QKV_OUT_PROJ_SEPARATE (str): Description. Default: 'save_qkv_out_proj_separate'.
        SAVE_DOT_EXCEPT_LOGITS_FFN1 (str): Description. Default: 'save_dot_except_logits_ffn1'.
        SAVE_DOT_EXCEPT_LOGITS (str): Description. Default: 'save_dot_except_logits'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    SAVE_NOTHING: str = "save_nothing"
    SAVE_UNET_ALL_CONV: str = "save_unet_all_conv"
    SAVE_UNET_CONV: str = "save_unet_conv"
    SAVE_EVERYTHING: str = "save_everything"
    SAVE_QKV_OUT_PROJ: str = "save_qkv_out_proj"
    SAVE_OUT_PROJ: str = "save_out_proj"
    SAVE_CONTEXT: str = "save_context"
    SAVE_CONTEXT_AND_OUT_PROJ: str = "save_encoded_and_out_proj"
    SAVE_DOT_ONLY: str = "save_dot_only"
    SAVE_DOT_WITH_NO_BATCH_DIM: str = "save_dot_with_no_batch_dims"
    SAVE_DOT_FOR_MLPERF_200B: str = "save_dot_for_mlperf_200b"
    SAVE_ITERATION_INPUT: str = "save_iteration_input"
    SAVE_TRANSFORMER_LAYER_OUTPUT: str = "save_transformer_layer_output"
    SAVE_QUANTIZED: str = "save_quantized"
    SAVE_QKV_OUT_PROJ_SEPARATE: str = "save_qkv_out_proj_separate"
    SAVE_DOT_EXCEPT_LOGITS_FFN1: str = "save_dot_except_logits_ffn1"
    SAVE_DOT_EXCEPT_LOGITS: str = "save_dot_except_logits"


class BaseActivation(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for activation.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class BaseNormalization(BaseModel):
    """Base class for normalization layers.

    Args:
        dim (int): Description. Default: 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dim: int = 0


class BatchNorm(BaseModel):
    """Batch normalization layer.

    Args:
        decay (float): Description. Default: 0.999.
        use_moving_avg_in_training (bool): Description. Default: False.
        set_padded_output_to_zero (bool): Description. Default: True.
        force_eval_mode (bool): Description. Default: False.
        gamma_init (WeightInit): Description. Default: 'dataclasses.field(default_factory=lambda : WeightInit.Constant(0.0))'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    decay: float = 0.999
    use_moving_avg_in_training: bool = False
    set_padded_output_to_zero: bool = True
    force_eval_mode: bool = False
    gamma_init: WeightInit = (
        "dataclasses.field(default_factory=lambda : WeightInit.Constant(0.0))"
    )


class BertModel(BaseModel):
    """None='```(None)```', mask_token_id: int=0, force_mask_generation: bool=False)`

    Args:
        lm_tpl (LayerTpl): Description. Default: 'template_field(transformer_models.TransformerLm)'.
        label_smoothing_prob (float): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    lm_tpl: LayerTpl = "template_field(transformer_models.TransformerLm)"
    label_smoothing_prob: Optional[float] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for BertModel.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class BiTemperedLoss(BaseModel):
    """Bi-tempered logitstic loss.

    Args:
        t1 (float): Description. Default: 1.0.
        t2 (float): Description. Default: 1.0.
        label_smoothing (float): Description. Default: 0.0.
        start_step (int): Description. Default: 0.
        end_step (int): Description. Default: 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    t1: float = 1.0
    t2: float = 1.0
    label_smoothing: float = 0.0
    start_step: int = 0
    end_step: int = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for BiTemperedLoss.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class Bias(BaseModel):
    """None=0.0)`

    Args:
        dims (int): Description. Default: 0.
        bias_init (float): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dims: int = 0
    bias_init: Optional[float] = None


class BregmanPCA(BaseModel):
    """Sequence[int]=0, activation_type: ActivationType='ActivationType', negative_slope: float=0.0, mean_beta: float=0.99, coefficients_lr: float=0.01, coefficients_beta: float=0.9, coefficients_steps: int=20, components_lr: float=0.01, components_beta: float=0.9, start_step: int=0, end_step: int=0, constant_lr_schedule: bool=True)`

    Args:
        num_components (int): Description. Default: 0.
        input_dims (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    num_components: int = 0
    input_dims: Optional[int] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for BregmanPCA.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class CausalDepthwiseConv1D(BaseModel):
    """Sequence[int]=0)`

    Args:
        kernel_size (int): Description. Default: 3.
        hidden_dims (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    kernel_size: int = 3
    hidden_dims: Optional[int] = None


class CifgLstmCellSimple(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for CifgLstmCellSimple.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class ClassificationMLPModel(BaseModel):
    """Language Model task with a simple MLP model.

    Args:
        mlp_tpl (LayerTpl): Description. Default: 'template_field(linears.MLPBlock)'.
        softmax_tpl (LayerTpl): Description. Default: 'template_field(embedding_softmax.SharedEmbeddingSoftmax)'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    mlp_tpl: LayerTpl = "template_field(linears.MLPBlock)"
    softmax_tpl: LayerTpl = "template_field(embedding_softmax.SharedEmbeddingSoftmax)"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for ClassificationMLPModel.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class ClassificationModel(BaseModel):
    """Classification task for images and video.

    Args:
        network_tpl (LayerTpl): Description. Default: 'template_field(resnets.ResNet)'.
        softmax_tpl (LayerTpl): Description. Default: 'template_field(embedding_softmax.FullSoftmax)'.
        input_field (str): Description. Default: 'image'.
        label_field (str): Description. Default: 'label_probs'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    network_tpl: LayerTpl = "template_field(resnets.ResNet)"
    softmax_tpl: LayerTpl = "template_field(embedding_softmax.FullSoftmax)"
    input_field: str = "image"
    label_field: str = "label_probs"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for ClassificationModel.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class Conformer(BaseModel):
    """None='```(None)```', model_dims: int=512, kernel_size: int=32, ff_activation_tpl: pax_fiddle.Config[activations.BaseActivation]='template_field(activations.Swish)', ff_residual_weight: float=0.5, ffn_dim_multiplier: int=4, atten_num_heads: int=8, layer_order: str='mhsa_before_conv', dropout_prob: float

    Args:
        input_dims (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: Optional[int] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for Conformer.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class Conv2D(BaseModel):
    """None='```(None)```', padding: str='SAME', tf_equivalent_padding: bool=False, is_causal: bool=False, weight_norm_tpl: pax_fiddle.Config[normalizations.BaseNormalization]='template_field(normalizations.IdentityNorm)')`

    Args:
        filter_shape (Sequence[int]): Description. Default: '(0, 0, 0, 0)'.
        filter_stride (Sequence[int]): Description. Default: '(0, 0)'.
        dilations (Sequence[int]): Description. Default: '(1, 1)'.
        bias (bool): Description. Default: False.
        bias_init (WeightInit): Description. Default: 'dataclasses.field(default_factory=lambda : WeightInit.Constant(0.0))'.
        kernel_init (WeightInit): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    filter_shape: Sequence[int] = "(0, 0, 0, 0)"
    filter_stride: Sequence[int] = "(0, 0)"
    dilations: Sequence[int] = "(1, 1)"
    bias: bool = False
    bias_init: WeightInit = (
        "dataclasses.field(default_factory=lambda : WeightInit.Constant(0.0))"
    )
    kernel_init: Optional[WeightInit] = None


class ConvBNAct(BaseModel):
    """None='template_field(normalizations.BatchNorm)', activation_tpl: pax_fiddle.Config[activations.BaseActivation]='template_field(activations.ReLU)')`

    Args:
        batch_norm_tpl (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    batch_norm_tpl: Optional[LayerTpl] = None


class ConvBNActWithPadding(BaseModel):
    """A block of conv-bn-activation layers with padding processing.

    Args:
        compat_with_lingvo (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    compat_with_lingvo: bool = False


class CubedReLU(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for CubedReLU.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.power(np.maximum(x, 0.0), 3)


class DepthwiseConv1D(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None


class DotProductAttention(BaseModel):
    """dict[str, int]=0, hidden_dim: int=0, num_heads: int=1, dim_per_head: int

    Args:
        input_dim (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dim: Optional[int] = None


class DotProductAttentionWithContext(BaseModel):
    """None='```(None)```', right_context: int

    Args:
        left_context (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    left_context: Optional[int] = None


class DotProductAttentionWithContextXL(BaseModel):
    """None='```(None)```', right_context: int

    Args:
        left_context (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    left_context: Optional[int] = None


class DotProductAttentionXL(BaseModel):
    """Transformer-XL multiheaded attention with relative positional embedding.

    Args:
        rel_pos_emb_dim (int): Description. Default: 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    rel_pos_emb_dim: int = 0


class Dropout(BaseModel):
    """None='```(None)```', noise_shape_broadcast_dims: Sequence[int]

    Args:
        keep_prob (float): Description. Default: 1.0.
        noise_shape (Sequence[int]): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    keep_prob: float = 1.0
    noise_shape: Optional[Sequence[int]] = None


class ELU(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for ELU.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.where(x > 0, x, 1.0 * (np.exp(x) - 1))


class Einsum(BaseModel):
    """Layer that computes an einsum and maybe a bias.

    Args:
        eqn (str): Description. Default: ''.
        w_shape (Sequence[int]): Description. Default: ().
        use_bias (bool): Description. Default: False.
        einsum_op_tpl (LayerTpl): Description. Default: 'template_field(base_ops.EinsumOp)'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    eqn: str = ""
    w_shape: Sequence[int] = ()
    use_bias: bool = False
    einsum_op_tpl: LayerTpl = "template_field(base_ops.EinsumOp)"


class EinsumOp(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None


class Embedding(BaseModel):
    """A simple embedding layer that performs embedding lookups from ids.

    Args:
        num_classes (int): Description. Default: 0.
        input_dims (int): Description. Default: 0.
        lookup_style (str): Description. Default: 'index'.
        scale_sqrt_depth (bool): Description. Default: False.
        set_nan_for_oob_id (bool): Description. Default: False.
        array_lookup (base_ops.ArrayLookup): Description. Default: 'instance_field(base_ops.ArrayLookup)'.
        einsum (base_ops.EinsumOp): Description. Default: 'instance_field(base_ops.EinsumOp)'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    num_classes: int = 0
    input_dims: int = 0
    lookup_style: str = "index"
    scale_sqrt_depth: bool = False
    set_nan_for_oob_id: bool = False
    array_lookup: base_ops.ArrayLookup = "instance_field(base_ops.ArrayLookup)"
    einsum: base_ops.EinsumOp = "instance_field(base_ops.EinsumOp)"


class FRnn(BaseModel):
    """None='base_layer.template_field(None)', reverse: bool=False, unroll: int=1)`

    Args:
        cell_tpl (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    cell_tpl: Optional[LayerTpl] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for FRnn.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class FeedForward(BaseModel):
    """None='```(None)```', bias_init: float

    Args:
        input_dims (int): Description. Default: 0.
        output_dims (int): Description. Default: 0.
        has_bias (bool): Description. Default: True.
        linear_tpl (LayerTpl): Description. Default: 'template_field(Linear)'.
        bias_tpl (LayerTpl): Description. Default: 'template_field(Bias)'.
        activation_tpl (pax_fiddle.Config[activations.BaseActivation]): Description. Default: 'template_field(activations.ReLU)'.
        weight_init (WeightInit): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: int = 0
    output_dims: int = 0
    has_bias: bool = True
    linear_tpl: LayerTpl = "template_field(Linear)"
    bias_tpl: LayerTpl = "template_field(Bias)"
    activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
        "template_field(activations.ReLU)"
    )
    weight_init: Optional[WeightInit] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for FeedForward.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class FullSoftmax(BaseModel):
    """None=0.0, bi_tempered_loss_tpl: LayerTpl

    Args:
        input_dims (int): Description. Default: 0.
        num_classes (int): Description. Default: 0.
        soft_cap_logits (float): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: int = 0
    num_classes: int = 0
    soft_cap_logits: Optional[float] = None


class GELU(BaseModel):
    """Gaussian Error Linear Unit (GELU) activation layer.

    Args:
        approximate (bool): Description. Default: True.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    approximate: bool = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for GELU.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return (
            0.5
            * x
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        )


class GShardSharedEmbeddingSoftmax(BaseModel):
    """None=0.0, logits_abs_max: float

    Args:
        input_dims (int): Description. Default: 0.
        num_classes (int): Description. Default: 0.
        use_tgt_labels_size_as_loss_denominator (bool): Description. Default: True.
        soft_cap_logits (float): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: int = 0
    num_classes: int = 0
    use_tgt_labels_size_as_loss_denominator: bool = True
    soft_cap_logits: Optional[float] = None


class GlobalPooling(BaseModel):
    """None='```(None)```', keepdims: bool=False)`

    Args:
        pooling_type (str): Description. Default: 'AVG'.
        pooling_dims (Sequence[int]): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    pooling_type: str = "AVG"
    pooling_dims: Optional[Sequence[int]] = None


class GroupNorm(BaseModel):
    """None='```(None)```', epsilon: float=0.001, set_padded_output_to_zero: bool=True, use_scale: bool=True, use_bias: bool=True)`

    Args:
        num_groups (int): Description. Default: 32.
        min_group_size (int): Description. Default: 1.
        cumulative (bool): Description. Default: False.
        input_rank (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    num_groups: int = 32
    min_group_size: int = 1
    cumulative: bool = False
    input_rank: Optional[int] = None


class GroupedQueryAttention(BaseModel):
    """None='```(None)```')`

    Args:
        input_dim (int): Description. Default: 0.
        hidden_dim (int): Description. Default: 0.
        num_heads (int): Description. Default: 1.
        num_kv_heads (int): Description. Default: 1.
        dim_per_head (int): Description. Default: 0.
        atten_dropout_prob (float): Description. Default: 0.0.
        atten_temp (float): Description. Default: 1.0.
        use_bias (bool): Description. Default: True.
        atten_logit_cap (float): Description. Default: 0.0.
        rope_min_max_timescales (tuple[int, int]): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dim: int = 0
    hidden_dim: int = 0
    num_heads: int = 1
    num_kv_heads: int = 1
    dim_per_head: int = 0
    atten_dropout_prob: float = 0.0
    atten_temp: float = 1.0
    use_bias: bool = True
    atten_logit_cap: float = 0.0
    rope_min_max_timescales: Optional[tuple[int, int]] = None


class Identity(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None


class IdentityNorm(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None


class LanguageModel(BaseModel):
    """Language Model base task.

    Args:
        lm_tpl (LayerTpl): Description. Default: 'template_field(transformer_models.TransformerLm)'.
        return_predictions (bool): Description. Default: False.
        decoder_tpl (DecoderHParams): Description. Default: 'base_layer.instance_field(GreedyDecoderHParams)'.
        model_type (LanguageModelType): Description. Default: 'LanguageModelType'.
        count_tokens (bool): Description. Default: False.
        apply_eval_sample_weights (bool): Description. Default: False.
        report_strict_acc (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    lm_tpl: LayerTpl = "template_field(transformer_models.TransformerLm)"
    return_predictions: bool = False
    decoder_tpl: DecoderHParams = "base_layer.instance_field(GreedyDecoderHParams)"
    model_type: LanguageModelType = "LanguageModelType"
    count_tokens: bool = False
    apply_eval_sample_weights: bool = False
    report_strict_acc: bool = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for LanguageModel.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class LanguageModelContinuousBatching(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for LanguageModelContinuousBatching.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class LanguageModelDPO(BaseModel):
    """Contains a pair of TransformerLM for direct preference optimization.

    Args:
        ref_mdl (transformer_models.TransformerLm): Description. Default: 'pax_fiddle.instance_field(transformer_models.TransformerLm)'.
        mdl (transformer_models.TransformerLm): Description. Default: 'pax_fiddle.instance_field(transformer_models.TransformerLm)'.
        beta (float): Description. Default: 0.1.
        token_counter (embedding_softmax.TokenCounter): Description. Default: 'pax_fiddle.instance_field(embedding_softmax.TokenCounter)'.
        apply_eval_sample_weights (bool): Description. Default: False.
        model_type (LanguageModelType): Description. Default: 'LanguageModelType'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    ref_mdl: transformer_models.TransformerLm = (
        "pax_fiddle.instance_field(transformer_models.TransformerLm)"
    )
    mdl: transformer_models.TransformerLm = (
        "pax_fiddle.instance_field(transformer_models.TransformerLm)"
    )
    beta: float = 0.1
    token_counter: embedding_softmax.TokenCounter = (
        "pax_fiddle.instance_field(embedding_softmax.TokenCounter)"
    )
    apply_eval_sample_weights: bool = False
    model_type: LanguageModelType = "LanguageModelType"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for LanguageModelDPO.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class LanguageModelType(BaseModel):
    """The different language model types based on the tokens visibility.

    Args:
        CAUSAL (str): Description. Default: 'causal'.
        PREFIX (str): Description. Default: 'prefix'.
        BIDIRECTIONAL (str): Description. Default: 'bidirectional'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    CAUSAL: str = "causal"
    PREFIX: str = "prefix"
    BIDIRECTIONAL: str = "bidirectional"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for LanguageModelType.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class LayerNorm(BaseModel):
    """Layer normalization.

    Args:
        direct_scale (bool): Description. Default: False.
        epsilon (float): Description. Default: 1e-06.
        use_scale (bool): Description. Default: True.
        use_bias (bool): Description. Default: True.
        reductions_in_fp32 (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    direct_scale: bool = False
    epsilon: float = 1e-06
    use_scale: bool = True
    use_bias: bool = True
    reductions_in_fp32: bool = False


class LayerNormalizedLstmCellSimple(BaseModel):
    """An implementation of layer normalized LSTM based on LSTMCellSimple.

    Args:
        layer_norm_epsilon (float): Description. Default: 1e-08.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    layer_norm_epsilon: float = 1e-08


class LayerwiseShardablePipelined(BaseModel):
    """None='base_layer.template_field(None)', num_microbatches: int

    Args:
        num_stages (int): Description. Default: 1.
        single_stage_body (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    num_stages: int = 1
    single_stage_body: Optional[LayerTpl] = None


class LeakyReLU(BaseModel):
    """Leaky ReLU activation layer.

    Args:
        negative_slope (float): Description. Default: 0.01.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    negative_slope: float = 0.01

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for LeakyReLU.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.where(x > 0, x, self.negative_slope * x)


class LightConv1D(BaseModel):
    """None='```(None)```', kernel_size: int

    Args:
        input_dims (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: Optional[int] = None


class Linear(BaseModel):
    """None='```(None)```', einsum_tpl: LayerTpl='template_field(base_ops.EinsumOp)')`

    Args:
        input_dims (int): Description. Default: 0.
        output_dims (int): Description. Default: 0.
        weight_init (WeightInit): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: int = 0
    output_dims: int = 0
    weight_init: Optional[WeightInit] = None


class LocalSelfAttention(BaseModel):
    """None='```(None)```', left_context: int

    Args:
        block_size (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    block_size: Optional[int] = None


class LocalSelfAttentionAlibi(BaseModel):
    """None=None, params_init: praxis.base_layer.WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (numpy.dtype): Description. Default: ....
        fprop_dtype (numpy.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[numpy.dtype] = None
    fprop_dtype: Optional[numpy.dtype] = None


class LocalSelfAttentionRelativeBias(BaseModel):
    """None=None, params_init: praxis.base_layer.WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (numpy.dtype): Description. Default: ....
        fprop_dtype (numpy.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[numpy.dtype] = None
    fprop_dtype: Optional[numpy.dtype] = None


class LocalSelfAttentionXL(BaseModel):
    """Local version of transformer-xl self attention.

    Args:
        rel_pos_emb_dim (int): Description. Default: 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    rel_pos_emb_dim: int = 0


class LstmCellSimple(BaseModel):
    """Simple LSTM cell.

    Args:
        inputs_arity (int): Description. Default: 1.
        num_input_nodes (int): Description. Default: 0.
        num_output_nodes (int): Description. Default: 0.
        num_hidden_nodes (int): Description. Default: 0.
        reset_cell_state (bool): Description. Default: False.
        cell_value_cap (float): Description. Default: 10.0.
        forget_gate_bias (float): Description. Default: 0.0.
        output_nonlinearity (bool): Description. Default: True.
        zo_prob (float): Description. Default: 0.0.
        bias_init (WeightInit): Description. Default: 'dataclasses.field(default_factory=lambda : WeightInit.Constant(0.0))'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    inputs_arity: int = 1
    num_input_nodes: int = 0
    num_output_nodes: int = 0
    num_hidden_nodes: int = 0
    reset_cell_state: bool = False
    cell_value_cap: float = 10.0
    forget_gate_bias: float = 0.0
    output_nonlinearity: bool = True
    zo_prob: float = 0.0
    bias_init: WeightInit = (
        "dataclasses.field(default_factory=lambda : WeightInit.Constant(0.0))"
    )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for LstmCellSimple.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class LstmFrnn(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for LstmFrnn.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class MLPBlock(BaseModel):
    """Multilayer perceptron block composed of multiple FeedForward layers.

    Args:
        num_layers (int): Description. Default: 3.
        hidden_dims (int): Description. Default: 128.
        activate_final (bool): Description. Default: True.
        ff_tpl (LayerTpl): Description. Default: 'template_field(FeedForward)'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    num_layers: int = 3
    hidden_dims: int = 128
    activate_final: bool = True
    ff_tpl: LayerTpl = "template_field(FeedForward)"


class MaskedLmDataAugmenter(BaseModel):
    """Performs data augmentation according to the BERT paper.

    Args:
        vocab_size (int): Description. Default: 0.
        mask_prob (float): Description. Default: 0.12.
        random_prob (float): Description. Default: 0.015.
        same_prob (float): Description. Default: 0.015.
        mask_token_id (int): Description. Default: -1.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    vocab_size: int = 0
    mask_prob: float = 0.12
    random_prob: float = 0.015
    same_prob: float = 0.015
    mask_token_id: int = -1


class MultitaskResidualAdapter(BaseModel):
    """None='template_field(normalizations.LayerNorm)', activation_tpl: pax_fiddle.Config[activations.BaseActivation]='template_field(activations.ReLU)')`

    Args:
        input_dims (int): Description. Default: 0.
        bottleneck_dims (int): Description. Default: 0.
        num_tasks (int): Description. Default: 1.
        norm_tpl (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: int = 0
    bottleneck_dims: int = 0
    num_tasks: int = 1
    norm_tpl: Optional[LayerTpl] = None


class Ngrammer(BaseModel):
    """Implements a generic N-grammer layer which looks up latent bi-gram id.

    Args:
        ngram_vocab_size (int): Description. Default: '768 * 256'.
        unigram_vocab_size (int): Description. Default: 0.
        ngram_emb_dim (int): Description. Default: 8.
        concat_ngrams (bool): Description. Default: True.
        num_heads (int): Description. Default: 0.
        dim_per_head (int): Description. Default: 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    ngram_vocab_size: int = "768 * 256"
    unigram_vocab_size: int = 0
    ngram_emb_dim: int = 8
    concat_ngrams: bool = True
    num_heads: int = 0
    dim_per_head: int = 0


class PerDimScale(BaseModel):
    """A layer to scale individual dims of the input.

    Args:
        dim (int): Description. Default: 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dim: int = 0


class PipelinedTransformer(BaseModel):
    """None='```(None)```', num_pipeline_microbatches: int

    Args:
        pipeline_stage (LayerTpl): Description. Default: 'template_field(StackedTransformer)'.
        circular_repeat (int): Description. Default: 1.
        num_pipeline_stages (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    pipeline_stage: LayerTpl = "template_field(StackedTransformer)"
    circular_repeat: int = 1
    num_pipeline_stages: Optional[int] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for PipelinedTransformer.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class Pooling(BaseModel):
    """Pooling layer, which by default performs max pooling.

    Args:
        window_shape (Sequence[int]): Description. Default: '(0, 0)'.
        window_stride (Sequence[int]): Description. Default: '(0, 0)'.
        pooling_type (str): Description. Default: 'MAX'.
        padding (str): Description. Default: 'SAME'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    window_shape: Sequence[int] = "(0, 0)"
    window_stride: Sequence[int] = "(0, 0)"
    pooling_type: str = "MAX"
    padding: str = "SAME"


class Pooling1D(BaseModel):
    """Pooling layer that operates over dimension 1 only; assume [B,T,...] inputs.

    Args:
        stride (int): Description. Default: 1.
        window (int): Description. Default: 0.
        pooling_type (str): Description. Default: 'AVG'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    stride: int = 1
    window: int = 0
    pooling_type: str = "AVG"


class PositionalEmbedding(BaseModel):
    """Generates position embedding for a given 1-d sequence.

    Args:
        min_timescale (int): Description. Default: 1.
        max_timescale (int): Description. Default: 10000.
        embedding_dims (int): Description. Default: 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    min_timescale: int = 1
    max_timescale: int = 10000
    embedding_dims: int = 0


class PositionalEmbedding2D(BaseModel):
    """Generates 2-d position embedding for sequence of flattened patches.

    Args:
        h (int): Description. Default: 0.
        w (int): Description. Default: 0.
        embedding_dims (int): Description. Default: 0.
        pos_transform (str): Description. Default: 'hwd->(hw)d'.
        num_prepend_cls_tokens (int): Description. Default: 0.
        num_append_cls_tokens (int): Description. Default: 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    h: int = 0
    w: int = 0
    embedding_dims: int = 0
    pos_transform: str = "hwd->(hw)d"
    num_prepend_cls_tokens: int = 0
    num_append_cls_tokens: int = 0


class RandomVectorQuantizer(BaseModel):
    """None='```(None)```', projection_dim: int=16, num_latent_classes: int

    Args:
        latent_dim (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    latent_dim: Optional[int] = None


class ReLU(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for ReLU.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.maximum(x, 0.0)


class ReLU6(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for ReLU6.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.minimum(np.maximum(x, 0.0), 6.0)


class RelativeBias(BaseModel):
    """A layer for Relative Attention Bias.

    Args:
        num_heads (int): Description. Default: 1.
        use_length_as_position (bool): Description. Default: True.
        relative_attention_num_buckets (int): Description. Default: 32.
        relative_attention_max_distance (int): Description. Default: 128.
        bidirectional (bool): Description. Default: False.
        use_xavier_init (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    num_heads: int = 1
    use_length_as_position: bool = True
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    bidirectional: bool = False
    use_xavier_init: bool = False


class Repeat(BaseModel):
    """None='base_layer.template_field(None)', x_times: int=0, unpack_summaries: bool=False, checkpoint_policy: AutodiffCheckpointType='AutodiffCheckpointType', unroll_in_decode: bool=False, sublayer_name: str='sub', optimizer_dims_mapping: SplitDimsMapping='```(None)```', collect_intermediate_outputs: bool=False, return_intermediate_outputs: bool=False, nd_prefix_shape: Sequence[int]

    Args:
        sub_tpl (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    sub_tpl: Optional[LayerTpl] = None


class ResNet(BaseModel):
    """None='pax_fiddle.fdl_field(default_factory=_res_net_output_spatial_pooling_params_default)', return_block_features: bool=False, entry_max_pool: bool=True)`

    Args:
        conv_params (LayerTpl): Description. Default: 'pax_fiddle.fdl_field(default_factory=_res_net_conv_params_default)'.
        block_params (LayerTpl): Description. Default: 'template_field(ResNetBlock)'.
        strides (Sequence[int]): Description. Default: '(1, 2, 2, 2)'.
        channels (Sequence[int]): Description. Default: '(256, 512, 1024, 2048)'.
        blocks (Sequence[int]): Description. Default: '(3, 4, 6, 3)'.
        kernels (Sequence[int]): Description. Default: '(3, 3, 3, 3)'.
        entryflow_conv_kernel (Sequence[int]): Description. Default: '(7, 7, 3)'.
        entryflow_conv_stride (Sequence[int]): Description. Default: '(2, 2)'.
        output_spatial_pooling_params (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    conv_params: LayerTpl = (
        "pax_fiddle.fdl_field(default_factory=_res_net_conv_params_default)"
    )
    block_params: LayerTpl = "template_field(ResNetBlock)"
    strides: Sequence[int] = "(1, 2, 2, 2)"
    channels: Sequence[int] = "(256, 512, 1024, 2048)"
    blocks: Sequence[int] = "(3, 4, 6, 3)"
    kernels: Sequence[int] = "(3, 3, 3, 3)"
    entryflow_conv_kernel: Sequence[int] = "(7, 7, 3)"
    entryflow_conv_stride: Sequence[int] = "(2, 2)"
    output_spatial_pooling_params: Optional[LayerTpl] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for ResNet.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class ResNetBlock(BaseModel):
    """ResNet Block as in https://arxiv.org/abs/1512.03385.

    Args:
        input_dim (int): Description. Default: 0.
        output_dim (int): Description. Default: 0.
        conv_params (LayerTpl): Description. Default: 'template_field(convolutions.ConvBNAct)'.
        kernel_size (int): Description. Default: 3.
        stride (int): Description. Default: 1.
        activation_tpl (pax_fiddle.Config[activations.BaseActivation]): Description. Default: 'template_field(activations.ReLU)'.
        residual_droppath_prob (float): Description. Default: 0.0.
        zero_init_residual (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dim: int = 0
    output_dim: int = 0
    conv_params: LayerTpl = "template_field(convolutions.ConvBNAct)"
    kernel_size: int = 3
    stride: int = 1
    activation_tpl: pax_fiddle.Config[activations.BaseActivation] = (
        "template_field(activations.ReLU)"
    )
    residual_droppath_prob: float = 0.0
    zero_init_residual: bool = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for ResNetBlock.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class RmsNorm(BaseModel):
    """None='```(None)```')`

    Args:
        epsilon (float): Description. Default: 1e-06.
        direct_scale (bool): Description. Default: True.
        intermediate_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    epsilon: float = 1e-06
    direct_scale: bool = True
    intermediate_dtype: Optional[jnp.dtype] = None


class RmsNormNoScale(BaseModel):
    """RMS normalization: https://arxiv.org/abs/1910.07467 without scale.

    Args:
        epsilon (float): Description. Default: 1e-06.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    epsilon: float = 1e-06


class SSM(BaseModel):
    """A generic S4D-SSM layer for (multiple) 1D input.

    Args:
        nheads (int): Description. Default: 0.
        dim (int): Description. Default: 0.
        l_max (int): Description. Default: 0.
        decode_num_samples (int): Description. Default: 0.
        step_size (float): Description. Default: 0.01.
        hippo_type (str): Description. Default: 'ss4d-1d'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    nheads: int = 0
    dim: int = 0
    l_max: int = 0
    decode_num_samples: int = 0
    step_size: float = 0.01
    hippo_type: str = "ss4d-1d"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for SSM.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class SSMGated(BaseModel):
    """Gated State Space Model, https://arxiv.org/pdf/2212.10544.pdf.

    Args:
        gss_fflayer_tpl (LayerTpl): Description. Default: 'template_field(linears.FeedForward)'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    gss_fflayer_tpl: LayerTpl = "template_field(linears.FeedForward)"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for SSMGated.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class SSMTransformer(BaseModel):
    """Transformer layer using SSM instead of self-attention.

    Args:
        ssm_tpl (LayerTpl): Description. Default: 'template_field(ssm.SSM)'.
        ssm_nheads (int): Description. Default: 0.
        ssm_dim (int): Description. Default: 0.
        ssm_l_max (int): Description. Default: 0.
        ssm_hippo_type (str): Description. Default: 'ss4d-1d-legs'.
        ssm_step_size (float): Description. Default: 0.1.
        decode_num_samples (int): Description. Default: 4.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    ssm_tpl: LayerTpl = "template_field(ssm.SSM)"
    ssm_nheads: int = 0
    ssm_dim: int = 0
    ssm_l_max: int = 0
    ssm_hippo_type: str = "ss4d-1d-legs"
    ssm_step_size: float = 0.1
    decode_num_samples: int = 4

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for SSMTransformer.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class SelfAttentionWithNormAndResidual(BaseModel):
    """None=True, residual_dropout_prob: float=0.0, residual_dropout_tpl: LayerTpl='template_field(stochastics.Dropout)', norm_policy: str

    Args:
        residual_weight (float): Description. Default: 1.0.
        input_weight (float): Description. Default: 1.0.
        self_atten_tpl (LayerTpl): Description. Default: 'template_field(DotProductAttentionWithContext)'.
        norm_tpl (LayerTpl): Description. Default: 'template_field(normalizations.LayerNorm)'.
        pre_layer_norm (bool): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    residual_weight: float = 1.0
    input_weight: float = 1.0
    self_atten_tpl: LayerTpl = "template_field(DotProductAttentionWithContext)"
    norm_tpl: LayerTpl = "template_field(normalizations.LayerNorm)"
    pre_layer_norm: Optional[bool] = None


class SequenceModel(BaseModel):
    """None='```(None)```')`

    Args:
        model_tpl (LayerTpl): Description. Default: 'template_field(transformer_models.TransformerEncoderDecoder)'.
        return_predictions (bool): Description. Default: False.
        decoder_tpl (DecoderHParams): Description. Default: 'base_layer.instance_field(GreedyDecoderHParams)'.
        label_smoothing_prob (float): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    model_tpl: LayerTpl = "template_field(transformer_models.TransformerEncoderDecoder)"
    return_predictions: bool = False
    decoder_tpl: DecoderHParams = "base_layer.instance_field(GreedyDecoderHParams)"
    label_smoothing_prob: Optional[float] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for SequenceModel.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class Sequential(BaseModel):
    """None='```(None)```')`

    Args:
        layers (Sequence[Callable[..., Any]]): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    layers: Optional[Sequence[Callable[..., Any]]] = None


class SharedEmbeddingSoftmax(BaseModel):
    """A softmax layer that also supports embedding lookups.

    Args:
        lookup_style (str): Description. Default: 'index'.
        scale_sqrt_depth (bool): Description. Default: False.
        array_lookup_tpl (LayerTpl): Description. Default: 'template_field(base_ops.ArrayLookup)'.
        einsum_tpl (LayerTpl): Description. Default: 'template_field(base_ops.EinsumOp)'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    lookup_style: str = "index"
    scale_sqrt_depth: bool = False
    array_lookup_tpl: LayerTpl = "template_field(base_ops.ArrayLookup)"
    einsum_tpl: LayerTpl = "template_field(base_ops.EinsumOp)"


class SiLU(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for SiLU.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x * (1 / (1 + np.exp(-x)))


class Sigmoid(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for Sigmoid.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return 1 / (1 + np.exp(-x))


class SigmoidCrossEntropy(BaseModel):
    """None=0.0, bias_init: float

    Args:
        input_dims (int): Description. Default: 0.
        num_classes (int): Description. Default: 0.
        soft_cap_logits (float): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: int = 0
    num_classes: int = 0
    soft_cap_logits: Optional[float] = None

    def __call__(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Forward pass for SigmoidCrossEntropy.

        Args:
            logits (np.ndarray): Input logits.
            labels (np.ndarray): Target labels.

        Returns:
            np.ndarray: Loss array.
        """
        # numerically stable sigmoid cross entropy
        # z = logits, x = labels
        # max(x, 0) - x * z + log(1 + exp(-abs(x)))
        return (
            np.maximum(logits, 0)
            - logits * labels
            + np.log(1 + np.exp(-np.abs(logits)))
        )


class SpectrumAugmenter(BaseModel):
    """Performs data augmentation as according to the SpecAug paper.

    Args:
        freq_mask_max_bins (int): Description. Default: 27.
        freq_mask_count (int): Description. Default: 2.
        use_dynamic_time_mask_max_frames (bool): Description. Default: True.
        time_mask_max_frames (int): Description. Default: 40.
        time_mask_count (int): Description. Default: 10.
        time_mask_max_ratio (float): Description. Default: 0.05.
        time_masks_per_frame (float): Description. Default: 0.0.
        augment_at_eval (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    freq_mask_max_bins: int = 27
    freq_mask_count: int = 2
    use_dynamic_time_mask_max_frames: bool = True
    time_mask_max_frames: int = 40
    time_mask_count: int = 10
    time_mask_max_ratio: float = 0.05
    time_masks_per_frame: float = 0.0
    augment_at_eval: bool = False


class SquaredReLU(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for SquaredReLU.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.square(np.maximum(x, 0.0))


class StackFrnn(BaseModel):
    """None='base_layer.template_field(None)', num_layers: int=1, num_input_nodes: int=0, num_output_nodes: int=0)`

    Args:
        frnn_tpl (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    frnn_tpl: Optional[LayerTpl] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for StackFrnn.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class StackedTransformer(BaseModel):
    """None='```(None)```', dropout_prob: float=0.0, atten_dropout_prob: float

    Args:
        use_cross_attention (bool): Description. Default: False.
        mask_self_attention (bool): Description. Default: False.
        num_layers (int): Description. Default: 0.
        model_dims (int): Description. Default: 0.
        hidden_dims (int): Description. Default: 0.
        num_heads (int): Description. Default: 0.
        dim_per_head (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    use_cross_attention: bool = False
    mask_self_attention: bool = False
    num_layers: int = 0
    model_dims: int = 0
    hidden_dims: int = 0
    num_heads: int = 0
    dim_per_head: Optional[int] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for StackedTransformer.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class StackedTransformerRepeated(BaseModel):
    """None='```(None)```', return_intermediate_outputs: bool=False, collect_intermediate_outputs: bool=False)`

    Args:
        block (LayerTpl): Description. Default: 'template_field(StackedTransformer)'.
        x_times (int): Description. Default: 0.
        checkpoint_policy (repeats.AutodiffCheckpointType): Description. Default: 'repeats.AutodiffCheckpointType'.
        unroll_in_decode (bool): Description. Default: True.
        repeat_layer_name (str): Description. Default: 'repeat'.
        sublayer_name (str): Description. Default: 'sub'.
        repeat_optimizer_dims_mapping (SplitDimsMapping): Description. Default: '(None)'.
        nd_prefix_shape (Sequence[int]): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    block: LayerTpl = "template_field(StackedTransformer)"
    x_times: int = 0
    checkpoint_policy: repeats.AutodiffCheckpointType = "repeats.AutodiffCheckpointType"
    unroll_in_decode: bool = True
    repeat_layer_name: str = "repeat"
    sublayer_name: str = "sub"
    repeat_optimizer_dims_mapping: SplitDimsMapping = "(None)"
    nd_prefix_shape: Optional[Sequence[int]] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for StackedTransformerRepeated.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class StackingOverTime(BaseModel):
    """Stacking applied along the time axis.

    Args:
        left_context (int): Description. Default: 0.
        right_context (int): Description. Default: 0.
        stride (int): Description. Default: 0.
        pad_with_left_frame (bool): Description. Default: False.
        pad_with_right_frame (bool): Description. Default: False.
        padding_reduce_option (str): Description. Default: 'reduce_min'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    left_context: int = 0
    right_context: int = 0
    stride: int = 0
    pad_with_left_frame: bool = False
    pad_with_right_frame: bool = False
    padding_reduce_option: str = "reduce_min"


class StochasticResidual(BaseModel):
    """Stochastic residual layer that randomly drops the residual branch.

    Args:
        residual_weight (float): Description. Default: 1.0.
        survival_prob (float): Description. Default: 1.0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    residual_weight: float = 1.0
    survival_prob: float = 1.0


class Swish(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for Swish.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x * (1 / (1 + np.exp(-x)))


class Tanh(BaseModel):
    """None=None, params_init: WeightInit=<factory>, skip_lp_regularization: bool

    Args:
        dtype (jnp.dtype): Description. Default: ....
        fprop_dtype (jnp.dtype): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    dtype: Optional[jnp.dtype] = None
    fprop_dtype: Optional[jnp.dtype] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for Tanh.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.tanh(x)


class TemporalShifting(BaseModel):
    """Shifts audio signals by a random amount during training.

    Args:
        shift_range_ms (float): Description. Default: 0.0.
        sample_rate (float): Description. Default: 16000.0.
        axis (int): Description. Default: 1.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    shift_range_ms: float = 0.0
    sample_rate: float = 16000.0
    axis: int = 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for TemporalShifting.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class TrainablePositionalEmbedding(BaseModel):
    """Generates trainable position embedding for a given 1-d sequence.

    Args:
        max_seq_length (int): Description. Default: 10240.
        lookup_style (str): Description. Default: 'matmul'.
        array_lookup_tpl (LayerTpl): Description. Default: 'template_field(base_ops.ArrayLookup)'.
        einsum_tpl (LayerTpl): Description. Default: 'template_field(base_ops.EinsumOp)'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    max_seq_length: int = 10240
    lookup_style: str = "matmul"
    array_lookup_tpl: LayerTpl = "template_field(base_ops.ArrayLookup)"
    einsum_tpl: LayerTpl = "template_field(base_ops.EinsumOp)"


class Transformer(BaseModel):
    """None='```(None)```', dim_per_head: int

    Args:
        input_dims (int): Description. Default: 0.
        hidden_dims (int): Description. Default: 0.
        num_heads (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: int = 0
    hidden_dims: int = 0
    num_heads: Optional[int] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for Transformer.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class TransformerEncoderDecoder(BaseModel):
    """None='template_field(embedding_softmax.PositionalEmbedding)', encoder_position_emb_tpl: LayerTpl

    Args:
        position_emb_tpl (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    position_emb_tpl: Optional[LayerTpl] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for TransformerEncoderDecoder.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class TransformerFeedForward(BaseModel):
    """Transformer feedforward layer with residual connection and dropout.

    Args:
        input_dims (int): Description. Default: 0.
        output_dims (int): Description. Default: 0.
        hidden_dims (int): Description. Default: 0.
        has_bias (bool): Description. Default: True.
        apply_padding_first (bool): Description. Default: False.
        activation_tpl (pax_fiddle.Config[activations_lib.BaseActivation]): Description. Default: 'template_field(activations_lib.ReLU)'.
        use_gated_activation (bool): Description. Default: False.
        fflayer_tpl (LayerTpl): Description. Default: 'template_field(linears.FeedForward)'.
        ln_tpl (LayerTpl): Description. Default: 'template_field(normalizations.LayerNorm)'.
        residual_dropout_prob (float): Description. Default: 0.0.
        relu_dropout_tpl (LayerTpl): Description. Default: 'template_field(stochastics.Dropout)'.
        relu_dropout_prob (float): Description. Default: 0.0.
        residual_dropout_tpl (LayerTpl): Description. Default: 'template_field(stochastics.Dropout)'.
        add_skip_connection (bool): Description. Default: True.
        residual_weight (float): Description. Default: 1.0.
        residual_droppath_prob (float): Description. Default: 0.0.
        norm_policy (str): Description. Default: 'pre'.
        internal_gshard_variance_scaling_fan_in_init (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: int = 0
    output_dims: int = 0
    hidden_dims: int = 0
    has_bias: bool = True
    apply_padding_first: bool = False
    activation_tpl: pax_fiddle.Config[activations_lib.BaseActivation] = (
        "template_field(activations_lib.ReLU)"
    )
    use_gated_activation: bool = False
    fflayer_tpl: LayerTpl = "template_field(linears.FeedForward)"
    ln_tpl: LayerTpl = "template_field(normalizations.LayerNorm)"
    residual_dropout_prob: float = 0.0
    relu_dropout_tpl: LayerTpl = "template_field(stochastics.Dropout)"
    relu_dropout_prob: float = 0.0
    residual_dropout_tpl: LayerTpl = "template_field(stochastics.Dropout)"
    add_skip_connection: bool = True
    residual_weight: float = 1.0
    residual_droppath_prob: float = 0.0
    norm_policy: str = "pre"
    internal_gshard_variance_scaling_fan_in_init: bool = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for TransformerFeedForward.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class TransformerFeedForwardMoe(BaseModel):
    """None='```(None)```', expert_capacity_dim: int=0, unadjusted_expert_capacity_factor: float=2.0, expert_weight_shards: int=1, second_expert_policy: str='all', internal_gshard_variance_scaling_fan_in_init: bool=True, explicit_fan_in_fan_out_axes: bool=False, moe_load_balance_loss_weight: float=1.0, gating_logit_cap: float=0.0, moe_gating_embedding_level: str='token', use_gated_activation: bool=False)`

    Args:
        input_dims (int): Description. Default: 0.
        hidden_dims (int): Description. Default: 0.
        apply_padding_first (bool): Description. Default: False.
        ln_tpl (LayerTpl): Description. Default: 'template_field(normalizations.LayerNorm)'.
        activation_tpl (pax_fiddle.Config[activations_lib.BaseActivation]): Description. Default: 'template_field(activations_lib.ReLU)'.
        relu_dropout_tpl (LayerTpl): Description. Default: 'template_field(stochastics.Dropout)'.
        relu_dropout_prob (float): Description. Default: 0.0.
        residual_dropout_tpl (LayerTpl): Description. Default: 'template_field(stochastics.Dropout)'.
        residual_dropout_prob (float): Description. Default: 0.0.
        add_skip_connection (bool): Description. Default: True.
        residual_weight (float): Description. Default: 1.0.
        norm_policy (str): Description. Default: 'pre'.
        residual_droppath_prob (float): Description. Default: 0.0.
        gating_func (str): Description. Default: 'top2'.
        num_experts (int): Description. Default: 0.
        num_groups (int): Description. Default: 0.
        min_group_size (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dims: int = 0
    hidden_dims: int = 0
    apply_padding_first: bool = False
    ln_tpl: LayerTpl = "template_field(normalizations.LayerNorm)"
    activation_tpl: pax_fiddle.Config[activations_lib.BaseActivation] = (
        "template_field(activations_lib.ReLU)"
    )
    relu_dropout_tpl: LayerTpl = "template_field(stochastics.Dropout)"
    relu_dropout_prob: float = 0.0
    residual_dropout_tpl: LayerTpl = "template_field(stochastics.Dropout)"
    residual_dropout_prob: float = 0.0
    add_skip_connection: bool = True
    residual_weight: float = 1.0
    norm_policy: str = "pre"
    residual_droppath_prob: float = 0.0
    gating_func: str = "top2"
    num_experts: int = 0
    num_groups: int = 0
    min_group_size: Optional[int] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for TransformerFeedForwardMoe.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class TransformerLm(BaseModel):
    """None='template_field(None)', post_attention_ngrammer_tpls: Sequence[LayerTpl]

    Args:
        position_emb_tpl (LayerTpl): Description. Default: 'template_field(embedding_softmax.PositionalEmbedding)'.
        model_dims (int): Description. Default: 0.
        stacked_transformer_tpl (LayerTpl): Description. Default: 'template_field(transformers.StackedTransformer)'.
        softmax_tpl (LayerTpl): Description. Default: 'template_field(embedding_softmax.SharedEmbeddingSoftmax)'.
        vocab_size (int): Description. Default: 0.
        packed_input (bool): Description. Default: False.
        model_type (LanguageModelType): Description. Default: 'LanguageModelType'.
        ngrammer_tpl (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    position_emb_tpl: LayerTpl = "template_field(embedding_softmax.PositionalEmbedding)"
    model_dims: int = 0
    stacked_transformer_tpl: LayerTpl = (
        "template_field(transformers.StackedTransformer)"
    )
    softmax_tpl: LayerTpl = "template_field(embedding_softmax.SharedEmbeddingSoftmax)"
    vocab_size: int = 0
    packed_input: bool = False
    model_type: LanguageModelType = "LanguageModelType"
    ngrammer_tpl: Optional[LayerTpl] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for TransformerLm.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class VQNgrammer(BaseModel):
    """Implements a VQ based ngrammer layer which looks up latent ngram id.

    Args:
        ngram_vocab_size (int): Description. Default: '768 * 256'.
        unigram_vocab_size (int): Description. Default: 0.
        ngram_emb_dim (int): Description. Default: 8.
        ngram_using_attention_scores (bool): Description. Default: False.
        causal_attention (bool): Description. Default: True.
        concat_ngrams (bool): Description. Default: False.
        num_clusters (int): Description. Default: 0.
        num_heads (int): Description. Default: 0.
        decay (float): Description. Default: 0.999.
        epsilon (float): Description. Default: 1e-06.
        dim_per_head (int): Description. Default: 0.
        use_cached_input_ids_to_cluster_ids (bool): Description. Default: False.
        enable_cache_updates (bool): Description. Default: True.
        full_update_cache_frequency (int): Description. Default: 0.
        full_update_cache_steps (tuple[int, ...]): Description. Default: ().
        prenormalize (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    ngram_vocab_size: int = "768 * 256"
    unigram_vocab_size: int = 0
    ngram_emb_dim: int = 8
    ngram_using_attention_scores: bool = False
    causal_attention: bool = True
    concat_ngrams: bool = False
    num_clusters: int = 0
    num_heads: int = 0
    decay: float = 0.999
    epsilon: float = 1e-06
    dim_per_head: int = 0
    use_cached_input_ids_to_cluster_ids: bool = False
    enable_cache_updates: bool = True
    full_update_cache_frequency: int = 0
    full_update_cache_steps: tuple[int, ...] = ()
    prenormalize: bool = False


class VanillaBlock(BaseModel):
    """Vanilla Block.

    Args:
        input_dim (int): Description. Default: 0.
        output_dim (int): Description. Default: 0.
        conv_params (LayerTpl): Description. Default: 'pax_fiddle.fdl_field(default_factory=_vanilla_block_conv_params_default)'.
        kernel_size (int): Description. Default: 3.
        stride (int): Description. Default: 1.
        negative_slope (float): Description. Default: 0.4.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    input_dim: int = 0
    output_dim: int = 0
    conv_params: LayerTpl = (
        "pax_fiddle.fdl_field(default_factory=_vanilla_block_conv_params_default)"
    )
    kernel_size: int = 3
    stride: int = 1
    negative_slope: float = 0.4


class VanillaNet(BaseModel):
    """None='template_field(poolings.GlobalPooling)', negative_slope: float=0.4)`

    Args:
        conv_params (LayerTpl): Description. Default: 'pax_fiddle.fdl_field(default_factory=_vanilla_net_conv_params_default)'.
        block_params (LayerTpl): Description. Default: 'template_field(VanillaBlock)'.
        strides (Sequence[int]): Description. Default: '(1, 2, 2, 2)'.
        channels (Sequence[int]): Description. Default: '(256, 512, 1024, 2048)'.
        blocks (Sequence[int]): Description. Default: '(3, 4, 6, 3)'.
        kernels (Sequence[int]): Description. Default: '(3, 3, 3, 3)'.
        entryflow_conv_kernel (Sequence[int]): Description. Default: '(7, 7)'.
        entryflow_conv_stride (Sequence[int]): Description. Default: '(2, 2)'.
        output_spatial_pooling_params (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    conv_params: LayerTpl = (
        "pax_fiddle.fdl_field(default_factory=_vanilla_net_conv_params_default)"
    )
    block_params: LayerTpl = "template_field(VanillaBlock)"
    strides: Sequence[int] = "(1, 2, 2, 2)"
    channels: Sequence[int] = "(256, 512, 1024, 2048)"
    blocks: Sequence[int] = "(3, 4, 6, 3)"
    kernels: Sequence[int] = "(3, 3, 3, 3)"
    entryflow_conv_kernel: Sequence[int] = "(7, 7)"
    entryflow_conv_stride: Sequence[int] = "(2, 2)"
    output_spatial_pooling_params: Optional[LayerTpl] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for VanillaNet.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class VectorQuantization(BaseModel):
    """Implements vector quantization (VQ)/online k-means clustering.

    Args:
        num_clusters (int): Description. Default: 0.
        num_heads (int): Description. Default: 0.
        decay (float): Description. Default: 0.999.
        epsilon (float): Description. Default: 1e-06.
        dim_per_head (int): Description. Default: 0.
        prenormalize (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    num_clusters: int = 0
    num_heads: int = 0
    decay: float = 0.999
    epsilon: float = 1e-06
    dim_per_head: int = 0
    prenormalize: bool = False


class VectorQuantizer(BaseModel):
    """None='```(None)```', latent_dim: int

    Args:
        num_latent_classes (int): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    num_latent_classes: Optional[int] = None


class VisionTransformer(BaseModel):
    """Vision transformer model.

    Args:
        entry_layers_tpl (LayerTpl): Description. Default: 'template_field(VitEntryLayers)'.
        transformer_layers_tpl (LayerTpl): Description. Default: 'template_field(transformers.StackedTransformer)'.
        exit_layers_tpl (LayerTpl): Description. Default: 'template_field(VitExitLayers)'.
        full_data_parallel_on_entry_exit (bool): Description. Default: False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    entry_layers_tpl: LayerTpl = "template_field(VitEntryLayers)"
    transformer_layers_tpl: LayerTpl = "template_field(transformers.StackedTransformer)"
    exit_layers_tpl: LayerTpl = "template_field(VitExitLayers)"
    full_data_parallel_on_entry_exit: bool = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for VisionTransformer.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return x


class VitEntryLayers(BaseModel):
    """None='template_field(embedding_softmax.TrainablePositionalEmbedding)', input_fc_has_bias: bool=True)`

    Args:
        pos_emb_shapes (tuple[int, int]): Description. Default: '(0, 0)'.
        patch_size (int): Description. Default: 0.
        input_dims (int): Description. Default: 0.
        output_dims (int): Description. Default: 0.
        pos_emb_dropout_prob (float): Description. Default: 0.0.
        prepend_cls_tokens (int): Description. Default: 0.
        append_cls_tokens (int): Description. Default: 0.
        pos_emb_tpl (LayerTpl): Description. Default: ....
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    pos_emb_shapes: tuple[int, int] = "(0, 0)"
    patch_size: int = 0
    input_dims: int = 0
    output_dims: int = 0
    pos_emb_dropout_prob: float = 0.0
    prepend_cls_tokens: int = 0
    append_cls_tokens: int = 0
    pos_emb_tpl: Optional[LayerTpl] = None


class VitExitLayers(BaseModel):
    """Exit block of ViT.

    Args:
        hidden_dim (int): Description. Default: 0.
        output_dim (int): Description. Default: 0.
        output_dropout_prob (float): Description. Default: 0.0.
        pooled (bool): Description. Default: True.
        pre_ln (bool): Description. Default: True.
        output_fc_tanh (bool): Description. Default: True.
        output_fc_has_bias (bool): Description. Default: True.
        pooling_tpl (LayerTpl): Description. Default: 'template_field(poolings.GlobalPooling)'.
        ln_tpl (LayerTpl): Description. Default: 'template_field(normalizations.LayerNorm)'.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    hidden_dim: int = 0
    output_dim: int = 0
    output_dropout_prob: float = 0.0
    pooled: bool = True
    pre_ln: bool = True
    output_fc_tanh: bool = True
    output_fc_has_bias: bool = True
    pooling_tpl: LayerTpl = "template_field(poolings.GlobalPooling)"
    ln_tpl: LayerTpl = "template_field(normalizations.LayerNorm)"
