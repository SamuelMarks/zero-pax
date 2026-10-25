"""Public API definitions for zero_pax."""

from __future__ import annotations

from zero_pax import _tensor_ops as _ops
from zero_pax.praxis import layers
from zero_pax.praxis.base import BasePraxisLayer
from zero_pax.praxis.layers import (
    ELU,
    GELU,
    SSM,
    AdaptedTransformerFeedForward,
    AttentionProjection,
    AutodiffCheckpointType,
    BaseActivation,
    BaseNormalization,
    BatchNorm,
    BertModel,
    Bias,
    BiTemperedLoss,
    BregmanPCA,
    CausalDepthwiseConv1D,
    CifgLstmCellSimple,
    ClassificationMLPModel,
    ClassificationModel,
    Conformer,
    Conv2D,
    ConvBNAct,
    ConvBNActWithPadding,
    CubedReLU,
    DepthwiseConv1D,
    DotProductAttention,
    DotProductAttentionWithContext,
    DotProductAttentionWithContextXL,
    DotProductAttentionXL,
    Dropout,
    Einsum,
    EinsumOp,
    Embedding,
    FeedForward,
    FRnn,
    FullSoftmax,
    GlobalPooling,
    GroupedQueryAttention,
    GroupNorm,
    GShardSharedEmbeddingSoftmax,
    Identity,
    IdentityNorm,
    LanguageModel,
    LanguageModelContinuousBatching,
    LanguageModelDPO,
    LanguageModelType,
    LayerNorm,
    LayerNormalizedLstmCellSimple,
    LayerwiseShardablePipelined,
    LeakyReLU,
    LightConv1D,
    Linear,
    LocalSelfAttention,
    LocalSelfAttentionAlibi,
    LocalSelfAttentionRelativeBias,
    LocalSelfAttentionXL,
    LstmCellSimple,
    LstmFrnn,
    MaskedLmDataAugmenter,
    MLPBlock,
    MultitaskResidualAdapter,
    Ngrammer,
    PerDimScale,
    PipelinedTransformer,
    Pooling,
    Pooling1D,
    PositionalEmbedding,
    PositionalEmbedding2D,
    RandomVectorQuantizer,
    RelativeBias,
    ReLU,
    ReLU6,
    Repeat,
    ResNet,
    ResNetBlock,
    RmsNorm,
    RmsNormNoScale,
    SelfAttentionWithNormAndResidual,
    SequenceModel,
    Sequential,
    SharedEmbeddingSoftmax,
    Sigmoid,
    SigmoidCrossEntropy,
    SiLU,
    SpectrumAugmenter,
    SquaredReLU,
    SSMGated,
    SSMTransformer,
    StackedTransformer,
    StackedTransformerRepeated,
    StackFrnn,
    StackingOverTime,
    StochasticResidual,
    Swish,
    Tanh,
    TemporalShifting,
    TrainablePositionalEmbedding,
    Transformer,
    TransformerEncoderDecoder,
    TransformerFeedForward,
    TransformerFeedForwardMoe,
    TransformerLm,
    VanillaBlock,
    VanillaNet,
    VectorQuantization,
    VectorQuantizer,
    VisionTransformer,
    VitEntryLayers,
    VitExitLayers,
    VQNgrammer,
)
from zero_pax.types import BaseParameter, Tensor, TensorOrScalar


def __call__(
    inputs: TensorOrScalar | None = None,
    *args: TensorOrScalar,
    **kwargs: TensorOrScalar,
) -> Tensor | None:
    """Module-level callable entrypoint returning `inputs` as a `Tensor` if provided.

    Args:
        inputs: Optional input tensor or scalar.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Converted `Tensor` or `None`.
    """
    if inputs is not None:
        return _ops.asarray(inputs)
    if args:
        return _ops.asarray(args[0])
    return None


def init_weights(
    layer: BasePraxisLayer | None = None,
    *args: TensorOrScalar,
    **kwargs: TensorOrScalar,
) -> dict[str, Tensor]:
    """Initialize and return a parameter weight dictionary for `layer` if provided.

    Args:
        layer: Optional `BasePraxisLayer` whose registered parameters should be exported.
        *args: Optional positional arguments.
        **kwargs: Optional keyword arguments.

    Returns:
        Dictionary mapping parameter names to `Tensor` objects.
    """
    if layer is not None:
        layer.setup()
        return {name: _ops.asarray(param()) for name, param in layer.params.items()}
    return {}


class Layer(BasePraxisLayer):
    """Base stateful layer container managing named `BaseParameter` instances.

    Inherits from `BasePraxisLayer` to unify the `zero_pax` and `zero_pax.praxis` hierarchies.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize the `Layer` with an optional `name`.

        Args:
            name: Optional string identifier for the layer.
        """
        super().__init__(name=name)

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Pass `inputs` through the base `Layer` by delegating to `BasePraxisLayer.__call__`.

        Args:
            inputs: Input tensor or scalar.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Converted `Tensor`.
        """
        return super().__call__(inputs, *args, **kwargs)


__all__ = [
    "AdaptedTransformerFeedForward",
    "AttentionProjection",
    "AutodiffCheckpointType",
    "BaseActivation",
    "BaseNormalization",
    "BaseParameter",
    "BasePraxisLayer",
    "BatchNorm",
    "BertModel",
    "BiTemperedLoss",
    "Bias",
    "BregmanPCA",
    "CausalDepthwiseConv1D",
    "CifgLstmCellSimple",
    "ClassificationMLPModel",
    "ClassificationModel",
    "Conformer",
    "Conv2D",
    "ConvBNAct",
    "ConvBNActWithPadding",
    "CubedReLU",
    "DepthwiseConv1D",
    "DotProductAttention",
    "DotProductAttentionWithContext",
    "DotProductAttentionWithContextXL",
    "DotProductAttentionXL",
    "Dropout",
    "ELU",
    "EXPORTED_NAMES",
    "Einsum",
    "EinsumOp",
    "Embedding",
    "FRnn",
    "FeedForward",
    "FullSoftmax",
    "GELU",
    "GShardSharedEmbeddingSoftmax",
    "GlobalPooling",
    "GroupNorm",
    "GroupedQueryAttention",
    "Identity",
    "IdentityNorm",
    "LanguageModel",
    "LanguageModelContinuousBatching",
    "LanguageModelDPO",
    "LanguageModelType",
    "Layer",
    "LayerNorm",
    "LayerNormalizedLstmCellSimple",
    "LayerwiseShardablePipelined",
    "LeakyReLU",
    "LightConv1D",
    "Linear",
    "LocalSelfAttention",
    "LocalSelfAttentionAlibi",
    "LocalSelfAttentionRelativeBias",
    "LocalSelfAttentionXL",
    "LstmCellSimple",
    "LstmFrnn",
    "MLPBlock",
    "MaskedLmDataAugmenter",
    "MultitaskResidualAdapter",
    "Ngrammer",
    "PerDimScale",
    "PipelinedTransformer",
    "Pooling",
    "Pooling1D",
    "PositionalEmbedding",
    "PositionalEmbedding2D",
    "RandomVectorQuantizer",
    "ReLU",
    "ReLU6",
    "RelativeBias",
    "Repeat",
    "ResNet",
    "ResNetBlock",
    "RmsNorm",
    "RmsNormNoScale",
    "SSM",
    "SSMGated",
    "SSMTransformer",
    "SelfAttentionWithNormAndResidual",
    "SequenceModel",
    "Sequential",
    "SharedEmbeddingSoftmax",
    "SiLU",
    "Sigmoid",
    "SigmoidCrossEntropy",
    "SpectrumAugmenter",
    "SquaredReLU",
    "StackFrnn",
    "StackedTransformer",
    "StackedTransformerRepeated",
    "StackingOverTime",
    "StochasticResidual",
    "Swish",
    "Tanh",
    "TemporalShifting",
    "Tensor",
    "TrainablePositionalEmbedding",
    "Transformer",
    "TransformerEncoderDecoder",
    "TransformerFeedForward",
    "TransformerFeedForwardMoe",
    "TransformerLm",
    "VQNgrammer",
    "VanillaBlock",
    "VanillaNet",
    "VectorQuantization",
    "VectorQuantizer",
    "VisionTransformer",
    "VitEntryLayers",
    "VitExitLayers",
    "__call__",
    "init_weights",
    "layers",
]

EXPORTED_NAMES: frozenset[str] = frozenset(__all__)
