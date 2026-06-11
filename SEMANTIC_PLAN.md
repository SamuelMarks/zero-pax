# Semantic & Mathematical Implementation Plan

This document outlines the exhaustive plan for implementing the pure-numpy mathematical semantics of the 108 Praxis layers.

## Global Objectives
* [x] Replace dummy Pydantic fields with strict `numpy` array validation.
* [x] Implement `__call__` / `forward` methods for all modules.
* [x] Define state initialization (`init_weights`, `init_states`) purely in NumPy.
* [x] Implement mathematical parity tests against expected formulas (100% test coverage).
* [x] No third-party dependencies besides `numpy` and `pydantic`.

## Activations
* [x] Implement `BaseActivation` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `CubedReLU` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `ELU` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `GELU` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LeakyReLU` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `ReLU` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `ReLU6` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SiLU` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Sigmoid` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SigmoidCrossEntropy` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SquaredReLU` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Swish` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Tanh` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests

## Attention
* [x] Implement `AttentionProjection` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `DotProductAttention` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `DotProductAttentionWithContext` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `DotProductAttentionWithContextXL` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `DotProductAttentionXL` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `GroupedQueryAttention` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LocalSelfAttention` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LocalSelfAttentionAlibi` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LocalSelfAttentionRelativeBias` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LocalSelfAttentionXL` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `PerDimScale` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `RelativeBias` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests

## Convolutions
* [x] Implement `CausalDepthwiseConv1D` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Conv2D` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `ConvBNAct` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `ConvBNActWithPadding` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `DepthwiseConv1D` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `GlobalPooling` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LightConv1D` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Pooling` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Pooling1D` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests

## Core & Base
* [x] Implement `AutodiffCheckpointType` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Bias` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Dropout` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Einsum` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `EinsumOp` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Identity` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LayerwiseShardablePipelined` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Linear` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `MLPBlock` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `MaskedLmDataAugmenter` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `MultitaskResidualAdapter` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Repeat` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Sequential` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SpectrumAugmenter` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `StackingOverTime` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `StochasticResidual` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `VanillaBlock` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `VitEntryLayers` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `VitExitLayers` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests

## Embeddings & Softmax
* [x] Implement `Embedding` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `FullSoftmax` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `GShardSharedEmbeddingSoftmax` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Ngrammer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `PositionalEmbedding` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `PositionalEmbedding2D` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `RandomVectorQuantizer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SharedEmbeddingSoftmax` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `TrainablePositionalEmbedding` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `VQNgrammer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `VectorQuantization` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `VectorQuantizer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests

## Models & Architectures
* [x] Implement `BertModel` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `BiTemperedLoss` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `BregmanPCA` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `ClassificationMLPModel` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `ClassificationModel` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Conformer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LanguageModel` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LanguageModelContinuousBatching` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LanguageModelDPO` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LanguageModelType` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `ResNet` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `ResNetBlock` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SequenceModel` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `VanillaNet` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests

## Normalizations
* [x] Implement `BaseNormalization` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `BatchNorm` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `GroupNorm` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `IdentityNorm` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LayerNorm` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LayerNormalizedLstmCellSimple` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `RmsNorm` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `RmsNormNoScale` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SelfAttentionWithNormAndResidual` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests

## RNNs & SSMs
* [x] Implement `CifgLstmCellSimple` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `FRnn` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LstmCellSimple` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `LstmFrnn` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SSM` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SSMGated` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `StackFrnn` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `TemporalShifting` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests

## Transformers
* [x] Implement `AdaptedTransformerFeedForward` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `FeedForward` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `PipelinedTransformer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `SSMTransformer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `StackedTransformer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `StackedTransformerRepeated` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `Transformer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `TransformerEncoderDecoder` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `TransformerFeedForward` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `TransformerFeedForwardMoe` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `TransformerLm` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests
* [x] Implement `VisionTransformer` semantics
  - [x] State/Weight initialization
  - [x] Forward pass (`__call__`) math
  - [x] Numerical unit tests

