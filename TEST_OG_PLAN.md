# Official Test Suite Porting Plan

## Preparation
- [x] Research and identify the correct official `google/praxis` or `google/paxml` repository and commit tag corresponding to the currently implemented API.
- [x] Clone or extract the test cases from the official repository for the 108 implemented Praxis layers.
- [x] Create `requirements-test.txt` and populate it with original testing dependencies (e.g., `praxis`, `paxml`, `jax`, `jaxlib`, `flax`, `tensorflow`, `pytest`).
- [x] Create a testing harness/fixture that dual-initializes layers from both `google/praxis` and `zero_pax` to compare their outputs.

## Phase 1: Basic Activations
- [x] ReLU
- [x] LeakyReLU
- [x] GELU
- [x] Swish
- [x] Sigmoid
- [x] Tanh
- [x] CubedReLU
- [x] SquaredReLU
- [x] ELU

## Phase 2: Normalization Layers
- [x] BatchNorm
- [x] LayerNorm
- [x] GroupNorm
- [x] RmsNorm
- [x] RmsNormNoScale
- [x] BaseNormalization
- [x] IdentityNorm

## Phase 3: Linear & Pooling
- [x] Linear
- [x] Pooling
- [x] Pooling1D
- [x] GlobalPooling

## Phase 4: Convolutions
- [x] Conv2D
- [x] ConvBNAct
- [x] ConvBNActWithPadding
- [x] DepthwiseConv1D
- [x] CausalDepthwiseConv1D
- [x] LightConv1D

## Phase 5: Attention Mechanisms
- [x] DotProductAttention
- [x] DotProductAttentionWithContext
- [x] DotProductAttentionWithContextXL
- [x] DotProductAttentionXL
- [x] LocalSelfAttention
- [x] LocalSelfAttentionAlibi
- [x] LocalSelfAttentionRelativeBias
- [x] LocalSelfAttentionXL
- [x] SelfAttentionWithNormAndResidual
- [x] AttentionProjection
- [x] GroupedQueryAttention

## Phase 6: Transformer & Language Model Blocks
- [x] TransformerFeedForward
- [x] TransformerFeedForwardMoe
- [x] AdaptedTransformerFeedForward
- [x] Transformer
- [x] TransformerEncoderDecoder
- [x] TransformerLm
- [x] StackedTransformer
- [x] StackedTransformerRepeated
- [x] PipelinedTransformer
- [x] SSMTransformer
- [x] VisionTransformer
- [x] LanguageModel
- [x] LanguageModelContinuousBatching
- [x] LanguageModelDPO
- [x] SequenceModel
- [x] BertModel

## Phase 7: Embeddings & Quantization
- [x] Embedding
- [x] SharedEmbeddingSoftmax
- [x] GShardSharedEmbeddingSoftmax
- [x] PositionalEmbedding
- [x] PositionalEmbedding2D
- [x] TrainablePositionalEmbedding
- [x] VectorQuantization
- [x] VectorQuantizer
- [x] RandomVectorQuantizer

## Phase 8: Recurrent & SSM Layers
- [x] LstmCellSimple
- [x] CifgLstmCellSimple
- [x] LayerNormalizedLstmCellSimple
- [x] FRnn
- [x] LstmFrnn
- [x] StackFrnn
- [x] SSM
- [x] SSMGated

## Phase 9: Complex Architectures (ResNet, Vanilla, etc.)
- [x] ResNet
- [x] ResNetBlock
- [x] VanillaNet
- [x] VanillaBlock
- [x] MLPBlock
- [x] Conformer
- [x] VitEntryLayers
- [x] VitExitLayers

## Phase 10: Utilities, Routing, & Wrappers
- [x] Identity
- [x] Bias
- [x] Dropout
- [x] Einsum
- [x] EinsumOp
- [x] FullSoftmax
- [x] SigmoidCrossEntropy
- [x] BiTemperedLoss
- [x] Sequential
- [x] Repeat
- [x] PerDimScale
- [x] StackingOverTime
- [x] StochasticResidual
- [x] TemporalShifting
- [x] MultitaskResidualAdapter
- [x] LayerwiseShardablePipelined
- [x] BregmanPCA
- [x] AutodiffCheckpointType
- [x] Ngrammer
- [x] VQNgrammer
- [x] MaskedLmDataAugmenter
- [x] SpectrumAugmenter

## Phase 11: Validation and Integration
- [x] Implement robust `numpy.allclose` comparisons to ensure 1-to-1 exactness of the forward passes across all parameter states.
- [x] Address floating-point differences (e.g. standardizing `rtol` and `atol` between `numpy` and `jax.numpy`).
- [x] Finalize test suite logic and ensure 100% equivalence coverage.
- [x] Add `requirements-test.txt` installation step to `ci.yml`.
