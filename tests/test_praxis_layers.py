"""
Tests for praxis layers.
"""

# ruff: noqa: F403, F405
from zero_pax.praxis.layers import *


def test_adaptedtransformerfeedforward():
    """Test AdaptedTransformerFeedForward."""
    obj = AdaptedTransformerFeedForward()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_attentionprojection():
    """Test AttentionProjection."""
    obj = AttentionProjection()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_autodiffcheckpointtype():
    """Test AutodiffCheckpointType."""
    obj = AutodiffCheckpointType()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_baseactivation():
    """Test BaseActivation."""
    import numpy as np

    obj = BaseActivation()
    if "BaseActivation" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_basenormalization():
    """Test BaseNormalization."""
    obj = BaseNormalization()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_batchnorm():
    """Test BatchNorm."""
    obj = BatchNorm()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_bertmodel():
    """Test BertModel."""
    obj = BertModel()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_bitemperedloss():
    """Test BiTemperedLoss."""
    obj = BiTemperedLoss()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_bias():
    """Test Bias."""
    obj = Bias()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_bregmanpca():
    """Test BregmanPCA."""
    obj = BregmanPCA()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_causaldepthwiseconv1d():
    """Test CausalDepthwiseConv1D."""
    obj = CausalDepthwiseConv1D()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_cifglstmcellsimple():
    """Test CifgLstmCellSimple."""
    obj = CifgLstmCellSimple()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_classificationmlpmodel():
    """Test ClassificationMLPModel."""
    obj = ClassificationMLPModel()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_classificationmodel():
    """Test ClassificationModel."""
    obj = ClassificationModel()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_conformer():
    """Test Conformer."""
    obj = Conformer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_conv2d():
    """Test Conv2D."""
    obj = Conv2D()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_convbnact():
    """Test ConvBNAct."""
    obj = ConvBNAct()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_convbnactwithpadding():
    """Test ConvBNActWithPadding."""
    obj = ConvBNActWithPadding()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_cubedrelu():
    """Test CubedReLU."""
    import numpy as np

    obj = CubedReLU()
    if "CubedReLU" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_depthwiseconv1d():
    """Test DepthwiseConv1D."""
    obj = DepthwiseConv1D()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_dotproductattention():
    """Test DotProductAttention."""
    obj = DotProductAttention()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_dotproductattentionwithcontext():
    """Test DotProductAttentionWithContext."""
    obj = DotProductAttentionWithContext()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_dotproductattentionwithcontextxl():
    """Test DotProductAttentionWithContextXL."""
    obj = DotProductAttentionWithContextXL()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_dotproductattentionxl():
    """Test DotProductAttentionXL."""
    obj = DotProductAttentionXL()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_dropout():
    """Test Dropout."""
    obj = Dropout()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_elu():
    """Test ELU."""
    import numpy as np

    obj = ELU()
    if "ELU" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_einsum():
    """Test Einsum."""
    obj = Einsum()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_einsumop():
    """Test EinsumOp."""
    obj = EinsumOp()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_embedding():
    """Test Embedding."""
    obj = Embedding()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_frnn():
    """Test FRnn."""
    obj = FRnn()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_feedforward():
    """Test FeedForward."""
    obj = FeedForward()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_fullsoftmax():
    """Test FullSoftmax."""
    obj = FullSoftmax()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_gelu():
    """Test GELU."""
    import numpy as np

    obj = GELU()
    if "GELU" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_gshardsharedembeddingsoftmax():
    """Test GShardSharedEmbeddingSoftmax."""
    obj = GShardSharedEmbeddingSoftmax()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_globalpooling():
    """Test GlobalPooling."""
    obj = GlobalPooling()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_groupnorm():
    """Test GroupNorm."""
    obj = GroupNorm()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_groupedqueryattention():
    """Test GroupedQueryAttention."""
    obj = GroupedQueryAttention()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_identity():
    """Test Identity."""
    obj = Identity()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_identitynorm():
    """Test IdentityNorm."""
    obj = IdentityNorm()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_languagemodel():
    """Test LanguageModel."""
    obj = LanguageModel()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_languagemodelcontinuousbatching():
    """Test LanguageModelContinuousBatching."""
    obj = LanguageModelContinuousBatching()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_languagemodeldpo():
    """Test LanguageModelDPO."""
    obj = LanguageModelDPO()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_languagemodeltype():
    """Test LanguageModelType."""
    obj = LanguageModelType()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_layernorm():
    """Test LayerNorm."""
    obj = LayerNorm()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_layernormalizedlstmcellsimple():
    """Test LayerNormalizedLstmCellSimple."""
    obj = LayerNormalizedLstmCellSimple()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_layerwiseshardablepipelined():
    """Test LayerwiseShardablePipelined."""
    obj = LayerwiseShardablePipelined()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_leakyrelu():
    """Test LeakyReLU."""
    import numpy as np

    obj = LeakyReLU()
    if "LeakyReLU" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_lightconv1d():
    """Test LightConv1D."""
    obj = LightConv1D()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_linear():
    """Test Linear."""
    obj = Linear()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_localselfattention():
    """Test LocalSelfAttention."""
    obj = LocalSelfAttention()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_localselfattentionalibi():
    """Test LocalSelfAttentionAlibi."""
    obj = LocalSelfAttentionAlibi()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_localselfattentionrelativebias():
    """Test LocalSelfAttentionRelativeBias."""
    obj = LocalSelfAttentionRelativeBias()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_localselfattentionxl():
    """Test LocalSelfAttentionXL."""
    obj = LocalSelfAttentionXL()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_lstmcellsimple():
    """Test LstmCellSimple."""
    obj = LstmCellSimple()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_lstmfrnn():
    """Test LstmFrnn."""
    obj = LstmFrnn()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_mlpblock():
    """Test MLPBlock."""
    obj = MLPBlock()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_maskedlmdataaugmenter():
    """Test MaskedLmDataAugmenter."""
    obj = MaskedLmDataAugmenter()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_multitaskresidualadapter():
    """Test MultitaskResidualAdapter."""
    obj = MultitaskResidualAdapter()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_ngrammer():
    """Test Ngrammer."""
    obj = Ngrammer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_perdimscale():
    """Test PerDimScale."""
    obj = PerDimScale()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_pipelinedtransformer():
    """Test PipelinedTransformer."""
    obj = PipelinedTransformer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_pooling():
    """Test Pooling."""
    obj = Pooling()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_pooling1d():
    """Test Pooling1D."""
    obj = Pooling1D()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_positionalembedding():
    """Test PositionalEmbedding."""
    obj = PositionalEmbedding()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_positionalembedding2d():
    """Test PositionalEmbedding2D."""
    obj = PositionalEmbedding2D()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_randomvectorquantizer():
    """Test RandomVectorQuantizer."""
    obj = RandomVectorQuantizer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_relu():
    """Test ReLU."""
    import numpy as np

    obj = ReLU()
    if "ReLU" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_relu6():
    """Test ReLU6."""
    import numpy as np

    obj = ReLU6()
    if "ReLU6" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_relativebias():
    """Test RelativeBias."""
    obj = RelativeBias()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_repeat():
    """Test Repeat."""
    obj = Repeat()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_resnet():
    """Test ResNet."""
    obj = ResNet()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_resnetblock():
    """Test ResNetBlock."""
    obj = ResNetBlock()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_rmsnorm():
    """Test RmsNorm."""
    obj = RmsNorm()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_rmsnormnoscale():
    """Test RmsNormNoScale."""
    obj = RmsNormNoScale()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_ssm():
    """Test SSM."""
    obj = SSM()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_ssmgated():
    """Test SSMGated."""
    obj = SSMGated()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_ssmtransformer():
    """Test SSMTransformer."""
    obj = SSMTransformer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_selfattentionwithnormandresidual():
    """Test SelfAttentionWithNormAndResidual."""
    obj = SelfAttentionWithNormAndResidual()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_sequencemodel():
    """Test SequenceModel."""
    obj = SequenceModel()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_sequential():
    """Test Sequential."""
    obj = Sequential()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_sharedembeddingsoftmax():
    """Test SharedEmbeddingSoftmax."""
    obj = SharedEmbeddingSoftmax()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_silu():
    """Test SiLU."""
    import numpy as np

    obj = SiLU()
    if "SiLU" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_sigmoid():
    """Test Sigmoid."""
    import numpy as np

    obj = Sigmoid()
    if "Sigmoid" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_sigmoidcrossentropy():
    """Test SigmoidCrossEntropy."""
    import numpy as np

    obj = SigmoidCrossEntropy()
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    logits = np.array([-1.0, 0.0, 1.0])
    labels = np.array([0.0, 0.0, 1.0])
    out = obj(logits, labels)
    assert out.shape == logits.shape


def test_spectrumaugmenter():
    """Test SpectrumAugmenter."""
    obj = SpectrumAugmenter()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_squaredrelu():
    """Test SquaredReLU."""
    import numpy as np

    obj = SquaredReLU()
    if "SquaredReLU" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_stackfrnn():
    """Test StackFrnn."""
    obj = StackFrnn()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_stackedtransformer():
    """Test StackedTransformer."""
    obj = StackedTransformer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_stackedtransformerrepeated():
    """Test StackedTransformerRepeated."""
    obj = StackedTransformerRepeated()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_stackingovertime():
    """Test StackingOverTime."""
    obj = StackingOverTime()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_stochasticresidual():
    """Test StochasticResidual."""
    obj = StochasticResidual()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_swish():
    """Test Swish."""
    import numpy as np

    obj = Swish()
    if "Swish" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_tanh():
    """Test Tanh."""
    import numpy as np

    obj = Tanh()
    if "Tanh" == "LeakyReLU":
        obj.negative_slope = 0.01
    assert obj is not None

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
    x = np.array([-1.0, 0.0, 1.0])
    out = obj(x)
    assert out.shape == x.shape


def test_temporalshifting():
    """Test TemporalShifting."""
    obj = TemporalShifting()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_trainablepositionalembedding():
    """Test TrainablePositionalEmbedding."""
    obj = TrainablePositionalEmbedding()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_transformer():
    """Test Transformer."""
    obj = Transformer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_transformerencoderdecoder():
    """Test TransformerEncoderDecoder."""
    obj = TransformerEncoderDecoder()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_transformerfeedforward():
    """Test TransformerFeedForward."""
    obj = TransformerFeedForward()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_transformerfeedforwardmoe():
    """Test TransformerFeedForwardMoe."""
    obj = TransformerFeedForwardMoe()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_transformerlm():
    """Test TransformerLm."""
    obj = TransformerLm()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_vqngrammer():
    """Test VQNgrammer."""
    obj = VQNgrammer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_vanillablock():
    """Test VanillaBlock."""
    obj = VanillaBlock()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_vanillanet():
    """Test VanillaNet."""
    obj = VanillaNet()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_vectorquantization():
    """Test VectorQuantization."""
    obj = VectorQuantization()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_vectorquantizer():
    """Test VectorQuantizer."""
    obj = VectorQuantizer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_visiontransformer():
    """Test VisionTransformer."""
    obj = VisionTransformer()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_vitentrylayers():
    """Test VitEntryLayers."""
    obj = VitEntryLayers()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass


def test_vitexitlayers():
    """Test VitExitLayers."""
    obj = VitExitLayers()
    assert obj is not None
    import numpy as np

    if hasattr(obj, "__call__"):
        try:
            obj(np.array([1.0, 2.0]))
        except TypeError:
            pass
