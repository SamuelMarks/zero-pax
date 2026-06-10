"""Module docstring."""

import numpy as np
import pytest
from zero_pax.praxis.layers import (
    Identity,
    Bias,
    Dropout,
    Einsum,
    EinsumOp,
    FullSoftmax,
    SigmoidCrossEntropy,
    BiTemperedLoss,
    Sequential,
    Repeat,
    PerDimScale,
    StackingOverTime,
    StochasticResidual,
    TemporalShifting,
    MultitaskResidualAdapter,
    LayerwiseShardablePipelined,
    BregmanPCA,
    AutodiffCheckpointType,
    Ngrammer,
    VQNgrammer,
    MaskedLmDataAugmenter,
    SpectrumAugmenter,
)


def test_identity():
    """test_identity docstring."""
    layer = Identity()
    inputs = np.array([1, 2, 3])
    np.testing.assert_allclose(layer(inputs), inputs)


def test_bias():
    """test_bias docstring."""
    layer = Bias(dims=3)
    inputs = np.array([[1, 2, 3]])
    b = np.array([1, 1, 1])
    out = layer(inputs, b=b)
    np.testing.assert_allclose(out, [[2, 3, 4]])

    out_def = layer(inputs)
    np.testing.assert_allclose(out_def, [[1, 2, 3]])


def test_dropout():
    """test_dropout docstring."""
    layer = Dropout(keep_prob=0.5)
    inputs = np.ones((2, 2))
    out = layer(inputs)
    np.testing.assert_allclose(out, np.ones((2, 2)) * 0.5)

    layer_id = Dropout(keep_prob=1.0)
    out_id = layer_id(inputs)
    np.testing.assert_allclose(out_id, inputs)


def test_einsum():
    """test_einsum docstring."""
    layer = Einsum(equation="ij,jk->ik")
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    out = layer(a, b)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, np.dot(a, b))

    layer_def = Einsum()
    out_def = layer_def(a, b)
    np.testing.assert_allclose(out_def, a)

    layer_op = EinsumOp(equation="ij,jk->ik")
    out_op = layer_op(a, b)
    assert out_op.shape == (2, 4)

    layer_op_def = EinsumOp()
    out_op_def = layer_op_def(a, b)
    np.testing.assert_allclose(out_op_def, a)


def test_fullsoftmax():
    """test_fullsoftmax docstring."""
    layer = FullSoftmax()
    logits = np.array([[0.0, 1.0, 0.0]])
    out = layer(logits)
    assert np.allclose(np.sum(out, axis=-1), 1.0)


def test_sigmoid_cross_entropy():
    """test_sigmoid_cross_entropy docstring."""
    layer = SigmoidCrossEntropy()
    logits = np.array([[-1.0, 2.0]])
    labels = np.array([[0.0, 1.0]])
    out = layer(logits, labels)
    assert out.shape == (1, 2)


def test_bitempered_loss():
    """test_bitempered_loss docstring."""
    layer = BiTemperedLoss()
    logits = np.array([[-1.0, 2.0]])
    labels = np.array([[0.0, 1.0]])
    out = layer(logits, labels)
    assert out == 0.0


def test_sequential():
    """test_sequential docstring."""

    class DummyAdd(object):
        """DummyAdd docstring."""

        def __call__(self, x):
            """__call__ docstring."""
            return x + 1

    layer = Sequential(layers=[DummyAdd(), DummyAdd()])
    out = layer(np.array([0]))
    assert out[0] == 2

    layer_empty = Sequential()
    out_empty = layer_empty(np.array([0]))
    assert out_empty[0] == 0


def test_repeat():
    """test_repeat docstring."""

    class DummyAdd(object):
        """DummyAdd docstring."""

        def __call__(self, x):
            """__call__ docstring."""
            return x + 1

    layer = Repeat(sub_layer=DummyAdd(), num_repeats=3)
    out = layer(np.array([0]))
    assert out[0] == 3

    layer_empty = Repeat()
    out_empty = layer_empty(np.array([0]))
    assert out_empty[0] == 0


def test_perdimscale():
    """test_perdimscale docstring."""
    layer = PerDimScale(dims=3)
    inputs = np.ones((2, 3))
    scale = np.array([1, 2, 3])
    out = layer(inputs, scale=scale)
    np.testing.assert_allclose(out, [[1, 2, 3], [1, 2, 3]])

    out_def = layer(inputs)
    np.testing.assert_allclose(out_def, inputs)


def test_stackingovertime():
    """test_stackingovertime docstring."""
    layer = StackingOverTime(left_context=1, right_context=1, stride=2)
    inputs = np.ones((2, 5, 3))  # B, T, C
    out = layer(inputs)
    assert out.shape == (2, 3, 9)


def test_stochasticresidual():
    """test_stochasticresidual docstring."""
    layer = StochasticResidual(residual_weight=0.5)
    inputs = np.ones((2, 2))
    residual = np.ones((2, 2)) * 2
    out = layer(inputs, residual)
    np.testing.assert_allclose(out, np.ones((2, 2)) * 2.0)


def test_temporalshifting():
    """test_temporalshifting docstring."""
    layer = TemporalShifting(shift=1)
    inputs = np.ones((2, 5, 3))
    out = layer(inputs)
    assert out.shape == (2, 5, 3)

    layer_neg = TemporalShifting(shift=-1)
    out_neg = layer_neg(inputs)
    assert out_neg.shape == (2, 5, 3)

    layer_zero = TemporalShifting(shift=0)
    out_zero = layer_zero(inputs)
    np.testing.assert_allclose(out_zero, inputs)


def test_routing_mocks():
    """test_routing_mocks docstring."""
    inputs = np.ones((2, 2))
    for cls in [
        MultitaskResidualAdapter,
        LayerwiseShardablePipelined,
        BregmanPCA,
        AutodiffCheckpointType,
        Ngrammer,
        VQNgrammer,
        MaskedLmDataAugmenter,
        SpectrumAugmenter,
    ]:
        layer = cls()
        out = layer(inputs)
        np.testing.assert_allclose(out, inputs)
