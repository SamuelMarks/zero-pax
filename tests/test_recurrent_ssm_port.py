"""Tests for the test_recurrent_ssm_port module."""

import numpy as np
import pytest
from zero_pax.praxis.layers import (
    LstmCellSimple,
    CifgLstmCellSimple,
    LayerNormalizedLstmCellSimple,
    FRnn,
    LstmFrnn,
    StackFrnn,
    SSM,
    SSMGated,
)


def test_lstmcellsimple():
    """Executes the test_lstmcellsimple test.

    Returns:
        The result of the test.
    """
    layer = LstmCellSimple(hidden_size=4)
    m = np.random.normal(size=(2, 4)).astype(np.float32)
    c = np.random.normal(size=(2, 4)).astype(np.float32)
    act = np.random.normal(size=(2, 8)).astype(np.float32)
    wm = np.random.normal(size=(12, 16)).astype(np.float32)
    b = np.random.normal(size=(16,)).astype(np.float32)

    (new_m, new_c), out = layer((m, c), act, wm=wm, b=b)
    assert new_m.shape == (2, 4)
    assert new_c.shape == (2, 4)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(new_m, out)

    # testing padding and reset mask independently
    pad = np.array([[1], [0]]).astype(np.float32)
    (pm, pc), pout = layer((m, c), act, padding=pad)
    np.testing.assert_allclose(pc[0], c[0])
    np.testing.assert_allclose(pm[0], m[0])

    reset = np.array([[1], [0]]).astype(np.float32)
    (rm, rc), rout = layer((m, c), act, reset_mask=reset)
    # The output is definitely not c[0] because it evaluated with 0 incoming state.
    # We just ensure it executed cleanly.
    assert rc.shape == (2, 4)


def test_cifglstmcellsimple():
    """Executes the test_cifglstmcellsimple test.

    Returns:
        The result of the test.
    """
    layer = CifgLstmCellSimple(hidden_size=4)
    m = np.random.normal(size=(2, 4)).astype(np.float32)
    c = np.random.normal(size=(2, 4)).astype(np.float32)
    act = np.random.normal(size=(2, 8)).astype(np.float32)
    wm = np.random.normal(size=(12, 12)).astype(np.float32)

    (new_m, new_c), out = layer((m, c), act, wm=wm)
    assert new_m.shape == (2, 4)

    pad = np.array([[1], [0]]).astype(np.float32)
    (pm, pc), _ = layer((m, c), act, padding=pad)
    np.testing.assert_allclose(pc[0], c[0])

    reset = np.array([[1], [0]]).astype(np.float32)
    (rm, rc), _ = layer((m, c), act, reset_mask=reset)
    assert rc.shape == (2, 4)


def test_layernormalizedlstmcellsimple():
    """Executes the test_layernormalizedlstmcellsimple test.

    Returns:
        The result of the test.
    """
    layer = LayerNormalizedLstmCellSimple(hidden_size=4)
    m = np.random.normal(size=(2, 4)).astype(np.float32)
    c = np.random.normal(size=(2, 4)).astype(np.float32)
    act = np.random.normal(size=(2, 8)).astype(np.float32)

    (new_m, new_c), out = layer((m, c), act)
    assert new_m.shape == (2, 4)

    pad = np.array([[1], [0]]).astype(np.float32)
    (pm, pc), _ = layer((m, c), act, padding=pad)
    np.testing.assert_allclose(pc[0], c[0])

    reset = np.array([[1], [0]]).astype(np.float32)
    (rm, rc), _ = layer((m, c), act, reset_mask=reset)
    assert rc.shape == (2, 4)


def test_frnn():
    """Executes the test_frnn test.

    Returns:
        The result of the test.
    """
    layer = FRnn(hidden_size=4)
    inputs = np.random.normal(size=(2, 5, 8)).astype(np.float32)
    out = layer(inputs)
    assert out.shape == (2, 5, 4)

    state0 = np.random.normal(size=(2, 4)).astype(np.float32)
    out2 = layer(inputs, state0=state0)
    assert out2.shape == (2, 5, 4)


def test_lstmfrnn():
    """Executes the test_lstmfrnn test.

    Returns:
        The result of the test.
    """
    layer = LstmFrnn(hidden_size=4)
    inputs = np.random.normal(size=(2, 5, 8)).astype(np.float32)
    out = layer(inputs)
    assert out.shape == (2, 5, 4)

    # testing state0
    m = np.random.normal(size=(2, 4)).astype(np.float32)
    c = np.random.normal(size=(2, 4)).astype(np.float32)
    out2 = layer(inputs, state0=(m, c))
    assert out2.shape == (2, 5, 4)


def test_stackfrnn():
    """Executes the test_stackfrnn test.

    Returns:
        The result of the test.
    """
    layer = StackFrnn(hidden_size=4, num_layers=2)
    inputs = np.random.normal(size=(2, 5, 8)).astype(np.float32)
    out = layer(inputs)
    assert out.shape == (2, 5, 4)


def test_ssm():
    """Executes the test_ssm test.

    Returns:
        The result of the test.
    """
    layer = SSM(hidden_size=4)
    inputs = np.random.normal(size=(2, 5, 8)).astype(np.float32)
    out = layer(inputs)
    assert out.shape == (2, 5, 4)


def test_ssmgated():
    """Executes the test_ssmgated test.

    Returns:
        The result of the test.
    """
    layer = SSMGated(hidden_size=4)
    inputs = np.random.normal(size=(2, 5, 8)).astype(np.float32)
    out = layer(inputs)
    assert out.shape == (2, 5, 4)
