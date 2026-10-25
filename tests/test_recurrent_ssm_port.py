"""Tests for recurrent and State Space Model layers in zero_pax.praxis.layers.recurrent_ssm."""

from __future__ import annotations

import numpy as np

from zero_pax.praxis.layers import (
    SSM,
    CifgLstmCellSimple,
    FRnn,
    LayerNorm,
    LayerNormalizedLstmCellSimple,
    Linear,
    LstmCellSimple,
    LstmFrnn,
    ReLU,
    SSMGated,
    SSMTransformer,
    StackFrnn,
)


def test_lstm_cells() -> None:
    """Verify `LstmCellSimple`, `CifgLstmCellSimple`, and `LayerNormalizedLstmCellSimple` numerically."""
    x = np.ones((2, 4), dtype=np.float32)
    c0 = np.ones((2, 4), dtype=np.float32) * 0.5
    m0 = np.zeros((2, 4), dtype=np.float32)
    pad = np.array([[0.0], [1.0]], dtype=np.float32)

    cell = LstmCellSimple(
        num_input_nodes=4,
        num_output_nodes=4,
        forget_gate_bias=1.0,
        cell_value_cap=5.0,
        reset_cell_state=True,
    )
    c1, m1 = cell(
        x,
        state0=(c0, m0),
        padding=pad,
        reset_mask=np.array([[1.0], [1.0]], dtype=np.float32),
    )
    f_expected = 1.0 / (1.0 + np.exp(-1.0))
    o_expected = 0.5
    c_expected_0 = f_expected * 0.5
    m_expected_0 = o_expected * np.tanh(c_expected_0)
    np.testing.assert_allclose(c1[0], np.full((4,), c_expected_0), atol=1e-5)
    np.testing.assert_allclose(m1[0], np.full((4,), m_expected_0), atol=1e-5)
    np.testing.assert_allclose(c1[1], c0[1], atol=1e-5)
    np.testing.assert_allclose(m1[1], m0[1], atol=1e-5)

    c_reset, _ = cell(
        x,
        state0=(c0, m0),
        reset_mask=np.zeros((2, 1), dtype=np.float32),
    )
    np.testing.assert_allclose(c_reset, np.zeros((2, 4)), atol=1e-5)

    c2, m2 = cell(x, state0=np.zeros((2, 8), dtype=np.float32))
    assert c2.shape == (2, 4) and m2.shape == (2, 4)
    c3, m3 = cell(np.zeros((2, 12), dtype=np.float32))
    assert c3.shape == (2, 4) and m3.shape == (2, 4)

    cifg = CifgLstmCellSimple(num_input_nodes=4, num_output_nodes=4, output_nonlinearity=False)
    wm_cifg = np.ones((8, 12), dtype=np.float32) * 0.5
    c_cifg, m_cifg = cifg(x, state0=(c0, m0), wm=wm_cifg)
    g_val = np.tanh(2.0)
    f_val = 1.0 / (1.0 + np.exp(-2.0))
    i_val = 1.0 - f_val
    o_val = f_val
    expected_c_cifg = f_val * 0.5 + i_val * g_val
    np.testing.assert_allclose(c_cifg, np.full((2, 4), expected_c_cifg), atol=1e-5)
    np.testing.assert_allclose(m_cifg, np.full((2, 4), o_val * expected_c_cifg), atol=1e-5)

    ln_lstm = LayerNormalizedLstmCellSimple(
        num_input_nodes=4, num_output_nodes=4, reset_cell_state=True
    )
    w_ln = np.arange(1, 129, dtype=np.float32).reshape((8, 16)) * 0.05
    c_ln, m_ln = ln_lstm(
        x, state0=(c0, m0), wm=w_ln, padding=pad, reset_mask=np.ones((2, 1), dtype=np.float32)
    )
    assert c_ln.shape == (2, 4) and m_ln.shape == (2, 4)
    np.testing.assert_allclose(c_ln[1], c0[1], atol=1e-5)
    assert np.any(np.asarray(m_ln[0]) != 0.0)


def test_frnn_lstmfrnn_and_stackfrnn() -> None:
    """Verify `FRnn`, `LstmFrnn`, and `StackFrnn` sequence unrolling and padding preservation."""
    seq = np.ones((2, 4, 3), dtype=np.float32)
    paddings = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    frnn = FRnn(hidden_size=3, reverse=False)
    out_frnn = frnn(seq, paddings=paddings)
    s0 = np.tanh(np.ones((3,)))
    s1 = np.tanh(s0 + 1.0)
    np.testing.assert_allclose(out_frnn[0, 0], s0, atol=1e-5)
    np.testing.assert_allclose(out_frnn[0, 1], s1, atol=1e-5)
    np.testing.assert_allclose(out_frnn[0, 2], s1, atol=1e-5)
    np.testing.assert_allclose(out_frnn[0, 3], s1, atol=1e-5)

    frnn_rev = FRnn(hidden_size=3, reverse=True, cell_tpl=ReLU())
    out_rev = frnn_rev(seq)
    np.testing.assert_allclose(out_rev[0, 3], np.full((3,), 1.0), atol=1e-5)
    np.testing.assert_allclose(out_rev[0, 0], np.full((3,), 4.0), atol=1e-5)

    frnn_lstm_cell = FRnn(
        hidden_size=3, cell_tpl=LstmCellSimple(num_input_nodes=3, num_output_nodes=3), reverse=True
    )
    assert frnn_lstm_cell(seq, paddings=paddings).shape == (2, 4, 3)

    lfrnn = LstmFrnn(num_input_nodes=3, num_output_nodes=3, reverse=False)
    wm = np.ones((6, 12), dtype=np.float32) * 0.2
    out_lf = lfrnn(seq, paddings=paddings, wm=wm)
    np.testing.assert_allclose(out_lf[0, 1], out_lf[0, 2], atol=1e-5)
    np.testing.assert_allclose(out_lf[0, 1], out_lf[0, 3], atol=1e-5)

    lfrnn_rev = LstmFrnn(num_input_nodes=3, num_output_nodes=3, reverse=True)
    assert lfrnn_rev(seq, state0=(np.zeros((2, 3)), np.zeros((2, 3))), wm=wm).shape == (2, 4, 3)

    sfrnn = StackFrnn(num_input_nodes=3, num_output_nodes=3, num_layers=2)
    out_stack = sfrnn(seq, paddings=paddings)
    assert out_stack.shape == (2, 4, 3)
    np.testing.assert_allclose(out_stack[0, 1], out_stack[0, 3], atol=1e-5)


def test_ssm_and_gated_variants() -> None:
    """Verify `SSM`, `SSMGated`, and `SSMTransformer` discretized state space recurrences."""
    inputs = np.ones((2, 4, 4), dtype=np.float32)
    paddings = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    ssm = SSM(nheads=2, head_dim=2, ssm_dim=2, hidden_size=4, hippo_type="legt")
    c_mat = np.ones((2, 4), dtype=np.float32)
    out = ssm(inputs, paddings=paddings, c_mat=c_mat)
    assert out.shape == (2, 4, 4)
    np.testing.assert_allclose(out[0, 1], out[0, 2], atol=1e-5)
    assert SSM(hippo_type="lagt")(inputs).shape == (2, 4, 4)

    ssmg = SSMGated(
        hidden_size=2,
        activation_fn="swish",
        ln_tpl=LayerNorm(dim=4),
        gss_fflayer_tpl=Linear(input_dims=4, output_dims=2),
    )
    out_g = ssmg(inputs, paddings=paddings)
    np.testing.assert_allclose(out_g, np.zeros((2, 4, 2)), atol=1e-5)
    np.testing.assert_allclose(SSMGated()(inputs), np.zeros((2, 4, 4)), atol=1e-5)

    ssmt = SSMTransformer(
        hidden_size=4,
        tr_atten_tpl=ReLU(),
        tr_fflayer_tpl=ReLU(),
    )
    out_t = ssmt(inputs)
    np.testing.assert_allclose(out_t, np.zeros((2, 4, 4)), atol=1e-5)
