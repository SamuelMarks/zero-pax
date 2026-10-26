"""Tests for recurrent and State Space Model layers in zero_pax.praxis.layers.recurrent_ssm."""

from __future__ import annotations

from tests import conftest as np
from zero_pax.praxis.layers import (
    SSM,
    CifgLstmCellSimple,
    FRnn,
    Identity,
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
from zero_pax.types import WeightInit


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

    lfrnn_rev = LstmFrnn(
        num_input_nodes=3,
        num_output_nodes=3,
        reverse=True,
        cell_tpl=CifgLstmCellSimple(num_input_nodes=3, num_output_nodes=3),
    )
    out_lfrnn_rev = lfrnn_rev(
        seq, state0=(np.zeros((2, 3)), np.zeros((2, 3))), wm=np.ones((6, 9), dtype=np.float32) * 0.2
    )
    assert out_lfrnn_rev.shape == (2, 4, 3)
    assert np.any(np.asarray(out_lfrnn_rev) != 0.0)

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
    out_lagt = SSM(hippo_type="lagt", params_init=WeightInit(method="ones", scale=0.5))(inputs)
    assert out_lagt.shape == (2, 4, 4)
    assert np.any(np.asarray(out_lagt) != 0.0)
    assert SSM(hippo_type="legs", l_max=2)(inputs).shape == (2, 4, 4)
    assert SSM(hippo_type="fourier")(inputs).shape == (2, 4, 4)

    var_inputs = np.arange(1, 33, dtype=np.float32).reshape((2, 4, 4)) * 0.1
    ssmg = SSMGated(
        hidden_size=2,
        activation_fn="swish",
        ln_tpl=LayerNorm(dim=4),
        gss_fflayer_tpl=Linear(
            input_dims=4, output_dims=2, weight_init=WeightInit(method="ones", scale=0.5)
        ),
        params_init=WeightInit(method="ones", scale=0.5),
    )
    out_g = ssmg(var_inputs, paddings=paddings)
    assert out_g.shape == (2, 4, 2)
    assert np.any(np.asarray(out_g) != 0.0)
    np.testing.assert_allclose(SSMGated()(inputs), np.zeros((2, 4, 4)), atol=1e-5)

    ssmt = SSMTransformer(
        hidden_size=4,
        tr_atten_tpl=ReLU(),
        tr_fflayer_tpl=ReLU(),
        params_init=WeightInit(method="ones", scale=0.5),
    )
    out_t = ssmt(var_inputs)
    assert out_t.shape == (2, 4, 4)
    assert np.any(np.asarray(out_t) != 0.0)

    lstm_tuple_inferred = LstmCellSimple(num_output_nodes=0, hidden_size=0)
    c_inf, m_inf = lstm_tuple_inferred(
        np.ones((2, 3), dtype=np.float32),
        state0=(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)),
    )
    np.testing.assert_allclose(c_inf, np.zeros((2, 3)))
    np.testing.assert_allclose(m_inf, np.zeros((2, 3)))

    lstm_no_bias = LstmCellSimple(enable_lstm_bias=False, cell_value_cap=None)
    c_nb, _ = lstm_no_bias(
        np.ones((2, 4), dtype=np.float32),
        state0=np.ones((2, 4), dtype=np.float32),
        padding=np.array([0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(c_nb[0], np.full((4,), 0.5))
    np.testing.assert_allclose(c_nb[1], np.ones((4,)))

    lstm_same_size = LstmCellSimple(hidden_size=2)
    c_ss, _ = lstm_same_size(
        np.ones((2, 2), dtype=np.float32),
        state0=np.ones((2, 2), dtype=np.float32),
    )
    np.testing.assert_allclose(c_ss, np.full((2, 2), 0.5))

    ln_lstm_no_bias = LayerNormalizedLstmCellSimple(enable_lstm_bias=False)
    c_ln_nb, _ = ln_lstm_no_bias(np.ones((2, 12), dtype=np.float32))
    np.testing.assert_allclose(c_ln_nb, np.zeros((2, 4)))

    np.testing.assert_allclose(
        FRnn()(np.ones((2, 4), dtype=np.float32)), np.tanh(np.ones((2, 4))), atol=1e-5
    )
    np.testing.assert_allclose(
        FRnn()(np.ones((4,), dtype=np.float32)), np.tanh(np.ones((4,))), atol=1e-5
    )
    out_frnn_lstm = FRnn(
        cell_tpl=LstmCellSimple(num_input_nodes=3, num_output_nodes=3), reverse=False
    )(np.ones((2, 3, 3), dtype=np.float32))
    np.testing.assert_allclose(out_frnn_lstm, np.zeros((2, 3, 3)))

    np.testing.assert_allclose(
        LstmFrnn()(np.ones((2, 4), dtype=np.float32)), np.zeros((2, 4)), atol=1e-5
    )
    np.testing.assert_allclose(
        StackFrnn(frnn_tpl=ReLU())(np.array([[[-1.0, 2.0]]], dtype=np.float32)),
        np.array([[[0.0, 2.0]]]),
    )
    np.testing.assert_allclose(SSM()(np.ones((2, 4), dtype=np.float32)), np.ones((2, 4)), atol=1e-5)
    ssm_bi = SSM(hidden_size=3, bidirectional=True)
    out_ssm_bi = ssm_bi(
        np.ones((2, 3, 3), dtype=np.float32),
        paddings=np.zeros((2, 3), dtype=np.float32),
        a_diag=np.array([-1.0]),
        b_mat=np.ones((3, 1)),
        c_mat=np.ones((1, 3)),
        d_vec=np.eye(3),
    )
    assert out_ssm_bi.shape == (2, 3, 3) and np.all(np.asarray(out_ssm_bi) > 1.0)

    out_ssm_bi_nopad = ssm_bi(
        np.ones((2, 3, 3), dtype=np.float32),
        a_diag=np.array([-1.0]),
        b_mat=np.ones((3, 1)),
        c_mat=np.ones((1, 3)),
        d_vec=np.eye(3),
    )
    np.testing.assert_allclose(out_ssm_bi_nopad, out_ssm_bi, atol=1e-5)

    out_ssmg = SSMGated(
        hidden_size=2,
        ssm_tpl=Identity(),
        gss_fflayer_tpl=Linear(input_dims=4, output_dims=2, weight_init=WeightInit(method="ones")),
        params_init=WeightInit(method="ones", scale=0.5),
    )(np.arange(1, 25, dtype=np.float32).reshape((2, 3, 4)) * 0.1)
    assert out_ssmg.shape == (2, 3, 2)
    assert np.any(np.asarray(out_ssmg) != 0.0)

    out_ssmt_real = SSMTransformer(
        hidden_size=3,
        params_init=WeightInit(method="ones", scale=0.5),
    )(np.arange(1, 19, dtype=np.float32).reshape((2, 3, 3)) * 0.1)
    assert out_ssmt_real.shape == (2, 3, 3)
    assert np.any(np.asarray(out_ssmt_real) != 0.0)

    lstm_zo = LstmCellSimple(
        inputs_arity=2,
        num_input_nodes=2,
        input_size=2,
        num_hidden_nodes=2,
        zo_prob=0.5,
        params_init=WeightInit(method="ones", scale=0.25),
    )
    c_zo, m_zo = lstm_zo(
        np.ones((2, 2), dtype=np.float32),
        state0=(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)),
    )
    assert c_zo.shape == (2, 2) and m_zo.shape == (2, 2)
    assert "inputs_arity" in lstm_zo.params

    frnn_unroll = FRnn(unroll=2, hidden_size=3)
    assert frnn_unroll(np.ones((2, 3, 3), dtype=np.float32)).shape == (2, 3, 3)
    assert "unroll_factor" in frnn_unroll.params

    stack_drop = StackFrnn(num_input_nodes=3, num_output_nodes=3, num_layers=2, dropout_prob=0.2)
    assert stack_drop(np.ones((2, 3, 3), dtype=np.float32)).shape == (2, 3, 3)

    ssm_heads = SSM(nheads=2, head_dim=2, decode_num_samples=2)
    assert ssm_heads(np.ones((2, 3, 4), dtype=np.float32)).shape == (2, 3, 4)

    ssmg_full = SSMGated(
        dss_dim=4,
        gss_mlp_dim=4,
        packed_input=True,
        allow_skip_cross_attention=True,
        relu_dropout_prob=0.1,
        ngrammer_tpl=Identity(),
        norm_policy="post",
        ssm_l_max=2,
        use_cross_attention=True,
        residual_dropout_prob=0.1,
        residual_droppath_prob=0.1,
        params_init=WeightInit(method="ones", scale=0.25),
    )
    assert ssmg_full(var_inputs).shape == (2, 4, 4)
    assert "ssm_gated_meta" in ssmg_full.params
