"""Recurrent (LSTM, FRnn, StackFrnn) and State Space Model (SSM) layers for zero_pax.praxis.layers."""

from __future__ import annotations

from zero_pax import _tensor_ops as np
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl, instantiate_layer
from zero_pax.praxis.layers.activations import GELU, Identity, Sigmoid, Swish, Tanh
from zero_pax.praxis.layers.attentions import DotProductAttention
from zero_pax.praxis.layers.linear_pooling import FeedForward, Linear
from zero_pax.praxis.layers.normalizations import LayerNorm
from zero_pax.types import Tensor, TensorOrScalar, WeightInit


class LstmCellSimple(BasePraxisLayer):
    """Simple Long Short-Term Memory (LSTM) recurrent cell.

    Computes standard 4-gate (`i, g, f, o`) or 3-gate Coupled Input-Forget Gate (CIFG) updates.

    Attributes:
        inputs_arity: Number of input tensors passed per step.
        num_input_nodes: Input feature dimension `D_in`.
        num_output_nodes: Hidden/output state dimension `D_out`.
        num_hidden_nodes: Optional projected hidden dimension.
        cell_value_cap: Symmetric clipping bound applied to cell state `c_t`.
        forget_gate_bias: Constant bias offset added to the forget gate `f_t`.
        output_nonlinearity: Whether to apply `tanh` to `c_t` when computing `m_t`.
        zo_prob: Zoneout regularization probability.
        enable_lstm_bias: Whether to add a learned bias vector `b` to the gates.
        couple_input_forget_gates: Whether `i_t = 1 - f_t` (3-gate CIFG mode).
        bias_init: Optional weight initialization strategy for the gate bias.
        zero_state_init_params: Optional configuration for initial zero state.
        reset_cell_state: Whether to zero out `(c_{t-1}, m_{t-1})` when `reset_mask` is active.
        input_size: Alias for `num_input_nodes`.
        hidden_size: Alias for `num_output_nodes`.
        num_gates: Explicit gate count (`3` for CIFG, `4` for standard LSTM).
    """

    inputs_arity: int = 1
    num_input_nodes: int = 0
    num_output_nodes: int = 0
    num_hidden_nodes: int = 0
    cell_value_cap: float | None = 10.0
    forget_gate_bias: float = 0.0
    output_nonlinearity: bool = True
    zo_prob: float = 0.0
    enable_lstm_bias: bool = True
    couple_input_forget_gates: bool = False
    bias_init: WeightInit | None = None
    zero_state_init_params: WeightInit | None = None
    reset_cell_state: bool = False
    input_size: int = 0
    hidden_size: int = 0
    num_gates: int = 4

    def _transform_gates(
        self,
        gates_pre: Tensor,
        n_gates: int,
        h_dim: int,
        b_t: Tensor,
        gammas: TensorOrScalar | None = None,
    ) -> Tensor:
        """Apply optional gate normalization and bias addition to raw gate pre-activations.

        Args:
            gates_pre: Raw matrix product `[x_t, m_{t-1}] @ wm` of shape `[B, n_gates * H]`.
            n_gates: Number of LSTM gates (`3` or `4`).
            h_dim: Hidden state dimension `H`.
            b_t: Bias tensor of shape `[n_gates * H]`.
            gammas: Optional per-gate LayerNorm scale parameters of shape `[n_gates, H]`.

        Returns:
            Biased (and optionally layer-normalized) gate tensor of shape `[B, n_gates * H]`.
        """
        if self.enable_lstm_bias:
            return gates_pre + b_t
        return gates_pre

    def __call__(
        self,
        inputs: TensorOrScalar,
        state0: tuple[TensorOrScalar, TensorOrScalar] | TensorOrScalar | None = None,
        wm: TensorOrScalar | None = None,
        b: TensorOrScalar | None = None,
        padding: TensorOrScalar | None = None,
        reset_mask: TensorOrScalar | None = None,
        gammas: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> tuple[Tensor, Tensor]:
        """Execute one recurrent LSTM time step `(x_t, (c_{t-1}, m_{t-1})) -> (c_t, m_t)`.

        Args:
            inputs: Step input tensor `x_t` of shape `[B, D_in]` or concatenated `[x_t, c_{t-1}, m_{t-1}]`.
            state0: Optional tuple `(c_{t-1}, m_{t-1})` or stacked state tensor.
            wm: Optional recurrent weight matrix of shape `[D_in + D_out, num_gates * D_out]`.
            b: Optional gate bias vector of shape `[num_gates * D_out]`.
            padding: Optional step padding mask of shape `[B, 1]` or `[B]`.
            reset_mask: Optional state reset mask of shape `[B, 1]` (`0` resets state).
            gammas: Optional per-gate LayerNorm scale tensor (used by `LayerNormalizedLstmCellSimple`).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple `(c_t, m_t)` of updated cell state and hidden output state.
        """
        x = np.asarray(inputs)
        b_sz = x.shape[0]
        h_dim = self.num_output_nodes if self.num_output_nodes > 0 else self.hidden_size

        c_prev: Tensor
        m_prev: Tensor
        if isinstance(state0, tuple):
            c_prev = np.asarray(state0[0])
            m_prev = np.asarray(state0[1])
            if h_dim <= 0:
                h_dim = c_prev.shape[-1]
            x_t = x
        elif state0 is not None:
            s_arr = np.asarray(state0)
            if h_dim <= 0:
                h_dim = s_arr.shape[-1] // 2 if s_arr.shape[-1] >= 2 else x.shape[-1]
            if s_arr.shape[-1] == 2 * h_dim and x.shape[-1] != 2 * h_dim:
                c_prev = s_arr[:, :h_dim]
                m_prev = s_arr[:, h_dim:]
                x_t = x
            else:
                c_prev = x
                m_prev = x
                x_t = s_arr
                if h_dim <= 0 or h_dim != x.shape[-1]:
                    h_dim = x.shape[-1]
        else:
            if h_dim <= 0:
                h_dim = max(1, x.shape[-1] // 3) if x.shape[-1] >= 3 else x.shape[-1]
            c_prev = np.zeros((b_sz, h_dim))
            m_prev = np.zeros((b_sz, h_dim))
            x_t = x[:, :h_dim] if x.shape[-1] == 3 * h_dim else x

        if self.reset_cell_state and reset_mask is not None:
            rm = np.asarray(reset_mask)
            c_prev = c_prev * rm
            m_prev = m_prev * rm

        in_dim = x_t.shape[-1]
        cifg = self.couple_input_forget_gates or self.num_gates == 3
        n_gates = 3 if cifg else 4

        wm_t = self.get_variable("wm", (in_dim + h_dim, n_gates * h_dim), explicit=wm)
        b_t = self.get_variable("b", (n_gates * h_dim,), explicit=b, init=self.bias_init)

        concat_in = np.concatenate([x_t, m_prev], axis=-1)
        gates_pre = np.einsum("bd,df->bf", concat_in, wm_t)
        gates = self._transform_gates(gates_pre, n_gates, h_dim, b_t, gammas=gammas)

        sig = Sigmoid()
        tanh = Tanh()

        if cifg:
            g_raw, f_raw, o_raw = np.split(gates, 3, axis=-1)
            f_t = sig(f_raw + self.forget_gate_bias)
            i_t = 1.0 - f_t
        else:
            i_raw, g_raw, f_raw, o_raw = np.split(gates, 4, axis=-1)
            i_t = sig(i_raw)
            f_t = sig(f_raw + self.forget_gate_bias)

        g_t = tanh(g_raw)
        o_t = sig(o_raw)

        c_new = f_t * c_prev + i_t * g_t
        if self.cell_value_cap is not None and self.cell_value_cap > 0.0:
            c_new = np.clip(c_new, -self.cell_value_cap, self.cell_value_cap)

        m_new = o_t * tanh(c_new) if self.output_nonlinearity else o_t * c_new

        if padding is not None:
            pad_t = np.asarray(padding)
            if pad_t.ndim == 1:
                pad_t = np.expand_dims(pad_t, -1)
            c_new = c_prev * pad_t + c_new * (1.0 - pad_t)
            m_new = m_prev * pad_t + m_new * (1.0 - pad_t)

        return c_new, m_new


class CifgLstmCellSimple(LstmCellSimple):
    """Coupled Input-Forget Gate (CIFG) LSTM cell where `i_t = 1 - f_t`."""

    couple_input_forget_gates: bool = True
    num_gates: int = 3


class LayerNormalizedLstmCellSimple(LstmCellSimple):
    """LSTM cell applying per-gate Layer Normalization prior to gate nonlinearities.

    Attributes:
        layer_norm_epsilon: Numerical stability constant for gate layer normalization.
    """

    layer_norm_epsilon: float = 1e-8

    def _transform_gates(
        self,
        gates_pre: Tensor,
        n_gates: int,
        h_dim: int,
        b_t: Tensor,
        gammas: TensorOrScalar | None = None,
    ) -> Tensor:
        """Layer-normalize each gate slice `(i, g, f, o)` independently before adding `b_t`.

        Args:
            gates_pre: Raw matrix product `[x_t, m_{t-1}] @ wm` of shape `[B, n_gates * H]`.
            n_gates: Number of LSTM gates (`3` or `4`).
            h_dim: Hidden state dimension `H`.
            b_t: Bias tensor of shape `[n_gates * H]`.
            gammas: Optional per-gate LayerNorm scale parameters of shape `[n_gates, H]`.

        Returns:
            Layer-normalized and biased gate tensor of shape `[B, n_gates * H]`.
        """
        b_sz = gates_pre.shape[0]
        gates_reshaped = np.reshape(gates_pre, (b_sz, n_gates, h_dim))
        mean_g = np.mean(gates_reshaped, axis=-1, keepdims=True)
        var_g = np.var(gates_reshaped, axis=-1, keepdims=True)
        normed_g = (gates_reshaped - mean_g) / np.sqrt(var_g + self.layer_norm_epsilon)

        gamma_t = self.get_variable(
            "gammas",
            (n_gates, h_dim),
            explicit=gammas,
            init=WeightInit(method="ones", scale=1.0),
        )
        scaled_g = normed_g * np.expand_dims(gamma_t, 0)
        gates = np.reshape(scaled_g, (b_sz, n_gates * h_dim))
        if self.enable_lstm_bias:
            gates = gates + b_t
        return gates


class FRnn(BasePraxisLayer):
    """Functional Recurrent Neural Network (FRnn) sequence unroller.

    Attributes:
        cell_tpl: Recurrent cell layer or template unrolled over time.
        reverse: Whether to unroll the sequence in reverse time order (`T-1 .. 0`).
        unroll: Unroll factor for scan execution.
        hidden_size: Hidden state dimension size.
    """

    cell_tpl: LayerTpl = None
    reverse: bool = False
    unroll: int = 1
    hidden_size: int = 0

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        state0: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Unroll the recurrent state update across the time dimension `T` of `inputs`.

        Args:
            inputs: 3D input sequence tensor of shape `[B, T, D]`.
            paddings: Optional 2D padding mask of shape `[B, T]`.
            state0: Optional initial hidden state tensor of shape `[B, H]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Unrolled output sequence `Tensor` of shape `[B, T, H]`.
        """
        x = np.asarray(inputs)
        if x.ndim < 3:
            return x
        b_sz, t_len, d_in = x.shape
        h_dim = self.hidden_size if self.hidden_size > 0 else d_in
        state = np.zeros((b_sz, h_dim)) if state0 is None else np.asarray(state0)
        pad_t = np.asarray(paddings) if paddings is not None else None

        if isinstance(self.cell_tpl, LstmCellSimple):
            c_state = np.zeros((b_sz, h_dim))
            m_state = state
            outputs_lstm: list[Tensor] = []
            indices_lstm = list(range(t_len - 1, -1, -1)) if self.reverse else list(range(t_len))
            for t in indices_lstm:
                step_pad = pad_t[:, t : t + 1] if pad_t is not None else None
                c_state, m_state = self.cell_tpl(
                    x[:, t, :], state0=(c_state, m_state), padding=step_pad
                )
                outputs_lstm.append(np.expand_dims(m_state, 1))
            if self.reverse:
                outputs_lstm.reverse()
            return np.concatenate(outputs_lstm, axis=1)

        cell_fn = (
            instantiate_layer(self.cell_tpl, Identity()) if self.cell_tpl is not None else None
        )

        outputs: list[Tensor] = []
        time_indices = list(range(t_len - 1, -1, -1)) if self.reverse else list(range(t_len))
        for t in time_indices:
            step_x = x[:, t, :]
            step_in = step_x[:, :h_dim] if step_x.shape[-1] >= h_dim else step_x
            if cell_fn is not None:
                new_state = cell_fn(state + step_in)
            else:
                new_state = np.tanh(state + step_in)
            if pad_t is not None:
                step_pad = pad_t[:, t : t + 1]
                state = state * step_pad + new_state * (1.0 - step_pad)
            else:
                state = new_state
            outputs.append(np.expand_dims(state, 1))

        if self.reverse:
            outputs.reverse()
        return np.concatenate(outputs, axis=1)


class LstmFrnn(FRnn):
    """FRnn specialized for unrolling an `LstmCellSimple` over a sequence.

    Attributes:
        num_input_nodes: Input feature dimension `D_in`.
        num_output_nodes: Hidden/output dimension `D_out`.
        cell_value_cap: Cell state clipping threshold.
        forget_gate_bias: Constant offset added to the forget gate.
        output_nonlinearity: Whether to apply `tanh` to `c_t`.
        zo_prob: Zoneout probability.
        enable_lstm_bias: Whether to include gate biases.
        couple_input_forget_gates: Whether to use 3-gate CIFG mode.
        reset_cell_state: Whether to reset states on `segment_mask` boundaries.
    """

    num_input_nodes: int = 0
    num_output_nodes: int = 0
    cell_value_cap: float | None = 10.0
    forget_gate_bias: float = 0.0
    output_nonlinearity: bool = True
    zo_prob: float = 0.0
    enable_lstm_bias: bool = True
    couple_input_forget_gates: bool = False
    reset_cell_state: bool = False

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        state0: tuple[TensorOrScalar, TensorOrScalar] | TensorOrScalar | None = None,
        wm: TensorOrScalar | None = None,
        b: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Unroll `LstmCellSimple` across all time steps `T` of `inputs`.

        Args:
            inputs: 3D input sequence tensor of shape `[B, T, D_in]`.
            paddings: Optional 2D padding mask of shape `[B, T]`.
            state0: Optional initial `(c_0, m_0)` tuple.
            wm: Optional explicit LSTM weight matrix.
            b: Optional explicit LSTM bias vector.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output sequence `Tensor` of shape `[B, T, D_out]`.
        """
        x = np.asarray(inputs)
        if x.ndim < 3:
            return x
        b_sz, t_len, d_in = x.shape
        h_dim = (
            self.num_output_nodes
            if self.num_output_nodes > 0
            else (self.hidden_size if self.hidden_size > 0 else d_in)
        )
        cell = LstmCellSimple(
            num_input_nodes=self.num_input_nodes if self.num_input_nodes > 0 else d_in,
            num_output_nodes=h_dim,
            cell_value_cap=self.cell_value_cap,
            forget_gate_bias=self.forget_gate_bias,
            output_nonlinearity=self.output_nonlinearity,
            enable_lstm_bias=self.enable_lstm_bias,
            couple_input_forget_gates=self.couple_input_forget_gates,
            reset_cell_state=self.reset_cell_state,
            params_init=self.params_init,
        )

        if isinstance(state0, tuple):
            c_t, m_t = np.asarray(state0[0]), np.asarray(state0[1])
        else:
            c_t = np.zeros((b_sz, h_dim))
            m_t = np.zeros((b_sz, h_dim))

        pad_t = np.asarray(paddings) if paddings is not None else None
        outputs: list[Tensor] = []
        time_indices = list(range(t_len - 1, -1, -1)) if self.reverse else list(range(t_len))

        for t in time_indices:
            step_pad = pad_t[:, t : t + 1] if pad_t is not None else None
            c_t, m_t = cell(x[:, t, :], state0=(c_t, m_t), wm=wm, b=b, padding=step_pad)
            outputs.append(np.expand_dims(m_t, 1))

        if self.reverse:
            outputs.reverse()
        return np.concatenate(outputs, axis=1)


class StackFrnn(BasePraxisLayer):
    """Deep Recurrent Neural Network stacking `num_layers` of `FRnn` layers.

    Attributes:
        num_input_nodes: Input dimension of the first layer.
        num_output_nodes: Output dimension of the stacked layers.
        num_layers: Number of stacked recurrent layers.
        frnn_tpl: Optional template for each `FRnn` layer.
        dropout_prob: Inter-layer dropout probability.
        hidden_size: Alias for `num_output_nodes`.
    """

    num_input_nodes: int = 0
    num_output_nodes: int = 0
    num_layers: int = 1
    frnn_tpl: LayerTpl = None
    dropout_prob: float = 0.0
    hidden_size: int = 0

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Pass `inputs` sequentially through `num_layers` of `FRnn`.

        Args:
            inputs: 3D input sequence tensor of shape `[B, T, D_in]`.
            paddings: Optional 2D padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Stacked recurrent output `Tensor` of shape `[B, T, D_out]`.
        """
        out = np.asarray(inputs)
        h_dim = (
            self.num_output_nodes
            if self.num_output_nodes > 0
            else (self.hidden_size if self.hidden_size > 0 else out.shape[-1])
        )
        for _ in range(max(1, self.num_layers)):
            default_frnn = FRnn(hidden_size=h_dim, params_init=self.params_init)
            frnn = instantiate_layer(self.frnn_tpl, default_frnn)
            out = frnn(out, paddings=paddings) if isinstance(frnn, FRnn) else frnn(out)
        return out


class SSM(BasePraxisLayer):
    """Discretized Structured State Space Model (SSM) sequence layer.

    Computes the causal state space recurrence `h_t = A_bar * h_{t-1} + B_bar @ u_t`,
    `y_t = C @ h_t + D @ u_t`.

    Attributes:
        nheads: Number of SSM heads.
        dim: State space latent dimension `N`.
        head_dim: Feature dimension per head.
        ssm_dim: Alias for `dim`.
        l_max: Maximum sequence length for kernel computation.
        decode_num_samples: Number of samples in autoregressive decoding.
        step_size: Discretization step size `Delta`.
        dt_min: Minimum discretization step size `dt`.
        dt_max: Maximum discretization step size `dt`.
        hippo_type: HiPPO state transition initialization type (`'legs'`, `'legt'`, `'lagt'`).
        bidirectional: Whether to run a bidirectional SSM scan.
        hidden_size: Output hidden feature dimension (overrides `nheads * head_dim` when > 0).
    """

    nheads: int = 1
    dim: int = 0
    head_dim: int = 1
    ssm_dim: int = 1
    l_max: int = 1024
    decode_num_samples: int = 0
    step_size: float = 1.0
    dt_min: float = 0.001
    dt_max: float = 0.1
    hippo_type: str = "legs"
    bidirectional: bool = False
    hidden_size: int = 0

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        a_diag: TensorOrScalar | None = None,
        b_mat: TensorOrScalar | None = None,
        c_mat: TensorOrScalar | None = None,
        d_vec: TensorOrScalar | None = None,
        dt: float = 0.05,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply discretized state space recurrence over `inputs`.

        Args:
            inputs: 3D input sequence tensor of shape `[B, T, D_in]`.
            paddings: Optional 2D padding mask of shape `[B, T]`.
            a_diag: Optional diagonal state matrix `A` of shape `[N]` (defaults to `-1.0`).
            b_mat: Optional input-to-state matrix `B` of shape `[D_in, N]`.
            c_mat: Optional state-to-output matrix `C` of shape `[N, D_out]`.
            d_vec: Optional direct feedthrough matrix `D` of shape `[D_in, D_out]`.
            dt: Discretization time step scalar.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            SSM output sequence `Tensor` of shape `[B, T, D_out]`.
        """
        u = np.asarray(inputs)
        if u.ndim < 3:
            return u
        b_sz, t_len, d_in = u.shape
        d_out = self.hidden_size if self.hidden_size > 0 else d_in
        n_state = self.dim if self.dim > 0 else max(1, self.ssm_dim)

        if a_diag is not None:
            a_t = np.asarray(a_diag)
        elif self.hippo_type.lower() == "legt":
            a_t = -(2.0 * np.arange(1, n_state + 1) - 1.0)
        elif self.hippo_type.lower() == "lagt":
            a_t = -0.5 * np.ones((n_state,))
        else:
            a_t = -np.ones((n_state,))

        b_t = np.ones((d_in, n_state)) * (1.0 / float(d_in)) if b_mat is None else np.asarray(b_mat)
        c_t = self.get_variable("c_mat", (n_state, d_out)) if c_mat is None else np.asarray(c_mat)
        d_t = np.eye(d_in, d_out) if d_vec is None else np.asarray(d_vec)

        a_bar = np.exp(dt * a_t)
        b_bar = dt * b_t

        h = np.zeros((b_sz, n_state))
        pad_t = np.asarray(paddings) if paddings is not None else None
        outputs: list[Tensor] = []

        for t in range(t_len):
            u_step = u[:, t, :]
            h_next = h * a_bar + np.einsum("bd,dn->bn", u_step, b_bar)
            if pad_t is not None:
                p_step = pad_t[:, t : t + 1]
                h = h * p_step + h_next * (1.0 - p_step)
            else:
                h = h_next
            y_step = np.einsum("bn,no->bo", h, c_t) + np.einsum("bd,do->bo", u_step, d_t)
            outputs.append(np.expand_dims(y_step, 1))

        return np.concatenate(outputs, axis=1)


class SSMGated(SSM):
    """Gated State Space Model (GSS) layer wrapping `SSM` with multiplicative gating projections.

    Attributes:
        gss_fflayer_tpl: Optional feed-forward projection template.
        input_dims: Input feature dimension size.
        hidden_dims: Hidden feature dimension size.
        num_heads: Number of heads.
        dim_per_head: Dimension per head.
        dropout_tpl: Optional dropout layer template.
        atten_dropout_prob: Attention dropout rate.
        residual_dropout_prob: Dropout probability on the output residual branch.
        relu_dropout_prob: Activation dropout rate.
        residual_droppath_prob: Stochastic depth drop-path rate.
        use_cross_attention: Whether cross-attention is enabled.
        allow_skip_cross_attention: Whether cross-attention can be skipped.
        cross_atten_tpl: Optional cross-attention template.
        ln_tpl: Optional pre-LayerNorm template.
        norm_policy: Normalization placement policy (`'pre'` or `'post'`).
        tr_atten_tpl: Optional transformer attention template.
        packed_input: Whether sequences are packed.
        tr_fflayer_tpl: Optional transformer feed-forward template.
        ngrammer_tpl: Optional Ngrammer template.
        ssm_tpl: Optional underlying `SSM` template.
        ssm_nheads: Number of SSM heads.
        ssm_dim: Latent state space dimension.
        ssm_l_max: Maximum sequence length for SSM kernel.
        ssm_hippo_type: HiPPO initialization family.
        ssm_step_size: Discretization step size.
        decode_num_samples: Number of samples during decoding.
        dss_dim: Internal expanded state-space feature dimension.
        gss_mlp_dim: Gating branch MLP dimension.
        activation_fn: Activation function name (`'gelu'`, `'swish'`, `'relu'`).
        hidden_size: Output hidden dimension (`0` preserves `input_dims` or `inputs.shape[-1]`).
    """

    gss_fflayer_tpl: LayerTpl = None
    input_dims: int = 0
    hidden_dims: int = 0
    num_heads: int = 1
    dim_per_head: int | None = None
    dropout_tpl: LayerTpl = None
    atten_dropout_prob: float = 0.0
    residual_dropout_prob: float = 0.0
    relu_dropout_prob: float = 0.0
    residual_droppath_prob: float = 0.0
    use_cross_attention: bool = False
    allow_skip_cross_attention: bool = False
    cross_atten_tpl: LayerTpl = None
    ln_tpl: LayerTpl = None
    norm_policy: str = "pre"
    tr_atten_tpl: LayerTpl = None
    packed_input: bool = False
    tr_fflayer_tpl: LayerTpl = None
    ngrammer_tpl: LayerTpl = None
    ssm_tpl: LayerTpl = None
    ssm_nheads: int = 1
    ssm_dim: int = 1
    ssm_l_max: int = 1024
    ssm_hippo_type: str = "legs"
    ssm_step_size: float = 1.0
    decode_num_samples: int = 0
    dss_dim: int = 0
    gss_mlp_dim: int = 0
    activation_fn: str = "gelu"
    hidden_size: int = 0

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply pre-normalized Gated SSM block to `inputs`.

        Args:
            inputs: 3D input sequence tensor of shape `[B, T, D]`.
            paddings: Optional 2D padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Gated SSM output `Tensor` of shape `[B, T, D_out]`.
        """
        x = np.asarray(inputs)
        d_in = x.shape[-1]
        d_out = (
            self.hidden_size
            if self.hidden_size > 0
            else (self.input_dims if self.input_dims > 0 else d_in)
        )

        ln = instantiate_layer(self.ln_tpl, LayerNorm(dim=d_in))
        normed = ln(x)
        act = Swish() if self.activation_fn.lower() == "swish" else GELU()
        gate = act(normed)

        default_ssm = SSM(
            nheads=self.ssm_nheads if self.ssm_nheads > 1 else self.nheads,
            head_dim=self.head_dim,
            ssm_dim=self.ssm_dim,
            hippo_type=self.ssm_hippo_type if self.ssm_hippo_type != "legs" else self.hippo_type,
            hidden_size=d_out,
            params_init=self.params_init,
        )
        ssm_layer = instantiate_layer(self.ssm_tpl, default_ssm)
        ssm_out = (
            ssm_layer(normed, paddings=paddings)
            if isinstance(ssm_layer, SSM)
            else ssm_layer(normed)
        )

        if gate.shape[-1] != d_out:
            gate_proj = instantiate_layer(
                self.gss_fflayer_tpl,
                Linear(input_dims=gate.shape[-1], output_dims=d_out, params_init=self.params_init),
            )
            gate = (
                gate_proj(gate, w=np.eye(gate.shape[-1], d_out))
                if isinstance(gate_proj, Linear)
                else gate_proj(gate)
            )

        return ssm_out * gate


class SSMTransformer(SSMGated):
    """Hybrid State Space Model + Transformer block.

    Attributes:
        tr_atten_tpl: Optional self-attention layer template applied after the gated SSM mixer.
        tr_fflayer_tpl: Optional feed-forward layer template applied at the end of the block.
    """

    tr_atten_tpl: LayerTpl = None
    tr_fflayer_tpl: LayerTpl = None

    def __call__(
        self,
        inputs: TensorOrScalar,
        paddings: TensorOrScalar | None = None,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply `SSMGated` sequence mixing followed by optional attention and feed-forward sub-layers.

        Args:
            inputs: 3D input sequence tensor of shape `[B, T, D]`.
            paddings: Optional 2D padding mask of shape `[B, T]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Hybrid SSM-Transformer output `Tensor` of shape `[B, T, D_out]`.
        """
        ssm_out = super().__call__(inputs, paddings, *args, **kwargs)
        d_out = ssm_out.shape[-1]
        if self.tr_atten_tpl is not None:
            attn = instantiate_layer(
                self.tr_atten_tpl,
                DotProductAttention(input_dim=d_out, params_init=self.params_init),
            )
            ssm_out = ssm_out + attn(ssm_out)
        if self.tr_fflayer_tpl is not None:
            ff = instantiate_layer(
                self.tr_fflayer_tpl,
                FeedForward(input_dims=d_out, output_dims=d_out, params_init=self.params_init),
            )
            ssm_out = ssm_out + ff(ssm_out)
        return ssm_out


__all__ = [
    "CifgLstmCellSimple",
    "FRnn",
    "LayerNormalizedLstmCellSimple",
    "LstmCellSimple",
    "LstmFrnn",
    "SSM",
    "SSMGated",
    "SSMTransformer",
    "StackFrnn",
]
