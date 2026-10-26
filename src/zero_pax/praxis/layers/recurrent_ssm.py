"""Recurrent (LSTM, FRnn, StackFrnn) and State Space Model (SSM) layers for zero_pax.praxis.layers."""

from __future__ import annotations

from zero_pax import _tensor_ops as _ops
from zero_pax.praxis.base import BasePraxisLayer, LayerTpl, instantiate_layer
from zero_pax.praxis.layers.activations import GELU, Identity, ReLU, Sigmoid, Swish, Tanh
from zero_pax.praxis.layers.attentions import DotProductAttention
from zero_pax.praxis.layers.linear_pooling import FeedForward, Linear
from zero_pax.praxis.layers.normalizations import LayerNorm
from zero_pax.praxis.layers.quantization_utils import Dropout, StochasticResidual
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
        x = self._to_tensor(inputs)
        if self.inputs_arity > 1:
            self.register_parameter("inputs_arity", float(self.inputs_arity))
        b_sz = x.shape[0]
        h_dim = (
            self.num_output_nodes
            if self.num_output_nodes > 0
            else (self.hidden_size if self.hidden_size > 0 else self.num_hidden_nodes)
        )

        c_prev: Tensor
        m_prev: Tensor
        if isinstance(state0, tuple):
            c_prev = _ops.asarray(state0[0])
            m_prev = _ops.asarray(state0[1])
            if h_dim <= 0:
                h_dim = c_prev.shape[-1]
            x_t = x
        elif state0 is not None:
            s_arr = _ops.asarray(state0)
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
            state_init = self.zero_state_init_params or WeightInit(method="zeros")
            c_prev = state_init.init_tensor((b_sz, h_dim), dtype=self.dtype)
            m_prev = state_init.init_tensor((b_sz, h_dim), dtype=self.dtype)
            x_t = x[:, :h_dim] if x.shape[-1] == 3 * h_dim else x

        if self.reset_cell_state and reset_mask is not None:
            rm = _ops.asarray(reset_mask)
            c_prev = c_prev * rm
            m_prev = m_prev * rm

        in_dim = (
            self.num_input_nodes
            if (self.num_input_nodes > 0 and self.num_input_nodes == x_t.shape[-1])
            else (
                self.input_size
                if (self.input_size > 0 and self.input_size == x_t.shape[-1])
                else x_t.shape[-1]
            )
        )
        cifg = self.couple_input_forget_gates or self.num_gates == 3
        n_gates = 3 if cifg else 4

        wm_t = self.get_variable("wm", (in_dim + h_dim, n_gates * h_dim), explicit=wm)
        b_t = self.get_variable("b", (n_gates * h_dim,), explicit=b, init=self.bias_init)

        concat_in = _ops.concatenate([x_t, m_prev], axis=-1)
        gates_pre = _ops.einsum("bd,df->bf", concat_in, wm_t)
        gates = self._transform_gates(gates_pre, n_gates, h_dim, b_t, gammas=gammas)

        sig = Sigmoid()
        tanh = Tanh()

        if cifg:
            g_raw, f_raw, o_raw = _ops.split(gates, 3, axis=-1)
            f_t = sig(f_raw + self.forget_gate_bias)
            i_t = 1.0 - f_t
        else:
            i_raw, g_raw, f_raw, o_raw = _ops.split(gates, 4, axis=-1)
            i_t = sig(i_raw)
            f_t = sig(f_raw + self.forget_gate_bias)

        g_t = tanh(g_raw)
        o_t = sig(o_raw)

        c_new = f_t * c_prev + i_t * g_t
        if self.cell_value_cap is not None and self.cell_value_cap > 0.0:
            c_new = _ops.clip(c_new, -self.cell_value_cap, self.cell_value_cap)

        m_new = o_t * tanh(c_new) if self.output_nonlinearity else o_t * c_new

        if 0.0 < self.zo_prob < 1.0:
            c_new = self.zo_prob * c_prev + (1.0 - self.zo_prob) * c_new
            m_new = self.zo_prob * m_prev + (1.0 - self.zo_prob) * m_new

        if padding is not None:
            pad_t = _ops.asarray(padding)
            if pad_t.ndim == 1:
                pad_t = _ops.expand_dims(pad_t, -1)
            c_new = c_prev * pad_t + c_new * (1.0 - pad_t)
            m_new = m_prev * pad_t + m_new * (1.0 - pad_t)

        return c_new, m_new


class CifgLstmCellSimple(LstmCellSimple):
    """Coupled Input-Forget Gate (CIFG) LSTM cell where `i_t = 1 - f_t`.

    Attributes:
        couple_input_forget_gates: Whether `i_t = 1 - f_t` (`True` by default).
        num_gates: Number of gates (`3` by default for CIFG).
    """

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
        gates_reshaped = _ops.reshape(gates_pre, (b_sz, n_gates, h_dim))
        mean_g = _ops.mean(gates_reshaped, axis=-1, keepdims=True)
        var_g = _ops.var(gates_reshaped, axis=-1, keepdims=True)
        normed_g = (gates_reshaped - mean_g) / _ops.sqrt(var_g + self.layer_norm_epsilon)

        gamma_t = self.get_variable(
            "gammas",
            (n_gates, h_dim),
            explicit=gammas,
            init=WeightInit(method="ones", scale=1.0),
        )
        scaled_g = normed_g * _ops.expand_dims(gamma_t, 0)
        gates = _ops.reshape(scaled_g, (b_sz, n_gates * h_dim))
        if self.enable_lstm_bias:
            gates = gates + b_t
        return gates


def _ensure_3d_sequence(x: Tensor) -> tuple[Tensor, int]:
    """Promote 1D or 2D tensors to 3D `[B, T, D]` and return `(x_3d, orig_ndim)`.

    Args:
        x: Input tensor of rank 1, 2, or 3+.

    Returns:
        Tuple of `(promoted_3d_tensor, original_ndim)`.
    """
    orig_ndim = x.ndim
    if orig_ndim == 1:
        return _ops.reshape(x, (1, 1, x.shape[0])), orig_ndim
    if orig_ndim == 2:
        return _ops.expand_dims(x, 1), orig_ndim
    return x, orig_ndim


def _restore_sequence_rank(out_3d: Tensor, orig_ndim: int) -> Tensor:
    """Restore a 3D `[B, T, D]` output tensor to `orig_ndim` when `T == 1`.

    Args:
        out_3d: 3D output sequence tensor.
        orig_ndim: Original input rank prior to `_ensure_3d_sequence`.

    Returns:
        Rank-restored `Tensor`.
    """
    if orig_ndim == 1:
        return out_3d[0, 0, :]
    if orig_ndim == 2:
        return out_3d[:, 0, :]
    return out_3d


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
            inputs: 3D input sequence tensor of shape `[B, T, D]` (or 1D/2D single-step input).
            paddings: Optional 2D padding mask of shape `[B, T]`.
            state0: Optional initial hidden state tensor of shape `[B, H]`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Unrolled output sequence `Tensor` of shape `[B, T, H]` (or rank-matched for 1D/2D input).
        """
        raw_x = self._to_tensor(inputs)
        if self.unroll > 1:
            self.register_parameter("unroll_factor", float(self.unroll))
        x, orig_ndim = _ensure_3d_sequence(raw_x)
        b_sz, t_len, d_in = x.shape
        h_dim = self.hidden_size if self.hidden_size > 0 else d_in
        state = _ops.zeros((b_sz, h_dim)) if state0 is None else _ops.asarray(state0)
        pad_t = _ops.asarray(paddings) if paddings is not None else None

        if isinstance(self.cell_tpl, LstmCellSimple):
            c_state = _ops.zeros((b_sz, h_dim))
            m_state = state
            outputs_lstm: list[Tensor] = []
            indices_lstm = list(range(t_len - 1, -1, -1)) if self.reverse else list(range(t_len))
            for t in indices_lstm:
                step_pad = pad_t[:, t : t + 1] if pad_t is not None else None
                c_state, m_state = self.cell_tpl(
                    x[:, t, :], state0=(c_state, m_state), padding=step_pad
                )
                outputs_lstm.append(_ops.expand_dims(m_state, 1))
            if self.reverse:
                outputs_lstm.reverse()
            return _restore_sequence_rank(_ops.concatenate(outputs_lstm, axis=1), orig_ndim)

        cell_fn = (
            self.get_sublayer("cell", self.cell_tpl, Identity())
            if self.cell_tpl is not None
            else None
        )

        outputs: list[Tensor] = []
        time_indices = list(range(t_len - 1, -1, -1)) if self.reverse else list(range(t_len))
        for t in time_indices:
            step_x = x[:, t, :]
            step_in = step_x[:, :h_dim] if step_x.shape[-1] >= h_dim else step_x
            if cell_fn is not None:
                new_state = cell_fn(state + step_in)
            else:
                new_state = _ops.tanh(state + step_in)
            if pad_t is not None:
                step_pad = pad_t[:, t : t + 1]
                state = state * step_pad + new_state * (1.0 - step_pad)
            else:
                state = new_state
            outputs.append(_ops.expand_dims(state, 1))

        if self.reverse:
            outputs.reverse()
        return _restore_sequence_rank(_ops.concatenate(outputs, axis=1), orig_ndim)


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
            inputs: 3D input sequence tensor of shape `[B, T, D_in]` (or 1D/2D single-step input).
            paddings: Optional 2D padding mask of shape `[B, T]`.
            state0: Optional initial `(c_0, m_0)` tuple.
            wm: Optional explicit LSTM weight matrix.
            b: Optional explicit LSTM bias vector.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output sequence `Tensor` of shape `[B, T, D_out]` (or rank-matched for 1D/2D input).
        """
        raw_x = self._to_tensor(inputs)
        x, orig_ndim = _ensure_3d_sequence(raw_x)
        b_sz, t_len, d_in = x.shape
        h_dim = (
            self.num_output_nodes
            if self.num_output_nodes > 0
            else (self.hidden_size if self.hidden_size > 0 else d_in)
        )
        default_cell = LstmCellSimple(
            num_input_nodes=self.num_input_nodes if self.num_input_nodes > 0 else d_in,
            num_output_nodes=h_dim,
            cell_value_cap=self.cell_value_cap,
            forget_gate_bias=self.forget_gate_bias,
            output_nonlinearity=self.output_nonlinearity,
            zo_prob=self.zo_prob,
            enable_lstm_bias=self.enable_lstm_bias,
            couple_input_forget_gates=self.couple_input_forget_gates,
            reset_cell_state=self.reset_cell_state,
            params_init=self.params_init,
            parent=self,
            name="lstm_cell",
        )
        resolved_cell = (
            instantiate_layer(self.cell_tpl, default_cell, parent=self, name="lstm_cell")
            if self.cell_tpl is not None
            else default_cell
        )
        cell = resolved_cell if isinstance(resolved_cell, LstmCellSimple) else default_cell

        if isinstance(state0, tuple):
            c_t, m_t = _ops.asarray(state0[0]), _ops.asarray(state0[1])
        else:
            c_t = _ops.zeros((b_sz, h_dim))
            m_t = _ops.zeros((b_sz, h_dim))

        pad_t = _ops.asarray(paddings) if paddings is not None else None
        outputs: list[Tensor] = []
        time_indices = list(range(t_len - 1, -1, -1)) if self.reverse else list(range(t_len))

        for t in time_indices:
            step_pad = pad_t[:, t : t + 1] if pad_t is not None else None
            c_t, m_t = cell(x[:, t, :], state0=(c_t, m_t), wm=wm, b=b, padding=step_pad)
            outputs.append(_ops.expand_dims(m_t, 1))

        if self.reverse:
            outputs.reverse()
        return _restore_sequence_rank(_ops.concatenate(outputs, axis=1), orig_ndim)


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
        out = self._to_tensor(inputs)
        in_d = self.num_input_nodes if self.num_input_nodes > 0 else out.shape[-1]
        out_dim = (
            self.num_output_nodes
            if self.num_output_nodes > 0
            else (self.hidden_size if self.hidden_size > 0 else in_d)
        )
        mid_dim = self.hidden_size if self.hidden_size > 0 else out_dim
        n_layers = max(1, self.num_layers)
        drop_layer = (
            self.get_sublayer("dropout", None, Dropout(keep_prob=max(0.0, 1.0 - self.dropout_prob)))
            if self.dropout_prob > 0.0
            else None
        )
        for idx in range(n_layers):
            layer_h_dim = out_dim if idx == n_layers - 1 else mid_dim
            default_frnn = FRnn(hidden_size=layer_h_dim, params_init=self.params_init)
            frnn = instantiate_layer(self.frnn_tpl, default_frnn, parent=self, name=f"frnn_{idx}")
            out = frnn(out, paddings=paddings) if isinstance(frnn, FRnn) else frnn(out)
            if drop_layer is not None:
                out = drop_layer(out)
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

    nheads: int = 0
    dim: int = 0
    head_dim: int = 0
    ssm_dim: int = 0
    l_max: int = 0
    decode_num_samples: int = 0
    step_size: float = 0.01
    dt_min: float = 0.001
    dt_max: float = 0.1
    hippo_type: str = "ss4d-1d"
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
            inputs: 3D input sequence tensor of shape `[B, T, D_in]` (or 1D/2D single-step input).
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
        raw_u = self._to_tensor(inputs)
        if self.decode_num_samples > 0:
            self.register_parameter("decode_num_samples", float(self.decode_num_samples))
        u, orig_ndim = _ensure_3d_sequence(raw_u)
        b_sz, t_len, d_in = u.shape
        d_out = (
            self.hidden_size
            if self.hidden_size > 0
            else (self.nheads * self.head_dim if (self.nheads > 0 and self.head_dim > 0) else d_in)
        )
        n_state = self.dim if self.dim > 0 else max(1, self.ssm_dim)

        if a_diag is not None:
            a_t = _ops.asarray(a_diag)
        elif self.hippo_type.lower() == "legt":
            a_t = -(2.0 * _ops.arange(1, n_state + 1) - 1.0)
        elif self.hippo_type.lower() == "lagt":
            a_t = -0.5 * _ops.ones((n_state,))
        elif self.hippo_type.lower() == "legs":
            a_t = -0.5 * _ops.arange(1, n_state + 1)
        elif self.hippo_type.lower() == "fourier":
            a_t = -1.0 / _ops.maximum(_ops.arange(1, n_state + 1), 1.0)
        else:
            a_t = -_ops.ones((n_state,))

        b_t = (
            _ops.ones((d_in, n_state)) * (1.0 / float(d_in))
            if b_mat is None
            else _ops.asarray(b_mat)
        )
        c_t = self.get_variable("c_mat", (n_state, d_out)) if c_mat is None else _ops.asarray(c_mat)
        d_t = _ops.eye(d_in, d_out) if d_vec is None else _ops.asarray(d_vec)

        raw_dt = dt if dt != 0.05 else max(dt, self.step_size)
        eff_dt = max(self.dt_min, min(self.dt_max, raw_dt))
        a_bar = _ops.exp(eff_dt * a_t)
        b_bar = eff_dt * b_t

        h = _ops.zeros((b_sz, n_state))
        pad_t = _ops.asarray(paddings) if paddings is not None else None
        outputs: list[Tensor] = []

        for t in range(t_len):
            if self.l_max > 0 and t > 0 and t % self.l_max == 0:
                h = _ops.zeros((b_sz, n_state))
            u_step = u[:, t, :]
            h_next = h * a_bar + _ops.einsum("bd,dn->bn", u_step, b_bar)
            if pad_t is not None:
                p_step = pad_t[:, t : t + 1]
                h = h * p_step + h_next * (1.0 - p_step)
            else:
                h = h_next
            y_step = _ops.einsum("bn,no->bo", h, c_t) + _ops.einsum("bd,do->bo", u_step, d_t)
            outputs.append(_ops.expand_dims(y_step, 1))

        forward_out = _ops.concatenate(outputs, axis=1)
        if not self.bidirectional:
            return _restore_sequence_rank(forward_out, orig_ndim)

        h_rev = _ops.zeros((b_sz, n_state))
        rev_outputs: list[Tensor] = []
        for t in range(t_len - 1, -1, -1):
            u_step = u[:, t, :]
            h_next = h_rev * a_bar + _ops.einsum("bd,dn->bn", u_step, b_bar)
            if pad_t is not None:
                p_step = pad_t[:, t : t + 1]
                h_rev = h_rev * p_step + h_next * (1.0 - p_step)
            else:
                h_rev = h_next
            y_rev = _ops.einsum("bn,no->bo", h_rev, c_t)
            rev_outputs.append(_ops.expand_dims(y_rev, 1))
        rev_outputs.reverse()
        return _restore_sequence_rank(
            forward_out + _ops.concatenate(rev_outputs, axis=1), orig_ndim
        )


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
    num_heads: int | None = None
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
    ssm_nheads: int = 0
    ssm_dim: int = 0
    ssm_l_max: int = 0
    ssm_hippo_type: str = "ss4d-1d-legs"
    ssm_step_size: float = 0.1
    decode_num_samples: int = 4
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
        x = self._to_tensor(inputs)
        d_in = x.shape[-1]
        d_out = (
            self.hidden_size
            if self.hidden_size > 0
            else (
                self.input_dims
                if self.input_dims > 0
                else (
                    self.dss_dim
                    if self.dss_dim > 0
                    else (self.gss_mlp_dim if self.gss_mlp_dim > 0 else d_in)
                )
            )
        )
        if self.packed_input or self.allow_skip_cross_attention or self.relu_dropout_prob > 0.0:
            self.register_parameter("ssm_gated_meta", float(d_out))

        if self.ngrammer_tpl is not None:
            ng_layer = self.get_sublayer("ngrammer", self.ngrammer_tpl, Identity())
            x = ng_layer(x)

        ln = self.get_sublayer("ln", self.ln_tpl, LayerNorm(dim=d_in))
        normed = ln(x) if self.norm_policy != "post" else x
        act_name = self.activation_fn.lower()
        act = Swish() if act_name == "swish" else (ReLU() if act_name == "relu" else GELU())
        gate = act(normed)

        default_ssm = SSM(
            nheads=self.ssm_nheads if self.ssm_nheads > 1 else self.nheads,
            head_dim=self.head_dim,
            ssm_dim=self.ssm_dim,
            l_max=self.ssm_l_max if self.ssm_l_max > 0 else self.l_max,
            decode_num_samples=self.decode_num_samples,
            step_size=self.ssm_step_size,
            hippo_type=self.ssm_hippo_type if self.ssm_hippo_type != "legs" else self.hippo_type,
            bidirectional=self.bidirectional,
            hidden_size=d_out,
            params_init=self.params_init,
        )
        ssm_layer = self.get_sublayer("ssm", self.ssm_tpl, default_ssm)
        ssm_out = (
            ssm_layer(normed, paddings=paddings)
            if isinstance(ssm_layer, SSM)
            else ssm_layer(normed)
        )

        if gate.shape[-1] != d_out or self.gss_fflayer_tpl is not None:
            gate_proj = self.get_sublayer(
                "gss_fflayer",
                self.gss_fflayer_tpl,
                Linear(
                    input_dims=gate.shape[-1],
                    output_dims=d_out,
                    weight_init=WeightInit(method="ones", scale=1.0),
                    params_init=self.params_init,
                ),
            )
            gate = gate_proj(gate)

        if ssm_out.shape[-1] != gate.shape[-1]:
            ssm_out = ssm_out[..., : gate.shape[-1]]
        out = ssm_out * gate

        if self.use_cross_attention or self.cross_atten_tpl is not None:
            eff_heads = max(1, self.num_heads if self.num_heads is not None else 1)
            c_attn = self.get_sublayer(
                "cross_atten",
                self.cross_atten_tpl,
                DotProductAttention(
                    input_dim=d_out,
                    num_heads=eff_heads,
                    dim_per_head=self.dim_per_head or max(1, d_out // eff_heads),
                    atten_dropout_prob=self.atten_dropout_prob,
                    params_init=self.params_init,
                ),
            )
            out = out + c_attn(out)

        default_drop: BasePraxisLayer = (
            Dropout(keep_prob=max(0.0, 1.0 - self.residual_dropout_prob))
            if self.residual_dropout_prob > 0.0
            else Identity()
        )
        drop_layer = self.get_sublayer("dropout", self.dropout_tpl, default_drop)
        out = drop_layer(out)
        if self.residual_droppath_prob > 0.0 and out.shape == x.shape:
            stoch = self.get_sublayer(
                "droppath",
                None,
                StochasticResidual(survival_prob=max(0.0, 1.0 - self.residual_droppath_prob)),
            )
            out = stoch(x, out) if isinstance(stoch, StochasticResidual) else stoch(out)
        if self.norm_policy == "post":
            out = ln(out)
        if self.tr_atten_tpl is not None:
            eff_heads = max(1, self.num_heads if self.num_heads is not None else 1)
            attn = self.get_sublayer(
                "tr_atten",
                self.tr_atten_tpl,
                DotProductAttention(
                    input_dim=d_out,
                    num_heads=eff_heads,
                    dim_per_head=self.dim_per_head or max(1, d_out // eff_heads),
                    params_init=self.params_init,
                ),
            )
            out = out + attn(out)
        if self.tr_fflayer_tpl is not None:
            f_hid = self.hidden_dims if self.hidden_dims > 0 else d_out
            ff = self.get_sublayer(
                "tr_fflayer",
                self.tr_fflayer_tpl,
                FeedForward(input_dims=d_out, output_dims=f_hid, params_init=self.params_init),
            )
            out = out + ff(out)
        return out


class SSMTransformer(SSMGated):
    """Hybrid State Space Model + Transformer block.

    Attributes:
        tr_atten_tpl: Optional self-attention layer template applied after the gated SSM mixer.
        tr_fflayer_tpl: Optional feed-forward layer template applied at the end of the block.
    """

    tr_atten_tpl: LayerTpl = None
    tr_fflayer_tpl: LayerTpl = None


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
