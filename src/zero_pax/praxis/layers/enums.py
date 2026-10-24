"""Enumeration and policy configuration classes for zero_pax.praxis.layers."""

from __future__ import annotations

from zero_pax import _tensor_ops as np
from zero_pax.praxis.base import BasePraxisLayer
from zero_pax.types import Tensor, TensorOrScalar


class AutodiffCheckpointType(BasePraxisLayer):
    """Configuration class defining activation rematerialization (checkpointing) policies.

    Attributes:
        SAVE_NOTHING: Save no intermediate activations.
        SAVE_UNET_ALL_CONV: Save all U-Net convolution outputs.
        SAVE_UNET_CONV: Save selected U-Net convolution outputs.
        SAVE_EVERYTHING: Save all intermediate activations.
        SAVE_QKV_OUT_PROJ: Save QKV and output projections.
        SAVE_OUT_PROJ: Save attention output projection.
        SAVE_CONTEXT: Save attention context vector.
        SAVE_CONTEXT_AND_OUT_PROJ: Save attention context and output projection.
        SAVE_DOT_ONLY: Save matrix multiplication dot products only.
        SAVE_DOT_WITH_NO_BATCH_DIM: Save dot products that do not have a batch dimension.
        SAVE_DOT_FOR_MLPERF_200B: Save dot products tuned for MLPerf 200B.
        SAVE_ITERATION_INPUT: Save loop iteration inputs.
        SAVE_TRANSFORMER_LAYER_OUTPUT: Save full transformer layer outputs.
        SAVE_QUANTIZED: Save quantized activations.
        SAVE_QKV_OUT_PROJ_SEPARATE: Save Q, K, V, and output projections separately.
        SAVE_DOT_EXCEPT_LOGITS_FFN1: Save dot products except logits and FFN1.
        SAVE_DOT_EXCEPT_LOGITS: Save dot products except final logits.
    """

    SAVE_NOTHING: str = "save_nothing"
    SAVE_UNET_ALL_CONV: str = "save_unet_all_conv"
    SAVE_UNET_CONV: str = "save_unet_conv"
    SAVE_EVERYTHING: str = "save_everything"
    SAVE_QKV_OUT_PROJ: str = "save_qkv_out_proj"
    SAVE_OUT_PROJ: str = "save_out_proj"
    SAVE_CONTEXT: str = "save_context"
    SAVE_CONTEXT_AND_OUT_PROJ: str = "save_encoded_and_out_proj"
    SAVE_DOT_ONLY: str = "save_dot_only"
    SAVE_DOT_WITH_NO_BATCH_DIM: str = "save_dot_with_no_batch_dims"
    SAVE_DOT_FOR_MLPERF_200B: str = "save_dot_for_mlperf_200b"
    SAVE_ITERATION_INPUT: str = "save_iteration_input"
    SAVE_TRANSFORMER_LAYER_OUTPUT: str = "save_transformer_layer_output"
    SAVE_QUANTIZED: str = "save_quantized"
    SAVE_QKV_OUT_PROJ_SEPARATE: str = "save_qkv_out_proj_separate"
    SAVE_DOT_EXCEPT_LOGITS_FFN1: str = "save_dot_except_logits_ffn1"
    SAVE_DOT_EXCEPT_LOGITS: str = "save_dot_except_logits"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """Return all valid checkpoint policy string values.

        Returns:
            Tuple of checkpoint policy strings.
        """
        return (
            "save_nothing",
            "save_unet_all_conv",
            "save_unet_conv",
            "save_everything",
            "save_qkv_out_proj",
            "save_out_proj",
            "save_context",
            "save_encoded_and_out_proj",
            "save_dot_only",
            "save_dot_with_no_batch_dims",
            "save_dot_for_mlperf_200b",
            "save_iteration_input",
            "save_transformer_layer_output",
            "save_quantized",
            "save_qkv_out_proj_separate",
            "save_dot_except_logits_ffn1",
            "save_dot_except_logits",
        )

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply checkpoint policy wrapper to `inputs`.

        Args:
            inputs: Input activation tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Checkpointed `Tensor` identical in value to `inputs`.
        """
        return np.asarray(inputs)


class LanguageModelType(BasePraxisLayer):
    """Configuration class defining language model attention masking modes.

    Attributes:
        CAUSAL: Autoregressive causal masking.
        PREFIX: Prefix language model masking (bidirectional prefix, causal target).
        BIDIRECTIONAL: Full bidirectional self-attention masking.
    """

    CAUSAL: str = "causal"
    PREFIX: str = "prefix"
    BIDIRECTIONAL: str = "bidirectional"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """Return all valid language model type strings.

        Returns:
            Tuple of valid language model mode strings.
        """
        return ("causal", "prefix", "bidirectional")

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Pass `inputs` through the language model type configuration.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Output `Tensor`.
        """
        return np.asarray(inputs)


__all__ = [
    "AutodiffCheckpointType",
    "LanguageModelType",
]
