"""Enumeration and policy configuration classes for zero_pax.praxis.layers."""

from __future__ import annotations

from enum import Enum


class AutodiffCheckpointType(str, Enum):
    """Enumeration defining activation rematerialization (checkpointing) policies.

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

    SAVE_NOTHING = "save_nothing"
    SAVE_UNET_ALL_CONV = "save_unet_all_conv"
    SAVE_UNET_CONV = "save_unet_conv"
    SAVE_EVERYTHING = "save_everything"
    SAVE_QKV_OUT_PROJ = "save_qkv_out_proj"
    SAVE_OUT_PROJ = "save_out_proj"
    SAVE_CONTEXT = "save_context"
    SAVE_CONTEXT_AND_OUT_PROJ = "save_encoded_and_out_proj"
    SAVE_DOT_ONLY = "save_dot_only"
    SAVE_DOT_WITH_NO_BATCH_DIM = "save_dot_with_no_batch_dims"
    SAVE_DOT_FOR_MLPERF_200B = "save_dot_for_mlperf_200b"
    SAVE_ITERATION_INPUT = "save_iteration_input"
    SAVE_TRANSFORMER_LAYER_OUTPUT = "save_transformer_layer_output"
    SAVE_QUANTIZED = "save_quantized"
    SAVE_QKV_OUT_PROJ_SEPARATE = "save_qkv_out_proj_separate"
    SAVE_DOT_EXCEPT_LOGITS_FFN1 = "save_dot_except_logits_ffn1"
    SAVE_DOT_EXCEPT_LOGITS = "save_dot_except_logits"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """Return all valid checkpoint policy string values.

        Returns:
            Tuple of checkpoint policy strings.
        """
        return tuple(member.value for member in cls)


class LanguageModelType(str, Enum):
    """Enumeration defining language model attention masking modes.

    Attributes:
        CAUSAL: Autoregressive causal masking.
        PREFIX: Prefix language model masking (bidirectional prefix, causal target).
        BIDIRECTIONAL: Full bidirectional self-attention masking.
    """

    CAUSAL = "causal"
    PREFIX = "prefix"
    BIDIRECTIONAL = "bidirectional"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """Return all valid language model type strings.

        Returns:
            Tuple of valid language model mode strings.
        """
        return tuple(member.value for member in cls)


__all__ = [
    "AutodiffCheckpointType",
    "LanguageModelType",
]
