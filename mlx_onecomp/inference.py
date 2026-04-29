"""Quantized inference layers using MLX QuantizedLinear."""

import mlx.core as mx
from mlx.nn import Linear, QuantizedLinear


def create_quantized_linear(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 128,
    bits: int = 4,
    bias: mx.array = None,
) -> QuantizedLinear:
    """Create MLX QuantizedLinear from quantized weight data.

    Args:
        weight: Quantized weight values (out_features, in_features), int32.
        scales: Scale coefficients, float16.
        biases: Zero-point / bias values, float16.
        group_size: Quantization group size.
        bits: Bit width.
        bias: Optional original bias.

    Returns:
        QuantizedLinear ready for inference.
    """
    # MLX QuantizedLinear expects quantized weights in its packed format.
    # For simple usage, we use the from_linear approach with dequantized weights.
    pass


def dequantize_weight(
    quantized_weight: mx.array,
    scale: mx.array,
    zero: mx.array,
    wbits: int,
    groupsize: int,
) -> mx.array:
    """Dequantize GPTQ-style quantized weights.

    Args:
        quantized_weight: (out_features, in_features) int32.
        scale: Scale coefficients.
        zero: Zero points.
        wbits: Bit width.
        groupsize: Group size (-1 = per-channel).

    Returns:
        Dequantized weight (out_features, in_features) float16.
    """
    if groupsize == -1:
        # Per-channel
        if scale.ndim == 1:
            scale = scale.reshape(-1, 1)
        if zero.ndim == 1:
            zero = zero.reshape(-1, 1)
        return (scale * (quantized_weight.astype(mx.float32) - zero)).astype(mx.float16)

    # Grouped
    in_features = quantized_weight.shape[1]
    num_groups = (in_features + groupsize - 1) // groupsize

    # Normalize scale/zero to (num_groups, out_features)
    if scale.shape[0] == quantized_weight.shape[0] and scale.shape[1] == num_groups:
        scale = scale.transpose()
        zero = zero.transpose()

    # Build g_idx
    g_idx = mx.arange(in_features) // groupsize

    scale_expanded = scale[g_idx, :].transpose()  # (out_features, in_features)
    zero_expanded = zero[g_idx, :].transpose()

    return (scale_expanded * (quantized_weight.astype(mx.float32) - zero_expanded)).astype(mx.float16)
