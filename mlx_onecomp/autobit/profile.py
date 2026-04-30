"""Sensitivity profiling for AutoBit.

Measures quantization error (MSE) of each Linear layer at different bit widths.
Produces a sensitivity table used by the ILP solver to find optimal bit allocation.
"""

import logging
import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_onecomp.quantizer.rtn._rtn import RTN

logger = logging.getLogger(__name__)

# Default bit widths to profile
DEFAULT_BITS = (2, 3, 4, 8)
DEFAULT_GROUPSIZE = 128


def _mse(a: mx.array, b: mx.array) -> float:
    """Mean squared error between two arrays."""
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


def profile_layer(
    weight: mx.array,
    bits_list: tuple[int, ...] = DEFAULT_BITS,
    groupsize: int = DEFAULT_GROUPSIZE,
) -> dict[int, float]:
    """Profile a single layer's sensitivity at different bit widths.

    Returns dict: {bits: mse}.
    """
    results = {}
    for bits in bits_list:
        quantizer = RTN(wbits=bits, groupsize=groupsize, sym=False)
        result = quantizer.quantize_weight(weight)
        mse_val = _mse(weight, result.dequantized_weight)
        results[bits] = mse_val
    return results


def sensitivity_profile(
    model,
    bits_list: tuple[int, ...] = DEFAULT_BITS,
    groupsize: int = DEFAULT_GROUPSIZE,
    layer_filter: Optional[callable] = None,
) -> dict[str, dict[int, float]]:
    """Profile all Linear layers in a model.

    Args:
        model: MLX model (dict-like or nn.Module).
        bits_list: Bit widths to test.
        groupsize: Quantization group size.
        layer_filter: Optional callable(name, module) -> bool to filter layers.

    Returns:
        {layer_name: {bits: mse, ...}, ...}
    """
    profile = {}
    total_layers = 0
    t0 = time.time()

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if layer_filter and not layer_filter(name, mod):
            continue

        total_layers += 1
        sens = profile_layer(mod.weight, bits_list, groupsize)
        profile[name] = sens

        # Log every 10 layers
        if total_layers % 10 == 0:
            best = min(sens, key=sens.get)
            logger.info(
                "  [%d] %s: best=%dbit (MSE=%.6f)",
                total_layers, name.split(".")[-1], best, sens[best],
            )

    elapsed = time.time() - t0
    logger.info(
        "Profiled %d layers in %.1fs (%.2fs/layer)",
        total_layers, elapsed, elapsed / max(total_layers, 1),
    )

    return profile
