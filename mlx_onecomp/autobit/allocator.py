"""Apply AutoBit allocation to a model.

Takes the optimal bit allocation from solve_bit_allocation() and
quantizes each layer with its assigned bit width.
"""

import gc
import logging
import time

import mlx.core as mx
import mlx.nn as nn

from mlx_onecomp.quantizer.rtn._rtn import RTN

logger = logging.getLogger(__name__)


def apply_allocation(
    model,
    allocation: dict[str, int],
    groupsize: int = 128,
) -> dict:
    """Apply per-layer bit allocation to a model.

    Args:
        model: MLX model.
        allocation: {layer_name: bits} from solve_bit_allocation().
        groupsize: Quantization group size.

    Returns:
        Dict with summary stats.
    """
    quantized = 0
    skipped = 0
    t0 = time.time()

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name not in allocation:
            skipped += 1
            continue

        bits = allocation[name]
        quantizer = RTN(wbits=bits, groupsize=groupsize, sym=False)
        result = quantizer.quantize_weight(mod.weight)
        mod.weight = result.dequantized_weight
        quantized += 1

        if quantized % 20 == 0:
            gc.collect()

    elapsed = time.time() - t0
    logger.info(
        "Applied allocation: %d layers quantized, %d skipped, %.1fs",
        quantized, skipped, elapsed,
    )

    # Summary
    bits_dist = {}
    for b in allocation.values():
        bits_dist[b] = bits_dist.get(b, 0) + 1

    return {
        "quantized": quantized,
        "skipped": skipped,
        "time": elapsed,
        "bits_distribution": bits_dist,
    }
