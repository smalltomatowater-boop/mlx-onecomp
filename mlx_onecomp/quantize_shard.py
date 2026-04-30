"""Memory-efficient shard-based quantization.

Processes safetensors shard files using mmap (lazy loading).
Only one tensor in memory at a time. Peak memory ~2-4GB per shard.

Usage:
    from mlx_onecomp.quantize_shard import quantize_shards

    quantize_shards(
        src_dir="/path/to/model",
        dst_dir="/path/to/quantized",
        method="rtn",
        wbits=4,
    )
"""

import gc
import glob
import json
import logging
import os
import shutil
import time

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from mlx_onecomp.quantizer.rtn._rtn import RTN
from mlx_onecomp.preprocessing.rotation import HadamardRotation, RandomRotation
from mlx_onecomp.autobit.profile import profile_layer
from mlx_onecomp.autobit.solver import solve_bit_allocation
from mlx_onecomp.autobit.allocator import apply_allocation

logger = logging.getLogger(__name__)

# Quantize targets: attention + MLP projection layers
QUANTIZE_PATTERNS = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".qkv_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)

# Non-quantize: cast bfloat16 to float16
FLOAT_CAST_PATTERNS = (
    "model.safetensors",
)


def should_quantize(name: str) -> bool:
    return any(name.endswith(s) for s in QUANTIZE_PATTERNS)


def quantize_tensor(weight: np.ndarray, wbits: int, groupsize: int) -> np.ndarray:
    """Quantize a single numpy weight tensor using RTN, return dequantized numpy."""
    w_mx = mx.array(weight.astype(np.float32))
    quantizer = RTN(wbits=wbits, groupsize=groupsize, sym=False)
    result = quantizer.quantize_weight(w_mx)
    dq_mx = result.dequantized_weight
    return np.array(dq_mx.astype(mx.float16))


def quantize_shard(
    shard_path: str,
    output_path: str,
    wbits: int = 4,
    groupsize: int = 128,
    allocation: dict | None = None,
    rotation: tuple | None = None,
):
    """Quantize a single safetensors shard file.

    Args:
        shard_path: Path to input shard.
        output_path: Path to output shard.
        wbits: Target bits (used when allocation is None).
        groupsize: Quantization group size.
        allocation: Optional {tensor_name: bits} for per-layer allocation.
        rotation: Optional (type, hidden_sizes) for rotation preprocessing.
    """
    t0 = time.time()
    quantized = 0
    unchanged = 0
    total_tensors = 0

    logger.info("Processing: %s", os.path.basename(shard_path))

    # Open with mmap (lazy loading)
    with safe_open(shard_path, framework="numpy") as f_in:
        tensors_to_save = {}

        for key in f_in.keys():
            total_tensors += 1
            logger.debug("[%d] Processing %s", total_tensors, key)

            if should_quantize(key):
                # Load tensor (mmap — only this tensor in memory)
                weight = f_in.get_tensor(key)

                # Apply rotation if applicable
                if rotation is not None:
                    rot_type, hidden_sizes = rotation
                    if weight.ndim == 2 and weight.shape[1] in hidden_sizes:
                        h = weight.shape[1]
                        if rot_type == "hadamard":
                            try:
                                rot = HadamardRotation(h)
                            except ValueError:
                                rot = RandomRotation(h)
                        else:
                            rot = RandomRotation(h)
                        weight = np.array(rot.rotate_weight_in(mx.array(weight)))

                # Determine bits for this layer
                if allocation is not None:
                    layer_bits = allocation.get(key, wbits)
                else:
                    layer_bits = wbits

                # Quantize
                dq = quantize_tensor(weight, layer_bits, groupsize)
                tensors_to_save[key] = dq
                quantized += 1
            else:
                # Keep as-is, but cast bfloat16 to float16
                weight = f_in.get_tensor(key)
                if str(weight.dtype) == "bfloat16":
                    tensors_to_save[key] = weight.astype(np.float16)
                else:
                    tensors_to_save[key] = weight
                unchanged += 1

            # Periodic GC every 50 tensors
            if total_tensors % 50 == 0:
                gc.collect()

        # Save shard
        logger.info("Saving %d tensors to %s", len(tensors_to_save), output_path)
        mx.save_safetensors(output_path, {k: mx.array(v) for k, v in tensors_to_save.items()})

    elapsed = time.time() - t0
    logger.info(
        "  Quantized: %d, Unchanged: %d, Total: %d, Time: %.1fs",
        quantized, unchanged, total_tensors, elapsed,
    )

    return {
        "quantized": quantized,
        "unchanged": unchanged,
        "total": total_tensors,
        "time": elapsed,
    }


def quantize_shards(
    src_dir: str,
    dst_dir: str,
    method: str = "rtn",
    wbits: int = 4,
    groupsize: int = 128,
    rotation: str | None = None,
    autobit: bool = False,
    autobit_target: float = 4.0,
) -> dict:
    """Quantize a multi-shard model.

    Processes one shard at a time using mmap. Peak memory ~2-4GB.

    Steps:
        1. If autobit: profile all shards (single pass)
        2. Solve allocation (if autobit)
        3. Quantize each shard sequentially

    Args:
        src_dir: Source model directory.
        dst_dir: Output directory.
        method: "rtn" only (GPTQ needs calibration, not supported here).
        wbits: Target bits.
        groupsize: Quantization group size.
        rotation: "hadamard", "random", or None.
        autobit: Enable AutoBit per-layer allocation.
        autobit_target: Target average bits.

    Returns:
        Dict with results.
    """
    results = {}
    t_start = time.time()

    if method != "rtn":
        raise ValueError(f"Shard-based quantization only supports RTN, got {method}")

    os.makedirs(dst_dir, exist_ok=True)

    # Find shard files
    shard_files = sorted(glob.glob(os.path.join(src_dir, "model-*.safetensors")))
    if not shard_files:
        # Single file model
        shard_files = [os.path.join(src_dir, "model.safetensors")]

    if not shard_files:
        raise FileNotFoundError(f"No safetensors found in {src_dir}")

    logger.info("Found %d shard(s)", len(shard_files))

    # Collect hidden sizes for rotation
    rotation_cfg = None
    if rotation is not None:
        hidden_sizes = set()
        with safe_open(shard_files[0], framework="numpy") as f:
            for key in f.keys():
                if should_quantize(key):
                    w = f.get_tensor(key)
                    if w.ndim == 2:
                        hidden_sizes.add(w.shape[1])
        rotation_cfg = (rotation, hidden_sizes)
        logger.info("Rotation: %s, hidden_sizes: %s", rotation, sorted(hidden_sizes))

    # AutoBit: profile + solve (first shard representative)
    allocation = None
    if autobit:
        logger.info("AutoBit: profiling first shard...")
        t0 = time.time()

        profile = {}
        layer_sizes = {}
        with safe_open(shard_files[0], framework="numpy") as f:
            for key in f.keys():
                if should_quantize(key):
                    weight = f.get_tensor(key)
                    profile[key] = profile_layer(mx.array(weight), bits_list=(2, 3, 4, 8), groupsize=groupsize)
                    layer_sizes[key] = weight.shape[0] * weight.shape[1]

        allocation = solve_bit_allocation(profile, layer_sizes, target_avg_bits=autobit_target)
        logger.info("AutoBit allocation solved in %.1fs", time.time() - t0)
        results["autobit_profile_time"] = time.time() - t0

    # Process each shard
    shard_results = []
    for i, shard_path in enumerate(shard_files):
        basename = os.path.basename(shard_path)
        output_path = os.path.join(dst_dir, basename)

        shard_result = quantize_shard(
            shard_path,
            output_path,
            wbits=wbits,
            groupsize=groupsize,
            allocation=allocation,
            rotation=rotation_cfg,
        )
        shard_results.append(shard_result)

        # Force GC after each shard
        del shard_result
        gc.collect()

    # Copy config files
    config_files = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "tokenizer.model", "special_tokens_map.json",
        "processor_config.json", "chat_template.jinja", "chat_template-instruct.jinja",
        "model.safetensors.index.json", "README.md",
    ]
    for f in config_files:
        src = os.path.join(src_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, f))
            logger.info("Copied %s", f)

    results["shards"] = shard_results
    results["total_time"] = time.time() - t_start
    results["total_quantized"] = sum(s["quantized"] for s in shard_results)
    results["output_dir"] = dst_dir

    # Size comparison
    src_size = sum(os.path.getsize(s) for s in shard_files) / 1024 / 1024 / 1024
    dst_size = sum(
        os.path.getsize(os.path.join(dst_dir, os.path.basename(s)))
        for s in shard_files
        if os.path.exists(os.path.join(dst_dir, os.path.basename(s)))
    ) / 1024 / 1024 / 1024
    results["src_size_gb"] = src_size
    results["dst_size_gb"] = dst_size

    logger.info(
        "Shard quantization complete: %.2fGB → %.2fGB in %.1fs",
        src_size, dst_size, results["total_time"],
    )

    return results
