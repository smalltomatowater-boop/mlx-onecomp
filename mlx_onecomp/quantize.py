"""End-to-end model quantization pipeline.

Combines block-wise quantization, rotation preprocessing,
AutoBit per-layer allocation, and LoRA recovery.

Usage:
    from mlx_onecomp import quantize_model

    result = quantize_model(
        model_path="mlx-community/lille-130m-instruct-fp16",
        method="gptq",
        wbits=4,
        rotation="hadamard",
        output_dir="/tmp/quantized",
    )
"""

import gc
import logging
import os
import shutil
import time

import mlx.core as mx
import mlx.nn as nn
import mlx_lm

from mlx_onecomp.pipeline.blockwise import BlockwisePipeline
from mlx_onecomp.preprocessing.rotation import HadamardRotation, RandomRotation
from mlx_onecomp.autobit.profile import sensitivity_profile
from mlx_onecomp.autobit.solver import solve_bit_allocation
from mlx_onecomp.autobit.allocator import apply_allocation

logger = logging.getLogger(__name__)


def quantize_model(
    model_path: str,
    output_dir: str,
    method: str = "gptq",
    wbits: int = 4,
    groupsize: int = 128,
    rotation: str | None = None,
    autobit: bool = False,
    autobit_target: float = 4.0,
    calibration_data=None,
    layer_filter=None,
) -> dict:
    """Run end-to-end quantization pipeline.

    Steps:
        1. Load model
        2. Rotation preprocessing (optional)
        3. AutoBit per-layer allocation (optional)
        4. Block-wise quantization
        5. Save model

    Args:
        model_path: HuggingFace or local model path.
        output_dir: Directory to save quantized model.
        method: "rtn" or "gptq".
        wbits: Target bit width (used when autobit=False).
        groupsize: Quantization group size.
        rotation: "hadamard", "random", or None.
        autobit: Enable AutoBit per-layer allocation.
        autobit_target: Target average bits for AutoBit.
        calibration_data: Iterable of text for GPTQ calibration.
        layer_filter: Optional (name, module) -> bool for AutoBit.

    Returns:
        Dict with step results and timing.
    """
    results = {}
    t_start = time.time()

    # Step 1: Load model
    logger.info("Loading model: %s", model_path)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_path)
    results["load_time"] = time.time() - t0
    logger.info("Model loaded in %.1fs", results["load_time"])

    # Step 2: Rotation preprocessing
    if rotation is not None:
        logger.info("Rotation: %s", rotation)
        t0 = time.time()

        all_hidden = set()
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                all_hidden.add(mod.weight.shape[1])

        for h in sorted(all_hidden):
            if rotation == "hadamard":
                try:
                    rot = HadamardRotation(h)
                    logger.info("  Hadamard rotation for hidden_size=%d", h)
                except ValueError:
                    rot = RandomRotation(h)
                    logger.info("  Random rotation (non-power-of-2) for hidden_size=%d", h)
            else:
                rot = RandomRotation(h)

            _apply_rotation(model, rot)

        results["rotation"] = {"time": time.time() - t0, "type": rotation}

    # Step 3: AutoBit per-layer allocation
    if autobit:
        logger.info("AutoBit allocation (target=%.1f bits)", autobit_target)
        t0 = time.time()

        profile = sensitivity_profile(
            model, bits_list=(2, 3, 4, 8),
            groupsize=groupsize, layer_filter=layer_filter,
        )

        layer_sizes = {}
        for name, mod in model.named_modules():
            if name in profile and isinstance(mod, nn.Linear):
                layer_sizes[name] = mod.weight.shape[0] * mod.weight.shape[1]

        allocation = solve_bit_allocation(profile, layer_sizes, target_avg_bits=autobit_target)
        alloc_result = apply_allocation(model, allocation, groupsize=groupsize)
        results["autobit"] = {**alloc_result, "time": time.time() - t0}
        logger.info(
            "AutoBit done: %.1fs, avg=%.1f bits, %d layers",
            results["autobit"]["time"], autobit_target, alloc_result["quantized"],
        )

        # Save intermediate model for LoRA training
        # (AutoBit already quantized weights in-place)
        os.makedirs(output_dir, exist_ok=True)
        model.save_weights(f"{output_dir}/model.safetensors")
        results["autobit"]["saved"] = True

    else:
        # Step 4: Block-wise quantization (standard RTN/GPTQ)
        logger.info("Block-wise %s quantization (%d-bit)", method, wbits)
        t0 = time.time()

        pipeline = BlockwisePipeline(model_path)
        pipeline.load()

        if method == "gptq":
            pipeline.calibrate(n_samples=4, seq_len=512)
        else:
            # RTN still needs calibration_inputs for block-wise processing
            # Create synthetic calibration data
            # Get hidden size from qkv_proj (attention input dimension)
            hidden = 640  # default
            for n, m in pipeline.model.named_modules():
                if isinstance(m, nn.Linear) and ("qkv_proj" in n or "q_proj" in n):
                    hidden = m.weight.shape[1]
                    break
            n_tokens = 128
            pipeline.calibration_inputs = mx.zeros((1, n_tokens, hidden))

        pb_result = pipeline.run(method=method, wbits=wbits, groupsize=groupsize)
        results["blockwise"] = {"layers": len(pipeline.quantized_blocks), "time": time.time() - t0}

        # Save via pipeline
        os.makedirs(output_dir, exist_ok=True)
        pipeline.save(output_dir)

    # Step 5: Save model (non-pipeline path)
    if not autobit:
        # Pipeline already saved
        pass
    else:
        model.save_weights(f"{output_dir}/model.safetensors")

    # Copy config files
    if os.path.isdir(model_path):
        for f in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
            src = os.path.join(model_path, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, f))

    gc.collect()
    results["total_time"] = time.time() - t_start
    logger.info("Quantization complete in %.1fs", results["total_time"])
    return results


def _apply_rotation(model, rotation):
    """Apply weight rotation to all Linear layers matching the rotation size."""
    h = rotation.Q.shape[0] if hasattr(rotation, "Q") else rotation.H.shape[0]

    for _, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        w = mod.weight
        if w.shape[1] == h:
            mod.weight = rotation.rotate_weight_in(w)
