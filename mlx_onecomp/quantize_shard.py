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
import struct
import tempfile
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


def should_quantize(name: str) -> bool:
    return any(name.endswith(s) for s in QUANTIZE_PATTERNS)


def _load_bf16_to_fp16(
    shard_path: str, data_offsets: list, shape: list
) -> np.ndarray:
    """Read bfloat16 tensor from safetensors, convert to float16 numpy."""
    start, end = data_offsets
    n_bytes = end - start
    with open(shard_path, "rb") as f:
        f.seek(start)
        raw = f.read(n_bytes)
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).astype(np.float16)


def _get_tensor_info(shard_path: str) -> dict:
    """Read safetensors header, return {name: (dtype, shape, data_offsets)}."""
    with open(shard_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    info = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        info[k] = (v["dtype"], v["shape"], v["data_offsets"])
    return info


def _load_tensor_numpy(shard_path: str, name: str, dtype: str, shape: list,
                        data_offsets: list) -> np.ndarray:
    """Load a tensor from safetensors, handling bfloat16."""
    if dtype == "BF16":
        return _load_bf16_to_fp16(shard_path, data_offsets, shape)
    # For fp16/fp32, use safetensors numpy framework
    with safe_open(shard_path, framework="numpy") as f:
        return f.get_tensor(name)


def _save_safetensors_batch(
    tensors: dict, path: str
) -> str:
    """Save a batch of tensors as a safetensors file (temp file)."""
    mx.save_safetensors(path, {k: mx.array(v) for k, v in tensors.items()})
    return path


def _merge_safetensors_batches(batch_paths: list, output_path: str):
    """Merge safetensors batch files into a single output file.

    Uses streaming merge: reads headers for the unified header,
    then concatenates tensor data without loading into memory.
    """
    # Read all headers to build unified header, adjusting offsets
    all_tensors = {}
    data_offset = 0
    for bp in batch_paths:
        with open(bp, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        # Adjust data offsets
        for key, value in header.items():
            if key == "__metadata__":
                continue
            offsets = list(value["data_offsets"])
            offsets[0] += data_offset
            offsets[1] += data_offset
            value["data_offsets"] = offsets
            all_tensors[key] = value
        # Track data size of this batch
        data_size = os.path.getsize(bp) - 8 - header_len
        data_offset += data_size

    # Build unified header bytes (compact format, matches safetensors)
    header_json = json.dumps(all_tensors, separators=(",", ":")).encode()
    full_header = struct.pack("<Q", len(header_json)) + header_json

    # Write merged file
    with open(output_path, "wb") as out:
        out.write(full_header)
        for bp in batch_paths:
            with open(bp, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                f.read(header_len)  # skip header
                while True:
                    chunk = f.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    out.write(chunk)


def _pack_int4_to_uint32(q_int: np.ndarray) -> np.ndarray:
    """Pack int4 values (0-15) into uint32 (8 values per uint32).

    Matches MLX's mx.quantize output format for safetensors.

    Input shape: (out_features, in_features) with values in [0, 15].
    Output shape: (out_features, in_features // 8) as uint32.
    """
    assert q_int.shape[-1] % 8 == 0, \
        f"in_features ({q_int.shape[-1]}) must be divisible by 8"
    reshaped = q_int.reshape(-1, 8).astype(np.uint32)
    packed = np.zeros(reshaped.shape[:-1], dtype=np.uint32)
    for i in range(8):
        packed |= (reshaped[:, i] << (4 * i))
    return packed.reshape(q_int.shape[0], q_int.shape[1] // 8)


def quantize_tensor(weight: np.ndarray, wbits: int, groupsize: int) -> dict:
    """Quantize a single numpy weight tensor using RTN.

    Returns dict compatible with MLX QuantizedLinear format:
        weight: uint32 packed quantized values
        scales: float32 scale tensor
        biases: float32 zero-point tensor (MLX calls them biases)
        wbits: bits per value
        groupsize: group size used
    """
    w_mx = mx.array(weight.astype(np.float32))
    quantizer = RTN(wbits=wbits, groupsize=groupsize, sym=False)
    result = quantizer.quantize_weight(w_mx)

    q_int = np.array(result.quantized_weight)
    scales = np.array(result.scale.astype(mx.float32))
    biases = np.array(result.zero.astype(mx.float32))

    packed = _pack_int4_to_uint32(q_int)

    return {
        "weight": packed,
        "scales": scales,
        "biases": biases,
        "wbits": wbits,
        "groupsize": groupsize,
    }


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
    BATCH_SIZE = 20

    logger.info("Processing: %s", os.path.basename(shard_path))

    # Read tensor metadata from header
    tensor_info = _get_tensor_info(shard_path)

    # Create temp dir for streaming batch saves (use TMPDIR env var, falls back to output dir)
    output_dir = os.path.dirname(output_path) or "."
    temp_base = os.environ.get("MLX_TEMP_DIR") or output_dir
    batch_dir = tempfile.mkdtemp(prefix="mlx_shard_", dir=temp_base)
    batch_paths = []
    batch = {}
    batch_idx = 0

    def flush_batch():
        nonlocal batch_idx
        if not batch:
            return
        batch_path = os.path.join(batch_dir, f"batch_{batch_idx:06d}.safetensors")
        _save_safetensors_batch(batch, batch_path)
        batch_paths.append(batch_path)
        batch.clear()
        batch_idx += 1
        gc.collect()

    for key, (dtype, shape, data_offsets) in tensor_info.items():
        total_tensors += 1
        logger.debug("[%d] Processing %s", total_tensors, key)

        if should_quantize(key):
            weight = _load_tensor_numpy(shard_path, key, dtype, shape, data_offsets)

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

            # Quantize and save in MLX QuantizedLinear format
            qresult = quantize_tensor(weight, layer_bits, groupsize)
            # key already ends with ".weight" — use it directly for the weight tensor
            batch[key] = qresult["weight"]
            # For scales/biases: strip ".weight" suffix and add the right suffix
            base = key[:-7]  # e.g. "...mlp.down_proj"
            batch[f"{base}.scales"] = qresult["scales"]
            batch[f"{base}.biases"] = qresult["biases"]
            quantized += 1
        else:
            weight = _load_tensor_numpy(shard_path, key, dtype, shape, data_offsets)
            batch[key] = weight
            unchanged += 1

        del weight
        if total_tensors % BATCH_SIZE == 0:
            flush_batch()

    flush_batch()

    # Merge all batch files into final output
    if batch_paths:
        logger.info("Merging %d batch files → %s", len(batch_paths), output_path)
        _merge_safetensors_batches(batch_paths, output_path)

    # Clean up temp dir (in output dir)
    shutil.rmtree(batch_dir, ignore_errors=True)

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
        tensor_info = _get_tensor_info(shard_files[0])
        for key, (dtype, shape, data_offsets) in tensor_info.items():
            if should_quantize(key) and len(shape) == 2:
                hidden_sizes.add(shape[1])
        rotation_cfg = (rotation, hidden_sizes)
        logger.info("Rotation: %s, hidden_sizes: %s", rotation, sorted(hidden_sizes))

    # AutoBit: profile + solve (first shard representative)
    allocation = None
    if autobit:
        logger.info("AutoBit: profiling first shard...")
        t0 = time.time()

        profile = {}
        layer_sizes = {}
        tensor_info = _get_tensor_info(shard_files[0])
        for key, (dtype, shape, data_offsets) in tensor_info.items():
            if should_quantize(key):
                weight = _load_tensor_numpy(shard_files[0], key, dtype, shape, data_offsets)
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

    # Update config.json with quantization metadata for MLX compatibility
    config_path = os.path.join(dst_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        quant_config = {
            "group_size": groupsize,
            "bits": wbits,
            "mode": "affine",
        }
        config["quantization"] = quant_config
        config["quantization_config"] = quant_config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Updated config.json with quantization metadata")

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
