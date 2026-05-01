"""RTN 4-bit quantization for Gemma-4 31B shards.

Pseudo-quantization: quantize→dequantize weights, store as FP16.
Reduces precision to 4-bit quality while keeping MLX/MLX-LM compatibility.
Processes shard by shard to fit in 32GB RAM.
"""

import gc
import time
import os
import shutil

import mlx.core as mx
from mlx_onecomp.quantizer.rtn._rtn import RTN

# Quantize targets: attention + MLP projection layers
QUANTIZE_KEYS = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)

# RTN config: 4-bit, groupsize=128, asymmetric
RTN_WBITS = 4
RTN_GROUPSIZE = 128


def should_quantize(name: str) -> bool:
    return any(name.endswith(s) for s in QUANTIZE_KEYS)


def quantize_shard(input_path: str, output_path: str):
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    print("  Loading shard...")
    t0 = time.time()
    tensors = mx.load(input_path)
    if not isinstance(tensors, dict):
        print(f"  ERROR: Expected dict, got {type(tensors)}")
        return
    print(f"  Loaded {len(tensors)} tensors in {time.time()-t0:.1f}s")

    quantizer = RTN(wbits=RTN_WBITS, groupsize=RTN_GROUPSIZE, sym=False)

    quantized_count = 0
    unchanged_count = 0
    total_start = time.time()

    for name, tensor in tensors.items():
        t0 = time.time()

        if should_quantize(name):
            w = tensor.astype(mx.float32)
            result = quantizer.quantize_weight(w)
            # Store dequantized weight as FP16 (pseudo-quantization)
            tensors[name] = result.dequantized_weight.astype(mx.float16)
            quantized_count += 1

            elapsed = time.time() - t0
            orig_mb = tensor.size * tensor.itemsize / 1024 / 1024
            print(f"  [{quantized_count}] {name}: {tensor.shape} ({orig_mb:.0f}MB) {elapsed:.1f}s")
        else:
            # Convert bfloat16 → float16 for compatibility
            if str(tensor.dtype) == "mlx.core.bfloat16":
                tensors[name] = tensor.astype(mx.float16)
            unchanged_count += 1

        # Force GC every 50 tensors
        if (quantized_count + unchanged_count) % 50 == 0:
            gc.collect()

    # Save
    print(f"\n  Saving to {output_path}...")
    mx.save_safetensors(output_path, {k: v for k, v in tensors.items()})
    print(f"  Quantized: {quantized_count}, Unchanged: {unchanged_count}")
    print(f"  Total time: {time.time()-total_start:.1f}s")

    orig_size = os.path.getsize(input_path) / 1024 / 1024 / 1024
    new_size = os.path.getsize(output_path) / 1024 / 1024 / 1024
    print(f"  Original: {orig_size:.2f}GB → Output: {new_size:.2f}GB")

    del tensors
    gc.collect()


def main():
    src_dir = "/Volumes/LLM_MODEL/gemma4-31B-heretic-uncensored-fp16"
    dst_dir = "/Volumes/LLM_MODEL/gemma4-31B-heretic-uncensored-mlx4bit"

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "processor_config.json", "chat_template.jinja", "chat_template-instruct.jinja",
                   "model.safetensors.index.json", "README.md", "valhalla.webp"]:
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  Copied {f}")

    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

    for shard in shards:
        src = os.path.join(src_dir, shard)
        dst = os.path.join(dst_dir, shard)

        if os.path.exists(dst):
            print(f"  Skipping (exists): {shard}")
            continue

        quantize_shard(src, dst)

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"Output: {dst_dir}")
    total_gb = sum(
        os.path.getsize(os.path.join(dst_dir, f))
        for f in shards if os.path.exists(os.path.join(dst_dir, f))
    ) / 1024 / 1024 / 1024
    print(f"Total size: {total_gb:.2f}GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
