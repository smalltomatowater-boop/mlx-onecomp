"""Benchmark shard quantization on Gemma-4 31B."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mlx_onecomp.quantize_shard import quantize_shards

SRC_DIR = "/Volumes/LLM_MODEL/gemma4-31B-heretic-uncensored-fp16"
DST_DIR = "/Volumes/LLM_MODEL/gemma4-quantized"


if __name__ == "__main__":
    import shutil

    print("=" * 60)
    print("  Gemma-4 31B Shard Quantization Benchmark")
    print("=" * 60)
    print(f"  Source:   {SRC_DIR}")
    print(f"  Output:   {DST_DIR}")
    print()

    # Clean previous output
    if os.path.exists(DST_DIR):
        print("  Removing old output...")
        shutil.rmtree(DST_DIR)
    os.makedirs(DST_DIR, exist_ok=True)

    result = quantize_shards(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        method="rtn",
        wbits=4,
        groupsize=128,
    )

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Total time:    {result['total_time']:.1f}s")
    print(f"  Source size:   {result['src_size_gb']:.2f} GB")
    print(f"  Output size:   {result['dst_size_gb']:.2f} GB")
    ratio = (result['dst_size_gb'] / result['src_size_gb']) * 100 if result['src_size_gb'] > 0 else 0
    print(f"  Compression:   {ratio:.1f}% ({result['src_size_gb']/result['dst_size_gb']:.1f}x smaller)" if result['dst_size_gb'] > 0 else "")
    print(f"  Total layers:  {result['total_quantized']}")
    for i, s in enumerate(result["shards"]):
        print(f"  Shard {i}:  {s['quantized']} quantized, "
              f"{s['unchanged']} unchanged, {s['time']:.1f}s")
    print(f"  Output: {DST_DIR}")
