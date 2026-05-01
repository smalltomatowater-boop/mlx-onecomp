"""Benchmark shard-based quantization with lille-130m."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from mlx_onecomp.quantize_shard import quantize_shards

MODEL_PATH = "/Users/taku/.cache/huggingface/hub/models--mlx-community--lille-130m-instruct-fp16/snapshots/b03dde2e407d7e19dc10a2aeecfe90288cee42c0"
OUTPUT_DIR = "/tmp/bench-quantized"


def run_bench(label: str, **kwargs):
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    t0 = time.time()
    result = quantize_shards(
        src_dir=MODEL_PATH,
        dst_dir=OUTPUT_DIR,
        **kwargs,
    )
    wall = time.time() - t0
    result["wall_time"] = wall

    src = result.get("src_size_gb", 0)
    dst = result.get("dst_size_gb", 0)
    tq = result.get("total_quantized", 0)
    print(f"[{label}] {wall:.1f}s  "
          f"{src:.2f}GB → {dst:.2f}GB  "
          f"quantized={tq}  "
          f"total_time={result.get('total_time', 0):.1f}s")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  Shard Quantization Benchmark — lille-130m")
    print("=" * 60)

    # Test 1: RTN 4-bit
    r1 = run_bench("RTN 4-bit", wbits=4, groupsize=128)

    # Test 2: RTN 8-bit
    r2 = run_bench("RTN 8-bit", wbits=8, groupsize=128)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  RTN 4-bit : {r1['wall_time']:.1f}s  "
          f"src={r1.get('src_size_gb', 0):.2f}GB → dst={r1.get('dst_size_gb', 0):.2f}GB")
    print(f"  RTN 8-bit : {r2['wall_time']:.1f}s  "
          f"src={r2.get('src_size_gb', 0):.2f}GB → dst={r2.get('dst_size_gb', 0):.2f}GB")
    print(f"  Quantized layers: {r1.get('total_quantized', '?')}")
    print(f"\nOutput dir: {OUTPUT_DIR}")
