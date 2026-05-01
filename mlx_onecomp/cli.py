"""CLI entry point for mlx-onecomp."""

import argparse
import glob
import logging
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="MLX-OneComp: LLM quantization on Apple Silicon"
    )
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument("--method", choices=["rtn", "gptq"], default="rtn")
    parser.add_argument("--wbits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--autobit", action="store_true")
    parser.add_argument("--autobit-target", type=float, default=4.0)
    parser.add_argument("--output", "-o", default="quantized_output")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Auto-detect: multi-shard safetensors → shard-based quantization
    if os.path.isdir(args.model):
        shards = sorted(glob.glob(os.path.join(args.model, "model-*.safetensors")))
        if shards:
            _shard_run(
                src_dir=args.model,
                dst_dir=args.output,
                method=args.method,
                wbits=args.wbits,
                groupsize=args.groupsize,
                autobit=args.autobit,
                autobit_target=args.autobit_target,
            )
            return

    # Single-file model → Runner pipeline
    _runner_run(
        model_path=args.model,
        output=args.output,
        method=args.method,
        wbits=args.wbits,
        groupsize=args.groupsize,
        autobit=args.autobit,
    )


def _shard_run(
    src_dir, dst_dir, method, wbits, groupsize, autobit, autobit_target
):
    from mlx_onecomp.quantize_shard import quantize_shards

    print(f"Shard-based quantization: {src_dir}")
    result = quantize_shards(
        src_dir=src_dir,
        dst_dir=dst_dir,
        method=method,
        wbits=wbits,
        groupsize=groupsize,
        autobit=autobit,
        autobit_target=autobit_target,
    )
    ratio = (result["dst_size_gb"] / result["src_size_gb"]) * 100
    print(f"Done: {result['src_size_gb']:.2f}GB → {result['dst_size_gb']:.2f}GB ({ratio:.1f}%)")


def _runner_run(model_path, output, method, wbits, groupsize, autobit):
    from mlx_onecomp.runner import Runner

    runner = Runner(model_path=model_path)

    if method == "rtn":
        runner.auto_run(wbits=wbits, groupsize=groupsize, method="rtn")
        runner.save(output)
    else:
        print("GPTQ via CLI requires calibration data. Use the Python API.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
