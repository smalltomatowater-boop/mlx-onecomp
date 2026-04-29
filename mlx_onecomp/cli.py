"""CLI entry point for mlx-onecomp."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="MLX-OneComp: LLM quantization on Apple Silicon")
    parser.add_argument("model_id", help="HuggingFace model ID or local path")
    parser.add_argument("--method", choices=["rtn", "gptq"], default="rtn")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--groupsize", type=int, default=-1)
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--mse", action="store_true")
    parser.add_argument("--output", "-o", default="quantized_output")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from mlx_onecomp.runner import Runner

    runner = Runner(model_path=args.model_id)

    if args.method == "rtn":
        runner.auto_run(
            wbits=args.wbits,
            groupsize=args.groupsize,
            method="rtn",
            sym=args.sym,
            mse=args.mse,
        )
        runner.save(args.output)
    else:
        print("GPTQ via CLI requires calibration data. Use the Python API.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
