"""Top-level runner for MLX-OneComp quantization.

Usage:
    from mlx_onecomp.runner import Runner

    runner = Runner(model_path="TinyLlama/TinyLlama-1.1B")
    runner.quantize(wbits=4, groupsize=128, method="gptq")
    runner.save("output_dir")
"""

import gc
import logging
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_onecomp.calibration.calibration import compute_hessian
from mlx_onecomp.quantizer.gptq._gptq import run_gptq
from mlx_onecomp.quantizer.rtn._rtn import RTN
from mlx_onecomp.inference import dequantize_weight

logger = logging.getLogger(__name__)


class Runner:
    """Quantize MLX models with GPTQ or RTN."""

    def __init__(self, model_path: str, dtype: str = "float16"):
        self.model_path = model_path
        self.dtype = mx.float16 if dtype == "float16" else mx.float32
        self.model = None
        self.tokenizer = None
        self.results = {}

    def load_model(self):
        """Load model with mlx-lm or safetensors."""
        try:
            import mlx_lm
            self.model, self.tokenizer = mlx_lm.load(self.model_path)
            logger.info("Model loaded via mlx_lm: %s", self.model_path)
        except ImportError:
            raise ImportError(
                "mlx-lm is required for model loading. "
                "Install with: pip install mlx-lm"
            )

    def _get_linear_layers(self) -> list[tuple[str, nn.Linear]]:
        """Find all Linear layers in the model."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append((name, module))
        return layers

    def quantize_rtn(
        self,
        wbits: int = 4,
        groupsize: int = -1,
        sym: bool = False,
        mse: bool = False,
    ):
        """Quantize all Linear layers with RTN."""
        if self.model is None:
            self.load_model()

        quantizer = RTN(wbits=wbits, groupsize=groupsize, sym=sym, mse=mse)
        layers = self._get_linear_layers()
        logger.info("RTN quantizing %d layers", len(layers))

        for name, module in layers:
            t0 = time.time()
            result = quantizer.quantize_weight(module.weight)
            result.quantization_time = time.time() - t0
            self.results[name] = result
            logger.info("  %s: %.1fs", name, result.quantization_time)

    def quantize_gptq(
        self,
        calibration_data: mx.array,
        wbits: int = 4,
        groupsize: int = -1,
        blocksize: int = 128,
        percdamp: float = 0.01,
        actorder: bool = False,
        sym: bool = True,
        mse: bool = False,
        batch_size: int = 8,
    ):
        """Quantize all Linear layers with GPTQ.

        Args:
            calibration_data: Activation inputs (num_samples, seq_len, hidden_size).
                              Must be pre-collected from the first transformer block.
            wbits: Bit width.
            groupsize: Group size (-1 = per-channel).
            blocksize: GPTQ block size.
            percdamp: Hessian damping.
            actorder: Sort by activation magnitude.
            sym: Symmetric quantization.
            mse: MSE grid search.
            batch_size: Calibration batch size.
        """
        if self.model is None:
            self.load_model()

        layers = self._get_linear_layers()
        logger.info("GPTQ quantizing %d layers (wbits=%d, groupsize=%d)", len(layers), wbits, groupsize)

        for name, module in layers:
            t0 = time.time()

            # Compute Hessian from calibration activations
            hessian = compute_hessian(calibration_data, batch_size=batch_size)

            weight = module.weight.astype(mx.float32)

            result_dict = run_gptq(
                hessian=hessian,
                weight=weight,
                blocksize=blocksize,
                percdamp=percdamp,
                wbits=wbits,
                groupsize=groupsize,
                actorder=actorder,
                mse=mse,
                sym=sym,
            )

            elapsed = time.time() - t0
            self.results[name] = {
                "wbits": wbits,
                "groupsize": groupsize,
                "sym": sym,
                "actorder": actorder,
                "quantization_time": elapsed,
                **result_dict,
            }
            logger.info("  %s: %.1fs", name, elapsed)

            del hessian
            gc.collect()

    def apply_quantization(self):
        """Replace Linear layers with dequantized weights in the model."""
        for name, result in self.results.items():
            parts = name.split(".")
            parent = self.model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            if isinstance(result, dict):
                # GPTQ result
                deq = dequantize_weight(
                    result["qweight"],
                    result["scales"],
                    result["qzeros"],
                    result["wbits"],
                    result["groupsize"],
                )
            else:
                # RTN result
                deq = result.dequantized_weight

            module = getattr(parent, parts[-1])
            module.weight = deq

        logger.info("Applied quantization to %d layers", len(self.results))

    def save(self, output_dir: str):
        """Save the quantized model."""
        import mlx_lm
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            # Save weights as safetensors
            mx.save_safetensors(
                str(output_path / "weights.safetensors"),
                dict(self.model.parameters()),
            )
            logger.info("Saved model to %s", output_dir)

    def auto_run(
        self,
        wbits: int = 4,
        groupsize: int = 128,
        method: str = "gptq",
        **kwargs,
    ):
        """One-command quantization (mirrors onecomp <model>)."""
        self.load_model()

        if method == "rtn":
            self.quantize_rtn(wbits=wbits, groupsize=groupsize, **kwargs)
        elif method == "gptq":
            # For GPTQ, calibration data must be provided separately
            raise ValueError(
                "GPTQ requires calibration_data. Use quantize_gptq() directly."
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.apply_quantization()
        logger.info("Quantization complete")
