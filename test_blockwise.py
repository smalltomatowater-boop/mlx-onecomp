"""Test block-wise pipeline with a small model."""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from mlx_onecomp.pipeline.blockwise import BlockwisePipeline

# Use a tiny model that fits in memory
MODEL = "mlx-community/lille-130m-instruct-fp16"

print(f"Testing BlockwisePipeline with {MODEL}")

pipe = BlockwisePipeline(MODEL)
pipe.load()

print(f"Found {len(pipe.blocks)} transformer blocks")

# Calibrate with wikitext
pipe.calibrate(n_samples=4, seq_len=512)

print(f"Calibration inputs shape: {pipe.calibration_inputs.shape}")

# Run GPTQ quantization
pipe.run(method="gptq", wbits=4, groupsize=128, batch_size=4)

print(f"Quantized {len(pipe.quantized_blocks)} layers")

# Save
pipe.save("/tmp/test_blockwise_gptq")
print("Done! Saved to /tmp/test_blockwise_gptq")
