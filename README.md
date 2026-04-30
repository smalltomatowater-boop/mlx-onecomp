# mlx-onecomp

OneCompression ported to Apple MLX — LLM quantization for Apple Silicon.

Based on [FujitsuResearch/OneCompression](https://github.com/FujitsuResearch/OneCompression).

## Why

OneCompression provides state-of-the-art LLM quantization (GPTQ, RTN, QEP), but only works on NVIDIA CUDA. This project ports the core quantization algorithms to Apple Silicon using the MLX framework.

## Features

- **Block-wise GPTQ/RTN pipeline** — process transformer blocks sequentially for memory efficiency
- **Rotation preprocessing** — Hadamard/Random rotation to reduce outliers before quantization
- **AutoBit** — ILP-based optimal per-layer bit allocation
- **LoRA fine-tuning** — post-quantization recovery with low-rank adapters
- **End-to-end pipeline** — `quantize_model()` ties everything together

## Installation

```bash
pip install mlx numpy
pip install mlx-lm  # for model loading
pip install scipy   # for AutoBit
```

## Quick Start

```python
from mlx_onecomp import quantize_model

result = quantize_model(
    model_path="mlx-community/lille-130m-instruct-fp16",
    output_dir="/tmp/quantized",
    method="rtn",
    wbits=4,
)
print(f"Quantized {result['blockwise']['layers']} layers in {result['total_time']:.1f}s")
```

### With AutoBit per-layer allocation

```python
result = quantize_model(
    model_path="mlx-community/lille-130m-instruct-fp16",
    output_dir="/tmp/quantized",
    autobit=True,
    autobit_target=4.0,
)
```

### Standalone components

```python
# RTN quantizer
from mlx_onecomp.quantizer.rtn import RTN
import mlx.core as mx

quantizer = RTN(wbits=4, groupsize=128, sym=False)
result = quantizer.quantize_weight(weight)

# GPTQ
from mlx_onecomp.quantizer.gptq import run_gptq
from mlx_onecomp.calibration import compute_hessian

hessian = compute_hessian(activations)
result = run_gptq(hessian=hessian, weight=weight, wbits=4, groupsize=128)

# Rotation preprocessing
from mlx_onecomp.preprocessing.rotation import HadamardRotation

rot = HadamardRotation(hidden_size=512)
rotated_weight = rot.rotate_weight_in(weight)

# AutoBit
from mlx_onecomp.autobit.profile import sensitivity_profile
from mlx_onecomp.autobit.solver import solve_bit_allocation
from mlx_onecomp.autobit.allocator import apply_allocation

profile = sensitivity_profile(model, bits_list=(2, 3, 4, 8))
layer_sizes = {n: m.weight.shape[0] * m.weight.shape[1]
               for n, m in model.named_modules()
               if isinstance(m, nn.Linear)}
allocation = solve_bit_allocation(profile, layer_sizes, target_avg_bits=4.0)
apply_allocation(model, allocation)

# LoRA recovery
from mlx_onecomp.lora_trainer import LoRATrainer

trainer = LoRATrainer(model, rank=8, lora_layers=4)
trainer.train(dataset, steps=100, lr=1e-4)
trainer.save_lora("/tmp/lora_weights")
```

## Benchmarks

lille-130m (130M params, 24 blocks, FP16):

| Pipeline | Time |
|---|---|
| RTN | 2.2s (120 layers) |
| AutoBit (4.0 bits avg) | 6.9s (profile + solve + apply) |
| GPTQ | 13.9s |

## Project Structure

```
mlx_onecomp/
├── __init__.py                   # exports quantize_model
├── quantize.py                   # End-to-end pipeline
├── lora_trainer.py               # LoRA fine-tuning
├── quantizer/
│   ├── gptq/_gptq.py             # GPTQ core
│   └── rtn/_rtn.py               # RTN quantizer
├── pipeline/
│   └── blockwise.py              # Block-wise processing
├── preprocessing/
│   └── rotation.py               # Hadamard/Random rotation
├── autobit/
│   ├── profile.py                # Sensitivity profiling
│   ├── solver.py                 # ILP allocation solver
│   └── allocator.py              # Apply allocation to model
├── calibration/                  # Hessian computation
└── inference.py                  # Dequantization
```

## Status

| Feature | Status |
|---|---|
| GPTQ core | Working |
| RTN quantizer | Working |
| Hessian computation | Working |
| Block-wise pipeline | Working |
| Rotation preprocessing | Working |
| AutoBit (ILP) | Working |
| LoRA SFT post-process | Working |
| End-to-end quantization | Working |

## Related

- [FujitsuResearch/OneCompression](https://github.com/FujitsuResearch/OneCompression) — Original CUDA implementation
- [Issue #7](https://github.com/FujitsuResearch/OneCompression/issues/7) — Apple Silicon / MPS support request
- [Apple MLX](https://github.com/ml-explore/mlx) — MLX framework

## License

MIT License. Based on FujitsuResearch/OneCompression (MIT, Copyright 2025-2026 Fujitsu Ltd.).
