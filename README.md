# mlx-onecomp

OneCompression ported to Apple MLX — LLM quantization for Apple Silicon.

Based on [FujitsuResearch/OneCompression](https://github.com/FujitsuResearch/OneCompression).

## Why

OneCompression provides state-of-the-art LLM quantization (GPTQ, RTN, QEP), but only works on NVIDIA CUDA. This project ports the core quantization algorithms to Apple Silicon using the MLX framework, enabling:

- **GPTQ quantization** with Hessian-based optimal weight compression
- **RTN quantization** for fast weight-only compression
- **Apple Silicon native** — no CUDA dependency

## Key Findings from the Port

### 1. MLX Cholesky Support

MLX provides full linear algebra support including `mx.linalg.cholesky` and `mx.linalg.cholesky_inv` — the core operations needed for GPTQ's inverse Hessian computation. However, as of MLX 0.31, Cholesky requires a **CPU stream** (not yet GPU-accelerated):

```python
L = mx.linalg.cholesky(H, upper=False, stream=mx.cpu)
```

This is a key finding for anyone attempting similar ports. PyTorch MPS also lacks Cholesky support, making MLX the only viable path on Apple Silicon.

### 2. MLX Array Immutability

MLX arrays are immutable — no in-place assignment (`A[i] = x`). The GPTQ inner loop requires heavy in-place mutation of weight matrices. Solution: **hybrid approach**.

| Component | Framework | Reason |
|---|---|---|
| Cholesky decomposition | MLX | Linear algebra via CPU stream |
| Hessian computation | MLX | Matrix multiplication |
| GPTQ inner loop | NumPy | Requires in-place mutation |
| RTN quantization | MLX | Pure functional operations |

### 3. OneCompression CUDA Dependencies

Analysis of the 30+ CUDA references in OneCompression:

- `torch.cuda.empty_cache()` (30+ calls) → **unnecessary** in MLX (unified memory)
- `device="cuda:0"` defaults → **not needed** (unified memory, no device transfers)
- `torch.linalg.cholesky` → `mx.linalg.cholesky(stream=mx.cpu)` works
- `fast_hadamard_transform` (CUDA-only) → needs MLX Metal kernel replacement

## Benchmarks

Tested on real Gemma-4 weights (vision encoder, 4304×1152, bfloat16):

| Method | Bits | MSE | Time |
|---|---|---|---|
| RTN | 4-bit | 0.000010 | <0.01s |
| RTN | 3-bit | 0.000042 | <0.01s |
| RTN | 2-bit | 0.000222 | <0.01s |
| GPTQ | 4-bit | 0.000021 | 0.39s |
| GPTQ | 3-bit | 0.000153 | 0.37s |
| GPTQ | 2-bit | 0.000768 | 0.38s |

Note: GPTQ uses synthetic Hessian here. With real calibration data, GPTQ significantly outperforms RTN.

## Installation

```bash
pip install mlx numpy
pip install mlx-lm  # optional, for model loading
```

## Usage

### RTN Quantization

```python
import mlx.core as mx
from mlx_onecomp.quantizer.rtn import RTN

# Load weight matrix
weight = mx.load("weights.safetensors")["layer.weight"]

# Quantize to 4-bit with group size 128
quantizer = RTN(wbits=4, groupsize=128, sym=False)
result = quantizer.quantize_weight(weight)

print(f"MSE: {result.dequantized_weight}")
print(f"Scale shape: {result.scale.shape}")
```

### GPTQ Quantization

```python
import mlx.core as mx
from mlx_onecomp.quantizer.gptq import run_gptq
from mlx_onecomp.calibration import compute_hessian

# Get calibration activations (from model forward pass)
activations = ...  # (num_samples, seq_len, hidden_size)

# Compute Hessian
hessian = compute_hessian(activations)

# Run GPTQ
result = run_gptq(
    hessian=hessian,
    weight=layer_weight,
    wbits=4,
    groupsize=128,
    sym=True,
)
```

## Project Structure

```
mlx_onecomp/
├── quantizer/
│   ├── gptq/_gptq.py      # GPTQ core (Cholesky + quantization loop)
│   └── rtn/_rtn.py         # RTN quantizer
├── calibration/
│   └── calibration.py      # Hessian computation + QEP cross-term
├── inference.py            # Dequantization utilities
├── runner.py               # Top-level quantization runner
└── cli.py                  # CLI entry point
```

## Status

| Feature | Status |
|---|---|
| GPTQ core | Working |
| RTN quantizer | Working |
| Hessian computation | Working |
| QEP cross-term | Working |
| Block-wise pipeline | TODO |
| Rotation preprocessing | TODO |
| AutoBit (ILP) | TODO |
| LoRA SFT post-process | TODO |
| End-to-end model quantization | TODO (needs blockwise pipeline) |

## Related

- [FujitsuResearch/OneCompression](https://github.com/FujitsuResearch/OneCompression) — Original CUDA implementation
- [Issue #7](https://github.com/FujitsuResearch/OneCompression/issues/7) — Apple Silicon / MPS support request
- [Apple MLX](https://github.com/ml-explore/mlx) — MLX framework

## License

MIT License. Based on FujitsuResearch/OneCompression (MIT, Copyright 2025-2026 Fujitsu Ltd.).
