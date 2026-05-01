# mlx-onecomp

OneCompression ported to Apple MLX — LLM quantization for Apple Silicon.

Based on [FujitsuResearch/OneCompression](https://github.com/FujitsuResearch/OneCompression).

## Why

OneCompression provides state-of-the-art LLM quantization (GPTQ, RTN, QEP), but only works on NVIDIA CUDA. This project ports the core quantization algorithms to Apple Silicon using the MLX framework.

## Features

- **Shard-based quantization** — process multi-shard models with 2-4GB RAM (RTN, 4-bit packed int4)
- **Block-wise GPTQ/RTN pipeline** — process transformer blocks sequentially for memory efficiency
- **Rotation preprocessing** — Hadamard/Random rotation to reduce outliers before quantization
- **AutoBit** — ILP-based optimal per-layer bit allocation
- **LoRA fine-tuning** — post-quantization recovery with low-rank adapters
- **End-to-end pipeline** — `quantize_model()` ties everything together

## Installation

```bash
pip install mlx numpy safetensors
pip install mlx-lm  # for model loading
pip install scipy   # for AutoBit
```

## Quick Start

### Shard-based quantization (low RAM, large models)

Process multi-shard safetensors models with just 2-4GB RAM:

```python
from mlx_onecomp.quantize_shard import quantize_shards

result = quantize_shards(
    src_dir="/path/to/model",
    dst_dir="/path/to/output",
    method="rtn",
    wbits=4,
    groupsize=128,
)
print(f"{result['src_size_gb']:.2f}GB → {result['dst_size_gb']:.2f}GB")
```

### CLI (one-line)

```bash
# Multi-shard model (auto-detect) — 2-4GB RAM
mlx-onecomp /path/to/model -o /path/to/output

# With options
mlx-onecomp /path/to/model -o /path/to/output --wbits 4 --groupsize 128
mlx-onecomp /path/to/model -o /path/to/output --wbits 2 --autobit
```

### End-to-end pipeline (single-file models)

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

### lille-130m (130M params, 24 blocks, FP16)

| Pipeline | Time |
|---|---|
| RTN | 2.2s (120 layers) |
| AutoBit (4.0 bits avg) | 6.9s (profile + solve + apply) |
| GPTQ | 13.9s |

### Gemma-4 31B (Shard-based RTN, 4-bit packed int4)

| Metric | Value |
|---|---|
| Source size | 58.25 GB (FP16) |
| Output size | 18.19 GB (4-bit RTN) |
| Compression | 31.2% (3.2x smaller) |
| Total time | 25.6 min (410 layers) |
| Peak RAM | ~2-4 GB per shard |

## Project Structure

```
mlx_onecomp/
├── __init__.py                   # exports quantize_model
├── quantize.py                   # End-to-end pipeline
├── quantize_shard.py             # Shard-based quantization (low RAM)
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
| Shard quantization (packed int4) | Gemma-4 31B tested |
| GPTQ core | Synthetic data verified |
| RTN quantizer | Synthetic data verified |
| Hessian computation | GPTQ path verified |
| Block-wise pipeline | lille-130m tested |
| Rotation preprocessing | Orthogonality verified |
| AutoBit (ILP) | lille-130m tested |
| LoRA SFT post-process | Synthetic data verified |
| End-to-end quantization | lille-130m tested |

## Related

- [FujitsuResearch/OneCompression](https://github.com/FujitsuResearch/OneCompression) — Original CUDA implementation
- [Issue #7](https://github.com/FujitsuResearch/OneCompression/issues/7) — Apple Silicon / MPS support request
- [Apple MLX](https://github.com/ml-explore/mlx) — MLX framework

## CLI

```bash
# Multi-shard model (auto-detect) — 2-4GB RAM
mlx-onecomp /path/to/model -o /path/to/output

# With options
mlx-onecomp /path/to/model -o /path/to/output --wbits 4 --groupsize 128
mlx-onecomp /path/to/model -o /path/to/output --wbits 3
mlx-onecomp /path/to/model -o /path/to/output --wbits 2 --autobit
```

## License

MIT License. Based on FujitsuResearch/OneCompression (MIT, Copyright 2025-2026 Fujitsu Ltd.).

## 日本語

[FujitsuResearch/OneCompression](https://github.com/FujitsuResearch/OneCompression)はNVIDIA CUDA版のみ提供されている高品質LLM量子化ライブラリです。このプロジェクトはApple Silicon向けにMLXへ移植したものです。

**主な特徴：**
- シェード単位の量子化 — 2-4GBのRAMで大容量モデルに対応（4bitパックint4）
- ブロック単位のGPTQ/RTN — メモリ効率のよい変換
- AutoBit — ILPによるレイヤーごとの最適ビット割り当て
- LoRA — 量子化後の微調整による品質回復

### クイックスタート

```bash
# 量子化（multi-shardモデルを自動検知）
mlx-onecomp /path/to/model -o /path/to/output

# ビット数指定（2/3/4/8bitから選択）
mlx-onecomp /path/to/model -o /path/to/output --wbits 4

# AutoBit（レイヤーごとに2-8bitを自動最適化）
mlx-onecomp /path/to/model -o /path/to/output --wbits 2 --autobit
```

### 実証テスト結果

| モデル | 量子化 | 圧縮前 | 圧縮後 | 時間 |
|---|---|---|---|---|
| Gemma-4 31B | 4-bit RTN | 58.25GB | 18.19GB | 25.6分 |
| lille-130m | RTN | - | - | 2.2秒 |
