"""Test AutoBit pipeline."""

import sys
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

import mlx_lm
import mlx.nn as nn

from mlx_onecomp.autobit.profile import sensitivity_profile
from mlx_onecomp.autobit.solver import solve_bit_allocation
from mlx_onecomp.autobit.allocator import apply_allocation

MODEL = "mlx-community/lille-130m-instruct-fp16"

print(f"Loading {MODEL}")
model, tokenizer = mlx_lm.load(MODEL)

# Profile only attention + FFN Linear layers (skip embedding)
def layer_filter(name, mod):
    return "qkv_proj" in name or "out_proj" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name

print("Profiling sensitivity (2/3/4/8 bit)...")
profile = sensitivity_profile(model, bits_list=(2, 3, 4, 8), layer_filter=layer_filter)

# Layer sizes
layer_sizes = {}
for name, mod in model.named_modules():
    if name in profile and isinstance(mod, nn.Linear):
        layer_sizes[name] = mod.weight.shape[0] * mod.weight.shape[1]

# Solve for 4-bit average
print("\nSolving ILP (target: 4.0 bits avg)...")
allocation = solve_bit_allocation(profile, layer_sizes, target_avg_bits=4.0)

print("\nAllocation:")
for layer, bits in allocation.items():
    short = layer.split(".")[-1]
    print(f"  {short:20s}: {bits} bits")

# Apply
print("\nApplying allocation...")
result = apply_allocation(model, allocation)
print(f"Done! {result}")
