"""Test LoRA trainer."""

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

import mlx_lm
from mlx_onecomp.lora_trainer import LoRATrainer

MODEL = "mlx-community/lille-130m-instruct-fp16"

print(f"Loading {MODEL}")
model, tokenizer = mlx_lm.load(MODEL)

# Create LoRA trainer for last 4 blocks
trainer = LoRATrainer(model, rank=8, lora_layers=4)
print(f"LoRA modules: {len(trainer.lora_modules)}")

# Quick test with synthetic data
class FakeDataset:
    def __init__(self, n=16):
        self.texts = [f"Sample text number {i} with more words for sequence" for i in range(n)]
    def __iter__(self):
        return iter(self.texts)

print("\nTraining LoRA (3 steps, synthetic data)...")
trainer.train(FakeDataset(), steps=3, batch_size=2, seq_len=16, lr=1e-4)

# Save
trainer.save_lora("/tmp/test_lora_output")
print("\nDone!")
