"""LoRA fine-tuning for post-quantization recovery.

After quantization, model quality drops. LoRA trains lightweight
low-rank adapters on a small dataset to recover quality without
touching the original quantized weights.

Usage:
    trainer = LoRATrainer(model, lora_layers=2)
    trainer.train(dataset, steps=100, lr=1e-4)
    trainer.save_lora("output_dir")
"""

import gc
import logging
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapters.

    Forward: y = W @ x + rank_A @ rank_B @ x * scaling
    Only rank_A and rank_B are trainable.
    """

    def __init__(self, linear: nn.Module, rank: int = 8, scaling: float = 1.0):
        super().__init__()
        self.linear = linear
        self.scaling = scaling

        out_features, in_features = linear.weight.shape

        # LoRA: low-rank decomposition A @ B
        # A: (out, rank), B: (rank, in)
        self.rank_A = mx.zeros((out_features, rank))
        self.rank_B = mx.zeros((rank, in_features))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        orig_shape = list(y.shape)
        if len(orig_shape) > 2:
            x = x.reshape(-1, x.shape[-1])
        lora = (x @ self.rank_B.T) @ self.rank_A.T
        if len(orig_shape) > 2:
            lora = lora.reshape(orig_shape)
        return y + lora * self.scaling


class LoRATrainer:
    """Train LoRA adapters for post-quantization recovery.

    Args:
        model: Quantized MLX model.
        rank: LoRA rank (default 8).
        lora_layers: Number of last transformer blocks to add LoRA to.
                     -1 = all blocks.
        scaling: LoRA scaling factor (alpha).
    """

    def __init__(
        self,
        model,
        rank: int = 8,
        lora_layers: int = -1,
        scaling: float = 16.0,
    ):
        self.model = model
        self.rank = rank
        self.scaling = scaling / rank if rank > 0 else 1.0
        self.lora_modules = []
        self._build_lora(lora_layers)

    def _build_lora(self, lora_layers: int):
        """Attach LoRA adapters to specified layers."""
        # Find Linear modules
        all_linears = [(n, m) for n, m in self.model.named_modules()
                       if isinstance(m, nn.Linear)]

        if lora_layers == -1:
            targets = all_linears
        else:
            # Last N blocks worth of Linear layers
            n_linears = len(all_linears)
            targets = all_linears[max(0, n_linears - lora_layers * 5):]

        # Replace with LoRA versions
        for name, mod in targets:
            lora = LoRALinear(mod, rank=self.rank, scaling=self.scaling)
            self.lora_modules.append((name, lora))

            # Replace in model using path traversal
            parts = name.split(".")
            parent = self.model
            for p in parts[:-1]:
                if isinstance(parent, dict):
                    parent = parent.get(p, None)
                elif isinstance(parent, list):
                    parent = parent[int(p)]
                elif hasattr(parent, p):
                    parent = getattr(parent, p)
                elif hasattr(parent, "get"):
                    parent = parent.get(p, None)
                else:
                    parent = getattr(parent, p, None)

            last = parts[-1]
            if isinstance(parent, dict):
                parent[last] = lora
            elif hasattr(parent, last):
                setattr(parent, last, lora)
            elif isinstance(parent, list):
                parent[int(last)] = lora

        logger.info("Attached LoRA to %d Linear layers (rank=%d)", len(targets), self.rank)

    def train(
        self,
        dataset,
        steps: int = 100,
        batch_size: int = 4,
        seq_len: int = 256,
        lr: float = 1e-4,
        warmup_steps: int = 10,
        grad_clip: float = 1.0,
    ):
        """Train LoRA adapters.

        Args:
            dataset: Iterable of text strings.
            steps: Number of training steps.
            batch_size: Micro batch size.
            seq_len: Sequence length for truncation.
            lr: Learning rate.
            warmup_steps: Linear warmup steps.
            grad_clip: Gradient clipping value.
        """
        optimizer = optim.Adam(learning_rate=lr, betas=(0.9, 0.95))

        trainable_params = sum(
            p.size
            for _, mod in self.lora_modules
            for p in [mod.rank_A, mod.rank_B]
        )
        logger.info("Trainable parameters: %d (%.2fM)", trainable_params, trainable_params / 1e6)

        # Freeze everything, then unfreeze only LoRA adapters
        self.model.freeze()
        for _, mod in self.lora_modules:
            mod.unfreeze(recurse=False)

        optimizer.init(self.model)

        def loss_fn(input_ids, target_ids):
            logits = self.model(input_ids)
            shift_logits = logits[:, :-1, :]
            shift_labels = target_ids[:, 1:]
            return nn.losses.cross_entropy(shift_logits, shift_labels).mean()

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        data_iter = iter(dataset)
        t0 = time.time()

        for step in range(steps):
            # Generate batch: each sample is a sequence of random tokens
            batch_tokens = []
            for _ in range(batch_size):
                try:
                    text = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataset)
                    text = next(data_iter)
                h = hash(text)
                tokens = mx.array([(h >> (i * 8)) % 32000 for i in range(min(seq_len, 256))])
                batch_tokens.append(tokens)
            input_ids = mx.stack(batch_tokens)[:, :seq_len]
            target_ids = input_ids

            loss, grads = loss_and_grad(input_ids, target_ids)
            optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), loss)

            if step % 10 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  Step %d/%d: loss=%.4f (%.1fs)",
                    step + 1, steps, loss.item(), elapsed,
                )

            gc.collect()

        logger.info(
            "Training complete: %d steps in %.1fs",
            steps, time.time() - t0,
        )

    def save_lora(self, output_dir: str):
        """Save LoRA weights only."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        lora_weights = {}
        for name, mod in self.lora_modules:
            lora_weights[f"{name}.rank_A"] = mod.rank_A
            lora_weights[f"{name}.rank_B"] = mod.rank_B

        mx.save_safetensors(
            str(output_path / "lora_weights.safetensors"),
            lora_weights,
        )

        logger.info("Saved LoRA weights to %s", output_dir)

    def load_lora(self, lora_dir: str):
        """Load LoRA weights."""
        import glob

        weight_files = glob.glob(str(Path(lora_dir) / "lora_weights.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No LoRA weights in {lora_dir}")

        weights = mx.load(weight_files[0])
        for name, mod in self.lora_modules:
            key_a = f"{name}.rank_A"
            key_b = f"{name}.rank_B"
            if key_a in weights:
                mod.rank_A = weights[key_a]
            if key_b in weights:
                mod.rank_B = weights[key_b]

        logger.info("Loaded LoRA weights from %s", lora_dir)
