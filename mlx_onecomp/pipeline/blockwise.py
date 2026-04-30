"""Block-wise GPTQ/RTN quantization pipeline.

Processes transformer blocks one at a time:
  1. Forward calibration data through the block (FP16) to collect activations
  2. Compute Hessian from activations
  3. Quantize the block's Linear weights (GPTQ or RTN)
  4. Forward calibration data through the quantized block for the next layer
  5. Repeat for all blocks

Memory-efficient: only one block's weights + activations are in memory.
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

logger = logging.getLogger(__name__)


def _get_block_layers(model) -> list[tuple[str, nn.Module]]:
    """Extract (block_index, layer_name, module) for all transformer blocks."""
    blocks = []

    def _walk(prefix, module):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                blocks.append((full, child))
            _walk(full, child)

    _walk("", model)
    return blocks


def _find_blocks(model) -> list[tuple[str, nn.Module]]:
    """Find transformer block containers in the model.

    Looks for common patterns: model.layers, model.model.layers, etc.
    Returns list of (block_name, block_module).
    """
    candidates = [
        ("model.layers", lambda m: m.layers),
        ("model.model.layers", lambda m: m.model.layers),
        ("transformer.layers", lambda m: m.get("layers") if isinstance(m, dict) else m.layers),
    ]

    for name, getter in candidates:
        try:
            blocks = getter(model)
            if blocks is not None:
                return [(f"{name}.{i}", b) for i, b in enumerate(blocks)]
        except (AttributeError, KeyError, TypeError):
            continue

    # Fallback: search for any attribute that's a list of modules
    children_fn = getattr(model, "children", None) or getattr(model, "items", None)
    if children_fn:
        for name, child in (children_fn() if callable(children_fn) else []):
            sub = list(child.children()) if hasattr(child, "children") else []
            if len(sub) > 4:
                return [(f"{name}.{i}", c) for i, c in enumerate(sub)]

    raise ValueError("Could not find transformer blocks in model")


class BlockwisePipeline:
    """Block-wise quantization pipeline for MLX models.

    Usage:
        pipeline = BlockwisePipeline("TinyLlama/TinyLlama-1.1B")
        pipeline.load()
        pipeline.calibrate(n_samples=128, seq_len=2048)
        pipeline.run(method="gptq", wbits=4, groupsize=128)
        pipeline.save("output_dir")
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.blocks = []
        self.calibration_inputs = None  # (n_samples, seq_len, hidden_size)
        self.quantized_blocks = {}  # block_idx → quantization results

    def load(self):
        """Load model and tokenizer via mlx-lm."""
        import mlx_lm

        self.model, self.tokenizer = mlx_lm.load(self.model_path)
        self.model.eval()
        self.blocks = _find_blocks(self.model)
        logger.info("Loaded model: %d blocks", len(self.blocks))

    def calibrate(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train",
        n_samples: int = 128,
        seq_len: int = 2048,
        seed: int = 42,
    ):
        """Prepare calibration data.

        Tokenizes text data and runs through embedding layer to get
        hidden states as the initial input to the first transformer block.
        """
        from datasets import load_dataset

        mx.random.seed(seed)
        logger.info("Loading calibration data: %s/%s", dataset_name, dataset_config)

        data = load_dataset(dataset_name, dataset_config, split=split)
        text = "\n\n".join(data["text"])

        tokens = self.tokenizer.encode(text)
        token_ids = mx.array(tokens, dtype=mx.int32)

        # Split into samples of seq_len
        n_tokens = token_ids.shape[0]
        n_samples = min(n_samples, n_tokens // seq_len)
        token_ids = token_ids[: n_samples * seq_len].reshape(n_samples, seq_len)

        logger.info("Calibration: %d samples, seq_len=%d", n_samples, seq_len)

        # Run through embedding layer to get hidden states
        # This is model-specific; we hook into the model's embedding
        self.calibration_inputs = self._embed(token_ids)

    def _embed(self, token_ids: mx.array) -> mx.array:
        """Run token IDs through the embedding layer to get hidden states."""
        # Try common embedding locations
        for attr in ["embed_tokens", "wte", "embedding", "embed", "tok_embeddings"]:
            embed = getattr(self.model, attr, None)
            if embed is not None:
                return embed(token_ids)

        # Try model['transformer'].tok_embeddings or model.model.embed_tokens
        for inner_name in ["transformer", "model"]:
            inner = self.model.get(inner_name) if isinstance(self.model, dict) else getattr(self.model, inner_name, None)
            if inner is not None:
                for attr in ["embed_tokens", "wte", "tok_embeddings", "embedding", "embed"]:
                    embed = getattr(inner, attr, None)
                    if embed is not None:
                        return embed(token_ids)

        raise ValueError("Could not find embedding layer")

    def run(
        self,
        method: str = "gptq",
        wbits: int = 4,
        groupsize: int = 128,
        blocksize: int = 128,
        percdamp: float = 0.01,
        actorder: bool = False,
        sym: bool = True,
        mse: bool = False,
        batch_size: int = 8,
    ):
        """Run block-wise quantization on all transformer blocks.

        Args:
            method: "gptq" or "rtn".
            wbits: Bit width (2, 3, 4, 8).
            groupsize: Group size for quantization (-1 = per-channel).
            blocksize: GPTQ block size (columns per iteration).
            percdamp: Hessian damping factor.
            actorder: Sort by Hessian diagonal (GPTQ only).
            sym: Symmetric quantization.
            mse: MSE grid search for optimal scale.
            batch_size: Batch size for Hessian computation.
        """
        if self.model is None:
            self.load()
        if self.calibration_inputs is None:
            raise ValueError("Run calibrate() first")

        acts = self.calibration_inputs
        total_blocks = len(self.blocks)
        t_total = time.time()

        for i, (block_name, block) in enumerate(self.blocks):
            t0 = time.time()
            logger.info(
                "[%d/%d] Processing %s", i + 1, total_blocks, block_name
            )

            # Collect activations for each Linear layer in this block
            layer_acts = self._collect_activations(block, acts)

            # Quantize each Linear layer
            linears = [
                (n, m) for n, m in block.named_modules() if isinstance(m, nn.Linear)
            ]

            for layer_name, linear in linears:
                la = layer_acts.get(layer_name)
                if la is None:
                    logger.warning("No activations for %s, skipping", layer_name)
                    continue

                if method == "gptq":
                    hessian = compute_hessian(la, batch_size=batch_size)
                    result = run_gptq(
                        hessian=hessian,
                        weight=linear.weight,
                        blocksize=blocksize,
                        percdamp=percdamp,
                        wbits=wbits,
                        groupsize=groupsize,
                        actorder=actorder,
                        mse=mse,
                        sym=sym,
                    )
                    # Replace weight with dequantized version
                    from mlx_onecomp.inference import dequantize_weight

                    dq = dequantize_weight(
                        result["qweight"],
                        result["scales"],
                        result["qzeros"],
                        wbits,
                        groupsize,
                    )
                    linear.weight = dq
                    self.quantized_blocks[f"{block_name}.{layer_name}"] = result

                elif method == "rtn":
                    quantizer = RTN(wbits=wbits, groupsize=groupsize, sym=sym, mse=mse)
                    result = quantizer.quantize_weight(linear.weight)
                    linear.weight = result.dequantized_weight
                    self.quantized_blocks[f"{block_name}.{layer_name}"] = result

                del la
                gc.collect()

            # Forward through quantized block for next layer's activations
            acts = self._forward_block(block, acts)

            elapsed = time.time() - t0
            logger.info(
                "[%d/%d] %s done in %.1fs", i + 1, total_blocks, block_name, elapsed
            )

            # Periodic GC
            if (i + 1) % 4 == 0:
                gc.collect()

        logger.info(
            "Quantization complete: %d blocks in %.1fs",
            total_blocks,
            time.time() - t_total,
        )

    def _collect_activations(
        self, block: nn.Module, inputs: mx.array
    ) -> dict[str, mx.array]:
        """Forward inputs through block, collecting activations at each Linear layer.

        Steps through the block manually to capture correct input for each Linear.
        """
        activations = {}

        # Find sub-modules
        attn = None
        ffn = None
        for name in ["self_attn", "attention", "attn"]:
            attn = block.get(name) if hasattr(block, "get") else getattr(block, name, None)
            if attn is not None:
                attn_name = name
                break
        for name in ["mlp", "feed_forward", "ffn"]:
            ffn = block.get(name) if hasattr(block, "get") else getattr(block, name, None)
            if ffn is not None:
                ffn_name = name
                break

        if attn is None or ffn is None:
            return activations

        # Find norms
        norm1 = getattr(block, "input_layernorm", None) or getattr(block, "ln1", None)
        if norm1 is None:
            norm1 = getattr(attn, "norm", None)
        norm2 = getattr(block, "post_attention_layernorm", None) or getattr(block, "ln2", None)
        if norm2 is None:
            norm2 = getattr(ffn, "norm", None)

        # Find activation fn for MLP
        act_fn = getattr(ffn, "act_fn", None)

        # Step 1: Attention
        h_attn = norm1(inputs) if norm1 is not None else inputs

        # Collect input for qkv/q/k/v projections
        for name, mod in attn.named_modules():
            if isinstance(mod, nn.Linear) and name in ("qkv_proj", "q_proj", "k_proj", "v_proj", "query_key_value"):
                activations[f"{attn_name}.{name}"] = h_attn

        # Full attention forward to get output
        h_attn_out = attn(h_attn)

        # Collect input for out_proj
        for name, mod in attn.named_modules():
            if isinstance(mod, nn.Linear) and name in ("out_proj", "o_proj", "dense"):
                activations[f"{attn_name}.{name}"] = h_attn_out

        x = inputs + h_attn_out

        # Step 2: FFN
        h_ffn = norm2(x) if norm2 is not None else x

        # Collect input for gate/up projections
        gate_out = None
        for name, mod in ffn.named_modules():
            if isinstance(mod, nn.Linear) and name in ("gate_proj", "w1", "fc1"):
                activations[f"{ffn_name}.{name}"] = h_ffn
                gate_out = mod(h_ffn)
                if act_fn is not None:
                    gate_out = act_fn(gate_out)

        for name, mod in ffn.named_modules():
            if isinstance(mod, nn.Linear) and name in ("up_proj", "w3", "fc2"):
                activations[f"{ffn_name}.{name}"] = h_ffn
                up_out = mod(h_ffn)

        # down_proj input = gate_act * up
        if gate_out is not None:
            mlp_hidden = gate_out * up_out
        else:
            mlp_hidden = h_ffn

        for name, mod in ffn.named_modules():
            if isinstance(mod, nn.Linear) and name in ("down_proj", "w2", "fc3"):
                activations[f"{ffn_name}.{name}"] = mlp_hidden

        return activations

    def _forward_block(self, block: nn.Module, inputs: mx.array) -> mx.array:
        """Forward inputs through a block, returning the output activations."""
        # MLX: no no_grad needed (gradients explicit)
        return block(inputs)

    def save(self, output_dir: str):
        """Save quantized model to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_weights(str(output_path / "weights.safetensors"))

        # Copy tokenizer files from source if possible
        if self.tokenizer is not None:
            try:
                self.tokenizer._tokenizer.save_model(str(output_path))
            except Exception:
                pass

        logger.info("Saved quantized model to %s", output_dir)
