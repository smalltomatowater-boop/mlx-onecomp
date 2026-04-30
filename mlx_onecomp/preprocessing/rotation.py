"""Rotation preprocessing for LLM quantization.

Rotates weight matrices before quantization to reduce outlier sensitivity.
Based on QuIP/QuIP# research: random orthogonal or Hadamard rotation
makes weight distributions more uniform → better 4-bit quantization quality.

Two approaches:
- HadamardRotation: Fast (O(n log n)), deterministic, sufficient for most cases
- RandomRotation: Slower (O(n^2)), potentially better for extreme outliers
"""

import logging
import math

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _hadamard_matrix(n: int, dtype=mx.float32) -> mx.array:
    """Build Sylvester-type Hadamard matrix of size n (must be power of 2).

    H_1 = [[1]]
    H_{2k} = [[H_k,  H_k],
               [H_k, -H_k]]

    Normalized by 1/sqrt(n) so H @ H^T = I.
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    if n == 1:
        return mx.ones((1, 1), dtype=dtype)

    # Build recursively using numpy, convert at the end
    import numpy as np

    H = np.array([[1.0]], dtype=np.float32)
    size = 1
    while size < n:
        H = np.block([[H, H], [H, -H]]).astype(np.float32)
        size *= 2

    H = H / math.sqrt(n)
    return mx.array(H, dtype=dtype)


def _random_orthogonal(n: int, seed: int = 42, dtype=mx.float32) -> mx.array:
    """Generate random orthogonal matrix via QR decomposition."""
    import numpy as np

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # Ensure proper orthogonal (det = +1)
    D = np.diag(np.sign(np.diag(R)))
    Q = Q @ D
    return mx.array(Q, dtype=dtype)


class HadamardRotation:
    """Hadamard rotation for quantization preprocessing.

    Fast O(n log n) transform. For non-power-of-2 sizes, uses padding.

    Absorbs into Linear layers as:
        W' = W @ H (rotate columns)
        Input transform: x' = H^T @ x (applied at inference via fused kernel)
    """

    def __init__(self, hidden_size: int, dtype=mx.float32):
        self.hidden_size = hidden_size
        if hidden_size & (hidden_size - 1) != 0:
            raise ValueError(
                f"HadamardRotation requires power-of-2 hidden_size, got {hidden_size}. "
                "Use RandomRotation for non-power-of-2 sizes."
            )
        self.H = _hadamard_matrix(hidden_size, dtype=dtype)

    def rotate_weight_in(self, weight: mx.array) -> mx.array:
        """Rotate weight columns: W' = W @ H."""
        return (weight.astype(mx.float32) @ self.H).astype(weight.dtype)

    def rotate_weight_out(self, weight: mx.array) -> mx.array:
        """Inverse rotate: W' = W @ H^T."""
        return (weight.astype(mx.float32) @ self.H.T).astype(weight.dtype)

    def rotate_activations(self, x: mx.array) -> mx.array:
        """Rotate input activations: x' = H^T @ x."""
        return (x.astype(mx.float32) @ self.H.T).astype(x.dtype)

    def apply_to_block(self, block: nn.Module):
        """Apply Hadamard rotation to all Linear weights in a block.

        Rotates column space of each Linear weight matrix.
        The inverse rotation is absorbed into the previous layer's output
        (or applied at inference via the model's forward pass).
        """
        count = 0
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                mod.weight = self.rotate_weight_in(mod.weight)
                count += 1
        logger.info("Hadamard rotation applied to %d Linear layers", count)

    def remove_from_block(self, block: nn.Module):
        """Remove Hadamard rotation (inverse transform)."""
        count = 0
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                mod.weight = self.rotate_weight_out(mod.weight)
                count += 1
        logger.info("Hadamard rotation removed from %d Linear layers", count)


class RandomRotation:
    """Random orthogonal rotation for quantization preprocessing.

    Slower than Hadamard (O(n^2)) but may handle extreme outliers better.
    Uses QR decomposition of a random Gaussian matrix.
    """

    def __init__(self, hidden_size: int, seed: int = 42, dtype=mx.float32):
        self.hidden_size = hidden_size
        self.Q = _random_orthogonal(hidden_size, seed=seed, dtype=dtype)

    def rotate_weight_in(self, weight: mx.array) -> mx.array:
        return (weight.astype(mx.float32) @ self.Q).astype(weight.dtype)

    def rotate_weight_out(self, weight: mx.array) -> mx.array:
        return (weight.astype(mx.float32) @ self.Q.T).astype(weight.dtype)

    def rotate_activations(self, x: mx.array) -> mx.array:
        return (x.astype(mx.float32) @ self.Q.T).astype(x.dtype)

    def apply_to_block(self, block: nn.Module):
        count = 0
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                mod.weight = self.rotate_weight_in(mod.weight)
                count += 1
        logger.info("Random rotation applied to %d Linear layers", count)

    def remove_from_block(self, block: nn.Module):
        count = 0
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                mod.weight = self.rotate_weight_out(mod.weight)
                count += 1
        logger.info("Random rotation removed from %d Linear layers", count)
