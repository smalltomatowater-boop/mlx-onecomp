"""GPTQ quantization core ported to Apple MLX.

Strategy:
- MLX for Cholesky decomposition (via CPU stream — Metal GPU not yet supported)
- NumPy for the inner GPTQ loop (requires in-place mutation)

Ported from FujitsuResearch/OneCompression onecomp/quantizer/gptq/_gptq.py
"""

import gc
import logging

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Cholesky inverse-Hessian via MLX
# ──────────────────────────────────────────────

def _compute_inverse_hessian(
    hessian: np.ndarray,
    percdamp: float,
    max_retries: int = 5,
) -> np.ndarray:
    """Compute upper-triangular Cholesky factor of inverse Hessian.

    Uses mx.linalg.cholesky / cholesky_inv via CPU stream.
    """
    H = hessian.copy().astype(np.float32)
    damp = percdamp * np.mean(np.diag(H))
    np.fill_diagonal(H, np.diag(H) + damp)

    damp_scale = 1.0
    for attempt in range(max_retries):
        try:
            H_mx = mx.array(H)
            L = mx.linalg.cholesky(H_mx, upper=False, stream=mx.cpu)
            break
        except Exception:
            damp_scale *= 10.0
            np.fill_diagonal(H, np.diag(H) + damp_scale * damp)
            logger.warning(
                "Cholesky failed (attempt %d/%d); extra damping %.2e",
                attempt + 1, max_retries, damp_scale * damp,
            )
    else:
        raise RuntimeError(
            f"Cholesky failed after {max_retries} attempts. Hessian may be ill-conditioned."
        )

    Hinv = mx.linalg.cholesky_inv(L, upper=False, stream=mx.cpu)
    result = mx.linalg.cholesky(Hinv, upper=True, stream=mx.cpu)
    return np.array(result, dtype=np.float32)


# ──────────────────────────────────────────────
# NumPy-based quantize/dequantize (for inner loop)
# ──────────────────────────────────────────────

def _np_quantize(x, scale, zero, maxq):
    if maxq < 0:
        return None
    return np.clip(np.round(x / scale) + zero, 0, maxq).astype(np.int32)


def _np_dequantize(quantized, scale, zero):
    return scale * (quantized.astype(np.float32) - zero)


def _np_quantize_trits(x, scale, zero):
    return (x > scale / 2).astype(np.float32) * scale + (x < zero / 2).astype(np.float32) * zero


# ──────────────────────────────────────────────
# Scale/zero finder (NumPy)
# ──────────────────────────────────────────────

class _ScaleFinder:
    """Find optimal scale/zero for quantization."""

    def __init__(self):
        self.maxq = None
        self.scale = None
        self.zero = None
        self.sym = None
        self.mse = None
        self.norm = None
        self.grid = None
        self.maxshrink = None

    def configure(self, bits, sym=True, mse=False, norm=2.4, grid=100, maxshrink=0.8):
        self.maxq = 2**bits - 1
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, W):
        """Find scale/zero for weight matrix W (out_features, group_cols)."""
        tmp = np.zeros(W.shape[0], dtype=np.float32)
        xmin = np.minimum(W.min(axis=1), tmp)
        xmax = np.maximum(W.max(axis=1), tmp)

        if self.sym:
            xmax = np.maximum(np.abs(xmin), xmax)
            xmin = np.where(xmin < 0, -xmax, xmin)

        dead = (xmin == 0) & (xmax == 0)
        xmin[dead] = -1
        xmax[dead] = 1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = np.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = np.round(-xmin / self.scale)

        if self.mse:
            best = np.full(W.shape[0], np.inf)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = np.round(-xmin1 / scale1) if not self.sym else self.zero
                q = _np_quantize(W, scale1.reshape(-1, 1), zero1.reshape(-1, 1), self.maxq)
                if q is not None:
                    dq = _np_dequantize(q, scale1.reshape(-1, 1), zero1.reshape(-1, 1), self.maxq)
                    err = np.sum(np.abs(dq - W) ** self.norm, axis=1)
                else:
                    continue
                improved = err < best
                if np.any(improved):
                    best[improved] = err[improved]
                    self.scale[improved] = scale1[improved]
                    self.zero[improved] = zero1[improved]

        self.scale = self.scale.reshape(-1, 1)
        self.zero = self.zero.reshape(-1, 1)


# ──────────────────────────────────────────────
# run_gptq: main quantization loop
# ──────────────────────────────────────────────

def run_gptq(
    hessian: mx.array,
    weight: mx.array,
    blocksize: int = 128,
    percdamp: float = 0.01,
    wbits: int = 4,
    groupsize: int = -1,
    actorder: bool = False,
    mse: bool = False,
    sym: bool = True,
    q_grid: int = 600,
    q_norm: float = 2.4,
) -> dict:
    """GPTQ quantization on weight matrix using Hessian.

    Args:
        hessian: (in_features, in_features) Hessian matrix.
        weight: (out_features, in_features) weight matrix.
        blocksize, percdamp, wbits, groupsize, actorder, mse, sym, q_grid, q_norm:
            Standard GPTQ parameters.

    Returns:
        dict with keys: qweight, scales, qzeros, perm (all mx.array).
    """
    hessian_f32 = hessian.astype(mx.float32)
    weight_f32 = weight.astype(mx.float32)
    H = np.array(hessian_f32, dtype=np.float32).copy()
    W = np.array(weight_f32, dtype=np.float32).copy()

    in_features = H.shape[0]
    out_features = W.shape[0]

    finder = _ScaleFinder()
    finder.configure(wbits, sym=sym, mse=mse, norm=q_norm, grid=q_grid)
    finder.find_params(W)

    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    perm = None
    invperm = None
    if actorder:
        perm = np.argsort(-np.diag(H))
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = np.argsort(perm)

    Q_int = np.zeros_like(W, dtype=np.int32)

    Hinv = _compute_inverse_hessian(H, percdamp)

    if groupsize != -1:
        num_groups = (in_features + groupsize - 1) // groupsize
        all_scales = np.zeros((out_features, num_groups), dtype=np.float32)
        all_zeros = np.zeros((out_features, num_groups), dtype=np.float32)

    maxq = 2**wbits - 1

    for i1 in range(0, in_features, blocksize):
        i2 = min(i1 + blocksize, in_features)
        count = i2 - i1

        W1 = W[:, i1:i2].copy()
        Q1 = np.zeros_like(W1, dtype=np.int32)
        Err1 = np.zeros_like(W1, dtype=np.float32)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if groupsize != -1:
                if (i1 + i) % groupsize == 0:
                    col_end = min(i1 + i + groupsize, in_features)
                    finder.find_params(W[:, i1 + i : col_end])
                    group_idx = (i1 + i) // groupsize
                    all_scales[:, group_idx] = finder.scale.squeeze(-1)
                    all_zeros[:, group_idx] = finder.zero.squeeze(-1)

            scale = finder.scale
            zero = finder.zero

            if maxq >= 0:
                q_int = _np_quantize(w.reshape(-1, 1), scale, zero, maxq)
                if q_int is not None:
                    q = _np_dequantize(q_int, scale, zero, maxq).flatten()
                    q_int = q_int.flatten()
                else:
                    q = _np_quantize_trits(w.reshape(-1, 1), scale, zero).flatten()
                    q_int = None
            else:
                q = _np_quantize_trits(w.reshape(-1, 1), scale, zero).flatten()
                q_int = None

            if q_int is not None:
                Q1[:, i] = q_int

            err1 = (w - q) / d
            W1[:, i:] -= err1.reshape(-1, 1) @ Hinv1[i, i:].reshape(1, -1)
            Err1[:, i] = err1

        Q_int[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    if actorder and invperm is not None:
        Q_int = Q_int[:, invperm]

    if groupsize != -1:
        scale = mx.array(all_scales.astype(np.float16).T)
        zero = mx.array(all_zeros.astype(np.int32).T)
    else:
        scale = mx.array(finder.scale.astype(np.float16))
        zero = mx.array(finder.zero.astype(mx.int32))

    result = {
        "qweight": mx.array(Q_int),
        "scales": scale,
        "qzeros": zero,
        "perm": mx.array(perm) if perm is not None else None,
    }

    del H, W, Q_int, Hinv
    gc.collect()

    return result


# ──────────────────────────────────────────────
# Pure MLX quantize/dequantize (for external use)
# ──────────────────────────────────────────────

def quantize(x: mx.array, scale: mx.array, zero: mx.array, maxq) -> mx.array | None:
    """Quantize float values to integers (MLX)."""
    maxq_val = maxq.item() if isinstance(maxq, mx.array) else maxq
    if maxq_val < 0:
        return None
    return mx.clip(mx.round(x / scale) + zero, 0, maxq_val).astype(mx.int32)


def dequantize(quantized: mx.array, scale: mx.array, zero: mx.array, maxq) -> mx.array:
    """Dequantize integer values back to float (MLX)."""
    return scale * (quantized.astype(mx.float32) - zero)
