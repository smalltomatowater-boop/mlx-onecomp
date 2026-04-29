"""RTN (Round-To-Nearest) quantizer ported to MLX.

Ported from FujitsuResearch/OneCompression onecomp/quantizer/rtn/
"""

import gc
import logging
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Quantize / Dequantize
# ──────────────────────────────────────────────

def _quantize(x, scale, zero_point, q_min, q_max):
    return mx.clip(mx.round(x / scale) + zero_point, q_min, q_max).astype(mx.int32)


def _dequantize(quantized, scale, zero_point):
    return (quantized.astype(mx.float32) - zero_point) * scale


def pseudo_quantize_tensor(
    w: mx.array,
    n_bit: int = 8,
    q_group_size: int = -1,
    zero_point: bool = True,
    perchannel: bool = True,
    mse: bool = False,
    norm: float = 2.4,
    grid: int = 100,
    maxshrink: float = 0.8,
) -> tuple:
    """Pseudo-quantize a weight tensor (RTN method).

    Returns (dequantized_weight, scale, zero_point, quantized_weight).
    """
    org_shape = w.shape

    if q_group_size > 0:
        if w.shape[-1] % q_group_size != 0:
            raise ValueError(
                f"Tensor shape {w.shape[-1]} must be divisible by group size {q_group_size}"
            )
        w = w.reshape(-1, w.shape[-1] // q_group_size, q_group_size)
    elif perchannel:
        w = w.reshape(-1, 1, w.shape[-1])
    else:
        w = w.reshape(1, 1, -1)

    sym = not zero_point
    q_max = 2**n_bit - 1
    q_min = 0

    tmp = mx.array(0.0)
    w_max = mx.maximum(w.max(axis=-1, keepdims=True), tmp)
    w_min = mx.minimum(w.min(axis=-1, keepdims=True), tmp)

    if sym:
        w_max = mx.maximum(mx.abs(w_min), w_max)
        w_min = -w_max

    dead = (w_min == 0) & (w_max == 0)
    w_min = mx.where(dead, mx.array(-1.0), w_min)
    w_max = mx.where(dead, mx.array(1.0), w_max)

    scale = mx.maximum((w_max - w_min) / q_max, 1e-5)

    if sym:
        zp = mx.full(scale.shape, (q_max + 1) / 2, dtype=scale.dtype)
    else:
        zp = mx.round(-w_min / scale)

    if mse:
        best = mx.full(w.shape[:-1], float("inf"))
        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid
            wmin1 = p * w_min
            wmax1 = p * w_max
            scale1 = mx.maximum((wmax1 - wmin1) / q_max, 1e-5)
            zp1 = zp if sym else mx.round(-wmin1 / scale1)
            q = mx.clip(mx.round(w / scale1) + zp1, q_min, q_max)
            dq = (q - zp1) * scale1
            err = mx.sum(mx.abs(dq - w) ** norm, axis=-1)
            improved = (err < best).reshape(err.shape + (1,))
            best = mx.where(err < best, err, best)
            scale = mx.where(improved, scale1, scale)
            zp = mx.where(improved, zp1, zp)

    w_int = _quantize(w, scale, zp, q_min, q_max)
    w_quant = _dequantize(w_int, scale, zp)

    w_quant = w_quant.reshape(org_shape)
    w_int = w_int.reshape(org_shape)
    scale = scale.squeeze(-1)
    zp = zp.squeeze(-1)

    return w_quant, scale, zp, w_int


# ──────────────────────────────────────────────
# RTN Result / Quantizer
# ──────────────────────────────────────────────

@dataclass
class RTNResult:
    dequantized_weight: mx.array = None
    wbits: int = None
    groupsize: int = None
    sym: bool = None
    quantized_weight: Optional[mx.array] = None
    scale: Optional[mx.array] = None
    zero: Optional[mx.array] = None


@dataclass
class RTN:
    """RTN quantizer — weight-only, no calibration/Hessian needed."""
    wbits: int = 4
    groupsize: int = -1
    sym: bool = False
    mse: bool = False
    norm: float = 2.4
    grid: int = 100

    def quantize_weight(self, weight: mx.array) -> RTNResult:
        """Quantize a weight matrix directly.

        Args:
            weight: (out_features, in_features) weight matrix.

        Returns:
            RTNResult
        """
        W = weight.astype(mx.float32)

        if self.groupsize > 0:
            if W.shape[-1] % self.groupsize != 0:
                raise ValueError(
                    f"groupsize={self.groupsize} does not divide in_features={W.shape[-1]}"
                )

        Q, scale, zero, Q_int = pseudo_quantize_tensor(
            W,
            n_bit=self.wbits,
            q_group_size=self.groupsize,
            zero_point=not self.sym,
            perchannel=True,
            mse=self.mse,
            norm=self.norm,
            grid=self.grid,
        )

        gc.collect()

        return RTNResult(
            dequantized_weight=Q,
            wbits=self.wbits,
            groupsize=self.groupsize,
            sym=self.sym,
            quantized_weight=Q_int,
            scale=scale.astype(mx.float16),
            zero=zero.astype(mx.float16),
        )
