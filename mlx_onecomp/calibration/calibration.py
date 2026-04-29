"""Calibration data loading and Hessian computation for MLX."""

import math
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def compute_hessian(
    activations: mx.array,
    batch_size: int = 8,
) -> mx.array:
    """Compute scaled Hessian (X^T X) from calibration activations.

    Args:
        activations: (num_samples, seq_len, hidden_size) float32.
        batch_size: Mini-batch size for memory efficiency.

    Returns:
        Hessian matrix (hidden_size, hidden_size), float32.
    """
    if activations.ndim == 2:
        activations = activations.reshape(1, activations.shape[0], -1)
    assert activations.ndim == 3

    hidden_size = activations.shape[-1]
    hessian = mx.zeros((hidden_size, hidden_size), dtype=mx.float32)
    nsamples = 0

    for i in range(0, activations.shape[0], batch_size):
        batch = activations[i : i + batch_size].astype(mx.float32)
        batch = batch.reshape(-1, hidden_size)

        inp = batch.transpose()  # (hidden_size, n)

        tmp = batch.shape[0]
        hessian = hessian * (nsamples / (nsamples + tmp))
        nsamples += tmp

        inp_scaled = math.sqrt(2 / nsamples) * inp
        hessian = hessian + inp_scaled @ inp_scaled.transpose()

    return hessian


def compute_delta_hatx(
    quant_activations: mx.array,
    original_activations: mx.array,
    batch_size: int = 8,
) -> mx.array:
    """Compute delta^T hat_X for QEP weight correction.

    delta = original - quant
    hat_X = quant
    """
    if quant_activations.ndim == 2:
        quant_activations = quant_activations.reshape(1, quant_activations.shape[0], -1)
    if original_activations.ndim == 2:
        original_activations = original_activations.reshape(1, original_activations.shape[0], -1)
    assert quant_activations.shape == original_activations.shape

    hidden_size = quant_activations.shape[-1]
    delta_hatx = mx.zeros((hidden_size, hidden_size), dtype=mx.float32)
    nsamples = 0

    for i in range(0, quant_activations.shape[0], batch_size):
        bq = quant_activations[i : i + batch_size].astype(mx.float32).reshape(-1, hidden_size)
        bo = original_activations[i : i + batch_size].astype(mx.float32).reshape(-1, hidden_size)

        delta = (bo - bq).transpose()
        hat_x = bq.transpose()

        tmp = delta.shape[1]
        delta_hatx = delta_hatx * (nsamples / (nsamples + tmp))
        nsamples += tmp

        scale = math.sqrt(2 / nsamples)
        delta_hatx = delta_hatx + (scale * delta) @ (scale * hat_x).transpose()

    return delta_hatx
