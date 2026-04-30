"""ILP solver for optimal bit allocation.

Formulates and solves:
    minimize  Σ_i Σ_b (sensitivity[i][b] * x[i][b])
    subject to Σ_i Σ_b (b * num_params[i] * x[i][b]) <= total_bit_budget
               Σ_b x[i][b] = 1  for each layer i  (exactly one bit width)
               x[i][b] ∈ {0, 1}

Uses scipy.optimize.milp (mixed-integer linear programming).
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def solve_bit_allocation(
    sensitivity: dict[str, dict[int, float]],
    layer_sizes: dict[str, int],
    target_avg_bits: float = 4.0,
    bits_choices: tuple[int, ...] = (2, 3, 4, 8),
) -> dict[str, int]:
    """Solve for optimal per-layer bit allocation.

    Args:
        sensitivity: {layer_name: {bits: mse}} from sensitivity_profile().
        layer_sizes: {layer_name: num_params} for each layer.
        target_avg_bits: Target average bit width (e.g., 4.0).
        bits_choices: Available bit widths.

    Returns:
        {layer_name: optimal_bits} allocation.
    """
    try:
        from scipy.optimize import linprog, milp, LinearConstraint, Bounds
    except ImportError:
        raise ImportError("scipy required for AutoBit: pip install scipy")

    layers = list(sensitivity.keys())
    n_layers = len(layers)
    n_bits = len(bits_choices)
    n_vars = n_layers * n_bits

    # Objective: minimize total MSE
    c = np.zeros(n_vars)
    for i, layer in enumerate(layers):
        for j, bits in enumerate(bits_choices):
            c[i * n_bits + j] = sensitivity[layer].get(bits, float("inf"))

    # Constraint 1: exactly one bit width per layer
    # For each layer i: Σ_j x[i][j] = 1
    A_eq = np.zeros((n_layers, n_vars))
    b_eq = np.ones(n_layers)
    for i in range(n_layers):
        for j in range(n_bits):
            A_eq[i, i * n_bits + j] = 1.0

    # Constraint 2: average bits <= target
    # Σ_i Σ_j (bits[j] * num_params[i] * x[i][j]) <= target * total_params
    total_params = sum(layer_sizes[l] for l in layers)
    A_ub = np.zeros((1, n_vars))
    for i, layer in enumerate(layers):
        for j, bits in enumerate(bits_choices):
            A_ub[0, i * n_bits + j] = bits * layer_sizes[layer]
    b_ub = np.array([target_avg_bits * total_params])

    # Bounds: all variables in [0, 1]
    bounds = Bounds(lb=0, ub=1)

    # Integer constraints: all variables are binary
    integrality = np.ones(n_vars)  # 1 = integer

    constraints = [
        LinearConstraint(A_eq, b_eq, b_eq),  # equality
        LinearConstraint(A_ub, -np.inf, b_ub),  # inequality
    ]

    result = milp(
        c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
    )

    if not result.success:
        logger.warning("ILP solver failed: %s. Falling back to uniform allocation.", result.message)
        return {layer: int(round(target_avg_bits)) for layer in layers}

    # Extract allocation
    allocation = {}
    x = result.x.reshape(n_layers, n_bits)
    for i, layer in enumerate(layers):
        chosen = int(np.argmax(x[i]))
        allocation[layer] = bits_choices[chosen]

    # Log summary
    avg = sum(allocation[l] * layer_sizes[l] for l in layers) / total_params
    bits_dist = {}
    for b in allocation.values():
        bits_dist[b] = bits_dist.get(b, 0) + 1
    logger.info(
        "AutoBit allocation: avg=%.2f bits, distribution=%s",
        avg, bits_dist,
    )

    return allocation
