"""Smoke test for GPTQ core: Cholesky inverse-Hessian + quantization loop."""

import mlx.core as mx
import numpy as np
from mlx_onecomp.quantizer.gptq._gptq import (
    _compute_inverse_hessian_mlx,
    _compute_inverse_hessian_numpy,
    run_gptq,
)


def test_inverse_hessian():
    """Test Cholesky inverse-Hessian via MLX/Metal."""
    mx.set_default_device(mx.cpu)

    A = mx.array(np.random.randn(64, 64).astype(np.float32))
    H = A @ A.transpose() + mx.eye(64) * 10.0

    Hinv = _compute_inverse_hessian_mlx(H, percdamp=0.01)
    assert Hinv.shape == (64, 64), f"Shape: {Hinv.shape}"
    print(f"  Inverse Hessian shape: {Hinv.shape} — PASS")


def test_inverse_hessian_numpy():
    """Test Cholesky inverse-Hessian numpy wrapper."""
    H = np.random.randn(64, 64).astype(np.float32)
    H = H @ H.T + np.eye(64) * 10.0

    Hinv = _compute_inverse_hessian_numpy(H, percdamp=0.01)
    assert Hinv.shape == (64, 64)
    print(f"  Numpy wrapper shape: {Hinv.shape} — PASS")


def test_run_gptq_small():
    """Test full GPTQ quantization on a small weight matrix."""
    mx.set_default_device(mx.cpu)

    out_features, in_features = 32, 64
    weight = mx.array(np.random.randn(out_features, in_features).astype(np.float32))

    X = np.random.randn(128, in_features).astype(np.float32)
    H = mx.array(X.T @ X + np.eye(in_features) * 5.0)

    result = run_gptq(
        hessian=H, weight=weight,
        blocksize=16, percdamp=0.01, wbits=4,
        groupsize=-1, sym=True,
    )

    assert result["qweight"].shape == (out_features, in_features)
    assert result["scales"] is not None
    assert result["qzeros"] is not None
    print(f"  qweight: {result['qweight'].shape}, scales: {result['scales'].shape} — PASS")


def test_run_gptq_grouped():
    """Test GPTQ with group quantization."""
    mx.set_default_device(mx.cpu)

    out_features, in_features = 16, 64
    weight = mx.array(np.random.randn(out_features, in_features).astype(np.float32))

    X = np.random.randn(64, in_features).astype(np.float32)
    H = mx.array(X.T @ X + np.eye(in_features) * 5.0)

    result = run_gptq(
        hessian=H, weight=weight,
        blocksize=16, percdamp=0.01, wbits=4,
        groupsize=32, sym=True,
    )

    assert result["qweight"].shape == (out_features, in_features)
    print(f"  grouped scales: {result['scales'].shape} — PASS")


def test_run_gptq_actorder():
    """Test GPTQ with activation ordering."""
    mx.set_default_device(mx.cpu)

    out_features, in_features = 16, 32
    weight = mx.array(np.random.randn(out_features, in_features).astype(np.float32))

    X = np.random.randn(64, in_features).astype(np.float32)
    H = mx.array(X.T @ X + np.eye(in_features) * 5.0)

    result = run_gptq(
        hessian=H, weight=weight,
        blocksize=16, percdamp=0.01, wbits=4,
        groupsize=-1, actorder=True, sym=True,
    )

    assert result["perm"] is not None
    print(f"  actorder perm: {result['perm'].shape} — PASS")


if __name__ == "__main__":
    print("Running GPTQ core tests...")
    test_inverse_hessian()
    test_inverse_hessian_numpy()
    test_run_gptq_small()
    test_run_gptq_grouped()
    test_run_gptq_actorder()
    print("\nAll GPTQ tests passed!")
