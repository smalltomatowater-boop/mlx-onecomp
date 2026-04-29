"""Smoke test for MLX-OneComp: RTN quantization on a small weight matrix."""

import mlx.core as mx
from mlx_onecomp.quantizer.rtn._rtn import RTN, pseudo_quantize_tensor


def test_pseudo_quantize():
    """Test basic pseudo-quantization."""
    mx.set_default_device(mx.cpu)

    w = mx.random.normal((128, 256))
    q, scale, zp, q_int = pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1, zero_point=False)

    assert q.shape == w.shape, f"Shape mismatch: {q.shape} != {w.shape}"
    assert q_int.shape == w.shape

    # Check quantized values are in valid range
    maxq = 2**4 - 1
    assert mx.all(q_int >= 0), "Negative quantized values"
    assert mx.all(q_int <= maxq), f"Quantized values exceed max ({maxq})"

    # Check dequantized is close to original (4-bit has error but should be correlated)
    error = mx.mean((q - w) ** 2).item()
    print(f"  MSE: {error:.6f}")
    assert error < 1.0, f"Quantization error too high: {error}"

    print("  pseudo_quantize_tensor: PASS")


def test_rtn_grouped():
    """Test RTN with group quantization."""
    mx.set_default_device(mx.cpu)

    w = mx.random.normal((64, 128))
    q, scale, zp, q_int = pseudo_quantize_tensor(w, n_bit=4, q_group_size=32, zero_point=True)

    assert q.shape == w.shape
    # scale should have shape (64, 4) = (out_features, num_groups)
    assert scale.shape == (64, 128 // 32), f"Scale shape: {scale.shape}"

    print("  RTN grouped: PASS")


def test_rtn_quantizer():
    """Test RTN quantizer class."""
    mx.set_default_device(mx.cpu)

    weight = mx.random.normal((128, 512))
    quantizer = RTN(wbits=4, groupsize=128, sym=False)
    result = quantizer.quantize_weight(weight)

    assert result.dequantized_weight.shape == weight.shape
    assert result.wbits == 4
    assert result.groupsize == 128

    error = mx.mean((result.dequantized_weight - weight) ** 2).item()
    print(f"  RTN MSE: {error:.6f}")
    print("  RTN quantizer: PASS")


if __name__ == "__main__":
    print("Running MLX-OneComp smoke tests...")
    test_pseudo_quantize()
    test_rtn_grouped()
    test_rtn_quantizer()
    print("\nAll tests passed!")
