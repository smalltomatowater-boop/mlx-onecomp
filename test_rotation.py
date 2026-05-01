"""Test rotation preprocessing."""

import sys
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

import mlx.core as mx
from mlx_onecomp.preprocessing.rotation import HadamardRotation, RandomRotation

print("=== Hadamard Rotation Tests ===")

# Test 1: Orthogonality (H @ H^T = I)
rot = HadamardRotation(512)
I = rot.H @ rot.H.T
identity = mx.eye(rot.hidden_size)
err = mx.abs(I - identity).max()
print(f"[1] Orthogonality H@H^T=I: max error = {err.item():.2e}", "OK" if err.item() < 1e-5 else "FAIL")

# Test 2: Roundtrip (rotate in → rotate out = original)
W = mx.random.normal((640, 512))
W2 = rot.rotate_weight_out(rot.rotate_weight_in(W))
err2 = mx.abs(W - W2).max()
print(f"[2] Roundtrip rotate: max error = {err2.item():.2e}", "OK" if err2.item() < 1e-4 else "FAIL")

# Test 3: Non-power-of-2 uses RandomRotation
rot3 = RandomRotation(640)
W3 = mx.random.normal((1024, 640))
W3b = rot3.rotate_weight_out(rot3.rotate_weight_in(W3))
err3 = mx.abs(W3 - W3b).max()
print(f"[3] Non-power-of-2 (RandomRotation): max error = {err3.item():.2e}", "OK" if err3.item() < 1e-4 else "FAIL")

# Test 4: Random rotation
print("\n=== Random Rotation Tests ===")
rrot = RandomRotation(512, seed=123)
QQt = rrot.Q @ rrot.Q.T
err4 = mx.abs(QQt - mx.eye(512)).max()
print(f"[4] Random orthogonality: max error = {err4.item():.2e}", "OK" if err4.item() < 1e-5 else "FAIL")

W4 = mx.random.normal((640, 512))
W4b = rrot.rotate_weight_out(rrot.rotate_weight_in(W4))
err5 = mx.abs(W4 - W4b).max()
print(f"[5] Random roundtrip: max error = {err5.item():.2e}", "OK" if err5.item() < 1e-4 else "FAIL")

# Test 5: Weight distribution before/after rotation
print("\n=== Distribution Effect ===")
W5 = mx.random.normal((1024, 512))
# Inject outliers
W5 = W5 + mx.array([10.0 if i % 50 == 0 else 0.0 for i in range(1024 * 512)]).reshape(1024, 512)
std_before = mx.abs(W5).max(axis=1).max()
W5r = rot.rotate_weight_in(W5)
std_after = mx.abs(W5r).max(axis=1).max()
print(f"[6] Outlier reduction: max abs {std_before.item():.1f} → {std_after.item():.1f}")

print("\nAll tests done!")
