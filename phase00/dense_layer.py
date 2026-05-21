import random
from matrices import Matrix

# ── Input: 3 đặc trưng ──
inputs = Matrix([[0.5], [0.8], [0.2]])   # 3x1

# ── Trọng số: 2 nơ-ron đầu ra, mỗi nơ-ron nhận 3 input ──
random.seed(42)
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])   # 2x3

# ── Bias: 2x1 ──
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    """Áp dụng ReLU lên từng phần tử của ma trận."""
    return Matrix([[max(0.0, val) for val in row] for row in m.data])

# ── Lan truyền tiến (forward pass) ──
pre_activation = weights @ inputs + bias   # (2x3) @ (3x1) + (2x1)
output = relu_matrix(pre_activation)

print("Input shape:", inputs.shape)
print("Weight shape:", weights.shape)
print("Pre-activation:\n", pre_activation.data)
print("Output (after ReLU):\n", output.data)
print("Output shape:", output.shape)
