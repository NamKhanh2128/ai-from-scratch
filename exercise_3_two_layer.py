import random
from matrices import Matrix

random.seed(123)

# Tầng 1: 3 input -> 4 hidden
W1 = Matrix([[random.uniform(-1, 1) for _ in range(3)] for _ in range(4)])  # 4x3
b1 = Matrix([[0.1] for _ in range(4)])                                      # 4x1

# Tầng 2: 4 hidden -> 2 output
W2 = Matrix([[random.uniform(-1, 1) for _ in range(4)] for _ in range(2)])  # 2x4
b2 = Matrix([[0.1] for _ in range(2)])                                      # 2x1

x = Matrix([[0.5], [0.8], [0.2]])   # input 3x1

def relu_matrix(m):
    return Matrix([[max(0.0, val) for val in row] for row in m.data])

h_pre = W1 @ x + b1
h = relu_matrix(h_pre)
y_pre = W2 @ h + b2
y = relu_matrix(y_pre)   # hoặc không cần activation ở output tuỳ bài toán

print("Hidden shape:", h.shape)
print("Output shape:", y.shape)
print("Output:\n", y.data)import random
from matrices import Matrix

random.seed(123)

# Tầng 1: 3 input -> 4 hidden
W1 = Matrix([[random.uniform(-1, 1) for _ in range(3)] for _ in range(4)])  # 4x3
b1 = Matrix([[0.1] for _ in range(4)])                                      # 4x1

# Tầng 2: 4 hidden -> 2 output
W2 = Matrix([[random.uniform(-1, 1) for _ in range(4)] for _ in range(2)])  # 2x4
b2 = Matrix([[0.1] for _ in range(2)])                                      # 2x1

x = Matrix([[0.5], [0.8], [0.2]])   # input 3x1

def relu_matrix(m):
    return Matrix([[max(0.0, val) for val in row] for row in m.data])

h_pre = W1 @ x + b1
h = relu_matrix(h_pre)
y_pre = W2 @ h + b2
y = relu_matrix(y_pre)   # hoặc không cần activation ở output tuỳ bài toán

print("Hidden shape:", h.shape)
print("Output shape:", y.shape)
print("Output:\n", y.data)
