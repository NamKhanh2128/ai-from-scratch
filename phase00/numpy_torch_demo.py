import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)
print("a + b =", a + b)
print("dot =", np.dot(a, b))
print("cosine =", np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Rank
A = np.array([[1, 2], [2, 4]])
print("Rank of A:", np.linalg.matrix_rank(A))

# QR
Q, R = np.linalg.qr(np.random.randn(3, 3))
print("Q orthogonal?", np.allclose(Q @ Q.T, np.eye(3)))

import torch
x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])
similarity = torch.dot(x, y)
similarity.backward()
print("x grad =", x.grad)
