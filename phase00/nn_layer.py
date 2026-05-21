import random
from vectors import Vector, Matrix

random.seed(42)
# Ma trận trọng số 2x3
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print("Input (3D):", input_vector)
print("Output (2D):", output)
print("Đây chính là một tầng neural network: ma trận trọng số × vector input.")
