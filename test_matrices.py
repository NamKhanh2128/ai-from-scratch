from matrices import Matrix

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =\n", (A + B).data)
print("A - B =\n", (A - B).data)
print("A @ B =\n", (A @ B).data)             # dùng @ (gọi __matmul__)
print("A^T =\n", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 (2x2) =\n", A.inverse_2x2().data)

I2 = Matrix.identity(2)
print("I2 =\n", I2.data)

# Kiểm tra A @ A^-1 ≈ I
A_inv = A.inverse_2x2()
should_be_I = A @ A_inv
print("A @ A^-1 =\n", should_be_I.data)