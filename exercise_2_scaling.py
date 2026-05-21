from vectors import Vector, Matrix

# Ma trận co giãn: x2 ở trục x, x3 ở trục y
scale = Matrix([[2, 0],
                [0, 3]])

# Vector ban đầu
v = Vector([1, 1])

# Áp dụng ma trận (phép nhân ma trận với vector)
result = scale @ v

print("Ma trận co giãn:")
print(scale)
print("\nVector gốc:", v)
print("Vector sau khi biến đổi:", result)
print("(x gấp đôi, y gấp ba)")
