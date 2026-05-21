from vectors import Vector, Matrix

a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print("a + b =", a + b)
print("a · b =", a.dot(b))
print("|a| =", round(a.magnitude(), 4))
print("cosine similarity =", round(a.cosine_similarity(b), 4))

# Ma trận xoay 90 độ
rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])
rotated = rotation_90 @ point
print("Original:", point)
print("Rotated 90°:", rotated)
