from matrices import Matrix

def inverse_3x3(M):
    if M.shape != (3,3):
        raise ValueError("Chỉ dùng cho 3x3")
    det = M.determinant()
    if det == 0:
        raise ValueError("Ma trận suy biến")
    # Ma trận phụ hợp (adjugate) = chuyển vị của ma trận cofactor
    cofactors = []
    for i in range(3):
        row = []
        for j in range(3):
            # Tạo minor 2x2 bỏ dòng i, cột j
            minor_data = [
                [M.data[r][c] for c in range(3) if c != j]
                for r in range(3) if r != i
            ]
            minor = Matrix(minor_data)
            cofactor = ((-1) ** (i+j)) * minor.determinant()
            row.append(cofactor)
        cofactors.append(row)
    adjugate = Matrix(cofactors).transpose()
    # Chia mỗi phần tử cho det
    inv_data = [[adjugate.data[r][c] / det for c in range(3)] for r in range(3)]
    return Matrix(inv_data)

M = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
M_inv = inverse_3x3(M)
print("M_inv =\n", M_inv.data)
I = M @ M_inv
print("M @ M_inv ≈\n", I.data)
# So sánh với numpy (nếu có)
