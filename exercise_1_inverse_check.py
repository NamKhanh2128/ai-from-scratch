from matrices import Matrix

matrices = [
    Matrix([[2, 3], [1, 4]]),
    Matrix([[5, 1], [2, 3]]),
    Matrix([[0, 1], [1, 0]])   # det = -1 ≠ 0
]

for i, M in enumerate(matrices):
    print(f"\nM{i+1} =\n{M.data}")
    if M.shape == (2,2):
        try:
            M_inv = M.inverse_2x2()
            I_approx = M @ M_inv
            print("M @ M_inv ≈\n", I_approx.data)
            # Làm tròn để kiểm tra
            ok = all(abs(I_approx.data[r][c] - (1 if r==c else 0)) < 1e-10 for r in range(2) for c in range(2))
            print("Đúng là ma trận đơn vị?", "Có" if ok else "Không")
        except ValueError as e:
            print("Lỗi:", e)
