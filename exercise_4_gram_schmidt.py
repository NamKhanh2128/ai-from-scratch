from vectors import Vector, gram_schmidt

# Ba vector độc lập tuyến tính
v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])

# Thực hiện Gram-Schmidt
basis = gram_schmidt([v1, v2, v3])

print("Kết quả Gram-Schmidt:")
for i, u in enumerate(basis):
    mag = u.magnitude()
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {mag:.6f}")

# Kiểm tra trực giao từng cặp
print("\nKiểm tra dot product từng cặp (phải ≈ 0):")
for i in range(len(basis)):
    for j in range(i+1, len(basis)):
        dot = basis[i].dot(basis[j])
        print(f"  u{i+1} · u{j+1} = {dot:.6f}")

# Kiểm tra tất cả có phải vector đơn vị
print("\nTất cả có độ dài 1?")
all_unit = all(abs(u.magnitude() - 1.0) < 1e-10 for u in basis)
print("  ->", "Đúng" if all_unit else "Sai")
