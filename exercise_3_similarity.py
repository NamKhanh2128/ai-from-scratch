import random
from vectors import Vector

# Số chiều và số lượng vector
dim = 50
num_vectors = 5

# Sinh ngẫu nhiên 5 vector (seed để kết quả ổn định)
random.seed(123)
vectors = []
for i in range(num_vectors):
    components = [random.gauss(0, 1) for _ in range(dim)]
    vectors.append(Vector(components))

# Tính cosine similarity giữa tất cả các cặp
best_sim = -2.0   # cosine nằm trong [-1, 1], nên -2 là thấp hơn hết
best_pair = (0, 0)

for i in range(num_vectors):
    for j in range(i+1, num_vectors):
        sim = vectors[i].cosine_similarity(vectors[j])
        if sim > best_sim:
            best_sim = sim
            best_pair = (i, j)

i, j = best_pair
print(f"Cặp vector giống nhất: vector {i} và vector {j}")
print(f"Cosine similarity = {best_sim:.4f}")

# In thêm một vài thành phần đầu của chúng để quan sát
print(f"\nVector {i} (5 phần tử đầu): {vectors[i].components[:5]}")
print(f"Vector {j} (5 phần tử đầu): {vectors[j].components[:5]}")
