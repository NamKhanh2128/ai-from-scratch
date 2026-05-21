from datasets import load_dataset

# Lấy toàn bộ tập train của IMDB
dataset = load_dataset("imdb", split="train")

# Chia 80% train, 20% tạm (sẽ tách tiếp thành val + test)
split = dataset.train_test_split(test_size=0.2, seed=42)

# Từ 80% train, chia tiếp thành train (80% của 80% = 64% tổng) và val (20% của 80% = 16% tổng)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)} samples")
print(f"Validation: {len(val_ds)} samples")
print(f"Test: {len(test_ds)} samples")
print(f"Total: {len(train_ds) + len(val_ds) + len(test_ds)}")
