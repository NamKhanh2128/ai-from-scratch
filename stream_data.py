from datasets import load_dataset

# Bật streaming – không tải toàn bộ Wikipedia
dataset = load_dataset(
    "wikimedia/wikipedia", "20220301.en",
    split="train",
    streaming=True
)

# In 5 tiêu đề đầu tiên
for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break
