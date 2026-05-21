from datasets import load_dataset

# Tải dataset IMDB (lần đầu sẽ download, sau dùng cache tại ~/.cache/huggingface/datasets/)
dataset = load_dataset("imdb")

print(dataset)
print("---")
# Xem một mẫu trong tập train
print(dataset["train"][0])
