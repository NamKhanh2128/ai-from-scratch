from datasets import load_dataset

# Tải tập train của IMDB
dataset = load_dataset("imdb", split="train")

# Lưu dưới các định dạng khác nhau
dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")

print("Files saved: CSV, JSON, Parquet")
