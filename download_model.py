from huggingface_hub import hf_hub_download, snapshot_download

# Tải file cấu hình
config_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Config cached at: {config_path}")

# Tải toàn bộ model
model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Full model saved to: {model_dir}")

