import os
from huggingface_hub import hf_hub_download

model_id = "sentence-transformers/all-mpnet-base-v2"
model_path = "models/huggingface/sentence-transformers/all-mpnet-base-v2"

# Create the directory if it doesn't exist
os.makedirs(model_path, exist_ok=True)

# List of files to download
files = ["config.json", "pytorch_model.bin"]

for file_name in files:
    print(f"Downloading {file_name}...")
    try:
        file_path = hf_hub_download(
            repo_id=model_id, filename=file_name, cache_dir=model_path
        )
        print(f"{file_name} downloaded and saved at: {file_path}")
    except Exception as e:
        print(f"Error downloading {file_name}: {str(e)}")
