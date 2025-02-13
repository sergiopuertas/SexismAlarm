import os
import requests

repo_id = os.getenv("MODEL_ID")
model_path = "/huggingface/sentence-transformers/all-mpnet-base-v2/"
os.makedirs(model_path, exist_ok=True)

files = [
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
]

base_url = f"https://huggingface.co/{repo_id}/resolve/main/"


def download_file(file_name):
    url = f"{base_url}{file_name}"
    response = requests.get(url)

    if response.status_code == 200:
        with open(os.path.join(model_path, file_name), "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download {file_name}: {response.status_code}")


for file in files:
    download_file(file)
