import os
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


def load_model(model_id, model_path):
    if not os.path.exists(model_path):
        model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    else:
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def encode_sentences(sentences, model, tokenizer):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.numpy()


def main():
    model_id = "sentence-transformers/all-mpnet-base-v2"
    model_path = "app/huggingface/sentence-transformers/all-mpnet-base-v2"

    model, tokenizer = load_model(model_id, model_path)

    if model is not None and tokenizer is not None:
        first_sentence = os.environ.get("FIRST_SENTENCE", "default first sentence")
        second_sentence = os.environ.get("SECOND_SENTENCE", "default second sentence")

        embeddings1 = encode_sentences([first_sentence], model, tokenizer)
        embeddings2 = encode_sentences([second_sentence], model, tokenizer)

        similarity_score = cosine_similarity(embeddings1, embeddings2)[0][0]
        print(f"Similitud: {similarity_score}")


if __name__ == "__main__":
    main()
