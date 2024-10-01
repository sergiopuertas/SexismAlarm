import numpy as np
import sys
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


def evaluate_similarity(embedding1, embedding2, threshold=0.5):
    """
    Evaluate if the two embeddings are similar based on a threshold.

    Args:
        embedding1 (np.ndarray): First embedding.
        embedding2 (np.ndarray): Second embedding.
        threshold (float): Similarity threshold.

    Returns:
        str: Message indicating whether the sentences are related or not.
    """
    similarity = cosine_similarity(embedding1, embedding2)
    print(f"Similarity between sentence 1 and sentence 2: {similarity}")

    if similarity > threshold:
        return "The sentences are related."
    else:
        return "The sentences are not related."


def main():
    """
    Main function to execute the similarity evaluation.
    """
    if len(sys.argv) != 3:
        print("Usage: python similarity.py <sentence1> <sentence2>")
        sys.exit(1)

    sentence1 = sys.argv[1]
    sentence2 = sys.argv[2]

    # Define the base model path
    model_path = "/models/huggingface/sentence-transformers/all-mpnet-base-v2/snapshots/84f2bcc00d77236f9e89c8a360a00fb1139bf47d"

    # Load the model
    model = SentenceTransformer(model_path)

    # Define the sentences
    sentences = [sentence1, sentence2]

    # Obtain embeddings for the sentences
    embeddings = model.encode(sentences)

    # Evaluate similarity
    result = evaluate_similarity(embeddings[0], embeddings[1])
    print(result)


if __name__ == "__main__":
    main()
