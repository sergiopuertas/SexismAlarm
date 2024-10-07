import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
emb_dim = 200
batch_size = 32


def extract_divide_data():
    """
    Load the dataset and split it into train, validation, and test sets.
    Returns:
        Tuple: Text and label data for training, validation, and test sets.
    """
    seed = 33
    df = pd.read_csv(f"data/dataset.csv")
    df = df.dropna(
        subset=["text", "label"]
    )  # Remove rows with missing 'text' or 'label'

    # Split dataset into train and test sets
    X_train, X_test = train_test_split(
        df, train_size=0.8, test_size=0.2, shuffle=True, random_state=seed
    )
    # Further split the training set to get a validation set
    X_train, X_val = train_test_split(
        X_train, test_size=0.25, shuffle=True, random_state=seed
    )

    return (
        df["text"],
        X_train["text"].tolist(),
        X_train["label"].tolist(),
        X_test["text"].tolist(),
        X_test["label"].tolist(),
        X_val["text"].tolist(),
        X_val["label"].tolist(),
    )


def load_embeddings(file_path):
    """
    Load GloVe embeddings from the specified file.
    Args:
        file_path (str): Path to the GloVe embedding file.
    Returns:
        dict: A dictionary mapping words to their vector representations.
    """
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.array(values[1:], dtype="float32")
            embeddings[word] = coeffs
    return embeddings


def vectorize_data(df):
    """
    Vectorize the text data using CountVectorizer.
    Args:
        df (pd.DataFrame): DataFrame containing the text data.
    Returns:
        Tuple: A fitted vectorizer, vocabulary, and vocabulary size.
    """
    vectorizer = CountVectorizer(max_features=20000)
    vectorizer.fit(df)
    vocab = vectorizer.vocabulary_
    vocab_size = len(vocab)
    return vectorizer, vocab, vocab_size


def create_weights_matrix(vocab, vocab_size):
    """
    Create the embedding matrix using the GloVe embeddings.
    Args:
        vocab (dict): Vocabulary of words.
        vocab_size (int): Size of the vocabulary.
    Returns:
        torch.FloatTensor: Embedding matrix with GloVe vectors.
    """
    embeddings = load_embeddings(f"model/train/glove.twitter.27B.200d.txt")
    emb_mat = np.zeros((vocab_size, emb_dim))
    for word, i in vocab.items():
        if word in embeddings:
            emb_mat[i] = embeddings[word]

    weights = torch.FloatTensor(emb_mat)
    return weights


def prepare_loaders(vectorizer, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Prepare data loaders for training, validation, and testing.
    Args:
        vectorizer (CountVectorizer): Fitted CountVectorizer.
        X_train (list): List of training texts.
        y_train (list): List of training labels.
        X_val (list): List of validation texts.
        y_val (list): List of validation labels.
        X_test (list): List of test texts.
        y_test (list): List of test labels.
    Returns:
        Tuple: DataLoader objects for training, validation, and test sets.
    """
    # Vectorize the texts and convert them to tensors
    X_train_vectorized = torch.tensor(
        np.array([vectorizer.transform([text]).toarray()[0] for text in X_train]),
        dtype=torch.long,
    )
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)

    X_val_vectorized = torch.tensor(
        np.array([vectorizer.transform([text]).toarray()[0] for text in X_val]),
        dtype=torch.long,
    )
    y_val_tensor = torch.tensor(y_val, dtype=torch.float)

    X_test_vectorized = torch.tensor(
        np.array([vectorizer.transform([text]).toarray()[0] for text in X_test]),
        dtype=torch.long,
    )
    y_test_tensor = torch.tensor(y_test, dtype=torch.float)

    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train_vectorized, y_train_tensor)
    val_dataset = TensorDataset(X_val_vectorized, y_val_tensor)
    test_dataset = TensorDataset(X_test_vectorized, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def main():
    # Load and prepare the data
    df, X_train, y_train, X_test, y_test, X_val, y_val = extract_divide_data()
    vectorizer, vocab, vocab_size = vectorize_data(df)

    # Create embedding weights
    weights = create_weights_matrix(vocab, vocab_size)
    pretrained_embeddings = torch.FloatTensor(weights)

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_loaders(
        vectorizer, X_train, y_train, X_val, y_val, X_test, y_test
    )
    torch.save(pretrained_embeddings, "model/train/embeddings.pt")
    torch.save(train_loader, "model/train/train_loader.pt")
    torch.save(train_loader, "model/train/val_loader.pt")
    torch.save(train_loader, "model/test_loader.pt")


if __name__ == "__main__":
    main()
