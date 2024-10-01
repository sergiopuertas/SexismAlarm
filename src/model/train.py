import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import StepLR
from model import LSTMModel

image_path = "./data/"

# Hyperparameters
emb_dim = 200
num_epochs = 20
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
dropout = 0.4


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
    embeddings = load_embeddings(f"model/glove.twitter.27B.200d.txt")
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


def train_model(device, train_loader, val_loader, model):
    """
    Train the LSTM model using the provided data loaders.
    Args:
        device (str): Device to train the model on ('cpu' or 'cuda').
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        model (torch.nn.Module): LSTM model.
    """
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # Learning rate scheduler to reduce LR during training
    scheduler = StepLR(optimizer, step_size=5, gamma=0.15)

    # Initialize Weights & Biases logging
    wandb.init(project="sexism_detector_lstm")

    best_val_loss = float("inf")
    patience = 4
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for texts, labels in tqdm(train_loader):
            texts = texts.to(device)
            labels = labels.float().to(device)

            # Forward pass
            outputs = model(texts)
            loss = nn.BCELoss()(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        val_loss = validate_model(model, val_loader)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        wandb.log({"loss": epoch_loss, "val_loss": val_loss})
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "model_trained.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def validate_model(model, val_loader):
    """
    Validate the model on the validation set.
    Args:
        model (torch.nn.Module): Trained model.
        val_loader (DataLoader): DataLoader for the validation data.
    Returns:
        float: Validation loss.
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts = texts.to(device)
            labels = labels.float().to(device)
            outputs = model(texts)
            loss = nn.BCELoss()(outputs.squeeze(), labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def main():
    """
    Main function to execute the training process.
    """
    # Leer la clave API desde el archivo key.txt
    with open("key.txt", "r", encoding="UTF-8") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    # Load and prepare the data
    df, X_train, y_train, X_test, y_test, X_val, y_val = extract_divide_data()
    vectorizer, vocab, vocab_size = vectorize_data(df)

    # Create embedding weights and initialize the model
    weights = create_weights_matrix(vocab, vocab_size)
    pretrained_embeddings = torch.FloatTensor(weights)

    # Crear el modelo pasando los embeddings preentrenados
    model = LSTMModel(emb_dim, pretrained_embeddings, dropout).to(device)

    # Prepare data loaders for training, validation, and test
    train_loader, val_loader, test_loader = prepare_loaders(
        vectorizer, X_train, y_train, X_val, y_val, X_test, y_test
    )
    torch.save(test_loader, "test_loader.pth")

    # Train the model
    train_model(device, train_loader, val_loader, model)


if __name__ == "__main__":
    main()
