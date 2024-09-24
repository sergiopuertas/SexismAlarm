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

image_path = "./data/"

# Params
emb_dim = 200
maxlen = 100
num_epochs = 20
batch_size = 16
learning_rate = 1e-3
weight_decay = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"


# Function to prepare the data
def prepare_data():
    seed = 33
    df = pd.read_csv(f"{image_path}dataset.csv")
    df = df.dropna(
        subset=["text", "label"]
    )  # Remove rows with NaN in 'text' or 'label'

    X_train, X_test = train_test_split(
        df, train_size=0.8, test_size=0.2, shuffle=True, random_state=seed
    )
    X_train, X_val = train_test_split(
        X_train, test_size=0.25, shuffle=True, random_state=seed
    )
    return (
        X_train["text"].tolist(),
        X_train["label"].tolist(),
        X_test["text"].tolist(),
        X_test["label"].tolist(),
        X_val["text"].tolist(),
        X_val["label"].tolist(),
    )


# Function to load GloVe embeddings
def load_embeddings(file_path):
    embeddings = {}
    with open(f"{image_path}{file_path}", "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.array(values[1:], dtype="float32")
            embeddings[word] = coeffs
    return embeddings


# Model definition
class LSTMModel(nn.Module):
    def __init__(self, emb_dim, weights):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(emb_dim, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)  # Returns output and states, only keep output
        x = self.fc1(x[:, -1, :])  # Take only last layer of LSTM
        x = self.fc2(x)
        return self.sigmoid(x)


def evaluate_model(model, val_loader):
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


def train_model(device, train_loader, val_loader, model):
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = StepLR(
        optimizer, step_size=5, gamma=0.15
    )  # LR reduction to adjust learning

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

            outputs = model(texts)
            loss = nn.BCELoss()(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        val_loss = evaluate_model(model, val_loader)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        wandb.log({"loss": epoch_loss, "val_loss": val_loss})
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def main():
    print("hola")
    # Load and prepare the data
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()
    embeddings = load_embeddings("glove.twitter.27B.200d.txt")
    print("loaded")
    vectorizer = CountVectorizer(max_features=10000)
    vectorizer.fit(X_train)

    vocab = vectorizer.vocabulary_
    vocab_size = len(vocab)

    # Create the embedding matrix
    emb_mat = np.zeros((vocab_size, emb_dim))
    for word, i in vocab.items():
        if word in embeddings:
            emb_mat[i] = embeddings[word]

    weights = torch.FloatTensor(emb_mat)
    model = LSTMModel(emb_dim, weights).to(device)

    # Vectorize the texts
    X_train_vectorized = torch.tensor(
        np.array([vectorizer.transform([text]).toarray()[0] for text in X_train]),
        dtype=torch.long,
    )
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)

    # Create the validation dataset
    X_val_vectorized = torch.tensor(
        np.array([vectorizer.transform([text]).toarray()[0] for text in X_val]),
        dtype=torch.long,
    )
    y_val_tensor = torch.tensor(y_val, dtype=torch.float)

    train_dataset = TensorDataset(X_train_vectorized, y_train_tensor)
    val_dataset = TensorDataset(X_val_vectorized, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_model(device, train_loader, val_loader, model)


if __name__ == "__main__":
    main()
