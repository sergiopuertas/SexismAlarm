import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from model import LSTMModel
from dataset import TextDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 5e-3
weight_decay = 1e-5
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def extract_divide_data():
    """
    Load the dataset and split it into train, validation, and test sets.
    Returns:
        Tuple: Text and label data for training, validation, and test sets.
    """
    seed = 141223
    df = pd.read_csv(f"data/dataset.csv")
    df = df.dropna(subset=["text", "label"])

    X_train, X_val = train_test_split(
        df, train_size=0.8, test_size=0.2, shuffle=True, random_state=seed
    )
    return (
        X_train["text"].tolist(),
        X_train["label"].tolist(),
        X_val["text"].tolist(),
        X_val["label"].tolist(),
    )


def tokenize_input(X_train, X_val, padding_word="PAD"):
    max_sequence_length = max(len(x) for x in X_train + X_val)
    X_train_tokens = [x.split() for x in X_train]
    X_val_tokens = [x.split() for x in X_val]

    text_tokens = X_train_tokens + X_val_tokens

    text_padded = pad_sequences(
        text_tokens,
        maxlen=max_sequence_length,
        padding="post",
        value=padding_word,
        dtype=object,
    )

    X_train_padded = text_padded[: len(X_train)]
    X_val_padded = text_padded[len(X_train) :]

    return X_train_padded, X_val_padded


def prepare_loaders():
    """
    Prepara los DataLoaders para entrenamiento y validación.
    """
    X_train, y_train, X_val, y_val = extract_divide_data()

    X_train, X_val = tokenize_input(X_train, X_val)

    train_dataset = TextDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TextDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def train_model(train_loader, val_loader, model):
    """
    Train the LSTM model using gradient accumulation.
    """
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = nn.BCELoss()

    wandb.init(project="sexism_detector_lstm")

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for i, (texts, labels) in enumerate(tqdm(train_loader, colour="magenta")):
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        val_loss = validate_model(model, val_loader)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
            },
            f"model/model_checkpoints/model_trained_{epoch}.pth",
        )
        wandb.log(
            {
                "loss": epoch_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                "model/model_trained.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def validate_model(model, val_loader):
    """
    Validate the model.
    """
    model.eval()
    val_loss = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for texts, labels in tqdm(val_loader, colour="green"):
            texts = texts.to(device)
            labels = labels.float().to(device)

            outputs = model(texts)
            outputs = outputs.squeeze()

            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    os.makedirs(f"model/model_checkpoints", exist_ok=True)
    with open(".key.txt", "r", encoding="UTF-8") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    train_loader, val_loader = prepare_loaders()

    model = LSTMModel().to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_model(train_loader, val_loader, model)


if __name__ == "__main__":
    main()
