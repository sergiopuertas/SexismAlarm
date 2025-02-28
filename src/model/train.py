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
from model import *
from dataset import TextDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Hyperparameters
batch_size = 32
num_epochs = 20
learning_rate = 1e-4
weight_decay = 5e-5
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
max_train_data = None
max_val_data = None


def extract_divide_data():
    """
    Load the dataset and split it into train, validation, and test sets.
    Returns:
        Tuple: Text and label data for training, validation, and test sets.
    """
    df = pd.read_csv(f"data/dataset.csv")
    df = df.dropna(subset=["text", "label"])

    # Limitar la cantidad de datos si se especifica un número
    if max_train_data:
        df = df[:max_train_data + max_val_data + 500]  # Se cargan datos suficientes para entrenamiento y validación

    return df["text"].tolist(), df["label"].tolist()


def tokenize_input(text, MAX_SEQUENCE_LENGTH=500):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index

    print("Found %s unique tokens." % len(word_index))
    text_padded = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        value=0,
        dtype="int32",
    )

    return text_padded, word_index


def prepare_loaders():
    """
    Prepara los DataLoaders para entrenamiento y validación.
    """
    X, y = extract_divide_data()

    X_padded, vocab = tokenize_input(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.3, shuffle=True, random_state=141223
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, shuffle=True, random_state=141223
    )

    # Limitar el número de datos para entrenamiento y validación
    if max_train_data:
        X_train, y_train = X_train[:max_train_data], y_train[:max_train_data]
    if max_val_data:
        X_val, y_val = X_val[:max_val_data], y_val[:max_val_data]

    train_dataset = TextDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    val_dataset = TextDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = TextDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    torch.save(vocab, "model/vocab.pt")
    torch.save(test_loader, "model/test_loader.pt")
    return train_loader, val_loader, test_loader, vocab


def train_model():
    """
    Train the LSTM model using gradient accumulation.
    """
    train_loader, val_loader, _, vocab = prepare_loaders()

    model = LSTMModel(vocab).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    wandb.init(project="sexism_detector_lstm")

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for i, (text, label) in enumerate(tqdm(train_loader, colour="magenta")):
            text, label = text.to(device), label.to(device).float()

            # Pasar texto y mask al modelo
            outputs = model(text)

            loss = criterion(outputs.squeeze(-1), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for texts, labels in tqdm(val_loader, colour="green"):
                texts, labels = texts.to(device), labels.to(device).float()

                # Pasar texto y mask al modelo
                outputs = model(texts)
                outputs = outputs.squeeze()

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
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


def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    with open(".key.txt", "r", encoding="UTF-8") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    train_model()


if __name__ == "__main__":
    main()
