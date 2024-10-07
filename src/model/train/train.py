import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import StepLR
from ..model import LSTMModel

# Hyperparameters
emb_dim = 200
num_epochs = 5
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
dropout = 0.4


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
    scheduler = StepLR(optimizer, step_size=2, gamma=0.15)

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                "model_trained.pth",
            )
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
    path = "model/train/"
    with open("key.txt", "r", encoding="UTF-8") as f:
        wandb_key = f.read().strip()

    wandb.login(key=wandb_key)

    train_loader = torch.load(f"{path}train_loader.pt")
    val_loader = torch.load(f"{path}val_loader.pt")
    pretrained_embeddings = torch.load(f"{path}embeddings.pt")

    model = LSTMModel(emb_dim, pretrained_embeddings, dropout).to(device)

    # Train the model
    train_model(device, train_loader, val_loader, model)


if __name__ == "__main__":
    main()
