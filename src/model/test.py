import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
from .train.train import prepare_loaders
from sklearn.model_selection import KFold
from model import LSTMModel
from tqdm import tqdm

embed_dim = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device):
    """
    Load a pre-trained model from the given path.
    Args:
        model_path (str): Path to the model.
        device (str): Device to load the model onto ('cpu' or 'cuda').
    Returns:
        torch.nn.Module: The loaded model.
    """
    model = LSTMModel(
        embed_dim, pretrained_embeddings=torch.load("model/train/embeddings.pt")
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def get_predictions(model, loader, device):
    """
    Get predictions from the model and true labels for evaluation.
    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader with test data.
        device (str): Device to use ('cpu' or 'cuda').
    Returns:
        np.array, np.array: Model predictions and true labels.
    """
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.round(outputs).cpu().numpy()  # Round for binary predictions
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix.
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve and compute AUC score.
    Args:
        y_true (np.array): Ground truth labels.
        y_pred_proba (np.array): Model probability predictions.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Plot Precision-Recall curve.
    Args:
        y_true (np.array): Ground truth labels.
        y_pred_proba (np.array): Model probability predictions.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="b", lw=2, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()


def evaluate_model(y_true, y_pred, y_pred_proba):
    """
    Evaluate the model by calculating and displaying key metrics.
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Model predictions.
        y_pred_proba (np.array): Model probability predictions.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Plot metrics
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_pred_proba)
    plot_precision_recall_curve(y_true, y_pred_proba)


def predict_phrase(model, vocab, text):
    text = torch.tensor(vocab[text.split()], device=device)
    label = torch.argmax(model(text.reshape(1, -1)), dim=1)
    return "sexist" if label == 1 else "non sexist"


def main():
    """
    Main function to load the model, perform cross-validation, and evaluate the model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model_path = "model/model_trained.pth"  # Asegúrate de que la ruta sea correcta
    model = load_model(model_path, device)

    # Load your dataset
    df = pd.read_csv("data/dataset.csv")
    df = df.dropna(subset=["text", "label"])  # Clean the dataset
    X = df["text"].tolist()
    y = df["label"].tolist()
    vectorizer = torch.load("model/vectorizer.pt")

    # Define KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=33)

    all_preds = []
    all_true = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # Prepare loaders for the current fold
        _, test_loader = prepare_loaders(vectorizer, X_train, y_train, X_test, y_test)
        # Get predictions for the current fold
        y_pred, y_true = get_predictions(model, test_loader, device)
        y_pred_proba = y_pred

        # Store predictions and true labels
        all_preds.extend(y_pred)
        all_true.extend(y_true)

        evaluate_model(y_true, y_pred, y_pred_proba)

    overall_accuracy = accuracy_score(all_true, all_preds)
    print(f"Overall Accuracy across all folds: {overall_accuracy:.4f}")


if __name__ == "__main__":
    main()
