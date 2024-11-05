import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from model import LSTMModel
from dataset import TextDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Hiperparámetros
emb_dim = 200
batch_size = 16
num_folds = 5  # Aún se usa para la validación cruzada


def load_model(model_path):
    """
    Load a pre-trained model from the given path.
    Args:
        model_path (str): Path to the model.
    Returns:
        torch.nn.Module: The loaded model.
    """
    model = LSTMModel()
    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def tokenize_input(padding_word="END"):
    df = pd.read_csv("data/dataset.csv")
    df = df.dropna(subset=["text", "label"])
    X = df["text"].tolist()
    y = df["label"].tolist()

    max_sequence_length = max(len(x) for x in X)
    text_tokens = [x.split() for x in X]

    text_padded = pad_sequences(
        text_tokens,
        maxlen=max_sequence_length,
        padding="post",
        value=padding_word,
        dtype=object,
    )
    return text_padded, y


def prepare_loader(X, y):
    dataset = TextDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.round(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    plot_confusion_matrix(y_true, y_pred)


def k_fold_cross_validation(model, num_folds):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=33)
    fold_results = []
    X, y = tokenize_input()
    for fold, (_, test_idx) in enumerate(kfold.split(X)):
        print(f"\nFold {fold + 1}/{num_folds}")

        X_test = X[test_idx]
        y_test = np.array(y)[test_idx]

        test_loader = prepare_loader(X_test, y_test)

        y_pred, y_true = get_predictions(model, test_loader)
        fold_results.append((y_true, y_pred))

    return fold_results


def main():
    model_path = "model/model_trained.pth"
    model = load_model(model_path)
    print("\nIniciando K-Fold Cross Validation...")
    k_fold_results = k_fold_cross_validation(model, num_folds)

    all_true = []
    all_pred = []

    for true, pred in k_fold_results:
        all_true.extend(true.tolist())
        all_pred.extend(pred.tolist())

    evaluate_model(np.array(all_true), np.array(all_pred))
    print(predict_sentiment(model, "women do not belong in politics"))


def predict_sentiment(net, sequence):
    """Predict the sentiment of a text sequence."""
    sequence_tensor = torch.tensor(sequence.split(), device=device).long()
    with torch.no_grad():
        output = net(sequence_tensor.unsqueeze(0))
    print(output)

    label = torch.round(output).item()

    return "sexist" if label == 1 else "non sexist"


if __name__ == "__main__":
    main()
