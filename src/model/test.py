import numpy as np
import torch
from tqdm import tqdm
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
from torch.utils.data import DataLoader


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Hiperparámetros
emb_dim = 200
batch_size = 64


def load_vocab(vocab_path):
    """
    Carga el vocabulario guardado desde un archivo.
    Args:
        vocab_path (str): Ruta al archivo del vocabulario.
    Returns:
        dict: Diccionario que mapea las palabras a índices.
    """
    vocab = torch.load(vocab_path)  # Cargamos el vocabulario guardado
    return vocab


def load_model(model_path, vocab):
    """
    Load a pre-trained model from the given path.
    Args:
        model_path (str): Path to the model.
    Returns:
        torch.nn.Module: The loaded model.
    """
    model = LSTMModel(vocab, embedding_dim=emb_dim)
    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    lengths = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            lengths.extend((inputs != 0).sum(dim=1).cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(lengths)


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

def predict_sentiment(net, sequence, vocab):
    """Predict the sentiment of a text sequence."""

    sequence_tokens = sequence.split()

    # Procesa la frase con el modelo
    sequence_indices = [vocab.get(word, 0) for word in sequence_tokens]
    sequence_tensor = torch.tensor(sequence_indices, device=device).long().unsqueeze(0)
    print(f"Sequence: {sequence_tensor}")
    with torch.no_grad():
        output = torch.sigmoid(net(sequence_tensor))
        print(output.item())

    label = torch.round(output).item()
    return "sexist" if label == 1 else "non sexist"


def main():
    vocab_path = "model/Versions/V2/vocab.pt"
    model_path = "model/Versions/V2/model_trained.pth"
    loader_path = "model/test_loader.pt"
    print(f"using:{device}")
    vocab = load_vocab(vocab_path)
    model = load_model(model_path, vocab)
    """loader = torch.load(loader_path,weights_only=False)

    all_true, all_pred, all_lengths = get_predictions(model, loader)
    unique_values, counts = np.unique(all_true, return_counts=True)
    print(unique_values,counts)
    all_pred = (all_pred >= 0.5)
    evaluate_model(np.array(all_true), np.array(all_pred))"""

    # Interactive loop for predictions
    while True:
        text = input("Enter a text to analyze: ")
        if text == "exit":
            break
        print(predict_sentiment(model, text, vocab))


if __name__ == "__main__":
    main()
