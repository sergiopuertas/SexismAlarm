import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        """
        Args:
            texts: A list where each entry is a list of words (tokens) for a text.
            labels: Optional; a list of labels corresponding to each text.
            embedding: TokenEmbedding instance to convert tokens to indices.
        """
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return text, label
        else:
            return text

