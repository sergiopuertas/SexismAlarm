import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, pretrained_embeddings, dropout=0.4):
        super().__init__()

        # Embedding layer with pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, padding_idx=0, freeze=True  # Disallow fine-tuning
        )

        hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Apply dropout to embeddings
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.layer_norm(lstm_out)

        lstm_out = lstm_out[:, -1, :]

        output = self.fc(lstm_out)
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
