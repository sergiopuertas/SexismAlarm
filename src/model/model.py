import torch
import torch.nn as nn

from src.model.token_emb import TokenEmbedding


class LSTMModel(nn.Module):
    def __init__(
        self,
        embedding_dim=200,
        pretrained_embeddings="model/glove.twitter.27B.200d.txt",
        dropout=0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            TokenEmbedding(pretrained_embeddings, embedding_dim).token_to_vec,
            padding_idx=0,
            freeze=True,
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

        self.apply(init_weights)

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Apply dropout to embeddings
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.layer_norm(lstm_out)

        lstm_out = lstm_out[:, -1, :]

        output = self.fc(lstm_out)
        output = torch.sigmoid(output)

        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
