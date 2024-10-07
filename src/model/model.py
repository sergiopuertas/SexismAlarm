import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, emb_dim, pretrained_embeddings, dropout):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        self.bilstm1 = nn.LSTM(
            input_size=emb_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.fc1 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm1(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return self.sigmoid(x)
