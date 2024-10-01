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

        self.bilstm2 = nn.LSTM(
            input_size=256,  # 128 units * 2 directions
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(256, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 16)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)

        x, _ = self.bilstm1(x)

        x, _ = self.bilstm2(x)

        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return self.sigmoid(x)
