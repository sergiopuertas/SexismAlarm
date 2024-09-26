import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, emb_dim, weights):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(emb_dim, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)  # Returns output and states, only keep output
        x = self.fc1(x[:, -1, :])  # Take only last layer of LSTM
        x = self.fc2(x)
        return self.sigmoid(x)

    # ADD ATTENTION OR OTHER STUFF
