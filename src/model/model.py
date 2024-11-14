import torch
import torch.nn as nn
import torch.nn.functional as F
from token_emb import TokenEmbedding


class LSTMModel(nn.Module):
    def __init__(self, vocab, embedding_dim=200, dropout=0.2, attention_dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            TokenEmbedding(vocab).idx_to_vec,
            padding_idx=0,
            freeze=True,
        ).to(torch.float32)

        hidden_size = 256
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout  # Dropout between LSTM layers
        )
        self.apply(init_weights)

        self.attention_weights = nn.Linear(hidden_size * 2, 1)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def attention(self, lstm_output):
        attn_scores = self.attention_weights(lstm_output)

        attn_scores = self.attention_dropout(attn_scores)
        attn_weights = F.softmax(attn_scores, dim=1)

        weighted_output = lstm_output * attn_weights
        out = weighted_output.sum(dim=1)

        attn_penalty = torch.norm(attn_weights, p=2).mean()

        return out, attn_penalty

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, _ = self.lstm(embedded)

        # Compute attention scores and get penalty
        out, attn_penalty = self.attention(lstm_output)

        out = self.layer_norm(out)

        out = self.dropout(out)
        out = self.fc(out)
        output = torch.sigmoid(out)

        return output, attn_penalty


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


