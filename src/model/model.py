import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LSTMModel(nn.Module):
    def __init__(self, vocab, embedding_dim=200, dropout=0.4, attention_dropout=0.2):
        """
        vocab: Vocabulario para construir la matriz de embeddings.
        embedding_dim: Dimensión de los embeddings.
        dropout: Dropout aplicado en el LSTM y en las capas fully connected.
        attention_dropout: Dropout aplicado sobre los pesos en la atención.
        """
        super(LSTMModel, self).__init__()

        # Capa de embeddings
        self.embedding = nn.Embedding.from_pretrained(
            TokenEmbedding(vocab).idx_to_vec,
            padding_idx=0,
            freeze=False,
        ).to(torch.float32)

        # LSTM bidireccional
        hidden_size = 256
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.apply(init_weights)

        # Self-Attention
        self.attention = SelfAttention(
            embed_dim=hidden_size * 2,
            dropout=attention_dropout
        )

        # LayerNorm, Dropout y Capa final Fully Connected
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, mask=None):
        """
        x: tensor de índices de tokens, shape (batch_size, seq_len)
        mask: tensor opcional de shape (batch_size, seq_len) con 1 para tokens reales y 0 para padding.
        """
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        lstm_output, _ = self.lstm(embedded)  # (batch, seq_len, hidden_size*2)

        attn_output, attn_weights = self.attention(lstm_output)
        # attn_output: (batch, seq_len, hidden_size*2)
        # attn_weights: (batch, seq_len, seq_len)

        attn_context = torch.matmul(attn_weights, attn_output)  # (batch, seq_len, hidden_size*2)

        out = attn_context.mean(dim=1)  # (batch, hidden_size*2)

        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)  # (batch, 1)
        out = out.squeeze(-1)  # (batch,)
        return out


class SelfAttention(nn.Module):
    """
    Implementa un mecanismo de self-attention simple:
    - Calcula proyecciones lineales para Q, K y V.
    - Aplica producto punto escalado para obtener scores de atención.
    - Utiliza softmax para normalizar y aplicar dropout.
    """

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Calcular scores: (batch, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Salida: (batch, seq_len, embed_dim)
        out = torch.bmm(attn_weights, V)
        return out, attn_weights

class TokenEmbedding:  # inspired in d2L
    def __init__(self, vocab, embedding_file="glove.twitter.27B.200d.txt"):
        self.idx_to_token = list(vocab.keys())
        self.token_to_idx = vocab
        self.idx_to_vec = self._load_embedding(embedding_file)
        self.unknown_idx = 0
    def _load_embedding(self, embedding_file):
        idx_to_vec = np.zeros((len(self.idx_to_token)+1, 200))
        with open(embedding_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in self.token_to_idx:
                    idx = self.token_to_idx[word]
                    idx_to_vec[idx] = np.array(values[1:], dtype="float32")
        return torch.tensor(idx_to_vec).to(device)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs
    def __len__(self):
        return len(self.idx_to_token)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.kaiming_normal_(m._parameters[param])
