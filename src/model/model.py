import torch
import torch.nn as nn
import torch.nn.functional as F
from token_emb import TokenEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, attention_dim, attention_dropout=0.2):
        """
        attention_dim: Dimensión de la representación (por ejemplo, hidden_size*2 para LSTM bidireccional)
        attention_dropout: Dropout aplicado sobre los pesos de atención
        """
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(attention_dim, attention_dim)
        self.key = nn.Linear(attention_dim, attention_dim)
        self.value = nn.Linear(attention_dim, attention_dim)
        self.dropout = nn.Dropout(attention_dropout)
        self.scale = math.sqrt(attention_dim)

    def forward(self, x, mask=None):
        """
        x: tensor de forma (batch_size, seq_len, attention_dim)
        mask: tensor de forma (batch_size, seq_len) con 1 para tokens reales y 0 para padding.
        """
        Q = self.query(x)  # (batch, seq_len, attention_dim)
        K = self.key(x)  # (batch, seq_len, attention_dim)
        V = self.value(x)  # (batch, seq_len, attention_dim)

        # Producto punto escalado: (batch, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # Salida de la atención: combinación ponderada de V
        attn_output = torch.bmm(attn_weights, V)  # (batch, seq_len, attention_dim)
        return attn_output, attn_weights


# Función para inicializar pesos
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)


# Modelo LSTM con Self-Attention
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
        # Paso 1: Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Paso 2: LSTM
        lstm_output, _ = self.lstm(embedded)  # (batch, seq_len, hidden_size*2)

        # Paso 3: Self-Attention
        attn_output, attn_weights = self.attention(lstm_output)
        # attn_output: (batch, seq_len, hidden_size*2)
        # attn_weights: (batch, seq_len, seq_len)

        attn_context = torch.matmul(attn_weights, attn_output)  # (batch, seq_len, hidden_size*2)

        # Mean pooling
        out = attn_context.mean(dim=1)  # (batch, hidden_size*2)

        # Normalización, Dropout y Capa final
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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.kaiming_normal_(m._parameters[param])
