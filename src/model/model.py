import torch
import torch.nn as nn
import torch.nn.functional as F
from token_emb import TokenEmbedding
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_attention(inputs, attn_weights, idx_to_word):
    """
    Visualiza los pesos de atención para una entrada dada.
    """
    for i, (text, weights) in enumerate(zip(inputs, attn_weights)):
        words = [idx_to_word[idx] for idx in text if idx != 0]  # Decodifica el texto
        plt.figure(figsize=(12, 2))
        sns.heatmap([weights[: len(words)]], annot=[words], fmt="", cmap="Blues")
        plt.title(f"Attention Weights for Sample {i}")
        plt.show()


class SelfAttention(nn.Module):
    def __init__(self, attention_dim, attention_dropout=0.2):
        super().__init__()
        self.query_layer = nn.Linear(attention_dim, attention_dim)
        self.key_layer = nn.Linear(attention_dim, attention_dim)
        self.value_layer = nn.Linear(attention_dim, attention_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Calcula la atención con producto punto escalado.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        return torch.matmul(attn_weights, value), attn_weights

    def forward(self, lstm_output, mask=None):
        """
        Aplica Self-Attention a las salidas de la LSTM.
        """
        query = self.query_layer(lstm_output)
        key = self.key_layer(lstm_output)
        value = self.value_layer(lstm_output)

        attn_output, attn_weights = self.scaled_dot_product_attention(
            query, key, value, mask
        )
        return attn_output, attn_weights


class LSTMModel(nn.Module):
    def __init__(self, vocab, embedding_dim=200, dropout=0.2, attention_dropout=0.2):
        super().__init__()
        # Embedding
        self.embedding = nn.Embedding.from_pretrained(
            TokenEmbedding(vocab).idx_to_vec,
            padding_idx=0,
            freeze=True,
        ).to(torch.float32)

        # LSTM
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
            attention_dim=hidden_size * 2, attention_dropout=attention_dropout
        )

        # LayerNorm, Dropout, Fully Connected
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, mask=None):
        """
        Forward pass con residual connections y mask.
        """
        embedded = self.embedding(x)
        lstm_output, _ = self.lstm(embedded)

        # Self-Attention con mask
        attn_output, attn_weights = self.attention(lstm_output, mask)

        # Residual Connection: Combina las salidas de LSTM y atención
        residual = lstm_output + attn_output

        # Agregación y LayerNorm
        out = torch.sum(attn_weights.unsqueeze(-1) * residual, dim=1)
        out = self.layer_norm(out)

        # Dropout y Fully Connected
        out = self.dropout(out)
        out = self.fc(out)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM):
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def create_padding_mask(seq, padding_idx=0):
    """
    Crea un mask para ignorar posiciones de padding.
    """
    return (seq != padding_idx).unsqueeze(1).unsqueeze(2)
