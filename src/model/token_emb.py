import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TokenEmbedding:
    def __init__(
        self, embedding_file="model/glove.twitter.27B.200d.txt", embedding_dim=200
    ):
        self.embedding_dim = embedding_dim
        self.embedding_file = embedding_file
        self.token_to_vec, self.unknown_vector = self._load_embedding()

    def _load_embedding(self):
        token_to_vec = {}
        unknown_vector = np.random.uniform(-0.05, 0.05, self.embedding_dim).astype(
            np.float32
        )

        with open(self.embedding_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                token_to_vec[word] = vector

        return (
            {k: torch.tensor(v, device=device) for k, v in token_to_vec.items()},
            torch.tensor(unknown_vector, device=device),
        )

    def __getitem__(self, token):
        return self.token_to_vec.get(token, self.unknown_vector)

    def __len__(self):
        return len(self.token_to_vec)
