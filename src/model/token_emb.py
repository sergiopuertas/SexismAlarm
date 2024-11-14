import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TokenEmbedding:  # inspired in d2L
    def __init__(self, vocab, embedding_file="model/glove.twitter.27B.200d.txt"):
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