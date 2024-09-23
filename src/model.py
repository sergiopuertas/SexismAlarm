from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Activation, MaxPool1D, LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

tokenizer = Tokenizer(num_words=10000)
emb_dim = 50
maxlen = 100
vocab = len(tokenizer.word_index) + 1
emb_mat = np.zeros((vocab, emb_dim))


def prepare_data():
    seed = 33
    data = pd.read_csv("data/dataset.csv")
    X_train, X_test = train_test_split(
        data, train_size=0.75, test_size=0.25, shuffle=True, random_state=seed
    )
    tokenizer.fit_on_texts(X_train)
    x_train = tokenizer.texts_to_sequences(X_train)
    x_test = tokenizer.texts_to_sequences(X_test)
    x_train = pad_sequences(x_train, padding="post", maxlen=maxlen)
    x_test = pad_sequences(x_test, padding="post", maxlen=maxlen)
    return x_train, x_test


def embed():
    file_path = "../Glove Twitter 27B/"

    with open(file_path + "glove.twitter.27B.200d.txt") as f:
        for line in f:
            word, *emb = line.split()
            if word in tokenizer.word_index:
                ind = tokenizer.word_index[word]
                emb_mat[ind] = np.array(emb, dtype="float32")[:emb_dim]


def build_model():
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab,
            output_dim=emb_dim,
            weights=[emb_mat],
            input_length=maxlen,
            trainable=False,
        )
    )
    model.add(MaxPool1D())
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    X_train, X_test = prepare_data()
    embed()
    model = build_model()


if __name__ == "__main__":
    main()
