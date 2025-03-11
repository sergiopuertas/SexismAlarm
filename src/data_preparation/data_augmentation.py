import pandas as pd
import nlpaug.augmenter.word as naw
import random
from data_preparation.data_cleaning import clean_csv

target_words = ["men", "man", "woman", "women","male","female", "males","females"]


def augment_text(text, aug_syn, repetitions):
    return [aug_syn.augment(text)[0] for _ in range(repetitions)]

def contains_target_words(text, target_words):
    return any(word in str(text) .lower() for word in target_words)

def generate_connected_phrases(concat_pool, label, num_phrases=20000):
    sentences = concat_pool['text'].tolist()
    connected = []

    while len(connected) < num_phrases:
        num = random.randint(2, 5)
        selected = random.sample(sentences, num)
        selected = [str(sentence) for sentence in selected]
        connected_text = " CONNECTED ".join(selected)
        connected.append({'text': connected_text, 'label': label, 'set': '6'})

    return pd.DataFrame(connected)


def main():
    """clean_csv(
        "data/full_v2.csv", "data/full_v2.csv", "data_preparation/abb_dict.txt"
        )
    print("limpio")"""

    df = pd.read_csv("data/full_v2.csv")

    sexist_original = df[df['label'] == 1]
    non_sexist_original = df[df['label'] == 0]

    aug_syn = naw.SynonymAug()
    textos_sexistas = sexist_original['text'].tolist()
    aumentados_sexistas = []

    for i, texto in enumerate(textos_sexistas):
        if i%2 == 0 :
            if not any(target in str(texto) for target in target_words):
                aumentados = augment_text(str(texto), aug_syn, 1)
                aumentados_sexistas.extend(aumentados)


    df_aumentado_sexista = pd.DataFrame({
        'text': aumentados_sexistas,
        'label': 1,
    })

    sexist_total = pd.concat([sexist_original, df_aumentado_sexista], ignore_index=True)
    sexist_connected = generate_connected_phrases(sexist_total, 1)
    non_sexist_connected = generate_connected_phrases(non_sexist_original, 0)

    final_df = pd.concat([
        sexist_total, non_sexist_original,
        sexist_connected, non_sexist_connected
    ], ignore_index=True)
    print("acabó")
    final_df.to_csv("data/dataset_full_v2.csv", index=False)


if __name__ == "__main__":
    main()