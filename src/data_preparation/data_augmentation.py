import nlpaug.augmenter.word as naw
import nltk
import random
from data_cleaning import clean_csv
import pandas as pd

nltk.download("averaged_perceptron_tagger")

word_repetition_map = {
    "women": 2,
    "man": 2,
    "woman": 1,
    "male": 1,
    "female": 1,
    "males": 1,
    "females": 1,
}


def augment_text(text, aug_syn, repetitions):
    # Aumentar una frase un número de veces determinado
    augmented_texts = [aug_syn.augment(text)[0] for _ in range(repetitions)]
    return augmented_texts


def generate_conjunction_phrases(sentences, label, max_combinations=5):
    # Generar nuevas frases concatenadas
    new_sentences = []
    for _ in range(len(sentences)):
        chosen_sentences = random.sample(sentences, random.randint(2, max_combinations))
        combined_sentence = " CONNECTED ".join(chosen_sentences)
        new_sentences.append([combined_sentence, label, "6"])
    return new_sentences


def augment_dataset(df, repetition_map):
    aug_syn = naw.SynonymAug()
    sexist_sentences = df[df["label"] == 1]["text"].tolist()
    non_sexist_sentences = df[df["label"] == 0]["text"].tolist()

    augmented_rows = []

    # Aumentar frases sexistas
    augmented_sexist = []
    for text in sexist_sentences:
        augmented_sexist.extend(
            [[aug_text, 1, "5"] for aug_text in augment_text(text, aug_syn, 3)]
        )
    all_sexist = pd.DataFrame(augmented_sexist, columns=["text", "label", "set"])[
        "text"
    ].tolist()
    augmented_rows.extend(generate_conjunction_phrases(all_sexist, 1))

    # Aumentar frases no sexistas
    augmented_non_sexist = []
    for text in non_sexist_sentences:
        repetitions = max([repetition_map.get(word, 0) for word in text.split()])
        if repetitions > 0:
            augmented_non_sexist.extend(
                [
                    [aug_text, 0, "5"]
                    for aug_text in augment_text(text, aug_syn, repetitions)
                ]
            )
    all_non_sexist = pd.DataFrame(
        augmented_non_sexist, columns=["text", "label", "set"]
    )["text"].tolist()
    augmented_rows.extend(generate_conjunction_phrases(all_non_sexist, 0))
    return pd.DataFrame(augmented_rows, columns=["text", "label", "set"])


if __name__ == "__main__":
    clean_csv(
        "data/full.csv", "data/full_balanced.csv", "data_preparation/abb_dict.txt"
    )

    df = pd.read_csv("data/full_balanced.csv")

    print("Generando datos aumentados...")
    augmented_df = augment_dataset(df, word_repetition_map)

    df_final = pd.concat([df, augmented_df], ignore_index=True)

    df_final.to_csv("data/dataset.csv", index=False)
