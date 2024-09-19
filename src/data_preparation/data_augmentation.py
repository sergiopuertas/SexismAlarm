import pandas as pd
import nlpaug.augmenter.word as naw
import nltk

nltk.download("averaged_perceptron_tagger")  # para el synonymaug


def rephrase_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    """
    aug_bert = naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased',
        action='substitute',
        top_k=20
    )
    """

    aug_syn = naw.SynonymAug()
    augmented_rows = []

    for i, row in df.iterrows():
        text = row[0]
        label = row["Label"]
        set_value = row["set"]

        print(f"Processing text nº{i+1}")

        for _ in range(4):
            # augmented_text = aug_bert.augment(text)
            augmented_text = aug_syn.augment(text)
            augmented_rows.append([augmented_text, label, set_value])
            print(augmented_text, label, set_value)

    augmented_df = pd.DataFrame(augmented_rows, columns=["Text", "Label", "Set"])
    augmented_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = "../data/only_sexist.csv"
    output_csv = "../data/only_sexist_augmented.csv"
    rephrase_csv(input_csv, output_csv)
    original = pd.read_csv("../data/full.csv")
    new = pd.read_csv("../data/only_sexist_changed.csv")
    df = pd.concat([original, new], ignore_index=True)
    df.to_csv("../data/dataset.csv")
