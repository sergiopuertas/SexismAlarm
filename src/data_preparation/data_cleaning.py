import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import contractions
import re

lmtzr = nltk.WordNetLemmatizer().lemmatize
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))


def remove_emoji(string):  # copied, reference
    emoji_pattern = re.compile(
        "["
        "U0001F600-U0001F64F"  # emoticons
        "U0001F300-U0001F5FF"  # symbols & pictographs
        "U0001F680-U0001F6FF"  # transport & map symbols
        "U0001F1E0-U0001F1FF"  # flags (iOS)
        "U00002702-U000027B0"
        "U000024C2-U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)


def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)


def get_wordnet_pos(treebank_tag):  # copied, reference
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def normalize_text(text):  # copied, reference
    word_pos = nltk.pos_tag(nltk.word_tokenize(text))
    lemm_words = [lmtzr(sw[0], get_wordnet_pos(sw[1])) for sw in word_pos]

    return [x.lower() for x in lemm_words]


def clean_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    synth_rows = list()
    for i, row in df.iterrows():
        text = row[0]
        label = row["Label"]
        set_value = row["set"]
        print(f"Processing text nº{i+1}")

        clean_text = text.lower()  # lowercase all
        clean_text = re.sub("http[s]?://\S+", "", clean_text)  # regex to remove urls
        clean_text = contractions.fix(clean_text)  # fix contractions
        clean_text = remove_emoji(clean_text)  # remove emoji
        clean_text = " ".join(clean_text.split())  # remove extra spaces
        clean_text = normalize_text(
            clean_text
        )  # normalize : tokenize, lemmatize and POS tagging
        clean_text = remove_stopwords(clean_text)  # remove stopwords

        synth_rows.append([clean_text, label, set_value])
        print(clean_text, label, set_value)

    augmented_df = pd.DataFrame(synth_rows, columns=["Text", "Label", "Set"])
    augmented_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = "../data/full.csv"
    output_csv = "../data/full_clean.csv"
    clean_csv(input_csv, output_csv)
