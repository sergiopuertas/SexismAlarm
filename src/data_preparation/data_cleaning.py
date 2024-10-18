import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons
import pandas as pd
import contractions
import re
import string

# Inicialización de NLTK
lmtzr = nltk.WordNetLemmatizer().lemmatize
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))


# Configuración de ekphrasis
text_processor = TextPreProcessor(
    normalize=["url", "email", "percent", "money", "phone", "time", "date", "number"],
    annotate={
        "hashtag",
        "allcaps",
        "elongated",
        "repeated",
        "emphasis",
        "censored",
    },
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,
    remove_tags=True,
    dicts=[emoticons],
)


# Función para cargar abreviaciones
def load_abbreviations(file_path):
    abbreviation_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            if ":" in line:
                abbr, expansion = line.split(":", 1)
                abbreviation_dict[abbr.strip()] = expansion.strip()
    return abbreviation_dict


# Función para eliminar stopwords
def remove_stopwords(text):
    filtered_words = [word for word in text if word not in stop_words]
    return " ".join(filtered_words)


# Obtener el tipo de palabra para lematización
def get_wordnet_pos(treebank_tag):
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


# Expansión de abreviaciones
def expand_abbreviations(text, abbreviation_dict):
    words = text.split()
    expanded_words = [abbreviation_dict.get(word.lower(), word) for word in words]
    return " ".join(expanded_words)


# Lematización y normalización
def normalize_text(text):
    word_pos = nltk.pos_tag(nltk.word_tokenize(text))
    lemm_words = [lmtzr(sw[0], get_wordnet_pos(sw[1])) for sw in word_pos]
    return [x.lower() for x in lemm_words]


# Función principal de limpieza de texto
def clean_txt(text, abbreviation_dict):
    clean_text = text.lower()  # Convertir a minúsculas
    clean_text = "".join(
        text_processor.pre_process_doc(clean_text)
    )  # Procesar con ekphrasis
    pattern = re.compile(
        r"[{}()\[\]<>*]"
    )  # Eliminar puntuación y caracteres no deseados
    clean_text = [word for word in clean_text if word not in string.punctuation]
    clean_text = [pattern.sub("", word) for word in clean_text]
    clean_text = "".join(clean_text)

    clean_text = " ".join(clean_text.split())  # Eliminar espacios extras
    # Expandir abreviaciones
    clean_text = expand_abbreviations(clean_text, abbreviation_dict)

    clean_text = normalize_text(clean_text)  # Tokenizar y lematizar
    clean_text = remove_stopwords(clean_text)  # Eliminar stopwords
    return clean_text


def clean_csv(input_csv, output_csv, abb_dict="data_preparation/abb_dict.txt"):
    abbreviation_dict = load_abbreviations(abb_dict)
    df = pd.read_csv(input_csv)
    synth_rows = list()
    for i, row in df.iterrows():
        text = row["text"]
        label = row["label"]
        set_value = row["set"]
        if i % 1000 == 0:
            print(f"Processing text nº{i}")

        clean_text = clean_txt(text, abbreviation_dict)
        synth_rows.append([clean_text, label, set_value])

    augmented_df = pd.DataFrame(synth_rows, columns=["text", "label", "set"])
    augmented_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = "data/full.csv"
    output_csv = "data/dataset.csv"
    clean_csv(input_csv, output_csv)
