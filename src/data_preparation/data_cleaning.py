import pandas as pd
import re
import string
import emoji
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
spell = Speller(lang="en")


def replace_emojis_with_text(text):
    return emoji.demojize(text)


# Función para cargar abreviaciones
def load_abbreviations(file_path):
    abbreviation_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            if ":" in line:
                abbr, expansion = line.split(":", 1)
                abbreviation_dict[abbr.strip()] = expansion.strip()
    return abbreviation_dict


# Expansión de abreviaciones
def expand_abbreviations(text, abbreviation_dict):
    words = text.split()
    expanded_words = [abbreviation_dict.get(word.lower(), word) for word in words]
    return " ".join(expanded_words)


# Limpieza de texto usando expresiones regulares y NLTK
def clean_txt(text, abbreviation_dict):
    clean_text = " ".join(
        [word if word == "CONNECTED" else word.lower() for word in text.split()]
    )
    clean_text = replace_emojis_with_text(clean_text)
    clean_text = re.sub(r"http\S+", "url", clean_text)  # URLs
    clean_text = re.sub(r"(\S+)@\S+", r"\1", clean_text)  # Correos electrónicos
    clean_text = re.sub(r"\b\d+(\.\d+)?\b", "number", clean_text)  # Números
    clean_text = re.sub(r"@(\w+)", r"\1", clean_text)  # Menciones de usuario
    clean_text = re.sub(r"mention(\d+)", "mention", clean_text)  # Menciones de usuario
    clean_text = re.sub(r"#(\w+)", r"\1", clean_text)  # Hashtags
    clean_text = re.sub(
        r"\b(?:jaja|jeje|haha|hehe|lol|lmao|lmfao)+\b", "laugh", clean_text
    )  # Risas
    clean_text = re.sub(
        r"(:\)|:\(|:/|:D|<3)", " emoji ", clean_text
    )  # Ejemplos simples de emojis
    clean_text = re.sub(r"(.)\1{2,}", r"\1\1", clean_text)  # Letras repetidas
    clean_text = re.sub(r"!{2,}", "!", clean_text)  # Exclamaciones repetidas
    clean_text = re.sub(r"\?{2,}", "?", clean_text)  # Interrogaciones repetidas
    clean_text = re.sub(
        r"\$\d+(\.\d+)?|\€\d+(\.\d+)?|\£\d+(\.\d+)?", "money", clean_text
    )  # Valores monetarios ($, €, £)
    clean_text = re.sub(
        r"\b\d{1,2}:\d{2}(?:\s?[ap]m)?\b", "time", clean_text
    )  # Horas y minutos (e.g., 10:30, 3pm)
    clean_text = re.sub(
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "date", clean_text
    )  # Fechas (e.g., 12-05-2021)
    clean_text = re.sub(
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        "day",
        clean_text,
    )  # Días de la semana
    clean_text = re.sub(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
        "month",
        clean_text,
    )  # Meses
    clean_text = re.sub(r"\d+%", "percent", clean_text)  # Porcentajes (e.g., 50%)
    clean_text = re.sub(r"\d+/\d+", "fraction", clean_text)  # Fracciones (e.g., 1/4)
    clean_text = re.sub(
        r"\b\d+[kKmM]\b", "number", clean_text
    )  # Números grandes (e.g., 5K, 2M)
    clean_text = re.sub(
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone number", clean_text
    )  # Teléfonos comunes
    clean_text = re.sub(
        r"\b\d{1,2}:\d{2}:\d{2}(?:\s?[ap]m)?\b", "time", clean_text
    )  # Horas con segundos (e.g., 10:30:45)
    clean_text = re.sub(
        r"\b(?:mon|tue|wed|thu|fri|sat|sun)\b", "day", clean_text
    )  # Abreviaturas de días (e.g., Mon, Tue)
    clean_text = re.sub(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", "month", clean_text
    )  # Abreviaturas de meses
    clean_text = re.sub(
        r"\b\d{4}\b", "year", clean_text
    )  # Años de cuatro dígitos (e.g., 2023)
    clean_text = re.sub(
        r"\b(?:p\.?m\.?|a\.?m\.?)\b", "", clean_text
    )  # Quita 'am' y 'pm' dejando solo la hora
    clean_text = re.sub(
        r"\b(?:tbd|n/a|unknown)\b", "unknown", clean_text
    )  # Estados indeterminados (e.g., TBD, N/A)

    clean_text = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", clean_text)
    clean_text = expand_abbreviations(clean_text, abbreviation_dict)

    # clean_text = spell(clean_text)
    clean_text = [lemmatizer.lemmatize(token) for token in clean_text]
    clean_text = "".join(clean_text)
    return clean_text


# Función para limpiar un archivo CSV
def clean_csv(input_csv, output_csv, abb_dict="data_preparation/abb_dict.txt"):
    print(1)
    abbreviation_dict = load_abbreviations(abb_dict)
    print(2)
    df = pd.read_csv(input_csv)
    print(3)
    synth_rows = []
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
