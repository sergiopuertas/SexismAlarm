import pandas as pd
import nlpaug.augmenter.word as naw
import random
from data_preparation.data_cleaning import clean_csv


def augment_text(text, aug_syn, repetitions):
    return [aug_syn.augment(text)[0] for _ in range(repetitions)]


def generate_connected_phrases(concat_pool, label, num_phrases=30000):
    sentences = concat_pool['text'].tolist()
    connected = []

    while len(connected) < num_phrases:
        num = random.randint(2, 10)
        selected = random.sample(sentences, num)
        connected_text = " CONNECTED ".join(str(selected))
        connected.append({'text': connected_text, 'label': label, 'set': '6'})

    return pd.DataFrame(connected)


def main():
    """clean_csv(
            "data/full.csv", "data/full_balanced.csv", "data_preparation/abb_dict.txt"
        )
    """

    df = pd.read_csv("data/full_balanced.csv")


    # Separar datos
    sexist_original = df[df['label'] == 1]
    non_sexist_original = df[df['label'] == 0]

    # Balancear datos sexistas (12k -> 30k)
    aug_syn = naw.SynonymAug()
    textos_sexistas = sexist_original['text'].tolist()
    mitad = len(textos_sexistas) // 2
    aumentados_sexistas = []

    for i, texto in enumerate(textos_sexistas):
        repeticiones = 1 if i < mitad else 2
        aumentados = augment_text(texto, aug_syn, repeticiones)
        aumentados_sexistas.extend(aumentados)

    df_aumentado_sexista = pd.DataFrame({
        'text': aumentados_sexistas,
        'label': 1,
        'set': '5'
    })

    sexist_total = pd.concat([sexist_original, df_aumentado_sexista], ignore_index=True)

    # Dividir cada categoría en mitades
    sexist_single = sexist_total.sample(n=10000, random_state=42)
    sexist_concat = sexist_total.drop(sexist_single.index)

    non_sexist_single = non_sexist_original.sample(n=10000, random_state=42)
    non_sexist_concat = non_sexist_original.drop(non_sexist_single.index)

    # Generar frases conectadas
    sexist_connected = generate_connected_phrases(sexist_concat, 1)
    non_sexist_connected = generate_connected_phrases(non_sexist_concat, 0)

    # Combinar todos los datos
    final_df = pd.concat([
        sexist_single, non_sexist_single,
        sexist_connected, non_sexist_connected
    ], ignore_index=True)

    # Guardar resultado
    final_df.to_csv("data/dataset_final.csv", index=False)


if __name__ == "__main__":
    main()