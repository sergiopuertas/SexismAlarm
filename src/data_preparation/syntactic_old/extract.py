import conllu


def extract_filtered_columns(file_path, target_deprels):
    # Leer el archivo CoNLL-U
    with open(file_path, "r", encoding="utf-8") as f:
        conllu_data = f.read()

    # Parsear el archivo CoNLL-U
    sentences = conllu.parse(conllu_data)

    filtered_data = []

    # Filtrar palabras según la relación de dependencia
    for sentence in sentences:
        sentence_data = []

        for token in sentence:
            if token["deprel"] in target_deprels:
                token_data = [
                    token["id"],
                    token["form"],  # FORM
                    "_",  # LEMMA
                    "_",  # UPOS
                    "_",  # XPOS
                    "_",  # FEATS
                    token["head"],  # HEAD
                    token["deprel"],  # DEPREL
                    "_",  # DEPS
                    "_",  # MISC
                ]
                sentence_data.append(token_data)

        if sentence_data:
            filtered_data.append(sentence_data)

    return filtered_data


target_deprels = ["nsubj", "obj", "iobj", "root", "amod", "advmod", "csubj"]

file_path = "ud-treebanks-v2.15/UD_English-GUM/en_gum-ud-dev.conllu"

filtered_words = extract_filtered_columns(file_path, target_deprels)

output_file_path = "ud-treebanks-v2.15/UD_English-GUM/en_gum-ud-selected.conllu"
with open(output_file_path, "w", encoding="utf-8") as f:
    for sentence in filtered_words:
        for token in sentence:
            f.write("\t".join(map(str, token)) + "\n")
        f.write("\n")
