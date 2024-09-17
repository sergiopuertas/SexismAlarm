import pandas as pd
import nlpaug.augmenter.word as naw
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords') # para stopwords
nltk.download('punkt') # para stopwords
nltk.download('averaged_perceptron_tagger') # para el synonymaug
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def rephrase_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    """
    aug_bert = naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased',
        action='substitute',
        top_k=20
    )
    """

    aug_syn = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=10,
                   aug_p=0.3, lang='eng',
                   stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None,
                   force_reload=False,
                   verbose=0)

    augmented_rows = []

    for i, row in df.iterrows():
        text = row[0]
        label = row['Label']
        set_value = row['set']

        print(f'Processing text nº{i+1}')

        for _ in range(4):
            # augmented_text = aug_bert.augment(text)
            augmented_text = aug_syn.augment(text)
            syn_augmented_text = remove_stopwords(augmented_text)
            augmented_rows.append([syn_augmented_text, label, set_value])
            print(augmented_text, label, set_value)

    augmented_df = pd.DataFrame(augmented_rows, columns=['Text', 'Label', 'Set'])
    augmented_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = '../data/only_sexist.csv'
    output_csv = '../data/only_sexist_changed.csv'
    # rephrase_csv(input_csv, output_csv)
