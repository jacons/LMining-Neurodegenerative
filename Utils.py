import re
from typing import Dict

import nltk as nltk
from nltk.corpus import stopwords
from pandas import DataFrame

nltk.download('punkt')  # <-- essential for tokenization
nltk.download('stopwords')  # <-- we run this command to download the stopwords in the project


def normalize_meshId(x):
    return x.upper().replace('MESH:', '')


def is_valid_relation(drug_id, disease_id, test_set) -> bool:
    try:
        return test_set[(
            normalize_meshId(drug_id),
            normalize_meshId(disease_id),
        )]
    except KeyError:
        return False


def preprocess_text(df_entities: DataFrame, id_to_wuid: Dict):
    def preprocess_text_(row) -> list[str]:
        text = row['text']
        entities = df_entities[df_entities['pmid'] == row['pmid']].sort_values(by='span_begin',
                                                                               ascending=False).iterrows()
        old_span_begin = None
        remove_numbers = lambda x: re.sub('[^a-zA-Z]', ' ', x)
        for i, entity_row in entities:
            preprocessed = remove_numbers(text[entity_row['span_end']:old_span_begin])
            text = f"{text[:entity_row['span_begin']]} {id_to_wuid[entity_row['id']]} {preprocessed} {text[old_span_begin:]}"
            old_span_begin = entity_row['span_begin']
        text = f"{remove_numbers(text[:old_span_begin])} {text[old_span_begin:]}"
        tokens = nltk.word_tokenize(text)
        return [w.lower().strip() for w in tokens if not w.lower() in stopwords.words("english")]

    return preprocess_text_
