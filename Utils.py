import re
from typing import Dict

import networkx as nx
import nltk as nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from pandas import DataFrame

from NodeRank import GraphRanker

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

        def remove_numbers(x):
            return re.sub('[^a-zA-Z]', ' ', x)

        for i, entity_row in entities:
            preprocessed = remove_numbers(text[entity_row['span_end']:old_span_begin])
            text = f"{text[:entity_row['span_begin']]} {id_to_wuid[entity_row['id']]}" \
                   f" {preprocessed} {text[old_span_begin:]}"
            old_span_begin = entity_row['span_begin']
        text = f"{remove_numbers(text[:old_span_begin])} {text[old_span_begin:]}"
        tokens = nltk.word_tokenize(text)
        return [w.lower().strip() for w in tokens if not w.lower() in stopwords.words("english")]

    return preprocess_text_


def plot_drugs(graph, cooc_g, max_n_diseases=3, max_n_drugs=5):
    i_diseases = [i_node for _, i_node in GraphRanker(graph=graph).rank_nodes() if
                  graph.index_to_info[i_node]['source']][:max_n_diseases]
    associations = {
        di: [i_neighbor for i_neighbor, history, cost in
             graph.find_nearest(di, predicate=lambda x: x['obj'] == 'drug', max=max_n_drugs)]
        for di in i_diseases
    }

    enriched_associations = {cooc_g.index_to_info[k]['mention']: [graph.index_to_info[i]['mention'] for i in v] for k, v
                             in associations.items()}

    g = nx.DiGraph()
    g.add_nodes_from(enriched_associations.keys())

    for k, v in enriched_associations.items():
        g.add_edges_from(([(k, t) for t in v]))
    g.edges(data=True)

    color_map = []
    for node in g:
        if node in enriched_associations.keys():
            color_map.append('#008080')
        else:
            color_map.append('#8B8000')

    plt.figure(figsize=(10, 10))
    nx.draw(g, node_color=color_map, with_labels=True)
