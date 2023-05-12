from enum import Enum

from matplotlib import pyplot as plt
from pandas import DataFrame
from tqdm import tqdm

from Graph_class import Graph
from Metrics.Coccurrences_class import Co_occurrencesGraph
from Metrics.Word2Vec_class import Word2VecGraph
from NodeRank import GraphRanker
from Utils import is_valid_relation


def test_drug_disease(graph: Graph, ts_set, firsts_selected_diseases: int = None, validity: int = 1):
    i_diseases = [i_node for _, i_node in GraphRanker(graph).rank_nodes() if graph.index_to_info[i_node]['source']][
                 :firsts_selected_diseases]  # intersection
    matched = 0
    predicate = lambda x: x['obj'] == 'drug'

    for i_disease in tqdm(i_diseases, desc="test_drug_disease"):
        nearest_neighbors = graph.find_nearest(i_disease, predicate=predicate, max_=validity)
        for i_neighbor, history, cost in nearest_neighbors:
            if is_valid_relation(disease_id=graph.index_to_info[i_disease]['id'],
                                 drug_id=graph.index_to_info[i_neighbor]['id'],
                                 test_set=ts_set):
                matched += 1
                break

    return matched / len(i_diseases)


graphType = Enum('GraphType', ['COOCCURRENCES', 'WORD2VEC'])


def build_graph_from_config(config, df_entities: DataFrame, texts):
    graph_type, kargs = config
    graph = None
    if graph_type.value == graphType.COOCCURRENCES.value:
        graph = Co_occurrencesGraph(df_entities)
    elif graph_type.value == graphType.WORD2VEC.value:
        graph = Word2VecGraph(df_entities, texts)
    if graph is not None:
        graph.populate_adj_matrix(**kargs)
    return graph


def gridsearch(configs, ts_set, texts, df_entities, firsts_selected_diseases=100, validity=5):
    ranking = []
    for i, config in enumerate(tqdm(configs, desc="gridsearch")):
        print(f'Config {i + 1} / {len(configs)}: {config}')
        graph = build_graph_from_config(config, df_entities, texts)
        precision = test_drug_disease(graph, ts_set, firsts_selected_diseases=firsts_selected_diseases,
                                      validity=validity)
        print(f'Precision {precision} for config {config}')
        ranking.append((precision, config))
    ranking.sort(key=lambda x: x[0], reverse=True)
    return ranking


def find_matches_drug_disease(graph: Graph, firsts_selected_diseases: int = None, validity: int = 1):
    i_diseases = [i_node for _, i_node in GraphRanker(graph).rank_nodes() if graph.index_to_info[i_node]['source']][
                 :firsts_selected_diseases]  # intersection
    matches = []
    predicate = lambda x: x['obj'] == 'drug'

    for i_disease in tqdm(i_diseases, desc="test_drug_disease"):
        nearest_neighbors = graph.find_nearest(i_disease, predicate=predicate, max_=validity)
        matched = 0
        for i_neighbor, history, cost in nearest_neighbors:
            if is_valid_relation(disease_id=graph.index_to_info[i_disease]['id'],
                                 drug_id=graph.index_to_info[i_neighbor]['id']):
                matched += 1
        matches.append(matched / len(nearest_neighbors))

    return matches


def plot_hist(graph, validity):
    matches = find_matches_drug_disease(graph, firsts_selected_diseases=100, validity=validity)

    plt.figure(figsize=(20, 10))
    plt.hist(matches, bins=len(set(matches)))
    plt.grid(True)
    plt.show()
