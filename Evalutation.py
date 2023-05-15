from enum import Enum
from random import sample
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.stats import stats
from tqdm import tqdm

from Graph_class import Graph
from Metrics.Coccurrences_class import Co_occurrencesGraph
from Metrics.Word2Vec_class import Word2VecGraph
from NodeRank import GraphRanker
from Utils import is_valid_relation, normalize_meshId


def test_DDAs(graph: Graph,
              top_z: int,
              test_set: Dict,
              max_diseases: int) -> float:
    """
    Given the knowledge graph, the method tests it on test_set
    :param graph: Knowledge graph
    :param test_set: Benchmark to test
    :param max_diseases:  max disease to take into consideration
    :param top_z: The top-z drug discovered by the algorthm
    :return: precision
    """

    i_diseases = [i_node for _, i_node in GraphRanker(graph).rank_nodes() if graph.index_to_info[i_node]['source']][
                 :max_diseases]  # intersection
    matched = 0

    # For all diseases that are contained in both knowledge graph and test set
    for i_disease in tqdm(i_diseases, desc="test_drug_disease"):

        # return the top-top_z drugs
        nearest_neighbors = graph.find_nearest(i_disease, predicate=lambda x: x['obj'] == 'drug', max_=top_z)
        for i_neighbor, _, _ in nearest_neighbors:
            if is_valid_relation(disease_id=graph.index_to_info[i_disease]['id'],
                                 drug_id=graph.index_to_info[i_neighbor]['id'],
                                 test_set=test_set):
                matched += 1
                break

    return matched / len(i_diseases)


def find_matches_drug_disease(graph: Graph,
                              top_z: int,
                              test_set: Dict,
                              max_diseases: int) -> list:
    """

    :param graph: Knowledge graph
    :param test_set:
    :param max_diseases: max disease to take into consideration
    :param top_z: the top-z drug discovered by the algorthm
    :return:
    """
    i_diseases = [i_node for _, i_node in GraphRanker(graph).rank_nodes() if graph.index_to_info[i_node]['source']][
                 :max_diseases]  # intersection

    matches = []

    # For all diseases that are contained in both knowledge graph and test set
    for i_disease in tqdm(i_diseases, desc="test_drug_disease"):

        # return the top-z drugs
        nearest_neighbors = graph.find_nearest(i_disease, predicate=lambda x: x['obj'] == 'drug', max_=top_z)

        matched = 0

        for i_neighbor, _, _ in nearest_neighbors:
            if is_valid_relation(disease_id=graph.index_to_info[i_disease]['id'],
                                 drug_id=graph.index_to_info[i_neighbor]['id'],
                                 test_set=test_set):
                matched += 1
        matches.append(matched / len(nearest_neighbors))

    return matches


graphType = Enum('GraphType', ['COOCCURRENCES', 'WORD2VEC'])


def create_knowledge_graph(config,
                           df_entities: DataFrame,
                           texts) -> Graph:
    """
    Given the configurations, the method builds the Knowledge graph
    :param config:
    :param df_entities:
    :param texts:
    :return:
    """
    graph_type, kargs = config
    graph = None

    if graph_type.value == graphType.COOCCURRENCES.value:
        graph = Co_occurrencesGraph(df_entities)

    elif graph_type.value == graphType.WORD2VEC.value:
        graph = Word2VecGraph(df_entities, texts)

    if graph is not None:
        graph.populate_adj_matrix(**kargs)

    return graph


def model_evaluation(configs,
                     ts_set: Dict,
                     texts,
                     top_z: int,
                     df_entities: DataFrame,
                     max_diseases: int = 100) -> list:
    """
    Given a set of configuration to try, the method finds the best one
    :param configs: configuration to try
    :param ts_set:
    :param top_z: The top-z drug discovered by the algorthm
    :param texts:
    :param df_entities: Sources information (documents)
    :param max_diseases: max disease to take into consideration
    :return: a sorted list of models
    """
    ranking = []

    # Iterate all configuration provided
    for i, config in enumerate(tqdm(configs, desc="gridsearch")):
        print(f'Config {i + 1} / {len(configs)}: {config}')

        # Build the knowledge graph
        graph = create_knowledge_graph(config, df_entities, texts)

        # Testing the knowledge graph on test_set
        precision = test_DDAs(graph=graph,
                              top_z=top_z,
                              test_set=ts_set,
                              max_diseases=max_diseases)

        print(f'Precision {precision} for config {config}')
        ranking.append((precision, config))

    # We sort the results by precision
    ranking.sort(key=lambda x: x[0], reverse=True)
    return ranking


def plot_hist(x,y):

    plt.figure(figsize=(8, 4))
    ax = plt.axes()
    ax.set_facecolor("#ececec")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white')
    ax.xaxis.grid(color='white')
    plt.bar(x, y, color="#2985e5", edgecolor="black")
    plt.xticks(range(len(x)), [str(i*10) + "%" for i in x])
    plt.xlabel("Precision")
    plt.ylabel("Num of disease")
    plt.title("Precision@Z for the first 100 most important diseases in the KG")
    plt.show()


def sample_disease2drugs(ddas: dict,
                         disease: str,
                         len_sampling: int) -> list[str]:
    """
    Given a DDAs dictionary sample k drugs random
    :param ddas: Disease-Drug Association provided by test set
    :param disease: A given disease
    :param len_sampling: length of sampling
    :return:
    """
    drug_ids = ddas[normalize_meshId(disease)]
    samples = sample(drug_ids, len_sampling) if len_sampling < len(drug_ids) else drug_ids
    return [normalize_meshId(id) for id in samples]


def sample_drugs(graph: Graph, len_sampling: int):
    """
    Given the Knowledge graph, return k drugs random
    :param graph: Knowledge graph
    :param len_sampling: length of sampling
    :return:
    """
    drug_ids = [normalize_meshId(id) for id, info in graph.id_to_info.items() if info['obj'] == 'drug']
    return sample(drug_ids, len_sampling) if len_sampling < len(drug_ids) else drug_ids


def pValues_DDAs(graph: Graph,
                 ddas: Dict,
                 max_diseases: int = None,
                 n_samples: int = 15) -> list[float]:
    """

    :param graph: Knowledge graph
    :param ddas:
    :param max_diseases: max disease to take into consideration
    :param n_samples:
    :param offset:
    :return:
    """

    # Make the intersection between Test and graph (disease)
    i_diseases = [i_node for _, i_node in GraphRanker(graph).rank_nodes() if graph.index_to_info[i_node]['source']]
    i_diseases = i_diseases[:max_diseases]

    pValues = []

    # Foreach disease
    for i_disease in tqdm(i_diseases, desc="find_pValues_drug_disease"):

        # Sample random drugs by test_set (ddas dictionary) for a given disease
        sample_true = sample_disease2drugs(disease=normalize_meshId(graph.node_ids[i_disease]),
                                           ddas=ddas,
                                           len_sampling=n_samples)

        # Sample random drugs by Knowledge graph
        sample_random = sample_drugs(graph=graph,
                                     len_sampling=n_samples)

        storage = set()

        def stop_condition(x):
            norm_id = normalize_meshId(x['id'])
            if norm_id in sample_true or norm_id in sample_random:
                storage.add(norm_id)
            return len(storage) >= len(sample_true) + len(sample_random)

        nearest_neighbors = graph.find_nearest(i_disease,
                                               predicate=lambda x: x['obj'] == 'drug',
                                               max_=None,
                                               stop_condition=stop_condition)

        scores_true, scores_random = [], []

        for rank, (i_neighbor, _, _) in enumerate(nearest_neighbors):

            id_node = normalize_meshId(graph.node_ids[i_neighbor])
            if id_node in sample_true:
                scores_true.append(rank + 1)

            if id_node in sample_random:
                scores_random.append(rank + 1)

        pValue = stats.ttest_ind(scores_true, scores_random)[1]

        pValues.append(0 if np.isnan(pValue) else pValue)

    return pValues
