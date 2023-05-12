import numpy as np
from tqdm import tqdm

from Graph_class import Graph


class Co_occurrencesGraph(Graph):

    def __init__(self, df_entities):
        super(Co_occurrencesGraph, self).__init__(df_entities)

    def populate_adj_matrix(self, **kargs):
        id_to_index = {id: i for i, id in enumerate(self.node_ids)}
        k = kargs['k']
        groups = self.df_entities.groupby('pmid')
        for name, group in tqdm(groups, desc="Co_occurrencesGraph populate_adj_matrix"):
            for _, curr in group.iterrows():
                neighbors = group[
                    (group['span_begin'] < curr['span_end'] + k) & (group['span_begin'] > curr['span_end'] + 1)]
                for _, neighbor in neighbors.iterrows():
                    current_entity_id = curr['id']
                    neighbor_entity_id = neighbor['id']
                    if current_entity_id != neighbor_entity_id:
                        current_entity_index = id_to_index[current_entity_id]
                        neighbor_entity_index = id_to_index[neighbor_entity_id]
                        self.adjacency_matrix[current_entity_index, neighbor_entity_index] += 1
        self.adjacency_matrix = self.adjacency_matrix + self.adjacency_matrix.T
        np.fill_diagonal(self.adjacency_matrix, 0)
        self.compute_scaled_adj_matrix()
