from typing import Callable

import numpy as np
import pandas as pd
import networkx as nx

from matplotlib import pyplot as plt
from pandas import DataFrame


class Graph:

    def __init__(self, df_entities: DataFrame) -> None:

        self.df_entities = df_entities
        self.node_ids = None
        self.adjacency_matrix = None
        self.id_to_info = None
        self.index_to_info = None
        self.scaled_adj_matrix = None
        self.reset()

    def reset(self):
        self.node_ids = self.df_entities['id'].unique()
        self.adjacency_matrix = np.zeros((len(self.node_ids), len(self.node_ids)))
        base_index_entities = set(self.df_entities.groupby(['id'])['mention'].agg(pd.Series.mode).index.values)
        self.id_to_info = {e['id']: e for e in
                           self.df_entities[self.df_entities['id'].isin(base_index_entities)].to_dict(orient='records')}
        self.index_to_info = {i: self.id_to_info[id] for i, id in enumerate(self.node_ids)}

    def populate_adj_matrix(self, **kargs) -> None:
        pass

    def compute_scaled_adj_matrix(self):
        self.scaled_adj_matrix = 1 - (self.adjacency_matrix - self.adjacency_matrix.min()) / (
                self.adjacency_matrix.max() - self.adjacency_matrix.min())

    def statistics(self, ranges_x=(0, 50), ranges_y=(0, 50)):
        print('Max co-occurrences: ', self.adjacency_matrix.max())
        print('Mean co-occurrences: ', self.adjacency_matrix.mean())
        print('Std of co-occurrences: ', self.adjacency_matrix.std())

        plt.imshow(self.adjacency_matrix[ranges_x[0]: ranges_x[1], ranges_y[0]: ranges_y[1]], cmap='gray')
        plt.colorbar()
        plt.show()

    def draw_example(self, range_nodes=(0, 10)):
        graph = nx.from_numpy_array(self.adjacency_matrix[range_nodes[0]:range_nodes[1], range_nodes[0]:range_nodes[1]])
        graph.edges(data=True)
        labels = dict(
            list([(id, info['mention']) for id, info in self.index_to_info.items()])[range_nodes[0]:range_nodes[1]])
        nx.draw(graph, labels=labels)

    def find_nearest(self, src_index, predicate: Callable, max_: int = 1):
        frontier = {_to: (v, []) for _to, v in enumerate(self.scaled_adj_matrix[src_index]) if v > 0}
        already_visited, result = {src_index}, []
        while len(frontier.keys()) > 0:
            _to, (value, queue) = sorted(frontier.items(), key=lambda item: item[1][0])[0]
            del frontier[_to]
            if predicate(self.index_to_info[_to]):
                result.append((_to, queue, value))
                if max_ is not None and len(result) == max_:
                    return result
            already_visited.add(_to)
            _from = _to
            for _to, v in enumerate(self.scaled_adj_matrix[_from]):
                if v > 0 and _to not in already_visited:
                    cost = value + v + 1
                    new_value = (cost, [*queue, _from])
                    if _to in frontier:
                        if frontier[_to][0] < cost:
                            frontier[_to] = new_value
                    else:
                        frontier[_to] = new_value

        return result
