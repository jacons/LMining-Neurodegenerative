import numpy as np

from Graph_class import Graph


class GraphRanker:

    def __init__(self, graph: Graph):
        self.graph = graph

    def __pagerank(self, iterations=200, a=0.85):
        m = self.graph.adjacency_matrix
        n = m.shape[0]
        v = np.ones(n) / n
        m_hat = a * m + (1 - a) / n
        for _ in range(iterations):
            v = m_hat @ v
            v /= v.sum()  # Values to percentage wrt the sum of all values. Avoiding number explosion
        return v

    def rank_nodes(self, max_=None, iterations=15, a=0.85):
        rank = self.__pagerank(iterations=iterations, a=a)
        arg_rank = rank.argsort()[::-1]
        return [(rank[arg], arg) for arg in arg_rank[:max_]]

    def rank_edges(self, max_=None):
        tam = np.tril(self.graph.adjacency_matrix)
        edges_rank = np.dstack(np.unravel_index(tam.ravel().argsort(), tam.shape))[0][::-1]
        tam_sum = tam.sum()
        return [(tam[n1, n2] / tam_sum, n1, n2) for n1, n2 in edges_rank[:max_]]

    def print_nodes_rank(self, *args, **kargs):
        print('Score\tName (ID)'.upper())
        nodes_rank = self.rank_nodes(*args, **kargs)
        for score, arg in nodes_rank:
            print(
                f'{round(score, 2)}%\t{self.graph.index_to_info[arg]["mention"]} '
                f'({self.graph.index_to_info[arg]["id"]})')

    def print_edges_rank(self, *args, **kargs):
        print('Score\tRelation'.upper())
        edges_rank = self.rank_edges(*args, **kargs)
        for score, n1, n2 in edges_rank:
            print(f'{round(score * 100, 2)}%',
                  f'{self.graph.index_to_info[n1]["mention"]} <=>'
                  f' {self.graph.index_to_info[n2]["mention"]}', sep='\t')
