import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from Graph_class import Graph


class TrainingCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.tqdm = None
        self.previous_total_loss = 0

    def on_epoch_end(self, model):
        if self.tqdm is None:
            self.tqdm = tqdm(total=model.epochs, desc='w2v training')
        total_loss = model.get_latest_training_loss()
        loss = total_loss - self.previous_total_loss
        self.previous_total_loss = total_loss
        tqdm.write(f'Loss after epoch {self.epoch + 1}/{model.epochs}: {loss}')
        self.epoch += 1
        self.tqdm.update(1)
        if self.epoch == model.epochs:
            self.tqdm.close()
            self.tqdm = None


class Word2VecGraph(Graph):

    def __init__(self, df_entities, texts):
        super(Word2VecGraph, self).__init__(df_entities)
        self.texts = texts

    def populate_adj_matrix(self, **kargs):
        w2v_model = Word2Vec(
            sentences=self.texts,
            min_count=kargs['min_count'],
            vector_size=kargs['vector_size'],
            window=kargs['window'],
            sg=kargs['sg'],
            epochs=kargs['epochs'],
            alpha=kargs['learning_rate'],
            compute_loss=True,
            callbacks=[TrainingCallback()]
        )
        w2v = w2v_model.wv

        for i_node in tqdm(range(len(self.adjacency_matrix)), desc="Word2VecGraph populate_adj_matrix"):
            word1 = id_to_wuid[self.index_to_info[i_node]['id']].lower()
            for j_node in range(i_node):
                word2 = id_to_wuid[self.index_to_info[j_node]['id']].lower()
                try:
                    self.adjacency_matrix[i_node, j_node] = w2v.similarity(word1, word2)
                except KeyError:
                    pass
        self.adjacency_matrix = self.adjacency_matrix + self.adjacency_matrix.T
        np.fill_diagonal(self.adjacency_matrix, 0)
        self.compute_scaled_adj_matrix()
