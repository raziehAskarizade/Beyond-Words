
import pandas as pd
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np


class CoOccurrenceGraphConstructor(GraphConstructor):

    def __init__(self, text: str, config: Config):
        super(CoOccurrenceGraphConstructor, self).__init__(text, config)
        self.nlp = spacy.load(self.config.spacy.pipeline)
        self.doc = self.nlp(self.text)
        self.unique_words, self.unique_map = self.__get_unique_words()
        self.co_occurrence_matrix = self.__get_co_occurrence_matrix()
        self.unique_word_vectors = self.__get_unique_words_vector()
        self.__create_graph()

    def __get_unique_words(self):
        unique_words = []
        for token in self.doc:
            unique_words.append(token.lemma_)
        unique_words = pd.Series(list(set(unique_words)))
        unique_map = pd.Series(range(len(unique_words)), index=unique_words)
        return unique_words, unique_map

    def __get_co_occurrence_matrix(self):
        token_lemmas = [t.lemma_ for t in self.doc]
        n_gram = 4
        dense_mat = torch.zeros((len(self.unique_words), len(self.unique_words)), dtype=torch.float32)
        for i in range(len(token_lemmas) - n_gram):
            n_gram_data = list(set(token_lemmas[i:i + n_gram]))
            n_gram_ids = self.unique_map[n_gram_data]
            grid_ids = [(x, y) for x in n_gram_ids for y in n_gram_ids if x != y]
            grid_ids = torch.tensor(grid_ids, dtype=torch.int)
            dense_mat[grid_ids[:, 0], grid_ids[:, 1]] += 1
        dense_mat = torch.nn.functional.normalize(dense_mat)
        sparse_mat = dense_mat.to_sparse_coo()
        return sparse_mat

    def __get_unique_words_vector(self):
        unique_word_ids = [self.nlp.vocab.strings[self.unique_words[i]] for i in range(len(self.unique_words))]
        unique_word_vectors = torch.zeros((len(self.unique_words), self.nlp.vocab.vectors_length), dtype=torch.float32)
        for i in range(len(self.unique_words)):
            word_id = unique_word_ids[i]
            if word_id in self.nlp.vocab.vectors:
                unique_word_vectors[i] = torch.tensor(self.nlp.vocab.vectors[word_id])
            else:
                # Write functionality to resolve word vector ((for now we use random vector)) 1000
                # use pretrain model to generate vector (heavy)
                # Over-fit a smaller model over spacy dictionary
                unique_word_vectors[i] = torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32)
        return unique_word_vectors

    def __create_graph(self):  # edge_label
        self.node_attr = self.unique_word_vectors
        self.edge_index = self.co_occurrence_matrix.indices()
        self.edge_attr = self.co_occurrence_matrix.values()
        self.graph = Data(x=self.node_attr, edge_index=self.edge_index,edge_attr=self.edge_attr)
