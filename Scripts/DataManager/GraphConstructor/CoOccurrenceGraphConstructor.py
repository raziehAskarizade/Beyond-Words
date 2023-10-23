
import pandas as pd
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np


class CoOccurrenceGraphConstructor(GraphConstructor):

    def __init__(self, config: Config):
        super(CoOccurrenceGraphConstructor, self).__init__(config)
        self.nlp = spacy.load(self.config.spacy.pipeline)

    def to_graph(self, text: str):
        doc = self.nlp(text)
        unique_words, unique_map = self.__get_unique_words(doc)
        unique_word_vectors = self.__get_unique_words_vector(unique_words)
        co_occurrence_matrix = self.__get_co_occurrence_matrix(doc, unique_words, unique_map)
        return self.__create_graph(unique_word_vectors, co_occurrence_matrix)

    @staticmethod
    def __get_unique_words(doc):
        unique_words = []
        for token in doc:
            unique_words.append(token.lemma_)
        unique_words = pd.Series(list(set(unique_words)))
        unique_map = pd.Series(range(len(unique_words)), index=unique_words)
        return unique_words, unique_map

    @staticmethod
    def __get_co_occurrence_matrix(doc, unique_words, unique_map):
        token_lemmas = [t.lemma_ for t in doc]
        n_gram = 4
        dense_mat = torch.zeros((len(unique_words), len(unique_words)), dtype=torch.float32)
        for i in range(len(token_lemmas) - n_gram):
            n_gram_data = list(set(token_lemmas[i:i + n_gram]))
            n_gram_ids = unique_map[n_gram_data]
            grid_ids = [(x, y) for x in n_gram_ids for y in n_gram_ids if x != y]
            grid_ids = torch.tensor(grid_ids, dtype=torch.int)
            dense_mat[grid_ids[:, 0], grid_ids[:, 1]] += 1
        dense_mat = torch.nn.functional.normalize(dense_mat)
        sparse_mat = dense_mat.to_sparse_coo()
        return sparse_mat

    def __get_unique_words_vector(self, unique_words):
        unique_word_ids = [self.nlp.vocab.strings[unique_words[i]] for i in range(len(unique_words))]
        unique_word_vectors = torch.zeros((len(unique_words), self.nlp.vocab.vectors_length), dtype=torch.float32)
        for i in range(len(unique_words)):
            word_id = unique_word_ids[i]
            if word_id in self.nlp.vocab.vectors:
                unique_word_vectors[i] = torch.tensor(self.nlp.vocab.vectors[word_id])
            else:
                # Write functionality to resolve word vector ((for now we use random vector)) 1000
                # use pretrain model to generate vector (heavy)
                # Over-fit a smaller model over spacy dictionary
                unique_word_vectors[i] = torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32)
        return unique_word_vectors

    @staticmethod
    def __create_graph(unique_word_vectors, co_occurrence_matrix):  # edge_label
        node_attr = unique_word_vectors
        edge_index = co_occurrence_matrix.indices()
        edge_attr = co_occurrence_matrix.values()
        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
