# Fardin Rastakhiz, Omid Davar @ 2023


import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config

import torch
import numpy as np
import os


class CoOccurrenceGraphConstructor(GraphConstructor):

    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(CoOccurrenceGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 load_preprocessed_data=False, naming_prepend='', use_compression=True, start_data_load=0, end_data_load=-1):
        super(CoOccurrenceGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, load_preprocessed_data, naming_prepend, use_compression, start_data_load, end_data_load)
        self.var.graph_num = len(self.raw_data)

        # farsi
        self.nlp = config.nlp
        self.token_lemma = config.token_lemma

        self.word_ids = self.get_word_by_id()

    def to_graph(self, text: str):
        # farsi
        doc = []
        token_list = self.token_lemma(text)
        for sentence in token_list.sentences:
            for token in sentence.words:
                doc.append((token.text, token.lemma))

        unique_words, unique_map = self.__get_unique_words(doc=doc)
        if len(unique_words) < 2:
            return
        unique_word_vectors = self.__get_unique_words_vector(unique_words)
        co_occurrence_matrix = self.__get_co_occurrence_matrix(
            doc, unique_words, unique_map)
        return self.__create_graph(unique_word_vectors, co_occurrence_matrix)

    @staticmethod
    def __get_unique_words(doc):
        unique_words = []
        unique_map = {}
        for token in doc:
            unique_words.append(token[0].lower())
        unique_words = set(unique_words)
        if len(unique_words) < 2:
            return unique_words, unique_map
        unique_words = pd.Series(list(unique_words))
        unique_map = pd.Series(range(len(unique_words)), index=unique_words)
        return unique_words, unique_map

    def __get_co_occurrence_matrix(self, doc, unique_words, unique_map):
        tokens = [t[0].lower() for t in doc]
        n_gram = 4
        g_length = doc.__len__() - n_gram
        dense_mat = torch.zeros(
            (len(unique_words), len(unique_words)), dtype=torch.float32)
        for i in range(g_length):
            n_gram_data = list(set(tokens[i:i + n_gram]))
            if len(n_gram_data) < 2:
                continue
            n_gram_ids = unique_map[n_gram_data]
            grid_ids = [(x, y)
                        for x in n_gram_ids for y in n_gram_ids if x != y]
            grid_ids = torch.tensor(grid_ids, dtype=torch.int)
            dense_mat[grid_ids[:, 0], grid_ids[:, 1]] += 1
        dense_mat = torch.nn.functional.normalize(dense_mat)
        sparse_mat = dense_mat.to_sparse_coo()
        return sparse_mat

    def __get_unique_words_vector(self, unique_words):
        unique_word_ids = []
        for i in range(len(unique_words)):
            unique_word_ids.append(self.nlp.get_word_id(unique_words[i]))
        unique_word_vectors = torch.zeros(
            (len(unique_words), self.nlp.get_dimension()), dtype=torch.float32)
        for i in range(len(unique_words)):
            word_id = unique_word_ids[i]
            if int(word_id) != -1:
                unique_word_vectors[i] = torch.tensor(
                    self.nlp.get_word_vector(unique_words[i]))
            else:
                # Write functionality to resolve word vector ((for now we use random vector)) 1000
                # use pretrain model to generate vector (heavy)
                # Over-fit a smaller model over stanza dictionary
                unique_word_vectors[i] = torch.zeros(
                    (self.nlp.get_dimension(),), dtype=torch.float32)
        return unique_word_vectors

    def __get_unique_words_ids(self, unique_words):
        unique_word_ids = []
        for i in range(len(unique_words)):
            unique_word_ids.append(self.nlp.get_word_id(unique_words[i]))
        unique_word_indices = [None for i in range(len(unique_words))]
        for i in range(len(unique_words)):
            word_id = unique_word_ids[i]
            if int(word_id) != -1:
                unique_word_indices[i] = word_id
            else:
                # Write functionality to resolve word vector ((for now we use random vector)) 1000
                # use pretrain model to generate vector (heavy)
                # Over-fit a smaller model over stanza dictionary
                unique_word_indices[i] = torch.zeros(
                    (self.nlp.get_dimension(),), dtype=torch.float32)
        return unique_word_indices

    def __create_graph(self, unique_word_vectors, co_occurrence_matrix):  # edge_label
        node_attr = unique_word_vectors
        edge_index = co_occurrence_matrix.indices()
        edge_attr = co_occurrence_matrix.values()
        graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
        return graph

    def to_graph_indexed(self, text: str):
        # farsi
        doc = []
        token_list = self.token_lemma(text)
        for sentence in token_list.sentences:
            for token in sentence.words:
                doc.append((token.text, token.lemma))

        unique_words, unique_map = self.__get_unique_words(doc=doc)
        if len(unique_words) < 2:
            return
        unique_word_ids = self.__get_unique_words_ids(unique_words)
        co_occurrence_matrix = self.__get_co_occurrence_matrix(
            doc, unique_words, unique_map)
        return self.__create_graph(unique_word_ids, co_occurrence_matrix)

    def get_word_by_id(self):
        words_id = {}
        for word in self.nlp.get_words():
            words_id[self.nlp.get_word_id(word)] = word
        return words_id

    def prepare_loaded_data(self, graph):
        nodes = torch.zeros(
            (len(graph.x), self.nlp.get_dimension()), dtype=torch.float32)

        for i in range(len(graph.x)):
            if self.word_ids.get(int(graph.x[i])) is not None:
                nodes[i] = torch.tensor(
                    self.nlp.get_word_vector(self.word_ids[int(graph.x[i])]))
        return Data(x=nodes, edge_index=graph.edge_index, edge_attr=graph.edge_attr)

    def draw_graph(self, idx: int):
        g = to_networkx(self.get_graph(idx), to_undirected=True)
        layout = nx.spring_layout(g)
        nx.draw(g, pos=layout)
        unique_words_dict = {i: self.unique_words[i]
                             for i in range(len(self.unique_words))}
        nx.draw_networkx_labels(self.get_graph(
            idx), pos=layout, labels=unique_words_dict)
