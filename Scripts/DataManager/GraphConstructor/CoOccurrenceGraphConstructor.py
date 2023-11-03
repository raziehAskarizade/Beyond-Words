import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os


class CoOccurrenceGraphConstructor(GraphConstructor):

    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(CoOccurrenceGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend='', use_compression=True, num_data_load=-1, device='cpu'):
        super(CoOccurrenceGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, lazy_construction, load_preprocessed_data, naming_prepend, use_compression, num_data_load, device)
        self.var.nlp_pipeline = self.config.spacy.pipeline
        self.var.graph_num = len(self.raw_data)
        self.nlp = spacy.load(self.var.nlp_pipeline)
        
    def setup(self, load_preprocessed_data = True):
        self.load_preprocessed_data = True
        if load_preprocessed_data:
            self.load_var()
            self.num_data_load = self.var.graph_num if self.num_data_load > self.var.graph_num else self.num_data_load
            if not self.lazy_construction:
                for i in range(self.num_data_load):
                    if i%100 == 0:
                        print(f' {i} graph loaded')
                    if (self._graphs[i] is None) and (i in self.var.graphs_name):
                        self.load_data_compressed(i)
                    else:
                        self._graphs[i] = self.to_graph(self.raw_data[i])
                        self.var.graphs_name[i] = f'{self.naming_prepend}_{i}'
                        self.save_data_compressed(i)
                self.var.save_to_file(os.path.join(self.save_path, f'{self.naming_prepend}_var.txt'))
        else:
            if not self.lazy_construction:
                save_start = 0
                self.num_data_load = len(self.raw_data) if self.num_data_load > len(self.raw_data) else self.num_data_load
                for i in range(self.num_data_load):
                    if i not in self._graphs:
                        if i % 100 == 0:
                            self.save_data_range(save_start, i)
                            save_start = i
                            print(f'i: {i}')
                        self._graphs[i] = self.to_graph(self.raw_data[i])
                        self.var.graphs_name[i] = f'{self.naming_prepend}_{i}'
                self.save_data_range(save_start, self.num_data_load)
            self.var.save_to_file(os.path.join(self.save_path, f'{self.naming_prepend}_var.txt'))

    def to_graph(self, text: str):
        doc = self.nlp(text)
        unique_words, unique_map = self.__get_unique_words(doc)
        if len(unique_words) < 2:
            return
        unique_word_vectors = self.__get_unique_words_vector(unique_words)
        co_occurrence_matrix = self.__get_co_occurrence_matrix(doc, unique_words, unique_map)
        return self.__create_graph(unique_word_vectors, co_occurrence_matrix)

    @staticmethod
    def __get_unique_words(doc):
        unique_words = []
        unique_map = {}
        for token in doc:
            unique_words.append(token.lower_)
        unique_words = set(unique_words)
        if len(unique_words) < 2:
            return unique_words, unique_map
        unique_words = pd.Series(list(unique_words))
        unique_map = pd.Series(range(len(unique_words)), index=unique_words)
        return unique_words, unique_map

    def __get_co_occurrence_matrix(self, doc, unique_words, unique_map):
        tokens = [t.lower_ for t in doc]
        n_gram = 4
        g_length = doc.__len__() - n_gram
        dense_mat = torch.zeros((len(unique_words), len(unique_words)), dtype=torch.float32)
        for i in range(g_length):
            n_gram_data = list(set(tokens[i:i + n_gram]))
            if len(n_gram_data) < 2:
                continue
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

    def __get_unique_words_ids(self, unique_words):
        unique_word_ids = [self.nlp.vocab.strings[unique_words[i]] for i in range(len(unique_words))]
        unique_word_indices = [None for i in range(len(unique_words))]
        for i in range(len(unique_words)):
            word_id = unique_word_ids[i]
            if word_id in self.nlp.vocab.vectors:
                unique_word_indices[i] = word_id
            else:
                # Write functionality to resolve word vector ((for now we use random vector)) 1000
                # use pretrain model to generate vector (heavy)
                # Over-fit a smaller model over spacy dictionary
                unique_word_indices[i] = torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32)
        return unique_word_indices

    def __create_graph(self, unique_word_vectors, co_occurrence_matrix):  # edge_label
        node_attr = unique_word_vectors
        edge_index = co_occurrence_matrix.indices()
        edge_attr = co_occurrence_matrix.values()
        graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
        return graph.to(self.device)

    def to_graph_indexed(self, text: str):
        doc = self.nlp(text)
        unique_words, unique_map = self.__get_unique_words(doc)
        if len(unique_words) < 2:
            return
        unique_word_ids = self.__get_unique_words_ids(unique_words)
        co_occurrence_matrix = self.__get_co_occurrence_matrix(doc, unique_words, unique_map)
        return self.__create_graph(unique_word_ids, co_occurrence_matrix)
    
    def convert_indexed_nodes_to_vector_nodes(self, graph):
        nodes = torch.zeros((len(graph.x), self.nlp.vocab.vectors_length), dtype=torch.float32)
        for i in range(len(graph.x)):
            if graph.x[i] in self.nlp.vocab.vectors:
                nodes[i] = torch.tensor(self.nlp.vocab.vectors[graph.x[i]])
        return Data(x=nodes , edge_index=graph.edge_index , edge_attr=graph.edge_attr)
    
    def draw_graph(self, idx: int):
        g = to_networkx(self.get_graph(idx), to_undirected=True)
        layout = nx.spring_layout(g)
        nx.draw(g, pos=layout)
        unique_words_dict = {i: self.unique_words[i] for i in range(len(self.unique_words))}
        nx.draw_networkx_labels(self.get_graph(idx), pos=layout, labels=unique_words_dict)