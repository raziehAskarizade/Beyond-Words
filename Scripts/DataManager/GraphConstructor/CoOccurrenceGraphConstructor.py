import pickle
from typing import List, Dict

import pandas as pd
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os


class CoOccurrenceGraphConstructor(GraphConstructor):

    class _Variables:
        nlp_pipeline: str = ''
        text_num: int = 0
        graphs_name: List[str] = []

        def save_to_file(self, filename: str):
            with open(filename, 'wb') as file:
                pickle.dump(self, file)

        @classmethod
        def load_from_file(cls, filename: str):
            with open(filename, 'rb') as file:
                obj = pickle.load(file)
            if isinstance(obj, cls):
                return obj
            else:
                raise ValueError("Invalid file content. Unable to recreate the object.")

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend=''):
        super(CoOccurrenceGraphConstructor, self).__init__(config)
        self.texts = texts
        self.save_path = os.path.join(self.config.data_root_dir, save_path)
        self.var = self._Variables()
        self.naming_prepend = naming_prepend
        self.graphs:  List[Data] = []
        if load_preprocessed_data:
            self.load_data()
        else:
            self.var.nlp_pipeline = self.config.spacy.pipeline
            self.var.text_num = len(texts)
            self.var.graphs_name = []
            self.nlp = spacy.load(self.var.nlp_pipeline)

            if not lazy_construction:
                for i in range(len(texts)):
                    if i not in self.graphs:
                        if i % 100 == 0:
                            print(f'i: {i}')
                        self.graphs.append(self.to_graph(self.texts[i]))
                        self.var.graphs_name.append(f'{self.naming_prepend}_{i}')
            self.save_data()

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

    @staticmethod
    def __get_co_occurrence_matrix(doc, unique_words, unique_map):
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

    @staticmethod
    def __create_graph(unique_word_vectors, co_occurrence_matrix):  # edge_label
        node_attr = unique_word_vectors
        edge_index = co_occurrence_matrix.indices()
        edge_attr = co_occurrence_matrix.values()
        return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

    def save_data(self):
        for i in range(len(self.graphs)):
            torch.save(self.graphs[i], os.path.join(self.save_path, f'{self.var.graphs_name[i]}.pt'))
        self.var.save_to_file(os.path.join(self.save_path, f'{self.naming_prepend}_var.txt'))

    def load_data(self):
        self.var = self.var.load_from_file(os.path.join(self.save_path, f'{self.naming_prepend}_var.txt'))
        for i in range(self.var.text_num):
            self.graphs.append(torch.load(os.path.join(self.save_path, f'{self.var.graphs_name[i]}.pt')))
