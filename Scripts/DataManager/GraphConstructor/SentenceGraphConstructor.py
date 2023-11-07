import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructor.SequentialGraphConstructor import SequentialGraphConstructor
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os


class SentenceGraphConstructor(SequentialGraphConstructor):
    
    class _Variables(SequentialGraphConstructor._Variables):
        def __init__(self):
            super(SentenceGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''
    def __init__(self, texts: List[str], save_path: str, config: Config,
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend='', use_general_node=False , use_compression=True):

        super(SentenceGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, lazy_construction, load_preprocessed_data,
                      naming_prepend , use_compression=use_compression, use_general_node=False)
        self.settings = {"token_sentence_weight" : 1, "token_token_weight" : 2 , "general_sentence_weight" : 2}
        self.sentence_use_general_node = use_general_node
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
        if len(doc) < 2:
            return
        if self.use_general_node:
            return self.__create_graph_with_general_node(doc)
        else:
            return self.__create_graph(doc)
    def to_graph_indexed(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        if self.use_general_node:
            return self.__create_graph_with_general_node(doc, for_compression=True)
        else:
            return self.__create_graph(doc, for_compression=True)
    def convert_indexed_nodes_to_vector_nodes(self, graph):
        if self.use_general_node:
            words = torch.zeros((len(graph['word'].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph['word'].x)):
                if graph['word'].x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
                else:
                    words[i] = torch.zeros((self.nlp.vocab.vectors_length) , dtype=torch.float32)
            graph['word'].x = words
            graph['general'].x = self.__build_initial_general_vector()
        else:
            words = torch.zeros((len(graph.x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph.x)):
                if graph.x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
                else:
                    words[i] = torch.zeros((self.nlp.vocab.vectors_length) , dtype=torch.float32)
            graph.x = words
        return graph
        

