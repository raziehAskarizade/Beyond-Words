
import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os



class DependencyGraphConstructor(GraphConstructor):

    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(DependencyGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''
    def __init__(self, texts: List[str], save_path: str, config: Config,
                load_preprocessed_data=False, naming_prepend='', use_node_dependencies: bool = False, use_compression=True, start_data_load=0, end_data_load=-1):

        super(DependencyGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, load_preprocessed_data,
                      naming_prepend , use_compression, start_data_load, end_data_load)
        self.settings = {"tokens_dep_weight" : 1,"dep_tokens_weight" : 1, "token_token_weight" : 2}
        self.use_node_dependencies = use_node_dependencies
        self.var.nlp_pipeline = self.config.spacy.pipeline
        self.var.graph_num = len(self.raw_data)
        self.nlp = spacy.load(self.var.nlp_pipeline)
        self.dependencies = self.nlp.get_pipe("parser").labels
        
    def to_graph(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        if self.use_node_dependencies:
            return self.__create_graph_with_node_dependencies(doc)
        else:
            return self.__create_graph(doc)

    def __create_graph(self , doc , for_compression=False):
        node_attr = torch.zeros((len(doc), self.nlp.vocab.vectors_length), dtype=torch.float32)
        if for_compression:
            node_attr = [-1 for i in range(len(doc))]
        edge_index = []
        edge_attr = []
        for token in doc:
            if token.dep_ != 'ROOT':
                token_id = self.nlp.vocab.strings[token.lemma_]
                if token_id in self.nlp.vocab.vectors:
                    if for_compression:
                        node_attr[token.i] = token_id
                    else:
                        node_attr[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
                edge_index.append([token.head.i, token.i])
                # edge_attr.append(vectorized_dep)
                edge_attr.append(self.settings["tokens_dep_weight"])
            # adding sequential edges between tokens - uncomment the codes for vectorized edges
            if token.i != len(doc):
                # using zero vectors for edge features
                edge_index.append([token.i , token.i + 1])
                # self.edge_attr.append(torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32))
                edge_attr.append(self.settings["token_token_weight"])
                edge_index.append([token.i + 1 , token.i])
                # self.edge_attr.append(torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32))
                edge_attr.append(self.settings["token_token_weight"])
        # self.node_tokens = node_tokens
        # self.node_attr = node_attr
        edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.int32) , 0 , 1)
        # self.edge_attr = edge_attr # vectorized edge attributes
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        return Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr)
    
    def __find_dep_index(self , dependency : str):
        for dep_idx in range(len(self.dependencies)):
            if self.dependencies[dep_idx] == dependency:
                return dep_idx
        return -1 # means not found
    def __build_initial_dependency_vectors(self , dep_length : int):
        # return torch.zeros((dep_length, self.nlp.vocab.vectors_length), dtype=torch.float32)
        # return torch.nn.functional.one_hot(torch.arange(0 , dep_length), num_classes=-1)
        return torch.arange(0 , dep_length)
    def __create_graph_with_node_dependencies(self , doc , for_compression=False):
        # nodes size is dependencies + tokens
        data = HeteroData()
        dep_length = len(self.dependencies)
        data['dep'].length = dep_length
        if for_compression:
            data['dep'].x = torch.full((dep_length,),-1, dtype=torch.float32)
            data['word'].x = [-1 for i in range(len(doc))]
        else:
            data['dep'].x = self.__build_initial_dependency_vectors(dep_length)
            data['word'].x = torch.zeros((len(doc) , self.nlp.vocab.vectors_length), dtype=torch.float32)
        word_dep_edge_index = []
        dep_word_edge_index = []
        word_word_edge_index = []
        word_dep_edge_attr = []
        dep_word_edge_attr = []
        word_word_edge_attr = []
        for token in doc:
            # node_tokens.append(token.lemma_)
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                if for_compression:
                    data['word'].x[token.i] = token_id
                else:
                    data['word'].x[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
            if token.dep_ != 'ROOT':
                dep_idx = self.__find_dep_index(token.dep_)
                # not found protection
                if dep_idx != -1:
                    # edge from head token to dependency node
                    word_dep_edge_index.append([token.head.i , dep_idx])
                    word_dep_edge_attr.append(self.settings["tokens_dep_weight"])
                    # edge from dependency node to the token
                    dep_word_edge_index.append([dep_idx , token.i])
                    dep_word_edge_attr.append(self.settings["dep_tokens_weight"])
            # adding sequential edges between tokens - uncomment the codes for vectorized edges
            if token.i != len(doc) - 1:
                # using zero vectors for edge features
                word_word_edge_index.append([token.i , token.i + 1])
                word_word_edge_attr.append(self.settings["token_token_weight"])
                word_word_edge_index.append([token.i + 1 , token.i])
                word_word_edge_attr.append(self.settings["token_token_weight"])
        data['dep' , 'dep_word' , 'word'].edge_index = torch.transpose(torch.tensor(dep_word_edge_index, dtype=torch.int32) , 0 , 1) if len(dep_word_edge_index) > 0 else torch.tensor([[],[]] , dtype=torch.int32)
        data['word' , 'word_dep' , 'dep'].edge_index = torch.transpose(torch.tensor(word_dep_edge_index, dtype=torch.int32) , 0 , 1) if len(word_dep_edge_index) > 0 else torch.tensor([[],[]] , dtype=torch.int32)
        data['word' , 'seq' , 'word'].edge_index = torch.transpose(torch.tensor(word_word_edge_index, dtype=torch.int32) , 0 , 1) if len(word_word_edge_index) > 0 else torch.tensor([[],[]] , dtype=torch.int32)
        data['dep' , 'dep_word' , 'word'].edge_attr = torch.tensor(dep_word_edge_attr, dtype=torch.float32) 
        data['word' , 'word_dep' , 'dep'].edge_attr = torch.tensor(word_dep_edge_attr, dtype=torch.float32)
        data['word' , 'seq' , 'word'].edge_attr = torch.tensor(word_word_edge_attr, dtype=torch.float32)
        return data
    def draw_graph(self , idx : int):
        # do it later if needed
        pass
    def to_graph_indexed(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        if self.use_node_dependencies:
            return self.__create_graph_with_node_dependencies(doc , for_compression=True)
        else:
            return self.__create_graph(doc, for_compression=True)

    def prepare_loaded_data(self, graph):
        if self.use_node_dependencies:
            words = torch.zeros((len(graph['word'].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph['word'].x)):
                if graph['word'].x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
            graph['word'].x = words
            graph['dep'].x = self.__build_initial_dependency_vectors(len(self.dependencies))
        else:
            words = torch.zeros((len(graph.x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph.x)):
                if graph.x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph.x[i]])
            graph.x = words
        for t in graph.edge_types:
            if len(graph[t].edge_index) == 0:
                graph[i].edge_index = torch.tensor([[],[]] , dtype=torch.int32)
        return graph
        

