
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
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend='' , use_node_dependencies: bool=False, use_compression=True):

        super(DependencyGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, lazy_construction, load_preprocessed_data,
                      naming_prepend , use_compression)
        self.settings = {"tokens_dep_weight" : 1,"dep_tokens_weight" : 1, "token_token_weight" : 2}
        self.use_node_dependencies = use_node_dependencies
        if self.load_preprocessed_data:
            if not self.lazy_construction:
                self.load_all_data()
            else:
                self.load_var()
        else:
            self.var.nlp_pipeline = self.config.spacy.pipeline
            self.var.graph_num = len(self.raw_data)
            self.nlp = spacy.load(self.var.nlp_pipeline)
            self.dependencies = self.nlp.get_pipe("parser").labels
            if not self.lazy_construction:
                for i in range(len(self.raw_data)):
                    if i not in self._graphs:
                        if i % 100 == 0:
                            print(f'i: {i}')
                        self._graphs[i] = self.to_graph(self.raw_data[i])
                        self.var.graphs_name[i] = f'{self.naming_prepend}_{i}'
                        # self.save_all_data()


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
            node_attr = torch.full((len(doc),),-1, dtype=torch.float32)
        edge_index = []
        edge_attr = []
        for token in doc:
            if token.dep_ != 'ROOT':
                token_id = self.nlp.vocab.strings[token.lemma_]
                if token_id in self.nlp.vocab.vectors:
                    if for_compression:
                        node_attr[token.i] = torch.tensor(token_id , dtype=torch.float32)
                    else:
                        node_attr[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
                edge_index.append([token.head.i, token.i])
                dep_id = self.nlp.vocab.strings[token.dep_]
                if dep_id in self.nlp.vocab.vectors:
                    vectorized_dep = self.nlp.vocab.vectors[dep_id]
                else:
                    vectorized_dep = torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32)
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
        edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.long) , 0 , 1)
        # self.edge_attr = edge_attr # vectorized edge attributes
        return Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr)
    
    def __find_dep_index(self , dependency : str):
        for dep_idx in range(len(self.dependencies)):
            if self.dependencies[dep_idx] == dependency:
                return dep_idx
        return -1 # means not found
    def __build_initial_dependency_vectors(self , dep_length : int):
        return torch.zeros((dep_length, self.nlp.vocab.vectors_length), dtype=torch.float32)        
    def __create_graph_with_node_dependencies(self , doc , for_compression=False):
        # nodes size is dependencies + tokens
        data = HeteroData()
        dep_length = len(self.dependencies)
        if for_compression:
            data['dep'].x = torch.full((dep_length,),-1, dtype=torch.float32)
            data['word'].x = torch.full((len(doc),),-1, dtype=torch.float32)
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
                    data['word'].x[token.i] = torch.tensor(token_id , dtype=torch.float32)
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
        data['dep' , 'dep_word' , 'word'].edge_index = torch.transpose(torch.tensor(dep_word_edge_index, dtype=torch.long) , 0 , 1)
        data['word' , 'word_dep' , 'dep'].edge_index = torch.transpose(torch.tensor(word_dep_edge_index, dtype=torch.long) , 0 , 1)
        data['word' , 'seq' , 'word'].edge_index = torch.transpose(torch.tensor(word_word_edge_index, dtype=torch.long) , 0 , 1)
        data['dep' , 'dep_word' , 'word'].edge_attr = dep_word_edge_attr
        data['word' , 'word_dep' , 'dep'].edge_attr = word_dep_edge_attr
        data['word' , 'seq' , 'word'].edge_attr = word_word_edge_attr
        return data
    def draw_graph(self , idx : int):
        node_tokens = []
        doc = self.nlp(self.raw_data[idx])
        for d in self.dependencies:
            node_tokens.append(d)
        for t in doc:
            node_tokens.append(t)
        graph_data = self.get_graph(idx)
        g = to_networkx(graph_data)
        layout = nx.spring_layout(g)
        nx.draw(g, pos=layout)
        words_dict = {i: node_tokens[i] for i in range(len(node_tokens))}
        # edge_labels_dict = {(graph_data.edge_index[0][i].item() , graph_data.edge_index[1][i].item()) : { "dep" : graph_data.edge_attr[i]} for i in range(len(graph_data.edge_attr))}
        # nx.set_edge_attributes(g , edge_labels_dict)
        nx.draw_networkx_labels(g, pos=layout, labels=words_dict)
        # nx.draw_networkx_edge_labels(g, pos=layout)
    def to_graph_indexed(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        if self.use_node_dependencies:
            return self.__create_graph_with_node_dependencies(doc , for_compression=True)
        else:
            return self.__create_graph(doc, for_compression=True)
        pass
    def convert_indexed_nodes_to_vector_nodes(self, graph):
        if self.use_node_dependencies:
            words = torch.zeros((len(graph['word'].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph['word'].x)):
                if graph['word'].x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
                else:
                    words[i] = torch.zeros((self.nlp.vocab.vectors_length) , dtype=torch.float32)
            graph['word'].x = words
            graph['dep'].x = self.__build_initial_dependency_vectors(len(self.dependencies))
        else:
            words = torch.zeros((len(graph.x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph.x)):
                if graph.x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
                else:
                    words[i] = torch.zeros((self.nlp.vocab.vectors_length) , dtype=torch.float32)
            graph.x = words
        return graph
        

