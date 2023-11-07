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


class SequentialGraphConstructor(GraphConstructor):
    
    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(SequentialGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''
    def __init__(self, texts: List[str], save_path: str, config: Config,
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend='', use_general_node=False , use_compression=True):

        super(SequentialGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, lazy_construction, load_preprocessed_data,
                      naming_prepend , use_compression)
        self.settings = {"tokens_general_weight" : 1, "token_token_weight" : 2 , "general_tokens_weight" : 2}
        self.use_general_node = use_general_node
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
            return self._create_graph_with_general_node(doc)
        else:
            return self._create_graph(doc)
    def _create_graph(self , doc , for_compression=False):
        docs_length = len(doc)
        node_attr = torch.zeros((len(doc), self.nlp.vocab.vectors_length), dtype=torch.float32)
        if for_compression:
            node_attr = torch.full((len(doc),),-1, dtype=torch.float32)
        edge_index = []
        edge_attr = []
        for token in doc:
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                if for_compression:
                    node_attr[token.i] = torch.tensor(token_id , dtype=torch.float32)
                else:
                    node_attr[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
            if token.i != len(doc) - 1:
                # using zero vectors for edge features
                edge_index.append([token.i , token.i + 1])
                edge_index.append([token.i + 1 , token.i])
                edge_attr.append(self.settings["token_token_weight"]) 
                edge_attr.append(self.settings["token_token_weight"]) 
        edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.long) , 0 , 1)
        return Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr)
    def _build_initial_general_vector(self):
        return torch.zeros((1 , self.nlp.vocab.vectors_length), dtype=torch.float32)
    def _create_graph_with_general_node(self , doc , for_compression=False):
        data = HeteroData()
        if for_compression:
            data['general'].x = torch.full((1,),-1, dtype=torch.float32)
            data['word'].x = torch.full((len(doc),),-1, dtype=torch.float32)
        else:
            data['general'].x = self._build_initial_general_vector()
            data['word'].x = torch.zeros((len(doc) , self.nlp.vocab.vectors_length), dtype=torch.float32)
        word_general_edge_index = []
        general_word_edge_index = []
        word_word_edge_index = []
        word_general_edge_attr = []
        general_word_edge_attr = []
        word_word_edge_attr = []
        for token in doc:
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                if for_compression:
                    data['word'].x[token.i] = torch.tensor(token_id , dtype=torch.float32)
                else:
                    data['word'].x[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
            word_general_edge_index.append([token.i , 0])
            word_general_edge_attr.append(self.settings["tokens_general_weight"])
            general_word_edge_index.append([0 , token.i])
            general_word_edge_attr.append(self.settings["general_tokens_weight"])
            # adding sequential edges between tokens - uncomment the codes for vectorized edges
            if token.i != len(doc) - 1:
                word_word_edge_index.append([token.i , token.i + 1])
                word_word_edge_attr.append(self.settings["token_token_weight"])
                word_word_edge_index.append([token.i + 1 , token.i])
                word_word_edge_attr.append(self.settings["token_token_weight"])
        data['general' , 'general_word' , 'word'].edge_index = torch.transpose(torch.tensor(general_word_edge_index, dtype=torch.long) , 0 , 1)
        data['word' , 'word_general' , 'general'].edge_index = torch.transpose(torch.tensor(word_general_edge_index, dtype=torch.long) , 0 , 1)
        data['word' , 'seq' , 'word'].edge_index = torch.transpose(torch.tensor(word_word_edge_index, dtype=torch.long) , 0 , 1)
        data['general' , 'general_word' , 'word'].edge_attr = general_word_edge_attr
        data['word' , 'word_general' , 'general'].edge_attr = word_general_edge_attr
        data['word' , 'seq' , 'word'].edge_attr = word_word_edge_attr
        return data
    def draw_graph(self , idx : int):
        node_tokens = []
        if self.use_general_node:
            node_tokens.append("gen_node")
        doc = self.nlp(self.raw_data[idx])
        for t in doc:
            node_tokens.append(t.lemma_)
        graph_data = self.get_graph(idx)
        g = to_networkx(graph_data)
        layout = nx.spring_layout(g)
        nx.draw(g, pos=layout)
        words_dict = {i: node_tokens[i] for i in range(len(node_tokens))}
        nx.draw_networkx_labels(g, pos=layout, labels=words_dict)
    def to_graph_indexed(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        if self.use_general_node:
            return self._create_graph_with_general_node(doc, for_compression=True)
        else:
            return self._create_graph(doc, for_compression=True)
    def convert_indexed_nodes_to_vector_nodes(self, graph):
        if self.use_general_node:
            words = torch.zeros((len(graph['word'].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph['word'].x)):
                if graph['word'].x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
                else:
                    words[i] = torch.zeros((self.nlp.vocab.vectors_length) , dtype=torch.float32)
            graph['word'].x = words
            graph['general'].x = self._build_initial_general_vector()
        else:
            words = torch.zeros((len(graph.x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph.x)):
                if graph.x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
                else:
                    words[i] = torch.zeros((self.nlp.vocab.vectors_length) , dtype=torch.float32)
            graph.x = words
        return graph
        

