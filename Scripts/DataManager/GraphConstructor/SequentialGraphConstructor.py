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
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend='', use_general_node=False , use_compression=True, num_data_load=-1 , num_general_nodes=1):

        super(SequentialGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, lazy_construction, load_preprocessed_data,
                      naming_prepend , use_compression, num_data_load)
        self.settings = {"tokens_general_weight" : 1, "token_token_weight" : 2 , "general_tokens_weight" : 2}
        self.use_general_node = use_general_node
        self.var.nlp_pipeline = self.config.spacy.pipeline
        self.var.graph_num = len(self.raw_data)
        self.nlp = spacy.load(self.var.nlp_pipeline)
        self.num_general_nodes = num_general_nodes
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
            node_attr = [-1 for i in range(len(doc))]
        edge_index = []
        edge_attr = []
        for token in doc:
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                if for_compression:
                    node_attr[token.i] = token_id
                else:
                    node_attr[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
            if token.i != len(doc) - 1:
                # using zero vectors for edge features
                edge_index.append([token.i , token.i + 1])
                edge_index.append([token.i + 1 , token.i])
                edge_attr.append(self.settings["token_token_weight"]) 
                edge_attr.append(self.settings["token_token_weight"]) 
        edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.int32) , 0 , 1)
        edge_attr = torch.nn.functional.normalize(torch.tensor(edge_attr, dtype=torch.float32), dim=0)
        return Data(x=node_attr, edge_index=edge_index,edge_attr=edge_attr)
    def _build_initial_general_vector(self , num : int = 1):
        return torch.zeros((num , self.nlp.vocab.vectors_length), dtype=torch.float32)
    def _create_graph_with_general_node(self , doc , for_compression=False):
        data = HeteroData()
        if for_compression:
            data['general'].x = torch.full((1,),0, dtype=torch.float32)
            data['word'].x = [-1 for i in range(len(doc))]
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
                    data['word'].x[token.i] = token_id
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
        data['general' , 'general_word' , 'word'].edge_index = torch.transpose(torch.tensor(general_word_edge_index, dtype=torch.int32) , 0 , 1)
        data['word' , 'word_general' , 'general'].edge_index = torch.transpose(torch.tensor(word_general_edge_index, dtype=torch.int32) , 0 , 1)
        data['word' , 'seq' , 'word'].edge_index = torch.transpose(torch.tensor(word_word_edge_index, dtype=torch.int32) , 0 , 1)
        data['general' , 'general_word' , 'word'].edge_attr = torch.nn.functional.normalize(torch.tensor(general_word_edge_attr, dtype=torch.float32), dim=0)
        data['word' , 'word_general' , 'general'].edge_attr = torch.nn.functional.normalize(torch.tensor(word_general_edge_attr, dtype=torch.float32), dim=0)
        data['word' , 'seq' , 'word'].edge_attr = torch.nn.functional.normalize(torch.tensor(word_word_edge_attr, dtype=torch.float32), dim=0)
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
    def prepare_loaded_data(self, graph):
        if self.use_general_node:
            words = torch.zeros((len(graph['word'].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph['word'].x)):
                if graph['word'].x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
            graph['word'].x = words
            graph['general'].x = self.__build_initial_general_vector(num=self.num_general_nodes)
            graph = self.__add_multiple_general_nodes(graph,False , self.num_general_nodes)
        else:
            words = torch.zeros((len(graph.x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            for i in range(len(graph.x)):
                if graph.x[i] in self.nlp.vocab.vectors:
                    words[i] = torch.tensor(self.nlp.vocab.vectors[graph.x[i]])
            graph.x = words
        return graph
    
    def _add_multiple_general_nodes(self,graph , use_sentence_nodes, num_general_nodes):
        if not use_sentence_nodes:
            graph['general'].x = self.__build_initial_general_vector(num=self.num_general_nodes)
            if self.num_general_nodes > 1:
                # connecting other general nodes
                word_general_edge_index = torch.transpose(torch.tensor(graph['general' , 'general_word' , 'word'].edge_index, dtype=torch.int32) , 0 , 1).tolist()
                general_word_edge_index = torch.transpose(torch.tensor(graph['word' , 'word_general' , 'general'].edge_index, dtype=torch.int32) , 0 , 1).tolist()
                general_word_edge_attr = graph['general' , 'general_word' , 'word'].edge_attr.tolist()
                word_general_edge_attr = graph['word' , 'word_general' , 'general'].edge_attr.tolist()
                for i in range(len(graph['word'].x)):
                    for j in range(1,num_general_nodes):
                        word_general_edge_index.append([i , j])
                        general_word_edge_index.append([j , i])
                        word_general_edge_attr.append(self.settings["general_token_weight"])
                        general_word_edge_attr.append(self.settings["general_token_weight"])
                graph['general' , 'general_word' , 'word'].edge_index = torch.transpose(torch.tensor(general_word_edge_index, dtype=torch.int32) , 0 , 1)
                graph['word' , 'word_general' , 'general'].edge_index = torch.transpose(torch.tensor(word_general_edge_index, dtype=torch.int32) , 0 , 1)
                graph['general' , 'general_word' , 'word'].edge_attr = torch.nn.functional.normalize(torch.tensor(general_word_edge_attr, dtype=torch.float32), dim=0)
                graph['word' , 'word_general' , 'general'].edge_attr = torch.nn.functional.normalize(torch.tensor(word_general_edge_attr, dtype=torch.float32), dim=0)
        else:
            graph['general'].x = self.__build_initial_general_vector(num=self.num_general_nodes)
            if self.num_general_nodes > 1:
                # connecting other general nodes
                sentence_general_edge_index = torch.transpose(torch.tensor(graph['general' , 'general_sentence' , 'sentence'].edge_index, dtype=torch.int32) , 0 , 1).tolist()
                general_sentence_edge_index = torch.transpose(torch.tensor(graph['sentence' , 'sentence_general' , 'general'].edge_index, dtype=torch.int32) , 0 , 1).tolist()
                general_sentence_edge_attr = graph['general' , 'general_sentence' , 'sentence'].edge_attr.tolist()
                sentence_general_edge_attr = graph['sentence' , 'word_general' , 'general'].edge_attr.tolist()
                for i in range(len(graph['sentence'].x)):
                    for j in range(1,num_general_nodes):
                        word_general_edge_index.append([i , j])
                        general_word_edge_index.append([j , i])
                        word_general_edge_attr.append(self.settings["general_sentence_weight"])
                        general_word_edge_attr.append(self.settings["general_sentence_weight"])
                graph['general' , 'general_sentence' , 'sentence'].edge_index = torch.transpose(torch.tensor(general_sentence_edge_index, dtype=torch.int32) , 0 , 1)
                graph['sentence' , 'sentence_general' , 'general'].edge_index = torch.transpose(torch.tensor(sentence_general_edge_index, dtype=torch.int32) , 0 , 1)
                graph['general' , 'general_sentence' , 'sentence'].edge_attr = torch.nn.functional.normalize(torch.tensor(general_sentence_edge_attr, dtype=torch.float32), dim=0)
                graph['sentence' , 'sentence_general' , 'general'].edge_attr = torch.nn.functional.normalize(torch.tensor(sentence_general_edge_attr, dtype=torch.float32), dim=0)
        return graph
        

