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


class TagsGraphConstructor(GraphConstructor):
    
    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(TagsGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend='' , use_compression=True):

        super(TagsGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, lazy_construction, load_preprocessed_data,
                      naming_prepend , use_compression)
        self.settings = {"tokens_tag_weight" : 1, "token_token_weight" : 2}
        if self.load_preprocessed_data:
            if not self.lazy_construction:
                self.load_all_data()
            else:
                self.load_var()
        else:
            self.var.nlp_pipeline = self.config.spacy.pipeline
            self.var.graph_num = len(self.raw_data)
            self.nlp = spacy.load(self.var.nlp_pipeline)
            self.tags = self.nlp.get_pipe("tagger").labels
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
        return self.__create_graph(doc)

    def __find_tag_index(self , tag : str):
        for tag_idx in range(len(self.tags)):
            if self.tags[tag_idx] == tag:
                return tag_idx
        return -1 # means not found
    
    def __build_initial_tag_vectors(self , tags_length : int):
        return torch.zeros((tags_length, self.nlp.vocab.vectors_length), dtype=torch.float32)  
    def __create_graph(self , doc , for_compression=False):
        data = HeteroData()
        tags_length = len(self.tags)
        if for_compression:
            data['tag'].x = torch.full((tags_length,),-1, dtype=torch.float32)
            data['word'].x = torch.full((len(doc),),-1, dtype=torch.float32)
        else:
            data['tag'].x = self.__build_initial_tag_vectors(tags_length)
            data['word'].x = torch.zeros((len(doc) , self.nlp.vocab.vectors_length), dtype=torch.float32)
        word_tag_edge_index = []
        tag_word_edge_index = []
        word_word_edge_index = []
        word_tag_edge_attr = []
        tag_word_edge_attr = []
        word_word_edge_attr = []
        
        # for idx in range(tags_length):
            # if vevtorizing of dependencies is needed, do it here
            # node_attr[idx] = sth ...
        for token in doc:
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                if for_compression:
                    data['word'].x[token.i] = torch.tensor(token_id , dtype=torch.float32)
                else:
                    data['word'].x[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
            tag_idx = self.__find_tag_index(token.tag_)
            if tag_idx != -1:
                word_tag_edge_index.append([token.i , tag_idx])
                word_tag_edge_attr.append(self.settings["tokens_tag_weight"])
                tag_word_edge_index.append([tag_idx , token.i])
                tag_word_edge_attr.append(self.settings["tokens_tag_weight"])
            # adding sequential edges between tokens - uncomment the codes for vectorized edges
            if token.i != len(doc) - 1:
                # using zero vectors for edge features
                word_word_edge_index.append([token.i , token.i + 1])
                word_word_edge_attr.append(self.settings["token_token_weight"])
                word_word_edge_index.append([token.i + 1, token.i])
                word_word_edge_attr.append(self.settings["token_token_weight"])
        data['tag', 'tag_word', 'word'].edge_index = torch.transpose(torch.tensor(tag_word_edge_index, dtype=torch.long) , 0 , 1)
        data['word', 'word_tag', 'tag'].edge_index = torch.transpose(torch.tensor(word_tag_edge_index, dtype=torch.long) , 0 , 1)
        data['word', 'seq', 'word'].edge_index = torch.transpose(torch.tensor(word_word_edge_index, dtype=torch.long) , 0 , 1)
        data['tag', 'tag_word', 'word'].edge_attr = tag_word_edge_attr
        data['word', 'word_tag', 'tag'].edge_attr = word_tag_edge_attr
        data['word', 'seq', 'word'].edge_attr = word_word_edge_attr
        return data
    def draw_graph(self , idx : int):
        node_tokens = []
        doc = self.nlp(self.raw_data[idx])
        for d in self.tags:
            node_tokens.append(d)
        for t in doc:
            node_tokens.append(t.lemma_)
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
        return self.__create_graph(doc , for_compression=True)
        pass
    def convert_indexed_nodes_to_vector_nodes(self, graph):
        words = torch.zeros((len(graph['word'].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
        for i in range(len(graph['word'].x)):
            if graph['word'].x[i] in self.nlp.vocab.vectors:
                words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
            else:
                words[i] = torch.zeros((self.nlp.vocab.vectors_length) , dtype=torch.float32)
        graph['word'].x = words
        graph['tag'].x = self.__build_initial_tag_vectors(len(self.tags))
        return graph
        

