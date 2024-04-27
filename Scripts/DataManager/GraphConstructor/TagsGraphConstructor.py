# Omid Davar @ 2023

import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config

import torch
import numpy as np
import os

import stanza
import fasttext


class TagsGraphConstructor(GraphConstructor):
    
    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(TagsGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 load_preprocessed_data=False, naming_prepend='' , use_compression=True, start_data_load=0, end_data_load=-1):

        super(TagsGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, load_preprocessed_data,
                      naming_prepend , use_compression,start_data_load, end_data_load)
        self.settings = {"tokens_tag_weight" : 1, "token_token_weight" : 2}
        self.var.nlp_pipeline = self.config.fa.pipeline
        self.var.graph_num = len(self.raw_data)

        # farsi
        self.nlp = fasttext.load_model(self.var.nlp_pipeline)
        self.token_lemma = stanza.Pipeline("fa")

        self.tags = self.nlp.get_pipe("tagger").labels
        

    def to_graph(self, text: str):
        # farsi
        doc = []
        token_list = self.token_lemma(text)
        for sentence in token_list.sentences:
            for token in sentence.words:
                doc.append((token.text,token.lemma))


        if len(doc) < 2:
            return
        return self.__create_graph(doc)

    def __find_tag_index(self , tag : str):
        for tag_idx in range(len(self.tags)):
            if self.tags[tag_idx] == tag:
                return tag_idx
        return -1 # means not found
    
    def __build_initial_tag_vectors(self , tags_length : int):
        # return torch.zeros((tags_length, self.nlp.vocab.vectors_length), dtype=torch.float32)  
        # return torch.nn.functional.one_hot(torch.arange(0, tags_length), num_classes=-1)
        return torch.arange(0, tags_length)
    def __create_graph(self , doc , for_compression=False):
        data = HeteroData()
        tags_length = len(self.tags)
        data['tag'].length = tags_length
        if for_compression:
            data['word'].x = [-1 for i in range(len(doc))]
            data['tag'].x = torch.full((tags_length,), -1, dtype=torch.float32)
        else:
            data['word'].x = torch.zeros((len(doc) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            data['tag'].x = self.__build_initial_tag_vectors(tags_length)
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
                    data['word'].x[token.i] = token_id
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
        data['tag', 'tag_word', 'word'].edge_index = torch.transpose(torch.tensor(tag_word_edge_index, dtype=torch.int32) , 0 , 1) if len(tag_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'word_tag', 'tag'].edge_index = torch.transpose(torch.tensor(word_tag_edge_index, dtype=torch.int32) , 0 , 1) if len(word_tag_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'seq', 'word'].edge_index = torch.transpose(torch.tensor(word_word_edge_index, dtype=torch.int32) , 0 , 1) if len(word_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['tag', 'tag_word', 'word'].edge_attr = torch.tensor(tag_word_edge_attr, dtype=torch.float32)
        data['word', 'word_tag', 'tag'].edge_attr = torch.tensor(word_tag_edge_attr, dtype=torch.float32)
        data['word', 'seq', 'word'].edge_attr = torch.tensor(word_word_edge_attr, dtype=torch.float32)
        return data
    def draw_graph(self , idx : int):
        # define it if needed later
        pass
    def to_graph_indexed(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        return self.__create_graph(doc , for_compression=True)

    def prepare_loaded_data(self, graph):
        words = torch.zeros((len(graph['word'].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
        for i in range(len(graph['word'].x)):
            if graph['word'].x[i] in self.nlp.vocab.vectors:
                words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
        graph['word'].x = words
        graph['tag'].x = self.__build_initial_tag_vectors(len(self.tags))
        for t in graph.edge_types:
            if len(graph[t].edge_index) == 0:
                graph[i].edge_index = torch.empty(2, 0, dtype=torch.int32)
        return graph
        

