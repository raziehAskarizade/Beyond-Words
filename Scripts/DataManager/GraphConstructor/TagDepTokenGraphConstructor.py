
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



class TagDepTokenGraphConstructor(GraphConstructor):

    class _Variables(GraphConstructor._Variables):
        def __init__(self):
            super(TagDepTokenGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''
    def __init__(self, texts: List[str], save_path: str, config: Config,
                 lazy_construction=True, load_preprocessed_data=False, naming_prepend='' , use_compression=True, use_sentence_nodes=False, use_general_node=True):

        super(TagDepTokenGraphConstructor, self)\
            .__init__(texts, self._Variables(), save_path, config, lazy_construction, load_preprocessed_data,
                      naming_prepend , use_compression)
        self.settings = {"dep_token_weight" : 1, "token_token_weight" : 2, "tag_token_weight" : 1, "general_token_weight" : 1, "general_sentence_weight" : 1, "token_sentence_weight" : 1}
        self.use_sentence_nodes = use_sentence_nodes
        self.use_general_node = use_general_node
        self.var.nlp_pipeline = self.config.spacy.pipeline
        self.var.graph_num = len(self.raw_data)
        self.nlp = spacy.load(self.var.nlp_pipeline)
        self.dependencies = self.nlp.get_pipe("parser").labels
        self.tags = self.nlp.get_pipe("tagger").labels
            
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
        if self.use_sentence_nodes:
            self.__create_graph_with_sentences(doc)
        else:
            self.__create_graph(doc,use_general_node=self.use_general_node)
    
    def __find_dep_index(self , dependency : str):
        for dep_idx in range(len(self.dependencies)):
            if self.dependencies[dep_idx] == dependency:
                return dep_idx
        return -1 # means not found
    def __build_initial_dependency_vectors(self , dep_length : int):
        return torch.zeros((dep_length, self.nlp.vocab.vectors_length), dtype=torch.float32)
    def __find_tag_index(self , tag : str):
        for tag_idx in range(len(self.tags)):
            if self.tags[tag_idx] == tag:
                return tag_idx
        return -1 # means not found
    def __build_initial_tag_vectors(self , tags_length : int):
        return torch.zeros((tags_length, self.nlp.vocab.vectors_length), dtype=torch.float32)
    def __build_initial_general_vector(self):
        return torch.zeros((1 , self.nlp.vocab.vectors_length), dtype=torch.float32)    
    def __create_graph_with_sentences(self , doc , for_compression=False):
        data = self.__create_graph(doc,for_compression,False)
        sentence_embeddings = [sent.vector for sent in doc.sents]
        data['sentence'].x = torch.tensor(sentence_embeddings, dtype=torch.float32)
        sentence_general_edge_index = []
        general_sentence_edge_index = []
        sentence_word_edge_index = []
        word_sentence_edge_index = []
        sentence_general_edge_attr = []
        general_sentence_edge_attr = []
        sentence_word_edge_attr = []
        word_sentence_edge_attr = []
        if self.use_general_node:
            for i , _x in enumerate(doc.sents):
                # connecting sentences to general node
                sentence_general_edge_index.append([i , 0])
                general_sentence_edge_index.append([0 , i])
                sentence_general_edge_attr.append(self.settings['general_sentence_weight'])
                general_sentence_edge_attr.append(self.settings['general_sentence_weight']) # different weight for directed edges can be set in the future
        for token in doc:
            # connecting words to sentences
            word_sentence_edge_index.append([token.i , token.sent.start])
            sentence_word_edge_index.append([token.sent.start , token.i])
            word_sentence_edge_attr.append(self.settings['token_sentence_weight'])
            word_sentence_edge_attr.append(self.settings['token_sentence_weight'])
        if self.use_general_node:
            data['general' , 'general_sentence' , 'sentence'].edge_index = torch.transpose(torch.tensor(general_sentence_edge_index, dtype=torch.long) , 0 , 1)
            data['sentence' , 'sentence_general' , 'general'].edge_index = torch.transpose(torch.tensor(sentence_general_edge_index, dtype=torch.long) , 0 , 1)
            data['general' , 'general_sentence' , 'sentence'].edge_attr = general_sentence_edge_attr
            data['sentence' , 'sentence_general' , 'general'].edge_attr = sentence_general_edge_attr
        data['word' , 'word_sentence' , 'sentence'].edge_index = torch.transpose(torch.tensor(sentence_word_edge_index, dtype=torch.long) , 0 , 1)
        data['sentence' , 'sentence_word' , 'word'].edge_index = torch.transpose(torch.tensor(word_sentence_edge_index, dtype=torch.long) , 0 , 1)
        data['word' , 'word_sentence' , 'sentence'].edge_attr = word_sentence_edge_attr
        data['sentence' , 'sentence_word' , 'word'].edge_attr = sentence_word_edge_attr
        return data
    def __create_graph(self , doc , for_compression=False, use_general_node=True):
        # nodes size is dependencies + tokens
        data = HeteroData()
        dep_length = len(self.dependencies)
        tag_length = len(self.tags)
        if for_compression:
            data['dep'].x = torch.full((dep_length,),-1, dtype=torch.float32)
            data['word'].x = torch.full((len(doc),),-1, dtype=torch.float32)
            data['tag'].x = torch.full((tag_length,), -1, dtype=torch.float32)
            if use_general_node:
                data['general'].x = torch.full((1,),-1, dtype=torch.float32)
        else:
            data['dep'].x = self.__build_initial_dependency_vectors(dep_length)
            data['word'].x = torch.zeros((len(doc) , self.nlp.vocab.vectors_length), dtype=torch.float32)
            data['tag'].x = self.__build_initial_tag_vectors(tag_length)
            if use_general_node:
                data['general'].x = self.__build_initial_general_vector()
        word_dep_edge_index = []
        dep_word_edge_index = []
        word_tag_edge_index = []
        tag_word_edge_index = []
        word_word_edge_index = []
        word_general_edge_index = []
        general_word_edge_index = []
        word_dep_edge_attr = []
        dep_word_edge_attr = []
        word_tag_edge_attr = []
        tag_word_edge_attr = []
        word_word_edge_attr = []
        word_general_edge_attr = []
        general_word_edge_attr = []
        for token in doc:
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                if for_compression:
                    data['word'].x[token.i] = torch.tensor(token_id , dtype=torch.float32)
                else:
                    data['word'].x[token.i] = torch.tensor(self.nlp.vocab.vectors[token_id])
            # adding dependency edges
            if token.dep_ != 'ROOT':
                dep_idx = self.__find_dep_index(token.dep_)
                if dep_idx != -1:
                    word_dep_edge_index.append([token.head.i , dep_idx])
                    word_dep_edge_attr.append(self.settings["dep_token_weight"])
                    dep_word_edge_index.append([dep_idx , token.i])
                    dep_word_edge_attr.append(self.settings["dep_token_weight"])
            # adding tag edges
            tag_idx = self.__find_tag_index(token.tag_)
            if tag_idx != -1:
                word_tag_edge_index.append([token.i , tag_idx])
                word_tag_edge_attr.append(self.settings["tag_token_weight"])
                tag_word_edge_index.append([tag_idx , token.i])
                tag_word_edge_attr.append(self.settings["tag_token_weight"])
            # adding sequence edges
            if token.i != len(doc) - 1:
                # using zero vectors for edge features
                word_word_edge_index.append([token.i , token.i + 1])
                word_word_edge_attr.append(self.settings["token_token_weight"])
                word_word_edge_index.append([token.i + 1 , token.i])
                word_word_edge_attr.append(self.settings["token_token_weight"])
            # adding general node edges
            if use_general_node:
                word_general_edge_index.append([token.i , 0])
                word_general_edge_attr.append(self.settings["general_token_weight"])
                general_word_edge_index.append([0 , token.i])
                general_word_edge_attr.append(self.settings["general_token_weight"])
        data['dep' , 'dep_word' , 'word'].edge_index = torch.transpose(torch.tensor(dep_word_edge_index, dtype=torch.long) , 0 , 1)
        data['word' , 'word_dep' , 'dep'].edge_index = torch.transpose(torch.tensor(word_dep_edge_index, dtype=torch.long) , 0 , 1)
        data['tag', 'tag_word', 'word'].edge_index = torch.transpose(torch.tensor(tag_word_edge_index, dtype=torch.long) , 0 , 1)
        data['word', 'word_tag', 'tag'].edge_index = torch.transpose(torch.tensor(word_tag_edge_index, dtype=torch.long) , 0 , 1)
        data['word' , 'seq' , 'word'].edge_index = torch.transpose(torch.tensor(word_word_edge_index, dtype=torch.long) , 0 , 1)
        data['dep' , 'dep_word' , 'word'].edge_attr = dep_word_edge_attr
        data['word' , 'word_dep' , 'dep'].edge_attr = word_dep_edge_attr
        data['tag', 'tag_word', 'word'].edge_attr = tag_word_edge_attr
        data['word', 'word_tag', 'tag'].edge_attr = word_tag_edge_attr
        data['word' , 'seq' , 'word'].edge_attr = word_word_edge_attr
        if use_general_node:
            data['general' , 'general_word' , 'word'].edge_index = torch.transpose(torch.tensor(general_word_edge_index, dtype=torch.long) , 0 , 1)
            data['word' , 'word_general' , 'general'].edge_index = torch.transpose(torch.tensor(word_general_edge_index, dtype=torch.long) , 0 , 1)
            data['general' , 'general_word' , 'word'].edge_attr = general_word_edge_attr
            data['word' , 'word_general' , 'general'].edge_attr = word_general_edge_attr
        return data
    def draw_graph(self , idx : int):
        # TODO : do this part if needed
        pass
    def to_graph_indexed(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        if self.use_sentence_nodes:
            return self.__create_graph_with_sentences(doc , for_compression=True)
        else:
            return self.__create_graph(doc,for_compression=True,use_general_node=self.use_general_node)
    def convert_indexed_nodes_to_vector_nodes(self, graph):
        words = torch.zeros((len(graph['word'].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
        for i in range(len(graph['word'].x)):
            if graph['word'].x[i] in self.nlp.vocab.vectors:
                words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
            else:
                words[i] = torch.zeros((self.nlp.vocab.vectors_length) , dtype=torch.float32)
        graph['word'].x = words
        graph['dep'].x = self.__build_initial_dependency_vectors(len(self.dependencies))
        graph['tag'].x = self.__build_initial_tag_vectors(len(self.tags))
        if self.use_general_node:
            graph['general'].x = self.__build_initial_general_vector()
        # sentences are not coded - we dont need to creat them
        return graph
        

