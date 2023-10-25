
import pandas as pd
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np


class DependencyGraphConstructor(GraphConstructor):

    def __init__(self, text: str, config: Config, use_node_dependencies: bool=False):
        super(DependencyGraphConstructor, self).__init__(config)
        self.nlp = spacy.load(self.config.spacy.pipeline)
        self.dependencies = self.nlp.get_pipe("parser").labels
        self.doc = self.nlp(text)
        self.settings = {"tokens_dep_weight" : 1,"dep_tokens_weight" : 1, "token_token_weight" : 2}
        self.unique_words, self.unique_map = self.__get_unique_words()
        self.unique_word_vectors = self.__get_unique_words_vector()
        if use_node_dependencies:
            self.graph = self.__create_graph_with_node_dependencies()
        else:
            self.graph = self.__create_graph()
    def to_graph(self, text: str):
        doc = self.nlp(text)
        unique_words, unique_map = self.__get_unique_words(doc)
        if len(unique_words) < 2:
            return
        unique_word_vectors = self.__get_unique_words_vector(unique_words)
        return self.__create_graph()
    def __get_unique_words(self):
        unique_words = []
        for token in self.doc:
            unique_words.append(token.lemma_)
        unique_words = pd.Series(list(set(unique_words)))
        unique_map = pd.Series(range(len(unique_words)), index=unique_words)
        return unique_words, unique_map    
    def __get_unique_words_vector(self):
        unique_word_ids = [self.nlp.vocab.strings[self.unique_words[i]] for i in range(len(self.unique_words))]
        unique_word_vectors = torch.zeros((len(self.unique_words), self.nlp.vocab.vectors_length), dtype=torch.float32)
        for i in range(len(self.unique_words)):
            word_id = unique_word_ids[i]
            if word_id in self.nlp.vocab.vectors:
                unique_word_vectors[i] = torch.tensor(self.nlp.vocab.vectors[word_id])
            else:
                # Write functionality to resolve word vector ((for now we use random vector)) 1000
                # use pretrain model to generate vector (heavy)
                # Over-fit a smaller model over spacy dictionary
                unique_word_vectors[i] = torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32)
        return unique_word_vectors

    def __create_graph(self):
        node_attr = torch.zeros((len(self.doc), self.nlp.vocab.vectors_length), dtype=torch.float32)
        node_tokens = []
        edge_index = []
        edge_attr = []
        for token in self.doc:
            node_tokens.append(token.lemma_)
            if token.dep_ != 'ROOT':
                token_id = self.nlp.vocab.strings[token.lemma_]
                if token_id in self.nlp.vocab.vectors:
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
            if token.i != len(self.doc):
                # using zero vectors for edge features
                edge_index.append([token.i , token.i + 1])
                # self.edge_attr.append(torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32))
                edge_attr.append(self.settings["token_token_weight"])
                edge_index.append([token.i + 1 , token.i])
                # self.edge_attr.append(torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32))
                edge_attr.append(self.settings["token_token_weight"])
        self.node_tokens = node_tokens
        self.node_attr = node_attr
        self.edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.long) , 0 , 1)
        self.edge_attr = edge_attr # vectorized edge attributes
        return Data(x=self.node_attr, edge_index=self.edge_index,edge_attr=self.edge_attr)
    
    def __find_dep_index(self , dependency : str):
        for dep_idx in range(len(self.dependencies)):
            if self.dependencies[dep_idx] == dependency:
                return dep_idx
        return -1 # means not found
                
    def __create_graph_with_node_dependencies(self):
        # nodes size is dependencies + tokens
        dep_length = len(self.dependencies)
        node_attr = torch.zeros((len(self.doc) + dep_length, self.nlp.vocab.vectors_length), dtype=torch.float32)
        node_tokens = []
        edge_index = []
        edge_attr = []
        for idx in range(dep_length):
            # if vevtorizing of dependencies is needed, do it here
            # node_attr[idx] = sth ...
            node_tokens.append(self.dependencies[idx])
        for token in self.doc:
            node_tokens.append(token.lemma_)
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                node_attr[token.i + dep_length - 1] = torch.tensor(self.nlp.vocab.vectors[token_id])
            if token.dep_ != 'ROOT':
                dep_idx = self.__find_dep_index(token.dep_)
                # not found protection
                if dep_idx != -1:
                    # edge from head token to dependency node
                    edge_index.append([token.head.i + dep_length - 1, dep_idx])
                    edge_attr.append(self.settings["tokens_dep_weight"])
                    # edge from dependency node to the token
                    edge_index.append([dep_idx, token.i + dep_length - 1])
                    edge_attr.append(self.settings["dep_tokens_weight"])
            # adding sequential edges between tokens - uncomment the codes for vectorized edges
            if token.i != len(self.doc):
                # using zero vectors for edge features
                edge_index.append([token.i + dep_length - 1 , token.i + dep_length])
                edge_attr.append(self.settings["token_token_weight"])
                edge_index.append([token.i + dep_length , token.i + dep_length - 1])
                edge_attr.append(self.settings["token_token_weight"])
        self.node_tokens = node_tokens
        self.node_attr = node_attr
        self.edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.long) , 0 , 1)
        self.edge_attr = edge_attr 
        return Data(x=self.node_attr, edge_index=self.edge_index,edge_attr=self.edge_attr)
        

