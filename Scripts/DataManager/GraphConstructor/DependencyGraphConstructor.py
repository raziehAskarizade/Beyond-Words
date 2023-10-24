
import pandas as pd
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np


class DependencyGraphConstructor(GraphConstructor):

    def __init__(self, text: str, config: Config, use_node_dependencies: bool=False):
        super(DependencyGraphConstructor, self).__init__(text, config)
        self.nlp = spacy.load(self.config.spacy.pipeline)
        self.doc = self.nlp(self.text)
        self.unique_words, self.unique_map = self.__get_unique_words()
        self.unique_word_vectors = self.__get_unique_words_vector()
        if use_node_dependencies:
            self.graph = self.__create_graph_with_node_dependencies()
        else:
            self.graph = self.__create_graph()

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
        edge_index = []
        edge_attr = []
        for token in self.doc:
            # node_attr.append([token.text, token.pos_])
            if token.dep_ != 'ROOT':
                edge_index.append([self.unique_map[token.head.lemma_], self.unique_map[token.lemma_]])
                dep_id = self.nlp.vocab.strings[token.dep_]
                if dep_id in self.nlp.vocab.vectors:
                    vectorized_dep = self.nlp.vocab.vectors[dep_id]
                else:
                    vectorized_dep = torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32)
                edge_attr.append(vectorized_dep)
        self.node_attr = self.unique_word_vectors
        self.edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.long) , 0 , 1)
        self.edge_attr = edge_attr # vectorized edge attributes
        return Data(x=self.node_attr, edge_index=self.edge_index,edge_attr=self.edge_attr)
    def __create_graph_with_node_dependencies(self):
        # Not implemented
        pass
        

