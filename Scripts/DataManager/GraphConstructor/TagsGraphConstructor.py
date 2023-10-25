
import pandas as pd
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor
from torch_geometric.data import Data
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np


class TagsGraphConstructor(GraphConstructor):

    def __init__(self, text: str, config: Config):
        super(TagsGraphConstructor, self).__init__(config)
        self.nlp = spacy.load(self.config.spacy.pipeline)
        self.tags = self.nlp.get_pipe("tagger").labels
        self.doc = self.nlp(text)
        self.settings = {"tokens_tag_weight" : 1, "token_token_weight" : 2}
        self.graph = self.__create_graph()
    def to_graph(self, text: str):
        self.doc = self.nlp(text)
        return self.__create_graph()
    def __find_tag_index(self , tag : str):
        for tag_idx in range(len(self.tags)):
            if self.tags[tag_idx] == tag:
                return tag_idx
        return -1 # means not found
    def __create_graph(self):
        tags_length = len(self.tags)
        node_attr = torch.zeros((len(self.doc) + tags_length, self.nlp.vocab.vectors_length), dtype=torch.float32)
        node_tokens = []
        edge_index = []
        edge_attr = []
        
        for idx in range(tags_length):
            # if vevtorizing of dependencies is needed, do it here
            # node_attr[idx] = sth ...
            node_tokens.append(self.tags[idx])
        for token in self.doc:
            node_tokens.append(token.lemma_)
            token_id = self.nlp.vocab.strings[token.lemma_]
            if token_id in self.nlp.vocab.vectors:
                node_attr[token.i + tags_length - 1] = torch.tensor(self.nlp.vocab.vectors[token_id])
            tag_idx = self.__find_tag_index(token.tag_)
            if tag_idx != -1:
                edge_index.append([token.i + tags_length - 1 , tag_idx])
                edge_attr.append(self.settings["tokens_tag_weight"])
            # adding sequential edges between tokens - uncomment the codes for vectorized edges
            if token.i != len(self.doc):
                # using zero vectors for edge features
                edge_index.append([token.i + tags_length - 1 , token.i + 1])
                # self.edge_attr.append(torch.zeros((self.nlp.vocab.vectors_length,), dtype=torch.float32))
                edge_attr.append(self.settings["token_token_weight"])
        self.node_tokens = node_tokens
        self.node_attr = node_attr
        self.edge_index = torch.transpose(torch.tensor(edge_index, dtype=torch.long) , 0 , 1)
        self.edge_attr = edge_attr # vectorized edge attributes
        return Data(x=self.node_attr, edge_index=self.edge_index,edge_attr=self.edge_attr)
    

        

