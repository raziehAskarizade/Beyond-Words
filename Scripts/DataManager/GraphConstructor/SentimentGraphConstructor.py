import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructor.TagDepTokenGraphConstructor import TagDepTokenGraphConstructor
from torch_geometric.data import Data , HeteroData
from Scripts.Configs.ConfigClass import Config
import spacy
import torch
import numpy as np
import os
from spacytextblob.spacytextblob import SpacyTextBlob


class SentimentGraphConstructor(TagDepTokenGraphConstructor):
    
    class _Variables(TagDepTokenGraphConstructor._Variables):
        def __init__(self):
            super(SentimentGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''
            
    def __init__(self, texts: List[str], save_path: str, config: Config,
                load_preprocessed_data=False, naming_prepend='' , use_compression=True,use_sentence_nodes=False, use_general_node=True, num_data_load=-1,num_general_nodes = 1):

        super(SentimentGraphConstructor, self)\
            .__init__(texts, save_path, config, load_preprocessed_data,
                      naming_prepend  , use_compression,use_sentence_nodes, use_general_node, num_data_load,num_general_nodes)
        # self.settings["token_sentiment_weight"] = 2
        self.nlp.add_pipe('spacytextblob')
        
    def to_graph(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        return self.__create_sentiment_graph(doc)
    def _build_initial_sentiment_vector(self):
        return torch.zeros((2 , self.nlp.vocab.vectors_length), dtype=torch.float32)   
    def __create_sentiment_graph(self , doc , for_compression=False):
        if for_compression:
            data = super().to_graph_indexed(doc)
        else:
            data = super().to_graph(doc)
        # adding sentiment nodes
        if for_compression:
            data['sentiment'].x = torch.full((2,),-1, dtype=torch.float32)
        else:
            data['sentiment'].x = self._build_initial_sentiment_vector()
        sentiment_word_edge_index = []
        word_sentiment_edge_index = []
        sentiment_word_edge_attr = []
        word_sentiment_edge_attr = []
        for token in doc:
            # if token._.blob.polarity != 0:
            #     print(token._.blob.polarity)
            #     print(token)
            if token._.blob.polarity > 0:
                word_sentiment_edge_index.append([token.i, 1])
                sentiment_word_edge_index.append([1, token.i])
                word_sentiment_edge_attr.append(abs(token._.blob.polarity))
                sentiment_word_edge_attr.append(abs(token._.blob.polarity))
            if token._.blob.polarity < 0:
                word_sentiment_edge_index.append([token.i, 0])
                sentiment_word_edge_index.append([0, token.i])
                word_sentiment_edge_attr.append(abs(token._.blob.polarity))
                sentiment_word_edge_attr.append(abs(token._.blob.polarity))
        if len(word_sentiment_edge_index) > 0:
            data['word' , 'word_sentiment' , 'sentiment'].edge_index = torch.transpose(torch.tensor(word_sentiment_edge_index, dtype=torch.int32) , 0 , 1) if len(word_sentiment_edge_index) > 0 else []
            data['sentiment' , 'sentiment_word' , 'word'].edge_index = torch.transpose(torch.tensor(sentiment_word_edge_index, dtype=torch.int32) , 0 , 1) if len(sentiment_word_edge_index) > 0 else []
            data['word' , 'word_sentiment' , 'sentiment'].edge_attr = torch.tensor(word_sentiment_edge_attr, dtype=torch.float32)
            data['sentiment' , 'sentiment_word' , 'word'].edge_attr = torch.tensor(sentiment_word_edge_attr, dtype=torch.float32)
        return data
                    
    def to_graph_indexed(self, text: str):
        doc = self.nlp(text)
        if len(doc) < 2:
            return
        return self.__create_sentiment_graph(doc , for_compression=True)
    
    def prepare_loaded_data(self, graph):
        words = torch.zeros((len(graph['word'].x) , self.nlp.vocab.vectors_length), dtype=torch.float32)
        for i in range(len(graph['word'].x)):
            if graph['word'].x[i] in self.nlp.vocab.vectors:
                words[i] = torch.tensor(self.nlp.vocab.vectors[graph['word'].x[i]])
        graph['word'].x = words
        if self.use_general_node:
            graph = self._add_multiple_general_nodes(graph , True , self.num_general_nodes)
        graph['sentiment'].x = self._build_initial_sentiment_vector()
        return graph
    

        

