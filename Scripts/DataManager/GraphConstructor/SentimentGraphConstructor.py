# Omid Davar @ 2023

import pickle
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
from torch_geometric.utils import to_networkx

from Scripts.DataManager.GraphConstructor.TagDepTokenGraphConstructor import TagDepTokenGraphConstructor
from torch_geometric.data import Data, HeteroData
from Scripts.Configs.ConfigClass import Config
import torch
import numpy as np
import os


class SentimentGraphConstructor(TagDepTokenGraphConstructor):

    class _Variables(TagDepTokenGraphConstructor._Variables):
        def __init__(self):
            super(SentimentGraphConstructor._Variables, self).__init__()
            self.nlp_pipeline: str = ''

    def __init__(self, texts: List[str], save_path: str, config: Config,
                 load_preprocessed_data=False, naming_prepend='', use_compression=True, use_sentence_nodes=False, use_general_node=True, start_data_load=0, end_data_load=-1, num_general_nodes=1):

        super(SentimentGraphConstructor, self)\
            .__init__(texts, save_path, config, load_preprocessed_data,
                      naming_prepend, use_compression, use_sentence_nodes, use_general_node, start_data_load, end_data_load, num_general_nodes)
        # self.settings["token_sentiment_weight"] = 2

    def persion_polarity(self, dataset_path="C:/Users/razieh/Downloads/Beyond-Words/data/PerSent.xlsx"):
        xlsx = pd.ExcelFile(dataset_path)
        df = xlsx.parse('Dataset')
        return df

    def to_graph(self, text: str):
        # farsi
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        token_list = self.token_lemma(text)
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for word in sentence.words:
                doc.append((idx, word.text, word.lemma,
                            word.upos, word.head, word.deprel))
        # if len(doc) < 2:
        #     return
        return self.__create_sentiment_graph(doc)

    def _build_initial_sentiment_vector(self):
        return torch.zeros((2, self.nlp.get_dimension()), dtype=torch.float32)

    def __create_sentiment_graph(self, doc, for_compression=False):
        if for_compression:
            data = super().to_graph_indexed(doc)
        else:
            data = super().to_graph(doc)
        # adding sentiment nodes
        if for_compression:
            data['sentiment'].x = torch.full((2,), -1, dtype=torch.float32)
        else:
            data['sentiment'].x = self._build_initial_sentiment_vector()
        sentiment_word_edge_index = []
        word_sentiment_edge_index = []
        sentiment_word_edge_attr = []
        word_sentiment_edge_attr = []

        df = self.persion_polarity()
        for i, token in enumerate(doc[1:]):
            if len(df[df['Words'] == token[2]]) != 0:
                row = df[df['Words'] == token[2]]
                polarity = max(row['Polarity'])
                if polarity > 0:
                    word_sentiment_edge_index.append([i, 1])
                    sentiment_word_edge_index.append([1, i])
                    word_sentiment_edge_attr.append(abs(polarity))
                    sentiment_word_edge_attr.append(abs(polarity))
                if polarity < 0:
                    word_sentiment_edge_index.append([i, 0])
                    sentiment_word_edge_index.append([0, i])
                    word_sentiment_edge_attr.append(abs(polarity))
                    sentiment_word_edge_attr.append(abs(polarity))
        data['word', 'word_sentiment', 'sentiment'].edge_index = torch.transpose(torch.tensor(
            word_sentiment_edge_index, dtype=torch.int32), 0, 1) if len(word_sentiment_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['sentiment', 'sentiment_word', 'word'].edge_index = torch.transpose(torch.tensor(
            sentiment_word_edge_index, dtype=torch.int32), 0, 1) if len(sentiment_word_edge_index) > 0 else torch.empty(2, 0, dtype=torch.int32)
        data['word', 'word_sentiment', 'sentiment'].edge_attr = torch.tensor(
            word_sentiment_edge_attr, dtype=torch.float32)
        data['sentiment', 'sentiment_word', 'word'].edge_attr = torch.tensor(
            sentiment_word_edge_attr, dtype=torch.float32)
        return data

    def to_graph_indexed(self, text: str):
        # farsi
        doc_sentences = []
        doc = []
        doc.append(doc_sentences)
        token_list = self.token_lemma(text)
        for idx, sentence in enumerate(token_list.sentences):
            doc_sentences.append((sentence.text, sentence.tokens[0].text, idx))
            for word in sentence.words:
                doc.append((idx, word.text, word.lemma,
                            word.upos, word.head, word.deprel))
        # if len(doc) < 2:
        #     return
        return self.__create_sentiment_graph(doc, for_compression=True)

    def prepare_loaded_data(self, graph):
        graph = super(SentimentGraphConstructor,
                      self).prepare_loaded_data(graph)
        graph['sentiment'].x = self._build_initial_sentiment_vector()
        for t in graph.edge_types:
            if graph[t].edge_index.shape[1] == 0:
                graph[t].edge_index = torch.empty(2, 0, dtype=torch.int32)
        return graph

    def remove_node_type_from_graphs(self, node_name: str):
        for i in range(len(self._graphs)):
            if self._graphs[i] is not None:
                if node_name in self._graphs[i].node_types:
                    del self._graphs[i][node_name]
                for edge_type in self._graphs[i].edge_types:
                    if edge_type[0] == node_name or edge_type[1] == node_name:
                        del self._graphs[i][edge_type]
