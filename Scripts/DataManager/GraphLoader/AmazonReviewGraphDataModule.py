from copy import copy
import numpy as np
from os import path
from typing import Dict

import pandas as pd
from torch_geometric.loader import DataLoader

from Scripts.Configs.ConfigClass import Config
from Scripts.DataManager.GraphConstructor.CoOccurrenceGraphConstructor import CoOccurrenceGraphConstructor
from Scripts.DataManager.GraphConstructor.TagsGraphConstructor import TagsGraphConstructor
from Scripts.DataManager.GraphConstructor.DependencyGraphConstructor import DependencyGraphConstructor
from Scripts.DataManager.GraphConstructor.SequentialGraphConstructor import SequentialGraphConstructor
from Scripts.DataManager.GraphConstructor.TagDepTokenGraphConstructor import TagDepTokenGraphConstructor
from Scripts.DataManager.GraphConstructor.SentenceGraphConstructor import SentenceGraphConstructor
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor, TextGraphType
from Scripts.DataManager.GraphLoader.GraphDataModule import GraphDataModule
from torch.utils.data.dataset import random_split
import torch
from Scripts.DataManager.Datasets.GraphConstructorDataset import GraphConstructorDataset



class AmazonReviewGraphDataModule(GraphDataModule):

    def __init__(self, config: Config, has_val: bool, has_test: bool, test_size=0.2, val_size=0.2, num_workers=2, drop_last=True, train_data_path='', test_data_path='', graphs_path='', batch_size = 32, device='cpu', shuffle = False, num_data_load=-1, graph_type: TextGraphType = TextGraphType.FULL, load_preprocessed_data = True, *args, **kwargs):
        
        super(AmazonReviewGraphDataModule, self)\
            .__init__(config, device, has_val, has_test, test_size, val_size, *args, **kwargs)

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.graph_type = graph_type
        self.graphs_path = graphs_path if graphs_path!='' else 'data/GraphData/AmazonReview'
        self.train_data_path = 'data/Amazon-Review/train_sm.csv' if train_data_path == '' else train_data_path
        self.test_data_path = 'data/Amazon-Review/test_sm.csv' if test_data_path == '' else test_data_path
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.labels = None
        self.dataset = None
        self.shuffle = shuffle
        self.df: pd.DataFrame = pd.DataFrame()
        self.__train_dataset, self.__val_dataset, self.__test_dataset = None, None, None
        self.train_df = pd.read_csv(path.join(self.config.root, self.train_data_path))
        self.test_df = pd.read_csv(path.join(self.config.root, self.test_data_path))
        self.train_df.columns = ['Polarity', 'Title', 'Review']
        self.test_df.columns = ['Polarity', 'Title', 'Review']
        self.train_df = self.train_df[['Polarity', 'Review']]
        self.test_df = self.test_df[['Polarity', 'Review']]
        self.df = pd.concat([self.train_df, self.test_df])
        self.num_data_load = num_data_load if num_data_load>0 else self.df.shape[0]
        self.num_data_load = num_data_load if self.num_data_load < self.df.shape[0] else self.df.shape[0] 
        self.df = self.df.iloc[:self.num_data_load]
        self.df.index = np.arange(0, self.num_data_load)
        # activate one line below

        labels = self.df['Polarity'][:self.num_data_load]
        labels = labels.apply(lambda p: 0 if p == 1 else 1).to_numpy()
        labels = torch.from_numpy(labels)
        self.labels = labels.to(torch.float32).view(-1, 1).to(self.device)
        # graph_constructor = self.graph_constructors[TextGraphType.CO_OCCURRENCE]

        self.num_classes = len(torch.unique(self.labels))
        
        self.graph_constructors = self.__set_graph_constructors(self.graph_type)
        
        self.dataset, self.num_node_features = {}, {}
        self.__train_dataset, self.__val_dataset, self.__test_dataset = {}, {}, {}
        self.__train_dataloader, self.__test_dataloader, self.__val_dataloader = {}, {}, {}
        for key in self.graph_constructors:
            self.graph_constructors[key].setup(load_preprocessed_data)
            self.dataset[key] = GraphConstructorDataset(self.graph_constructors[key], self.labels)
            
            self.__train_dataset[key], self.__val_dataset[key], self.__test_dataset[key] =\
                random_split(self.dataset[key], [1-self.val_size-self.test_size, self.val_size, self.test_size])
            
            self.__train_dataloader[key] =  DataLoader(self.__train_dataset[key], batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] =  DataLoader(self.__test_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] =  DataLoader(self.__val_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            
        self.set_active_graph(key)
        
    def set_active_graph(self, graph_type: TextGraphType = TextGraphType.CO_OCCURRENCE):
        assert graph_type in self.dataset, 'The provided key is not valid'
        self.active_key = graph_type
        sample_graph = self.graph_constructors[self.active_key].get_first()
        self.num_node_features = sample_graph.num_features
    
    
    
    def prepare_data(self):
        pass
        
    def setup(self, stage: str):
        pass

    def teardown(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return self.__train_dataloader[self.active_key ]

    def test_dataloader(self):
        return self.__test_dataloader[self.active_key ]

    def val_dataloader(self):
        return self.__val_dataloader[self.active_key ]

    def __set_graph_constructors(self, graph_type: TextGraphType):
        graph_type = copy(graph_type)
        graph_constructors: Dict[TextGraphType, GraphConstructor] = {}
        
        if TextGraphType.CO_OCCURRENCE in graph_type:
            graph_constructors[TextGraphType.CO_OCCURRENCE] = self.__get_co_occurrence_graph()
            graph_type = graph_type - TextGraphType.CO_OCCURRENCE
        
        tag_dep_seq_sent = TextGraphType.DEPENDENCY | TextGraphType.TAGS | TextGraphType.SEQUENTIAL | TextGraphType.SENTENCE
        if tag_dep_seq_sent in graph_type:
            graph_constructors[tag_dep_seq_sent] = self.__get_full_graph()
            graph_type = graph_type - tag_dep_seq_sent
            
        tag_dep_seq = TextGraphType.DEPENDENCY | TextGraphType.TAGS | TextGraphType.SEQUENTIAL
        if tag_dep_seq in graph_type:
            graph_constructors[tag_dep_seq] = self.__get_dep_and_tag_graph()
            graph_type = graph_type - tag_dep_seq
        
        if TextGraphType.DEPENDENCY in graph_type:
            graph_constructors[TextGraphType.DEPENDENCY] = self.__get_dependency_graph()
            graph_type = graph_type - (TextGraphType.DEPENDENCY | TextGraphType.SEQUENTIAL)
        
        if TextGraphType.TAGS in graph_type:
            graph_constructors[TextGraphType.TAGS] = self.__get_tag_graph()
            graph_type = graph_type - (TextGraphType.TAGS | TextGraphType.SEQUENTIAL)
            
        if TextGraphType.SENTENCE in graph_type:
            graph_constructors[TextGraphType.SENTENCE] = self.__get_sentence_graph()
            graph_type = graph_type - (TextGraphType.SENTENCE | TextGraphType.SEQUENTIAL)
        
        if TextGraphType.SEQUENTIAL in graph_type:
            graph_constructors[TextGraphType.SEQUENTIAL] = self.__get_Sequential_graph()
            graph_type = graph_type - TextGraphType.SEQUENTIAL
            
        return graph_constructors

    def __get_co_occurrence_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return CoOccurrenceGraphConstructor(self.df['Review'][:self.num_data_load], path.join(self.graphs_path, 'co_occ'), self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load)
    
    def __get_dependency_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return DependencyGraphConstructor(self.df['Review'][:self.num_data_load], path.join(self.graphs_path, 'dep'), self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load , use_node_dependencies=True)
    
    def __get_sequential_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return SequentialGraphConstructor(self.df['Review'][:self.num_data_load], path.join(self.graphs_path, 'seq'), self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load , use_general_node=True)
    
    def __get_dep_and_tag_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return TagDepTokenGraphConstructor(self.df['Review'][:self.num_data_load], path.join(self.graphs_path, 'dep_and_tag'), self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load, use_sentence_nodes=False , use_general_node=True)
    
    def __get_tags_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return TagsGraphConstructor(self.df['Review'][:self.num_data_load], path.join(self.graphs_path, 'tags'), self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load)
    
    def __get_full_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return TagDepTokenGraphConstructor(self.df['Review'][:self.num_data_load], path.join(self.graphs_path, 'full'), self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load, use_sentence_nodes=True , use_general_node=True)
    
    def __get_sentence_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return SentenceGraphConstructor(self.df['Review'][:self.num_data_load], path.join(self.graphs_path, 'sents'), self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load, use_general_node=True)
    
    def zero_rule_baseline(self):
        return f'zero_rule baseline: {(len(self.labels[self.labels>0.5])* 100.0 / len(self.labels))  : .2f}%'