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
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor, TextGraphType
from Scripts.DataManager.GraphLoader.GraphDataModule import GraphDataModule
from torch.utils.data.dataset import random_split
import torch
from Scripts.DataManager.Datasets.GraphConstructorDataset import GraphConstructorDataset



class AmazonReviewGraphDataModule(GraphDataModule):

    def __init__(self, config: Config, has_val: bool, has_test: bool, test_size=0.2, val_size=0.2, num_workers=2,
                 drop_last=True, train_data_path='', test_data_path='', graphs_path='', batch_size = 32,
                 device='cpu', shuffle = False, num_data_load=-1,
                 graph_type: TextGraphType = TextGraphType.CO_OCCURRENCE | TextGraphType.DEPENDENCY | TextGraphType.TAGS | TextGraphType.FULL | TextGraphType.SENTENCE | TextGraphType.SEQUENTIAL, *args, **kwargs):
        # kwargs['num_workers'] = num_workers
        # kwargs['batch_size'] = batch_size
        # kwargs['shuffle'] = shuffle
        super(AmazonReviewGraphDataModule, self)\
            .__init__(config, device, has_val, has_test, test_size, val_size, *args, **kwargs)

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.graph_type = graph_type
        self.train_data_path = 'data/Amazon-Review/train_sm.csv' if train_data_path == '' else train_data_path
        self.test_data_path = 'data/Amazon-Review/test_sm.csv' if test_data_path == '' else test_data_path
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.labels = None
        self.dataset = None
        self.shuffle = shuffle
        self.num_node_features = 0
        self.num_classes = 0
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
        self.graph_constructors = self.__set_graph_constructors(self.graph_type)
        # activate one line below
        graph_constructor = self.graph_constructors[TextGraphType.CO_OCCURRENCE]
        # graph_constructor = self.graph_constructors[TextGraphType.DEPENDENCY]
        # graph_constructor = self.graph_constructors[TextGraphType.SEQUENTIAL]
        # graph_constructor = self.graph_constructors[TextGraphType.FULL]
        # graph_constructor = self.graph_constructors[TextGraphType.SENTENCE]
        # graph_constructor = self.graph_constructors[TextGraphType.TAGS]
        graph_constructor.setup()
        print(f'self.num_data_load: {self.num_data_load}')
        labels = self.df['Polarity'][:self.num_data_load]
        labels = labels.apply(lambda p: 0 if p == 1 else 1).to_numpy()
        labels = torch.from_numpy(labels)
        self.labels = labels.to(torch.float32).view(-1, 1).to(self.device)
        # graph_constructor = self.graph_constructors[TextGraphType.CO_OCCURRENCE]
        
        print(f'self.labels.shape: {self.labels.shape}')
        self.dataset = GraphConstructorDataset(graph_constructor, self.labels)
        sample_graph = graph_constructor.get_first()
        self.num_node_features = sample_graph.num_features
        self.num_classes = len(torch.unique(self.labels))
        self.__train_dataset, self.__val_dataset, self.__test_dataset =\
            random_split(self.dataset, [1-self.val_size-self.test_size, self.val_size, self.test_size])
        self.__train_dataloader =  DataLoader(self.__train_dataset, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
        self.__test_dataloader =  DataLoader(self.__test_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)
        self.__val_dataloader =  DataLoader(self.__val_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)
        
    def prepare_data(self):
        pass
        
    def setup(self, stage: str):
        pass

    def teardown(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return self.__train_dataloader

    def test_dataloader(self):
        return self.__test_dataloader

    def val_dataloader(self):
        return self.__val_dataloader 

    def __set_graph_constructors(self, graph_type: TextGraphType):
        
        graph_constructors: Dict[TextGraphType, GraphConstructor] = {}
        if TextGraphType.CO_OCCURRENCE in graph_type:
            graph_constructors[TextGraphType.CO_OCCURRENCE] = self.__get_co_occurrence_graph()
        if TextGraphType.DEPENDENCY in graph_type:
            graph_constructors[TextGraphType.DEPENDENCY] = self.__get_dependency_graph()
        if TextGraphType.SEQUENTIAL in graph_type:
            graph_constructors[TextGraphType.SEQUENTIAL] = self.__get_sequential_graph()
        if TextGraphType.TAGS in graph_type:
            graph_constructors[TextGraphType.TAGS] = self.__get_tags_graph()
        if TextGraphType.FULL in graph_type:
            graph_constructors[TextGraphType.FULL] = self.__get_full_graph()
        if TextGraphType.SENTENCE in graph_type:
            pass
        return graph_constructors

    def __get_co_occurrence_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return CoOccurrenceGraphConstructor(self.df['Review'][:self.num_data_load], 'data/GraphData/AmazonReview', self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load, device=self.device)
    
    def __get_dependency_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return DependencyGraphConstructor(self.df['Review'][:self.num_data_load], 'data/GraphData/AmazonReview', self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load, device=self.device , use_node_dependencies=True)
    
    def __get_sequential_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return SequentialGraphConstructor(self.df['Review'][:self.num_data_load], 'data/GraphData/AmazonReview', self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load, device=self.device , use_general_node=True)
    
    def __get_tags_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return TagsGraphConstructor(self.df['Review'][:self.num_data_load], 'data/GraphData/AmazonReview', self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load, device=self.device)
    
    def __get_full_graph(self):
        print(f'self.num_data_load: {self.num_data_load}')
        return TagDepTokenGraphConstructor(self.df['Review'][:self.num_data_load], 'data/GraphData/AmazonReview', self.config, lazy_construction=False, load_preprocessed_data=True, naming_prepend='graph', num_data_load=self.num_data_load, device=self.device, use_sentence_nodes=False , use_general_node=True)
    
    def zero_rule_baseline(self):
        return f'zero_rule baseline: {(len(self.labels[self.labels>0.5])* 100.0 / len(self.labels))  : .2f}%'