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
from Scripts.DataManager.GraphConstructor.SentimentGraphConstructor import SentimentGraphConstructor
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor, TextGraphType
from Scripts.DataManager.GraphLoader.GraphDataModule import GraphDataModule
from torch.utils.data.dataset import random_split
import torch
from Scripts.DataManager.Datasets.GraphConstructorDataset import GraphConstructorDataset, GraphConstructorDatasetRanged

class TwitterGraphDataModule(GraphDataModule):

    def __init__(self, config: Config, has_val: bool, has_test: bool, test_size=0.2, val_size=0.2, num_workers=2, drop_last=True, data_path='', graphs_path='', batch_size = 32, device='cpu', shuffle = False, start_data_load=0, end_data_load=-1, graph_type: TextGraphType = TextGraphType.FULL, load_preprocessed_data = True, reweights={}, *args, **kwargs):
        # Sample reweight [None,None,None,None,[(("word" , "seq" , "word") , 5)]]
        # 5 is weight in above code
        # (("word" , "seq" , "word") , 5)
        super(TwitterGraphDataModule, self)\
            .__init__(config, device, has_val, has_test, test_size, val_size, *args, **kwargs)

        self.reweights = reweights
        self.start_data_load = start_data_load
        self.end_data_load = end_data_load
        self.load_preprocessed_data = load_preprocessed_data
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.graph_type = graph_type
        self.graphs_path = graphs_path if graphs_path != '' else 'data/GraphData/Sentiment140'
        self.data_path = 'data/Sentiment140/sentiment140.csv' if data_path == '' else data_path
        self.labels = None
        self.dataset = None
        self.shuffle = shuffle
        self.df: pd.DataFrame = pd.DataFrame()
        self.__train_dataset, self.__val_dataset, self.__test_dataset = None, None, None

        
    def load_labels(self):
        self.df = pd.read_csv(
            path.join(self.config.root, self.data_path) , encoding='latin-1')

        self.df.columns = ['Polarity', 'Time', 'Date', 'Query', 'User' , 'Tweet']
        self.df = self.df[['Polarity', 'Tweet']]
        self.end_data_load = self.end_data_load if self.end_data_load > 0 else self.df.shape[0]
        self.end_data_load = self.end_data_load if self.end_data_load < self.df.shape[0] else self.df.shape[0]
        # removing tweets less than 3 words
        self.df = self.df[self.df['Tweet'].str.split().str.len().gt(2)]  
        # balancing the dataset
        g = self.df.groupby('Polarity')
        self.df = g.apply(lambda x: x.sample(self.end_data_load if self.end_data_load > 0 else g.size().min()).reset_index(drop=True))
        self.df = self.df.sample(frac=1 , random_state=1).reset_index(drop=True) # using seed in shuffling
    
        self.df = self.df.iloc[:self.end_data_load]
        self.df.index = np.arange(0, self.end_data_load)
        # activate one line below
        labels = self.df['Polarity'][:self.end_data_load]
        labels = labels.apply(lambda p: 0 if p == 0 else 1).to_numpy()
        labels = torch.from_numpy(labels)
        self.labels = labels.to(torch.float32).view(-1, 1).to(self.device)
        self.num_classes = len(torch.unique(self.labels))
        
    def load_graphs(self):
        self.graph_constructors = self.__set_graph_constructors(self.graph_type)
        
        self.dataset, self.num_node_features = {}, {}
        self.__train_dataset, self.__val_dataset, self.__test_dataset = {}, {}, {}
        self.__train_dataloader, self.__test_dataloader, self.__val_dataloader = {}, {}, {}
        for key in self.graph_constructors:
            self.graph_constructors[key].setup(self.load_preprocessed_data)
            # reweighting
            if key in self.reweights:
                for r in self.reweights[key]:
                    self.graph_constructors[key].reweight_all(r[0] , r[1])
            self.dataset[key] = GraphConstructorDataset(self.graph_constructors[key], self.labels)
            self.__train_dataset[key], self.__val_dataset[key], self.__test_dataset[key] =\
                random_split(self.dataset[key], [1-self.val_size-self.test_size, self.val_size, self.test_size])
            
            self.__train_dataloader[key] =  DataLoader(self.__train_dataset[key], batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] =  DataLoader(self.__test_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] =  DataLoader(self.__val_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            
        self.set_active_graph(key)
        
    
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size
        
        for key in self.graph_constructors:
            self.__train_dataloader[key] =  DataLoader(self.__train_dataset[key], batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] =  DataLoader(self.__test_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] =  DataLoader(self.__val_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            
        self.set_active_graph(key)    
    
    def get_data(self, datamodule):
        self.labels = datamodule.labels
        self.num_classes = datamodule.num_classes
        self.graph_constructors = datamodule.graph_constructors
        self.dataset, self.num_node_features = datamodule.dataset, datamodule.num_node_features
        self.__train_dataset, self.__val_dataset, self.__test_dataset = datamodule.__train_dataset, datamodule.__val_dataset, datamodule.__test_dataset
        self.__train_dataloader, self.__test_dataloader, self.__val_dataloader = datamodule.__train_dataloader, datamodule.__test_dataloader, datamodule.__val_dataloader
        self.set_active_graph(datamodule.active_key )
        
    def set_active_graph(self, graph_type: TextGraphType = TextGraphType.CO_OCCURRENCE):
        assert graph_type in self.dataset, 'The provided key is not valid'
        self.active_key = graph_type
        sample_graph = self.graph_constructors[self.active_key].get_first()
        self.num_node_features = sample_graph.num_features
           
    def create_sub_data_loader(self, begin: int, end: int):
        for key in self.graph_constructors:            
            dataset = GraphConstructorDatasetRanged(self.graph_constructors[key], self.labels, begin, end)
            train_dataset, val_dataset, test_dataset =\
                random_split(dataset, [1-self.val_size-self.test_size, self.val_size, self.test_size])
                
            self.__train_dataloader[key] =  DataLoader(train_dataset, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] =  DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] =  DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            
        self.set_active_graph(key)
        
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
            
        if TextGraphType.SENTIMENT in graph_type:
            graph_constructors[TextGraphType.SENTIMENT] = self.__get_sentiment_graph()
            graph_type = graph_type - TextGraphType.SENTIMENT
            
        return graph_constructors

    def __get_co_occurrence_graph(self):
        print(f'self.end_data_load: {self.end_data_load}')
        return CoOccurrenceGraphConstructor(self.df['Tweet'][:self.end_data_load], path.join(self.graphs_path, '140_co_occ'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load)
    
    def __get_dependency_graph(self):
        print(f'self.end_data_load: {self.end_data_load}')
        return DependencyGraphConstructor(self.df['Tweet'][:self.end_data_load], path.join(self.graphs_path, '140_dep'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load , use_node_dependencies=True)
    
    def __get_sequential_graph(self):
        print(f'self.end_data_load: {self.end_data_load}')
        return SequentialGraphConstructor(self.df['Tweet'][:self.end_data_load], path.join(self.graphs_path, '140_seq_gen'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load , use_general_node=True)
    
    def __get_dep_and_tag_graph(self):
        print(f'self.end_data_load: {self.end_data_load}')
        return TagDepTokenGraphConstructor(self.df['Tweet'][:self.end_data_load], path.join(self.graphs_path, '140_dep_and_tag'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load, use_sentence_nodes=False , use_general_node=True)
    
    def __get_tags_graph(self):
        print(f'self.end_data_load: {self.end_data_load}')
        return TagsGraphConstructor(self.df['Tweet'][:self.end_data_load], path.join(self.graphs_path, '140_tags'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load)
    
    def __get_full_graph(self):
        print(f'self.end_data_load: {self.end_data_load}')
        return TagDepTokenGraphConstructor(self.df['Tweet'][:self.end_data_load], path.join(self.graphs_path, '140_full'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load, use_sentence_nodes=True , use_general_node=True)
    
    def __get_sentence_graph(self):
        print(f'self.end_data_load: {self.end_data_load}')
        return SentenceGraphConstructor(self.df['Tweet'][:self.end_data_load], path.join(self.graphs_path, '140_sents_gen'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load, use_general_node=True)
    def __get_sentiment_graph(self):
        print(f'self.end_data_load: {self.end_data_load}')
        return SentimentGraphConstructor(self.df['Tweet'][:self.end_data_load], path.join(self.graphs_path, '140_sentiment'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load, use_sentence_nodes=True , use_general_node=True)
    def zero_rule_baseline(self):
        return f'zero_rule baseline: {(len(self.labels[self.labels>0.5])* 100.0 / len(self.labels))  : .2f}%'