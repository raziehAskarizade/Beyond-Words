from os import path

import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from Scripts.Configs.ConfigClass import Config
from Scripts.DataManager.GraphConstructor.CoOccurrenceGraphConstructor import CoOccurrenceGraphConstructor
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor, TextGraphType
from Scripts.DataManager.GraphLoader.GraphLoader import GraphLoader
from torch.utils.data.dataset import random_split
import torch


class GraphConstructorDataset(Dataset):

    def __init__(self, graph_constructor: GraphConstructor, graph_labels):
        self.graph_constructor = graph_constructor
        self.graph_labels = graph_labels

    def __getitem__(self, index):
        x = self.graph_constructor.get_graph(index)
        y = self.graph_labels[index]
        return x, y

    def __len__(self):
        return self.graph_constructor.var.graph_num


class AmazonReviewGraphLoader(GraphLoader):

    def __init__(self, config: Config, has_val: bool, has_test: bool, test_size=0.2, val_size=0.2, num_workers=2,
                 drop_last=True, train_data_path='', test_data_path='', graphs_path='', batch_size = 128,
                 device='cpu', graph_type: TextGraphType = TextGraphType.CO_OCCURRENCE, *args, **kwargs):
        super(AmazonReviewGraphLoader, self)\
            .__init__(device, has_val, has_test, test_size, val_size, *args, **kwargs)

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.config = config
        self.graph_type = graph_type
        self.train_data_path = 'data/Amazon-Review/train_sm.csv' if train_data_path == '' else train_data_path
        self.test_data_path = 'data/Amazon-Review/test_sm.csv' if test_data_path == '' else test_data_path
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.labels = None
        self.dataset = None
        self.num_node_features = 0
        self.num_classes = 0
        self.df: pd.DataFrame = pd.DataFrame()
        self.__train_dataset, self.__val_dataset, self.__test_dataset = None, None, None

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.train_df = pd.read_csv(path.join(self.config.root, self.train_data_path))
        self.test_df = pd.read_csv(path.join(self.config.root, self.test_data_path))
        self.train_df.columns = ['Polarity', 'Title', 'Review']
        self.test_df.columns = ['Polarity', 'Title', 'Review']
        self.train_df = self.train_df[['Polarity', 'Review']]
        self.test_df = self.test_df[['Polarity', 'Review']]
        self.df = pd.concat([self.train_df, self.test_df])
        labels = self.df['Polarity']
        labels = labels.apply(lambda p: 0 if p == 1 else 1).to_numpy()
        labels = torch.from_numpy(labels)
        self.labels = labels.to(torch.float32).view(-1, 1)
        graph_constructor = self.__get_co_occurrence_graph()
        self.dataset = GraphConstructorDataset(graph_constructor, self.labels)
        sample_graph = graph_constructor.get_first()
        self.num_node_features = sample_graph.num_features
        self.num_classes = len(torch.unique(self.labels))
        self.__train_dataset, self.__val_dataset, self.__test_dataset =\
            random_split(self.dataset, [1-self.val_size-self.test_size, self.val_size, self.test_size])

    def teardown(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(self.__train_dataset, batch_size=self.batch_size, drop_last=self.drop_last,
                          shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.__test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.__val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=True)

    def __set_graph_constructors(self, graph_type: TextGraphType):
        graph_constructors = {}
        if TextGraphType.CO_OCCURRENCE in graph_type:
            graph_constructors[TextGraphType.CO_OCCURRENCE] = self.__get_co_occurrence_graph()
        if TextGraphType.DEPENDENCY in graph_type:
            pass
        if TextGraphType.SEQUENTIAL in graph_type:
            pass
        if TextGraphType.TAGS in graph_type:
            pass
        return graph_constructors

    def __get_co_occurrence_graph(self):
        return CoOccurrenceGraphConstructor(self.train_df['Review'], 'AmazonReview', self.config,
                                         lazy_construction=True,
                                         load_preprocessed_data=True, naming_prepend='graph')