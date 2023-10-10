from typing import List, Set, Tuple
from abc import ABC, abstractmethod
from torch_geometric.utils import augmentation
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.utils import subgraph, train_test_split_edges


class GraphLoader(ABC):

    def __init__(self, nodes_x, edge_index, device,
                 test_size=0.2, _val_size=0.15, *args, **kwargs):
        self.nodes = nodes_x.to(device)
        self.edge_index = edge_index.to(device)
        self.edge_count = self.edge_index.shape[1]
        self.device = device
        self._test_size = test_size
        self._val_size = _val_size

        self.train_indices, self.test_indices, self.val_indices = self.create_split_indices()

        self.edge_indices_train = self.create_sub_graph_edges(self.train_indices)
        self.edge_indices_test = self.create_sub_graph_edges(self.test_indices)
        self.edge_indices_val = self.create_sub_graph_edges(self.test_indices)

    @abstractmethod
    def get_train_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

    @abstractmethod
    def get_val_data(self):
        pass

    @abstractmethod
    def extract_random_sub_edges_graph(self, edge_count):
        pass

    @abstractmethod
    def update(self, nodes_x, edge_index, device, *args, **kwargs):
        pass

    @abstractmethod
    def update_edge_index(self, edge_index):
        pass

    @abstractmethod
    def update_nodes(self, nodes_x):
        pass

    def create_split_indices(self):
        shuffled_indices = torch.randperm(self.nodes.shape[0], device=self.device)
        x_train, x_val = train_test_split(shuffled_indices, test_size=self._val_size)
        x_train, x_test = train_test_split(x_train, test_size=self._test_size)
        return x_train, x_test, x_val

    def create_sub_graph_edges(self, node_mask):
        new_edge_indices = subgraph(node_mask, self.edge_index)[0]
        map_values = torch.tensor(range(len(node_mask)))
        node_map = pd.DataFrame(data=map_values, index=node_mask.cpu().numpy(), dtype=int)
        new_edge_indices[0] = \
            torch.tensor(np.squeeze(node_map.loc[new_edge_indices[0].cpu()].values), device=self.device)
        new_edge_indices[1] = \
            torch.tensor(np.squeeze(node_map.loc[new_edge_indices[1].cpu()].values), device=self.device)
        return new_edge_indices


class NodeLabeledGraphLoader(GraphLoader, ABC):

    def __init__(self, nodes_x, nodes_y, edge_index, device, *args, **kwargs):
        super(NodeLabeledGraphLoader, self).__init__(nodes_x, edge_index, device, *args, **kwargs)
        self.node_labels = nodes_y.to(device)

    @abstractmethod
    def update_node_labels(self, node_y):
        pass
