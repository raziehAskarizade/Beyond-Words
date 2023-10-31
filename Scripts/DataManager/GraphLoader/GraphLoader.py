from typing import List, Set, Tuple, Any
from abc import ABC, abstractmethod

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.utils import augmentation
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.utils import subgraph, train_test_split_edges
from torch_geometric.data import Data

from Scripts.Configs.ConfigClass import Config
from Scripts.Utils.GraphCollection.GraphCollection import GraphCollection


class GraphLoader(LightningDataModule, ABC):

    def __init__(self, config: Config, device, has_val: bool, has_test: bool, test_size=0.2, val_size=0.15, *args, **kwargs):
        super(GraphLoader, self).__init__(has_val, has_test, **kwargs)
        self.config = config
        self.test_size = test_size
        self.val_size = val_size
        self.device = device

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage: str):
        pass

    @abstractmethod
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    @abstractmethod
    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @abstractmethod
    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @abstractmethod
    def teardown(self, stage: str) -> None:
        pass

    # def predict_dataloader(self) -> EVAL_DATALOADERS:
    # def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
    # def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
    # def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:


class HomogeneousGraphLoader(GraphLoader):

    def __init__(self, graph: Data, device, test_size=0.2, val_size=0.15, *args, **kwargs):
        super(HomogeneousGraphLoader, self).__init__(device, test_size, val_size, *args, **kwargs)
        self.graph = graph.to(device)
        self.nodes = self.graph.x
        self.edge_index = self.graph.edge_index
        self.edge_count = self.graph.num_edges
        self.node_count = self.graph.num_nodes
        self._sub_node_tensor_index = torch.arange(self.node_count)
        self._sub_edge_tensor_index = torch.arange(self.edge_count)

        self.train_indices, self.test_indices, self.val_indices = self.create_split_indices()

        self.edge_indices_train = self.create_sub_graph_edges(self.train_indices)
        self.edge_indices_test = self.create_sub_graph_edges(self.test_indices)
        self.edge_indices_val = self.create_sub_graph_edges(self.test_indices)

    @abstractmethod
    def train_dataloader(self):
        pass

    @abstractmethod
    def test_dataloader(self):
        pass

    @abstractmethod
    def val_dataloader(self):
        pass

    @abstractmethod
    def update(self, nodes_x, edge_index, device, *args, **kwargs):
        pass

    @abstractmethod
    def update_edge_index(self, edge_index):
        pass

    @abstractmethod
    def update_nodes(self, new_nodes: torch.Tensor):
        self.nodes = new_nodes
        self.node_count = len(self.nodes)
        self._sub_node_tensor_index = torch.arange(self.node_count)
        removable_edges = [i for i in range(self.edge_index.shape[1])
                           if (self.edge_index[0, i] not in self._sub_node_tensor_index
                               or self.edge_index[1, i] not in self._sub_node_tensor_index)]
        del self.edge_index[removable_edges]
        self.edge_count = self.edge_index.shape[1]
        self._sub_edge_tensor_index = torch.arange(self.edge_count)

        self.train_indices, self.test_indices, self.val_indices = self.create_split_indices()

        self.edge_indices_train = self.create_sub_graph_edges(self.train_indices)
        self.edge_indices_test = self.create_sub_graph_edges(self.test_indices)
        self.edge_indices_val = self.create_sub_graph_edges(self.test_indices)

    def create_split_indices(self):
        self._sub_node_tensor_index = self._sub_node_tensor_index[
            torch.randperm(self._sub_node_tensor_index.shape[0], device=self.device)]
        x_train, x_val = train_test_split(self._sub_node_tensor_index, test_size=self.val_size)
        x_train, x_test = train_test_split(x_train, test_size=self.test_size)
        return x_train, x_test, x_val

    def create_sub_graph_edges(self, node_mask):
        new_edge_indices = subgraph(node_mask, self.edge_index[self._sub_edge_tensor_index])[0]
        map_values = torch.tensor(range(len(node_mask)))
        node_map = pd.DataFrame(data=map_values, index=node_mask.cpu().numpy(), dtype=int)
        new_edge_indices[0] = \
            torch.tensor(np.squeeze(node_map.loc[new_edge_indices[0].cpu()].values), device=self.device)
        new_edge_indices[1] = \
            torch.tensor(np.squeeze(node_map.loc[new_edge_indices[1].cpu()].values), device=self.device)
        return new_edge_indices

    def use_sub_graph_by_nodes(self, node_index):
        self._sub_node_tensor_index = node_index
        self._sub_edge_tensor_index = self.create_sub_graph_edges(node_index)

    def use_sub_graph_by_edges(self, edges_tensor_index):
        self._sub_edge_tensor_index = edges_tensor_index
        self._sub_node_tensor_index = torch.unique(self._sub_edge_tensor_index)


class CollectionGraphLoader(GraphLoader):

    def __init__(self, graphs: GraphCollection, device, test_size=0.2, val_size=0.15, *args, **kwargs):
        super(CollectionGraphLoader, self).__init__(device, test_size, val_size, *args, **kwargs)
        self.graphs = graphs

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass


class KnowledgeGraphLoader(GraphLoader):
    pass
