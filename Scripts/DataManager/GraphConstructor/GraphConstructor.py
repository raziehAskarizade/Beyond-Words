import os
import pickle
from os import path
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, Any, List, Dict

from Scripts.Configs.ConfigClass import Config


class GraphConstructor(ABC):

    class _Variables(ABC):
        def __init__(self):
            self.graphs_name: Dict[int, str] = {}
            self.graph_num: int = 0

        def save_to_file(self, filename: str):
            with open(filename, 'wb') as file:
                pickle.dump(self, file)

        @classmethod
        def load_from_file(cls, filename: str):
            with open(filename, 'rb') as file:
                obj = pickle.load(file)
            if isinstance(obj, cls):
                return obj
            else:
                raise ValueError("Invalid file content. Unable to recreate the object.")

    def __init__(self, raw_data, variables: _Variables, save_path: str, config: Config, lazy_construction: bool,
                 load_preprocessed_data: bool, naming_prepend: str = ''):
        self.raw_data = raw_data
        self.config: Config = config
        self.lazy_construction = lazy_construction
        self.load_preprocessed_data = load_preprocessed_data
        self.device = config.device
        self.var = variables
        self.save_path = os.path.join(config.data_root_dir, save_path)
        self.naming_prepend = naming_prepend
        # self.node_attr, self.node_label, self.edge_index, self.edge_attr, self.edge_label = None, None, None, None, None
        # self.data = Data(x=self.node_attr, y=self.node_label, edge_index=self.edge_index)
        self._graphs: Dict[int, Data] = {}

    @abstractmethod
    def to_graph(self, raw_data):
        pass

    def get_graph(self, idx: int):
        if idx not in self._graphs:
            if self.load_preprocessed_data:
                self.load_data(idx)
            else:
                self._graphs[idx] = self.to_graph(self.raw_data[idx])
                self.var.graphs_name[idx] = f'{self.naming_prepend}_{idx}'
        return self._graphs[idx]

    # @abstractmethod
    # def set_graph(self, idx: int):
    #     pass
    #
    # @abstractmethod
    # def set_graphs(self, ids: List | Tuple | range | np.array | torch.Tensor | any):
    #     pass

    def get_graphs(self, ids: List | Tuple | range | np.array | torch.Tensor | any):
        not_loaded_ids = [idx for idx in ids if idx not in self._graphs]
        if len(not_loaded_ids) > 0 and self.load_preprocessed_data:
            self.load_data_list(not_loaded_ids)
        else:
            for idx in not_loaded_ids:
                self._graphs[idx] = self.to_graph(self.raw_data[idx])
                self.var.graphs_name[idx] = f'{self.naming_prepend}_{idx}'
        return {idx:self._graphs[idx] for idx in ids}

    def get_first(self):
        return self._graphs[next(iter(self._graphs))]

    def save_all_data(self):
        for i in range(len(self._graphs)):
            torch.save(self._graphs[i], path.join(self.save_path, f'{self.var.graphs_name[i]}.pt'))
        self.var.save_to_file(path.join(self.save_path, f'{self.naming_prepend}_var.txt'))

    def load_all_data(self):
        self.var = self.var.load_from_file(path.join(self.save_path, f'{self.naming_prepend}_var.txt'))
        for i in range(self.var.graph_num):
            self._graphs[i] = torch.load(path.join(self.save_path, f'{self.var.graphs_name[i]}.pt'))

    def load_var(self):
        self.var = self.var.load_from_file(path.join(self.save_path, f'{self.naming_prepend}_var.txt'))

    def load_data(self, idx: int):
        self._graphs[idx] = torch.load(path.join(self.save_path, f'{self.var.graphs_name[idx]}.pt'))

    def load_data_list(self, ids: List | Tuple | range | np.array | torch.Tensor | any):
        if torch.max(torch.tensor(ids) < self.var.graph_num) == 1:
            print(f'Index is out of range, indices should be more than 0 and less than {self.var.graph_num}')
            return

        for i in ids:
            self._graphs[i] = torch.load(path.join(self.save_path, f'{self.var.graphs_name[i]}.pt'))
