import os
import pickle
from os import path
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, Any, List, Dict

from torch_geometric.utils import to_networkx

from Scripts.Configs.ConfigClass import Config
from enum import Enum
from flags import Flags


class TextGraphType(Flags):
    CO_OCCURRENCE = 1
    DEPENDENCY = 2
    SEQUENTIAL = 4
    TAGS = 8


class GraphConstructor(ABC):

    class _Variables(ABC):
        def __init__(self):
            self.graphs_name: Dict[int, str] = {}
            self.graph_num: int = 0
            self.device = 'cpu'

        def save_to_file(self, filename: str):
            with open(filename, 'wb') as file:
                pickle.dump(self, file)

        @classmethod
        def load_from_file(cls, filename: str):
            print(f'filename: {filename}')
            with open(filename, 'rb') as file:
                obj = pickle.load(file)
            if isinstance(obj, cls):
                return obj
            else:
                raise ValueError("Invalid file content. Unable to recreate the object.")

    def __init__(self, raw_data, variables: _Variables, save_path: str, config: Config, lazy_construction: bool,
                 load_preprocessed_data: bool, naming_prepend: str = '', use_compression=True, num_data_load=-1, device='cpu'):
        
        self.raw_data = raw_data
        self.num_data_load = num_data_load if num_data_load > 0 else len(self.raw_data)
        self.config: Config = config
        self.lazy_construction = lazy_construction
        self.load_preprocessed_data = load_preprocessed_data
        self.var = variables
        self.var.device = device
        self.device = device
        self.save_path = os.path.join(config.root, save_path)
        self.naming_prepend = naming_prepend
        self.use_compression = use_compression
        # self.node_attr, self.node_label, self.edge_index, self.edge_attr, self.edge_label = None, None, None, None, None
        # self.data = Data(x=self.node_attr, y=self.node_label, edge_index=self.edge_index)
        self._graphs: List = [None for r in raw_data]

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def to_graph(self, raw_data):
        pass

    # below method returns torch geometric Data model with indexed nodes from spacy vocab
    @abstractmethod
    def to_graph_indexed(self, raw_data):
        pass

    # below method gets graph loaded from indexed files and gives complete graph
    @abstractmethod
    def convert_indexed_nodes_to_vector_nodes(self , graph):
        pass

    def get_graph(self, idx: int):
        if self._graphs[idx] is None:
            if self.load_preprocessed_data:
                if self.use_compression:
                    self.load_data_compressed(idx)
                else:
                    self.load_data(idx)
            else:
                self._graphs[idx] = self.to_graph(self.raw_data[idx])
                self.var.graphs_name[idx] = f'{self.naming_prepend}_{idx}'
        return self._graphs[idx].to(self.device)

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
        return {idx: self._graphs[idx] for idx in ids}

    def get_first(self):
        return self.get_graph(0)

    def save_all_data(self):
        for i in range(len(self._graphs)):
            torch.save(self._graphs[i], path.join(self.save_path, f'{self.var.graphs_name[i]}.pt'))
        self.var.save_to_file(path.join(self.save_path, f'{self.naming_prepend}_var.txt'))

    def load_all_data(self):
        self.load_var()
        for i in range(self.var.graph_num):
            self._graphs[i] = torch.load(path.join(self.save_path, f'{self.var.graphs_name[i]}.pt'))

    def load_var(self):
        self.var = self.var.load_from_file(path.join(self.save_path, f'{self.naming_prepend}_var.txt'))
        self.var.device = self.device

    def load_data(self, idx: int):
        self._graphs[idx] = torch.load(path.join(self.save_path, f'{self.var.graphs_name[idx]}.pt'))
        self._graphs[idx].to(self.device)

    def load_data_list(self, ids: List | Tuple | range | np.array | torch.Tensor | any):
        if torch.max(torch.tensor(ids) >= self.var.graph_num) == 1:
            print(f'Index is out of range, indices should be more than 0 and less than {self.var.graph_num}')
            return

        for i in ids:
            self._graphs[i] = torch.load(path.join(self.save_path, f'{self.var.graphs_name[i]}.pt'))
            self._graphs[i].to(self.device)

    def draw_graph(self, idx: int):
        g = to_networkx(self.get_graph(idx), to_undirected=True)
        layout = nx.spring_layout(g)
        nx.draw(g, pos=layout)

    def save_all_data_compressed(self):
        for i in range(len(self._graphs)):
            graph = self.to_graph_indexed(self.raw_data[i])
            try:
                torch.save(graph.to('cpu'), path.join(self.save_path, f'{self.var.graphs_name[i]}_compressed.pt'))
            except AttributeError:
                torch.save(graph, path.join(self.save_path, f'{self.var.graphs_name[i]}_compressed.pt'))
        self.var.save_to_file(path.join(self.save_path, f'{self.naming_prepend}_var.txt'))


    def save_data_range(self, start: int, end: int):
        for i in range(start, end):
            graph = self.to_graph_indexed(self.raw_data[i])
            try:
                torch.save(graph.to('cpu'), path.join(self.save_path, f'{self.var.graphs_name[i]}_compressed.pt'))
            except AttributeError:
                torch.save(graph, path.join(self.save_path, f'{self.var.graphs_name[i]}_compressed.pt'))

    def load_all_data_comppressed(self):
        self.load_var()
        for i in range(self.var.graph_num):
            if i % 100 == 0 : 
                print(f'data loading {i}')
            self._graphs[i] = self.convert_indexed_nodes_to_vector_nodes(torch.load(path.join(self.save_path, f'{self.var.graphs_name[i]}_compressed.pt')))
            self._graphs[i].to(self.device)

    def load_data_range(self, start: int, end: int):
        for i in range(start, end):
            self._graphs[i] = self.convert_indexed_nodes_to_vector_nodes(torch.load(path.join(self.save_path, f'{self.var.graphs_name[i]}_compressed.pt')))
            self._graphs[i].to(self.device)

    def save_data_compressed(self , idx: int):
        graph = self.to_graph_indexed(self.raw_data[idx])
        try:
            torch.save(graph.to('cpu'), path.join(self.save_path, f'{self.var.graphs_name[idx]}_compressed.pt'))
        except AttributeError:
            torch.save(graph, path.join(self.save_path, f'{self.var.graphs_name[idx]}_compressed.pt'))

    def load_data_compressed(self , idx: int):
        basic_graph = torch.load(path.join(self.save_path, f'{self.var.graphs_name[idx]}_compressed.pt'))
        self._graphs[idx] = self.convert_indexed_nodes_to_vector_nodes(basic_graph)
        self._graphs[idx].to(self.device)