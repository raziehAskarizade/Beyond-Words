import copy

from GraphAugmentor import GraphAugmentor
from Scripts.Models.Helpers.GraphLoader.GraphLoader import GraphLoader
from torch_geometric.utils import augmentation
from typing import Tuple, Optional, Union


class IncreaseEdgeGraphAugmentor(GraphAugmentor):
    def __init__(self, add_ratio: float, undirected: bool, training: bool = True, inplace: bool = False):
        super(IncreaseEdgeGraphAugmentor, self).__init__('IncreaseEdgeGraphAugmentor', inplace)
        self._add_ratio = add_ratio
        self._undirected = undirected
        self._training = training

    def augment(self, graph_loader: GraphLoader):
        augmented_graph_loader = graph_loader if self.inplace else copy.copy(graph_loader)
        num_nodes = len(augmented_graph_loader.nodes)
        new_edge_index = augmentation.add_random_edge(graph_loader.edge_index, self._add_ratio,
                                                      self._undirected,  num_nodes, self._training)
        augmented_graph_loader.update_edge_index(new_edge_index[0])

