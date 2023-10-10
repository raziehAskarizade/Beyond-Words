import copy

from GraphAugmentor import GraphAugmentor
from Scripts.Models.Helpers.GraphLoader.GraphLoader import GraphLoader
import torch
import pandas as pd
import numpy as np


class ReduceEdgeGraphAugmentor(GraphAugmentor):

    def __init__(self, reduce_ratio: float, training: bool = True, inplace: bool = False):
        super(ReduceEdgeGraphAugmentor, self).__init__('ReduceEdgeGraphAugmentor', inplace)
        self._reduce_ratio = reduce_ratio
        self._training = training

    def augment(self, graph_loader: GraphLoader):
        augmented_graph_loader = graph_loader if self.inplace else copy.copy(graph_loader)
        new_edge_index = self.extract_random_sub_edges_graph(augmented_graph_loader.edge_count)
        augmented_graph_loader.update_edge_index(new_edge_index[0])


    def extract_random_sub_edges_graph(self, edge_count, edge_index):
        random_indices = torch.randint(0, self.edge_index.shape[1], (edge_count,), device=self.device)
        my_new_edges = self.edge_index[:, random_indices]
        node_indices = torch.unique(torch.reshape(my_new_edges, (-1,)))
        my_new_nodes = self.nodes[node_indices]
        my_new_labels = self.node_labels[node_indices]
        map_values = torch.tensor((range(len(node_indices))))
        node_map = pd.DataFrame(map_values, node_indices.cpu().numpy(), dtype=int)
        cpu_edges = my_new_edges.cpu()
        my_new_edges[0] = torch.tensor(np.squeeze(node_map.loc[cpu_edges[0]].values), device=self.device)
        my_new_edges[1] = torch.tensor(np.squeeze(node_map.loc[cpu_edges[1]].values), device=self.device)
        return my_new_nodes, my_new_labels, my_new_edges