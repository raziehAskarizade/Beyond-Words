import torch
import numpy as np
import pandas as pd


def extract_sub_graph(graph_handler):
    num_sub_graph_edges = 100
    random_indices = torch.randint(0, graph_handler.edge_count, (num_sub_graph_edges,))
    edges: np.array = graph_handler.get_edges(random_indices).numpy()
    node_indices = np.unique(np.reshape(edges, -1))
    node_map = pd.DataFrame(data=list(range(len(node_indices))), index=node_indices, dtype=int)
    nodes = torch.Tensor(graph_handler.get_nodes(node_indices))
    edges[0] = np.squeeze(node_map.loc[edges[0]])
    edges[1] = np.squeeze(node_map.loc[edges[1]])
    edges = torch.Tensor(edges).int()
    return nodes, edges, node_indices, node_map
