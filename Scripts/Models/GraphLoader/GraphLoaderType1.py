
from Scripts.Models.GraphLoader.GraphLoader import NodeLabeledGraphLoader
import torch
from sklearn.model_selection import train_test_split


class GraphLoaderType1(NodeLabeledGraphLoader):

    def __init__(self, nodes_x, nodes_y, edge_index, batch_size, device,
                 test_size=0.2, val_size=0.15, *args, **kwargs):
        self.device = device
        self.batch_size = batch_size
        self.num_features = nodes_x.shape[1]
        self.num_classes = len(torch.unique(nodes_y))
        super(GraphLoaderType1, self)\
            .__init__(nodes_x, nodes_y, edge_index, device, test_size, val_size, *args, **kwargs)

    def shuffle_train_test(self):
        self.train_indices, self.test_indices = train_test_split(
            torch.concat([self.train_indices, self.test_indices]), test_size=self._test_size)

    def get_train_data(self):

        return self.nodes[self.train_indices], self.node_labels[self.train_indices], self.edge_indices_train

    def get_test_data(self):
        return self.nodes[self.test_indices], self.node_labels[self.test_indices], self.edge_indices_test

    def get_val_data(self):
        return self.nodes[self.val_indices], self.node_labels[self.val_indices], self.edge_indices_val

    def update(self, nodes_x, edge_index, device, *args, **kwargs):
        pass

    def update_edge_index(self, edge_index):
        self.edge_index = edge_index

    def update_nodes(self, nodes_x):
        pass

    def update_node_labels(self, node_y):
        pass

