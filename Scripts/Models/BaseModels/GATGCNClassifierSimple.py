
from torch_geometric.nn import GATv2Conv, GCNConv, GCN2Conv
from torch_geometric.nn import Sequential as GSequential
import torch.nn.functional as F
from torch import nn


class GNNClassifier(nn.Module):

    def __init__(self, input_feature: int, class_counts: int, dropout=0.1, *args, **kwargs):
        super(GNNClassifier, self).__init__(*args, **kwargs)
        self.input_features = input_feature
        self.num_classes = class_counts
        self.encoder = GSequential('x, edge_index', [
            (GCNConv(input_feature, 256), 'x, edge_index->x1'),
            (nn.ReLU(), 'x1->x1'),
            (GCNConv(256, 128), 'x1, edge_index -> x2'),
            (nn.ReLU(), 'x2->x2'),
            (GCNConv(128, 64), 'x2, edge_index -> x3'),
            (nn.ReLU(), 'x3->x3'),
            (GCNConv(64, 32), 'x3, edge_index -> x3'),
            (nn.ReLU(), 'x3->x3'),
            (GATv2Conv(32, 32, 4, dropout=dropout), 'x3, edge_index->x3'),
            # (GATv2Conv(128, 64, 2, dropout=dropout), 'x2, edge_index->x2'),
            (nn.ReLU(), 'x3->x3'),
            (GCN2Conv(128, 0.5, 0.1, 2), 'x3, x2, edge_index->x3'),
            (nn.ReLU(), 'x3->x3'),
            (GCNConv(128, 256), 'x3, edge_index->x3'),
            (nn.ReLU(), 'x3->x3'),
            (GCN2Conv(256, 0.5, 0.1, 2), 'x3, x1, edge_index->x3'),
            (nn.ReLU(), 'x3->x3')
        ])
        self.classifier = GCNConv(256, class_counts)

        for module in self.children():
            print(module.named_modules())

    def forward(self, x, edge_indices):
        x = self.encoder(x, edge_indices)
        x = self.classifier(x, edge_indices)
        return F.softmax(x, dim=1)

