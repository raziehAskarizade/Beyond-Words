
from torch_geometric.nn import GATv2Conv, GCNConv, GCN2Conv
from torch_geometric.nn import Sequential as GSequential
from torch import nn


class GraphAutoEncoderModel(nn.Module):

    def __init__(self, input_feature: int, out_features: int, dropout=0.1, *args, **kwargs):
        super(GraphAutoEncoderModel, self).__init__(*args, **kwargs)
        self.input_features = input_feature
        self.num_out_features = out_features
        self.encoder = GSequential('node, edge_index, edge_weights', [
            (GCNConv(input_feature, 256), 'x, edge_index, edge_weights ->x1, edge_weights'),
            (nn.ReLU(), 'x1->x1'),
            (GCNConv(256, 128), 'x1, edge_index, edge_weights -> x2, edge_weights'),
            (nn.ReLU(), 'x2->x2'),
            (GCNConv(128, 64), 'x2, edge_index, edge_weights -> x3, edge_weights'),
            (nn.ReLU(), 'x3->x3'),
            (GCNConv(64, 32), 'x3, edge_index, edge_weights -> x3, edge_weights'),
            (nn.ReLU(), 'x3->x3'),
            (GATv2Conv(32, 32, 4, dropout=dropout), 'x3, edge_index, edge_weights->x3, edge_weights'),
            # (GATv2Conv(128, 64, 2, dropout=dropout), 'x2, edge_index->x2'),
            (nn.ReLU(), 'x3->x3'),
            (GCN2Conv(128, 0.5, 0.1, 2), 'x3, x2, edge_index, edge_weights->x3, edge_weights'),
            (nn.ReLU(), 'x3->x3'),
            (GCNConv(128, 256), 'x3, edge_index->x3'),
            (nn.ReLU(), 'x3->x3'),
            (GCN2Conv(256, 0.5, 0.1, 2), 'x3, x1, edge_index, edge_weights->x3, edge_weights'),
            (nn.ReLU(), 'x3->x3')
        ])

        self.output_layer = GCNConv(256, self.num_out_features)

        for module in self.children():
            print(module.named_modules())

    def forward(self, x, edge_indices):
        x = self.encoder(x, edge_indices)
        return self.output_layer(x, edge_indices)

