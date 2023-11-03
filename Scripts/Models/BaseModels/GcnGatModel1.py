import torch
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, GCNConv, GCN2Conv, DenseGCNConv, dense_diff_pool
from torch_geometric.nn import Sequential as GSequential
from torch_geometric.utils import to_dense_adj
from torch import nn

class GcnGatModel1(nn.Module):
    r"""
    This class is for graph level classification or graph level regression
    """

    def __init__(self, input_feature: int, out_features: int, dropout=0.1, *args, **kwargs):
        super(GcnGatModel1, self).__init__(*args, **kwargs)
        self.input_features = input_feature
        self.num_out_features = out_features
        self.encoder = GSequential('x, edge_index, edge_weights', [
            (GCNConv(input_feature, 256), 'x, edge_index, edge_weights ->x1'),
            (nn.ReLU(), 'x1->x1'),
            (GCNConv(256, 128), 'x1, edge_index, edge_weights -> x2'),
            (nn.ReLU(), 'x2->x2'),
            (GCNConv(128, 64), 'x2, edge_index, edge_weights -> x3'),
            (nn.ReLU(), 'x3->x3'),
            (GCNConv(64, 32), 'x3, edge_index, edge_weights -> x3'),
            (nn.ReLU(), 'x3->x3'),
            (GATv2Conv(32, 32, 4, dropout=dropout), 'x3, edge_index ->x3'),
            # (GATv2Conv(128, 64, 2, dropout=dropout), 'x2, edge_index->x2'),
            (nn.ReLU(), 'x3->x3'),
            (GCN2Conv(128, 0.5, 0.1, 2), 'x3, x2, edge_index, edge_weights->x3'),
            (nn.ReLU(), 'x3->x3'),
            (GCNConv(128, 256), 'x3, edge_index->x3'),
            (nn.ReLU(), 'x3->x3'),
            (GCN2Conv(256, 0.5, 0.1, 2), 'x3, x1, edge_index, edge_weights->x3'),
            (nn.ReLU(), 'x3->x3')
        ])

        self.pooling_layer1 = GCNConv(256, 5)
        self.pooling_layer2 = DenseGCNConv(256, 1)
        self.output_layer = Linear(256, self.num_out_features)

    def forward(self, x):
        x1 = self.encoder(x.x, x.edge_index, x.edge_attr)
        ci = torch.tensor([x[i].x.shape[0] for i in range(len(x))], dtype=torch.int, device=x1.device).cumsum(0, dtype=torch.int)
        x2 = [x1[0 if i == 0 else ci[i - 1]:ci[i]] for i in range(len(ci))]
        x3 = torch.zeros((len(x2), 256), dtype=x1.dtype, device=x1.device)
        for i in range(len(ci)):
            s = self.pooling_layer1(x2[i], x[i].edge_index, x[i].edge_attr)
            adj = to_dense_adj(edge_index=x[i].edge_index, max_num_nodes=x[i].x.shape[0], edge_attr=x[i].edge_attr)
            nodes, adj, _, _ = dense_diff_pool(x2[i], adj, s=s)
            s = self.pooling_layer2(nodes, adj)
            nodes, _, _, _ = dense_diff_pool(nodes, adj, s=s)
            x3[i] = torch.squeeze(nodes)

        return self.output_layer(x3)