
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, GCNConv, GCN2Conv, BatchNorm, MemPooling
from torch_geometric.nn import Sequential as GSequential
from torch import nn


class HomogeneousDeepGraphEmbedding2(nn.Module):
    r"""
    This class is for graph level classification or graph level regression
    """

    def __init__(self, input_feature: int, out_features: int, base_hidden_feature: int=256, dropout=0.1, *args, **kwargs):
        super(HomogeneousDeepGraphEmbedding2, self).__init__(*args, **kwargs)
        self.input_features = input_feature
        self.num_out_features = out_features
        self.bsh: int = base_hidden_feature
        bsh2: int = int(self.bsh/2)
        
        self.attention = GSequential('x3, x4, edge_index, edge_weights', [
            (GATv2Conv(input_feature, int(input_feature/2), 2, dropout=dropout), 'x4, edge_index ->x4'),
            (BatchNorm(input_feature), 'x4->x4'),
            (nn.ReLU(), 'x4->x4'),
            
            (GCN2Conv(input_feature, 0.5, 0.1, 2), 'x4, x3, edge_index, edge_weights->x3'),
            (BatchNorm(input_feature), 'x3->x3'),
            (nn.ReLU(), 'x3->x3'),
            (GCNConv(input_feature, self.bsh), 'x3, edge_index, edge_weights -> x3'),
            (BatchNorm(self.bsh), 'x3->x3'),
            (nn.ReLU(), 'x3->x3'),
            
            (GATv2Conv(self.bsh, bsh2, 2, dropout=dropout), 'x3, edge_index ->x3'),
            (BatchNorm(self.bsh), 'x3->x3'),
            (nn.ReLU(), 'x3->x3'),
            (lambda x3, x4: (x3, x4), 'x3, x4 -> x3, x4')
        ])
        
        self.mem_pool = MemPooling(self.bsh, bsh2, 4, 2)
        self.output_layer = Linear(self.bsh, self.num_out_features)

    def forward(self, x):
        x_att, x4 = self.attention(x.x, x.x, x.edge_index, x.edge_attr)
        
        x_pooled, S = self.mem_pool(x_att, x.batch)
        x_pooled = x_pooled.view(x_pooled.shape[0], -1)
        
        return self.output_layer(x_pooled)