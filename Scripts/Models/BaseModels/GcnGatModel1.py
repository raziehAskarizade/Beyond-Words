
import torch
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, GCNConv, GCN2Conv, DenseGCNConv, dense_diff_pool, BatchNorm
from torch_geometric.nn import Sequential as GSequential
from torch import nn

class GcnGatModel1(nn.Module):
    r"""
    This class is for graph level classification or graph level regression
    """

    def __init__(self, input_feature: int, out_features: int, base_hidden_feature: int=256, dropout=0.1, *args, **kwargs):
        super(GcnGatModel1, self).__init__(*args, **kwargs)
        self.input_features = input_feature
        self.num_out_features = out_features
        self.bsh: int = base_hidden_feature
        bsh2: int = int(self.bsh/2)
        bsh4: int = int(self.bsh/4)
        bsh8: int = int(self.bsh/8)
        
        self.encoder = GSequential('x, edge_index, edge_weights', [
            (GCNConv(input_feature, self.bsh), 'x, edge_index, edge_weights ->x1'),
            (BatchNorm(self.bsh), 'x1->x1'),
            (nn.ReLU(), 'x1->x1'),
            (nn.Dropout(dropout), 'x1->x1'),
            (GCNConv(self.bsh, self.bsh), 'x1, edge_index, edge_weights ->x1'),
            (BatchNorm(self.bsh), 'x1->x1'),
            (nn.ReLU(), 'x1->x1'),
            (nn.Dropout(dropout), 'x1->x1'),
            (GCNConv(self.bsh, self.bsh), 'x1, edge_index, edge_weights ->x1'),
            (BatchNorm(self.bsh), 'x1->x1'),
            (nn.ReLU(), 'x1->x1'),
            (nn.Dropout(dropout), 'x1->x1'),
            
            (GCNConv(self.bsh, bsh2), 'x1, edge_index, edge_weights -> x2'),
            (BatchNorm(bsh2), 'x2->x2'),
            (nn.ReLU(), 'x2->x2'),
            (nn.Dropout(dropout), 'x2->x2'),
            (GCNConv(bsh2, bsh2), 'x2, edge_index, edge_weights -> x2'),
            (BatchNorm(bsh2), 'x2->x2'),
            (nn.ReLU(), 'x2->x2'),
            (nn.Dropout(dropout), 'x2->x2'),
            (GCNConv(bsh2, bsh2), 'x2, edge_index, edge_weights -> x2'),
            (BatchNorm(bsh2), 'x2->x2'),
            (nn.ReLU(), 'x2->x2'),
            (nn.Dropout(dropout), 'x2->x2'),
            
            (GCNConv(bsh2, bsh4), 'x2, edge_index, edge_weights -> x3'),
            (BatchNorm(bsh4), 'x3->x3'),
            (nn.ReLU(), 'x3->x3'),
            (nn.Dropout(dropout), 'x3->x3'),
            (GCNConv(bsh4, bsh4), 'x3, edge_index, edge_weights -> x3'),
            (BatchNorm(bsh4), 'x3->x3'),
            (nn.ReLU(), 'x3->x3'),
            (nn.Dropout(dropout), 'x3->x3'),
            (GCNConv(bsh4, bsh4), 'x3, edge_index, edge_weights -> x3'),
            (BatchNorm(bsh4), 'x3->x3'),
            (nn.ReLU(), 'x3->x3'),
            (nn.Dropout(dropout), 'x3->x3'),
            
            (GCNConv(bsh4, bsh8), 'x3, edge_index, edge_weights -> x4'),
            (BatchNorm(bsh8), 'x4->x4'),
            (nn.ReLU(), 'x4->x4'),
            (nn.Dropout(dropout), 'x4->x4'),
            (GCNConv(bsh8, bsh8), 'x4, edge_index, edge_weights -> x4'),
            (BatchNorm(bsh8), 'x4->x4'),
            (nn.ReLU(), 'x4->x4'),
            (nn.Dropout(dropout), 'x4->x4'),
            (GCNConv(bsh8, bsh8), 'x4, edge_index, edge_weights -> x4'),
            (BatchNorm(bsh8), 'x4->x4'),
            (nn.ReLU(), 'x4->x4'),
            (lambda x1, x2, x3, x4: (x1, x2, x3, x4), 'x1, x2, x3, x4 -> x1, x2, x3, x4')
        ])
        
        self.attention = GSequential('x3, x4, edge_index, edge_weights', [
            (GATv2Conv(bsh8, bsh8, 2, dropout=dropout), 'x4, edge_index ->x4'),
            (BatchNorm(bsh4), 'x4->x4'),
            (nn.ReLU(), 'x4->x4'),
            
            (GCN2Conv(bsh4, 0.5, 0.1, 2), 'x4, x3, edge_index, edge_weights->x3'),
            (BatchNorm(bsh4), 'x3->x3'),
            (nn.ReLU(), 'x3->x3'),
            (GCNConv(bsh4, bsh4), 'x3, edge_index, edge_weights -> x3'),
            (BatchNorm(bsh4), 'x3->x3'),
            (nn.ReLU(), 'x3->x3'),
            
            (GATv2Conv(bsh4, bsh4, 2, dropout=dropout), 'x3, edge_index ->x3'),
            (BatchNorm(bsh2), 'x3->x3'),
            (nn.ReLU(), 'x3->x3'),
            (lambda x3, x4: (x3, x4), 'x3, x4 -> x3, x4')
        ])
        
        self.decoder = GSequential('x1, x2, x3, edge_index, edge_weights', [
            
            (GCN2Conv(bsh2, 0.5, 0.1, 2), 'x3, x2, edge_index, edge_weights->x2'),
            (BatchNorm(bsh2), 'x2->x2'),
            (nn.ReLU(), 'x2->x2'),
            (nn.Dropout(dropout), 'x2->x2'),
            (GCNConv(bsh2, bsh2), 'x2, edge_index, edge_weights -> x2'),
            (BatchNorm(bsh2), 'x2->x2'),
            (nn.ReLU(), 'x2->x2'),
            (nn.Dropout(dropout), 'x2->x2'),
            (GCNConv(bsh2, self.bsh), 'x2, edge_index->x2'),
            (BatchNorm(self.bsh), 'x2->x2'),
            (nn.ReLU(), 'x2->x2'),
            (nn.Dropout(dropout), 'x2->x2'),
            
            (GCN2Conv(self.bsh, 0.5, 0.1, 2), 'x2, x1, edge_index, edge_weights->x1'),
            (BatchNorm(self.bsh), 'x1->x1'),
            (nn.ReLU(), 'x1->x1'),
            (nn.Dropout(dropout), 'x1->x1'),
            (GCNConv(self.bsh, self.bsh), 'x1, edge_index, edge_weights ->x1'),
            (BatchNorm(self.bsh), 'x1->x1'),
            (nn.ReLU(), 'x1->x1'),
            (nn.Dropout(dropout), 'x1->x1'),
            (GCNConv(self.bsh, self.bsh), 'x1, edge_index, edge_weights ->x1'),
            (BatchNorm(self.bsh), 'x1->x1'),
            (nn.ReLU(), 'x1->x1'),
            (nn.Dropout(dropout), 'x1->x1'),
            (lambda x1, x2, x3: (x1, x2, x3), 'x1, x2, x3 -> x, x2, x3')
        ])

        self.pooling_layer1 = GCNConv(self.bsh, 5)
        self.pooling_layer2 = DenseGCNConv(self.bsh, 1)
        self.output_layer = Linear(self.bsh, self.num_out_features)

    def forward(self, x):
        x1, x2, x3, x_enc = self.encoder(x.x, x.edge_index, x.edge_attr)
        x_att, x4 = self.attention(x3, x_enc, x.edge_index, x.edge_attr)
        x_dec, x2, x3 = self.decoder(x1, x2, x_att, x.edge_index, x.edge_attr)
        
        all_s = self.pooling_layer1(x_dec, x.edge_index, x.edge_attr)
        
        all_x = [x[i] for i in range(len(x))]
        ci = [all_x[i].x.shape[0] for i in range(len(x))]
        # ci = torch.tensor([x[i].x.shape[0] for i in range(len(x))], dtype=torch.int, device=x_dec.device).cumsum(0, dtype=torch.int)
        x_list = torch.split(x_dec, ci)
        # x_list = [x_dec[0 if i == 0 else ci[i - 1]:ci[i]] for i in range(len(ci))]
        s_i = torch.split(all_s, ci)
        # s_i = [all_s[0 if i == 0 else ci[i - 1]:ci[i]] for i in range(len(ci))]
        x_pooled = torch.zeros((len(x_list), self.bsh), dtype=x_dec.dtype, device=x_dec.device)
        
        for i in range(len(ci)):
            s = s_i[i]# self.pooling_layer1(x2[i], x[i].edge_index, x[i].edge_attr)
            adj = torch.zeros((ci[i], ci[i]), device=x_dec.device)
            # adj = torch.zeros((x[i].x.shape[0], x[i].x.shape[0]), device=x_dec.device)
            adj[all_x[i].edge_index[0], all_x[i].edge_index[1]] = x[i].edge_attr
            # adj = to_dense_adj(edge_index=x[i].edge_index, max_num_nodes=x[i].x.shape[0], edge_attr=x[i].edge_attr)
            nodes, adj, _, _ = dense_diff_pool(x_list[i], adj, s=s)
            s = self.pooling_layer2(nodes, adj)
            nodes, _, _, _ = dense_diff_pool(nodes, adj, s=s)
            x_pooled[i] = torch.squeeze(nodes)

        # return x1
        return self.output_layer(x_pooled)