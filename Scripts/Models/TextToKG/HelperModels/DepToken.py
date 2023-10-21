import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn.parameter import Parameter, UninitializedParameter

# Its better to use  Pytorch Geometric

from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GCN2Conv, GAT
from Scripts.Models.Layers.GraphAttention import GraphAttentionLayer


class DepTokenEmbedding(nn.Module):

    def __init__(self, input_feature: int, dropout=0.1, *args, **kwargs):
        super(DepTokenEmbedding, self).__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            GCNConv(input_feature, 256),
            GCN2Conv(256, 0.5, 0.2, 2),
            GCNConv(256, 128),
            GCNConv(128, 64),
            GCNConv(64, 32),
            GATv2Conv(32, 32, 2, dropout=dropout),
            nn.ReLU()
            )

        self.decoder = nn.Sequential(
            GCNConv(64, 128),
            GCNConv(128, 256),
            GCN2Conv(256, 0.5, 0.2, 2),
            GCNConv(256, input_feature)
            )

    def forward(self, x, edge_indices):
        x = self.encoder(x, edge_indices)
        x = self.decoder(x, edge_indices)
        return F.sigmoid(x)
