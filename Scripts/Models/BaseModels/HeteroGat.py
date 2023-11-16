

from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm

class HeteroGat(nn.Module):
    
    def __init__(self, in_feature, out_feature, dropout = 0.2, num_heads: int = 1) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(in_feature, int(out_feature/num_heads), heads=num_heads, edge_dim=1, add_self_loops=False)
        self.batch_norm = BatchNorm(out_feature)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weights: Tensor) -> Tensor:

        x = self.conv1(x, edge_index, edge_weights)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        return x