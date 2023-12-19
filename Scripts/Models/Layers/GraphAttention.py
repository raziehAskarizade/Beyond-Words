# Fardin Rastakhiz @ 2023

import torch
from torch import nn
import torch.nn.functional as F
from Scripts.Utils.Node import Node


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 k_attention=4, dropout=0.0,
                 bias: bool = True, device=None, dtype=None):
        super(GraphAttentionLayer, self).__init__()
        self.k_attention = k_attention
        self.value_update = nn.Linear(in_features, out_features, bias, device, dtype)
        self.neighbor_update = nn.Linear(in_features, out_features, bias, device, dtype)
        self.fc2 = nn.Linear(2*out_features, out_features, bias, device, dtype)

        self.transform_layers = nn.ModuleList()
        self.attention_layer = nn.ModuleList()
        for i in range(self.k_attention):
            self.transform_layers.append(nn.Linear(in_features, out_features, bias, device, dtype))
            self.attention_layer.append(nn.Linear(2*out_features, out_features, bias, device, dtype))

    def forward(self, x_node: Node):
        x = [self.value_update(x_node.value)]
        x_a = [self.calculate_attention(x_node.value, x_node.value)]
        for neighbor in x_node.neighbors:
            x_a.append(self.calculate_attention(x_node.value, neighbor))
            x.append(self.neighbor_update(neighbor))
        x_a = torch.exp(torch.Tensor(x_a))
        x_a = x_a / torch.sum(x_a, dim=1)
        x = torch.sum(x_a*x, dim=1)
        return F.relu(x)

    def calculate_attention(self, value, neighbor):
        x_ai = []
        for i in range(self.k_attention):
            x_a_1 = self.transform_layer[i](value)
            x_a_2 = self.transform_layer[i](neighbor)
            x_a_3 = self.attention_layer[i](torch.concat([x_a_1, x_a_2], dim=1))
            x_a_3 = F.relu(x_a_3)
            x_ai.append(x_a_3)
        return torch.concat(x_ai)
