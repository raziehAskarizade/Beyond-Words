
import torch.nn.functional as F
from torch import Tensor
import torch
from torch.nn import Linear, ModuleDict, ModuleList
from torch_geometric.nn import BatchNorm, MemPooling
from torch_geometric.data import HeteroData
from typing import Dict
from Scripts.Models.GraphNodeEmbedding.HeteroDeepGraphNodeEmbedding1 import HeteroDeepGraphNodeEmbedding1


class HeteroMempool2(torch.nn.Module):
    def __init__(self, 
                 metadata,
                 out_features: int,
                 hidden_feature: int=256,
                 pooling_node_types: Dict[str, int] = {'word':300},
                 num_clusters: int = 1,
                 node_embedding_layer: int= 4,
                 ):
        
        super(HeteroMempool2, self).__init__()
        self.num_out_features = out_features
        self.bsh: int = hidden_feature
        self.node_embedding_layer: int = node_embedding_layer
        
        self.pooling_node_types: Dict[str, int] = pooling_node_types
        
        self.hetero_models = ModuleList()
        self.hetero_models.add_module(f'DeepGraphNodeEmbedding1_{0}', 
                                        HeteroDeepGraphNodeEmbedding1(300, 128, metadata, 128, dropout=0.2))
        for i in range(1, self.node_embedding_layer):
            self.hetero_models.add_module(f'DeepGraphNodeEmbedding1_{i}', 
                                            HeteroDeepGraphNodeEmbedding1(128, 128, metadata, 128, dropout=0.2))
            
        
        self.num_clusters = num_clusters
        self.mem_pools = ModuleDict()
        for key in self.pooling_node_types:
            self.mem_pools[key] = MemPooling(self.pooling_node_types[key], self.bsh, 2, self.num_clusters)
        # self.mem_pool = MemPooling(self.bsh, self.bsh, 2, self.num_clusters)
                
        self.linear_1 = Linear(self.num_clusters*self.bsh, 64)
        self.linear_2 = Linear(len(self.pooling_node_types)*64, 64)
        self.linear_3 = Linear(64, 64)
        self.linear_4 = Linear(64, 64)
        self.batch_norm_1 = BatchNorm(64)
        
        self.output_layer = Linear(64, self.num_out_features)

    def forward(self, x: HeteroData) -> Tensor:
        for i in range(self.node_embedding_layer):
            x_out_gat, x_out_encoder = self.hetero_model(x)
            x.x_dict = x_out_gat
            
        x_pools = []
        for key in self.pooling_node_types:
            x_pooled, S = self.mem_pools[key](x_out_gat[key], x[key].batch)
            x_pooled = x_pooled.view(x_pooled.shape[0], -1)
            x_pooled = F.relu(self.linear_1(x_pooled))
            x_pools.append(x_pooled)
        x_pooled = torch.concat(x_pools, dim=1)
        x_pooled = F.relu(self.linear_2(x_pooled))
        x_pooled = F.relu(self.batch_norm_1(self.linear_3(x_pooled)))
        x_pooled = F.relu(self.linear_4(x_pooled))
        out = self.output_layer(x_pooled)
        return out, x_out_encoder
    