# Fardin Rastakhiz @ 2023

import torch.nn.functional as F
from torch import Tensor
import torch
from torch_geometric.nn import to_hetero, PairNorm
from torch_geometric.data import HeteroData
from Scripts.Models.BaseModels.HeteroGat import HeteroGat
from Scripts.Models.BaseModels.HeteroLinear import HeteroLinear


class HeteroDeepGraphNodeEmbedding1(torch.nn.Module):
    def __init__(self,
                 input_feature: int,
                 output_feature: int,
                 metadata,
                 hidden_feature: int=256,
                 dropout=0.1,
                 has_dep=True,
                 has_tag=True,
                 has_sentence=True,
                 edge_type_count=9):

        super(HeteroDeepGraphNodeEmbedding1, self).__init__()
        self.input_features: int = input_feature
        self.hidden_feature: int = hidden_feature
        self.output_feature: int = output_feature
        self.has_dep = has_dep
        self.has_tag = has_tag
        self.has_sentence = has_sentence
        self.edge_type_count = edge_type_count

        self.part_weight_norm = torch.nn.LayerNorm((self.edge_type_count,))
        self.norm = PairNorm()
        self.drop = torch.nn.Dropout(0.2)
        self.hetero_linear_1 = to_hetero(HeteroLinear(self.input_features, self.hidden_feature, dropout), metadata)
        
        self.hetero_gat_1 = to_hetero(HeteroGat(self.hidden_feature, self.hidden_feature, dropout, num_heads=2), metadata)
        self.hetero_gat_2 = to_hetero(HeteroGat(self.hidden_feature, self.hidden_feature, dropout, num_heads=2), metadata)
        self.hetero_gat_3 = to_hetero(HeteroGat(self.hidden_feature, self.output_feature, dropout, num_heads=2), metadata)
        
        self.hetero_linear_2 = to_hetero(HeteroLinear(self.output_feature, self.output_feature, dropout, use_batch_norm=True), metadata)
        self.hetero_linear_3 = to_hetero(HeteroLinear(self.output_feature, self.output_feature, dropout, use_dropout=False), metadata)
        
        
        self.dep_embedding = torch.nn.Embedding(45, self.input_features)
        self.tag_embedding = torch.nn.Embedding(50, self.input_features)
        self.dep_unembedding = torch.nn.Linear(self.output_feature, 45)
        self.tag_unembedding = torch.nn.Linear(self.output_feature, 50)
        
        self.pw1 = torch.nn.Parameter(torch.randn([self.edge_type_count,], dtype=torch.float32), requires_grad=True)
        self.pw2 = torch.nn.Parameter(torch.randn([self.edge_type_count,], dtype=torch.float32), requires_grad=True)


    def forward(self, x: HeteroData) -> Tensor:
        x_dict, edge_attr_dict, edge_index_dict = self.preprocess_data(x)
        edge_attr_dict = self.update_weights(edge_attr_dict, self.pw1)
        
        x_dict = self.hetero_linear_1(x_dict)
        
        x_dict = self.hetero_gat_1(x_dict, edge_index_dict, edge_attr_dict)
        # x_dict = self.normalize(x_dict, x)
        
        edge_attr_dict = self.update_weights(edge_attr_dict, self.pw2)
        
        x_dict = self.hetero_gat_2(x_dict, edge_index_dict, edge_attr_dict)
        # x_dict = self.normalize(x_dict, x)

        x_out_gat = self.hetero_gat_3(x_dict, edge_index_dict, edge_attr_dict)
                
        x_out_encoder = self.hetero_linear_2(x_out_gat)
        x_out_encoder = self.hetero_linear_3(x_out_encoder)
        
        if self.has_dep: 
            x_out_encoder['dep'] = self.dep_unembedding(x_out_encoder['dep'])
        if self.has_tag:
            x_out_encoder['tag'] = self.tag_unembedding(x_out_encoder['tag'])
        
        return x_out_gat, x_out_encoder

    def preprocess_data(self, x):
        x_dict = {key: x.x_dict[key] for key in x.x_dict}
        if self.has_dep and x_dict['dep'].ndim==1:
            x_dict['dep'] = self.dep_embedding(x_dict['dep'])
        if self.has_tag and x_dict['tag'].ndim==1:
            x_dict['tag'] = self.tag_embedding(x_dict['tag'])

        edge_attr_dict = x.edge_attr_dict
        edge_index_dict = x.edge_index_dict

        if self.has_sentence:
            shape1 = edge_index_dict[('sentence', 'sentence_word', 'word')].shape[1]
            shape2 = edge_attr_dict[('word', 'word_sentence', 'sentence')].shape[0]
            if shape1 != shape2:
                edge_attr_dict[('sentence', 'sentence_word', 'word')] = edge_attr_dict[('word', 'word_sentence', 'sentence')][shape1:shape2]
                edge_attr_dict[('word', 'word_sentence', 'sentence')] = edge_attr_dict[('word', 'word_sentence', 'sentence')][:shape1]

        for key in x.edge_attr_dict:
            edge_attr_dict[key] = self.get_scale_same(1.0, edge_attr_dict[key])

        return x_dict, edge_attr_dict, edge_index_dict

    def normalize(self, x_dict, x):
        for key in x_dict:
            x_dict[key] = self.norm(x_dict[key], x[key].batch)
        return x_dict

    def update_weights(self, edge_attr_dict, part_weights):
        self.part_weight_norm(part_weights)
        part_weights = F.relu(part_weights)
        for i, key in enumerate(edge_attr_dict):
            edge_attr = edge_attr_dict[key]
            if edge_attr is None or edge_attr == ('word', 'seq', 'word'):
                continue
            edge_attr_dict[key]= edge_attr * part_weights[i]
        return edge_attr_dict


    def get_scale_same(self, scale:float, attributes: Tensor):
        if attributes is None or len(attributes) == 0:
            return
        attributes = scale * torch.ones_like(attributes)
        return attributes