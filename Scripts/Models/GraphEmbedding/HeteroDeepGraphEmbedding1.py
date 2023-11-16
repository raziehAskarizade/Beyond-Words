
import torch.nn.functional as F
from torch import Tensor
import torch
from torch.nn import Linear
from torch_geometric.nn import BatchNorm, MemPooling, to_hetero, PairNorm
from torch_geometric.data import HeteroData
from Scripts.Models.BaseModels.HeteroGat import HeteroGat
from Scripts.Models.BaseModels.HeteroLinear import HeteroLinear


class HeteroDeepGraphEmbedding1(torch.nn.Module):
    
    def __init__(self,
                 input_feature: int, out_features: int,
                 metadata,
                 hidden_feature: int=256,
                 device = 'cpu',
                 dropout=0.1):

        super(HeteroDeepGraphEmbedding1, self).__init__()
        self.input_features = input_feature
        self.num_out_features = out_features
        self.bsh: int = hidden_feature

        self.part_weight_norm = torch.nn.LayerNorm((9,))
        self.norm = PairNorm()
        self.drop = torch.nn.Dropout(0.2)
        self.hetero_linear_1 = to_hetero(HeteroLinear(input_feature, self.bsh, dropout), metadata)
        
        self.hetero_gat_1 = to_hetero(HeteroGat(self.bsh, self.bsh, dropout, num_heads=2), metadata)
        self.hetero_gat_2 = to_hetero(HeteroGat(self.bsh, self.bsh, dropout, num_heads=2), metadata)
        self.hetero_gat_3 = to_hetero(HeteroGat(self.bsh, self.bsh, dropout, num_heads=2), metadata)
        
        self.hetero_linear_2 = to_hetero(HeteroLinear(self.bsh, input_feature, dropout, use_batch_norm=True), metadata)
        self.hetero_linear_3 = to_hetero(HeteroLinear(input_feature, input_feature, dropout, use_dropout=False), metadata)
        
        self.mem_pool = MemPooling(self.bsh, self.bsh, 2, 1)
        
        self.linear_1 = Linear(self.bsh, 64)
        self.linear_2 = Linear(64, 64)
        self.linear_3 = Linear(64, 64)
        self.batch_norm_1 = BatchNorm(64)
        
        self.output_layer = Linear(64, self.num_out_features)

        self.dep_embedding = torch.nn.Embedding(45, self.input_features)
        self.tag_embedding = torch.nn.Embedding(50, self.input_features)
        self.dep_unembedding = torch.nn.Linear(self.input_features, 45)
        self.tag_unembedding = torch.nn.Linear(self.input_features, 50)
        
        self.pw1 = torch.nn.Parameter(torch.randn([9,], dtype=torch.float32), requires_grad=True)
        self.pw2 = torch.nn.Parameter(torch.randn([9,], dtype=torch.float32), requires_grad=True)

    def forward(self, x: HeteroData) -> Tensor:
        x_dict, edge_attr_dict, edge_index_dict = self.preprocess_data(x)
        edge_attr_dict = self.update_weights(edge_attr_dict, self.pw1)
        
        x_dict = self.hetero_linear_1(x_dict)
        
        x_dict = self.hetero_gat_1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = self.normalize(x_dict, x)
        
        edge_attr_dict = self.update_weights(edge_attr_dict, self.pw2)
        
        x_dict = self.hetero_gat_2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = self.normalize(x_dict, x)

        x_dict = self.hetero_gat_3(x_dict, edge_index_dict, edge_attr_dict)
        
        x_pooled, S = self.mem_pool(x_dict['word'], x['word'].batch)
        
        x_pooled = x_pooled.view(x_pooled.shape[0], -1)
        x_pooled = F.relu(self.linear_1(x_pooled))
        x_pooled = F.relu(self.batch_norm_1(self.linear_2(x_pooled)))
        x_pooled = F.relu(self.linear_3(x_pooled))
        out = self.output_layer(x_pooled)
        x_dict = self.hetero_linear_2(x_dict)
        x_dict = self.hetero_linear_3(x_dict)
        x_dict['dep'] = self.dep_unembedding(x_dict['dep'])
        x_dict['tag'] = self.tag_unembedding(x_dict['tag'])
        return out, x_dict

    def preprocess_data(self, x):
        x_dict = {key: x.x_dict[key] for key in x.x_dict}
        x_dict['dep'] = self.dep_embedding(x_dict['dep'])
        x_dict['tag'] = self.tag_embedding(x_dict['tag'])

        edge_attr_dict = x.edge_attr_dict
        edge_index_dict = x.edge_index_dict

        shape1 = edge_index_dict[('sentence', 'sentence_word', 'word')].shape[1]
        shape2 = edge_attr_dict[('word', 'word_sentence', 'sentence')].shape[0]
        if shape1 != shape2:
            edge_attr_dict[('sentence', 'sentence_word', 'word')] = edge_attr_dict[('word', 'word_sentence', 'sentence')][shape1:shape2]
            edge_attr_dict[('word', 'word_sentence', 'sentence')] = edge_attr_dict[('word', 'word_sentence', 'sentence')][:shape1]

        for key in x.edge_attr_dict:
            edge_attr_dict[key] = self.get_scale_same(1.0, edge_attr_dict[key])

        return x_dict, edge_attr_dict, edge_index_dict

    def normalize(self, x_dict, x):
        # for key in x_dict:
        #     x_dict[key] = self.norm(x_dict[key], x[key].batch)
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