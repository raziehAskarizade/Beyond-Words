from typing import List
import torch
from torch import nn


class HeteroLossArgs:
    def __init__(self, y, x_dict):
        self.y = y
        self.x_dict = x_dict
        
class HeteroLoss1(torch.nn.Module):
    def __init__(self, exception_keys: List[str], enc_factor=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bce_loss=  nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.exception_keys = exception_keys
        self.enc_factor = enc_factor
    
    def forward(self, out_pred, out_main, target_query_emb, main_query_emb):
        pred_y = self.location_sigmoid(out_pred.y)
        main_y = self.location_sigmoid(out_main.y)
        
        classification_loss = self.mse_loss(pred_y, main_y)
        query_loss = self.similarity(target_query_emb, main_query_emb)
        loss = classification_loss * query_loss
        
        enc_loss = 0
        if self.enc_factor > 0:
            x_dict_keys = [k for k in out_pred.x_dict.keys() if k not in self.exception_keys]
            for key in x_dict_keys:
                tensor1 = out_pred.x_dict[key]
                tensor2 = out_main.x_dict[key]
                if tensor2.ndim == 1 and tensor2.dtype is torch.long:
                    tensor2 = torch.nn.functional.one_hot(input=tensor2.to(torch.long), num_classes=tensor1.shape[1]).to(torch.float32)
                std1, mean1 = torch.std_mean(tensor1, dim=1)
                std2, mean2 = torch.std_mean(tensor2, dim=1)
                loss += self.mse_loss(mean1, mean2) + self.mse_loss(std1, std2)
            
        return loss + self.enc_factor * enc_loss
    
    def similarity(self, query_i, query_j):
        return torch.cosine_similarity(query_i, query_j)
    
    def location_sigmoid(self, pred_rank):
        v = torch.exp(-0.5*(pred_rank-5))
        return 2*torch.tanh(0.5*v)-1
    