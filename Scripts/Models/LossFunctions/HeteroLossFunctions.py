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
    
    def forward(self, out_pred, out_main):
        loss = self.bce_loss(out_pred.y, out_main.y)
        x_dict_keys = [k for k in out_pred.x_dict.keys() if k not in self.exception_keys]
        
        for key in x_dict_keys:
            tensor1 = out_pred.x_dict[key]
            tensor2 = out_main.x_dict[key]
            if tensor2.ndim == 1 and tensor2.dtype is torch.long:
                tensor2 = torch.nn.functional.one_hot(input=tensor2.to(torch.long), num_classes=tensor1.shape[1]).to(torch.float32)
            loss += self.enc_factor * self.mse_loss(tensor1, tensor2)
        return loss
    