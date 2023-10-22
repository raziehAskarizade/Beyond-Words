from abc import ABC, abstractmethod
from Scripts.Helpers.enums import Optimizer, LossType
import torch
from torch import nn


class ModelManager(ABC):

    def __init__(self, device=torch.device('cpu'), lr=0.01, l2_norm=0.001):
        self.device = device
        self.lr = lr
        self.l2_norm = l2_norm
        self.history = dict()

    @abstractmethod
    def train(self, epoch_num: int = 100, lr: float = None, l2_norm: float = None, optimizer: Optimizer = None):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self, node_x, edge_index):
        pass

    @staticmethod
    def _create_optimizer(model, lr, l2_norm, optimizer=Optimizer.ADAM):
        if optimizer == Optimizer.ADAM:
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_norm)
        elif optimizer == Optimizer.SGD:
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_norm)
        elif optimizer == Optimizer.RMS_PROP:
            return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_norm)
        elif optimizer == Optimizer.ADA_GRAD:
            return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2_norm)
        elif optimizer == Optimizer.ADA_MAX:
            return torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=l2_norm)
        else:
            raise NotImplementedError("The optimizer type is not implemented!")

    @staticmethod
    def _create_loss_func(loss_type):
        if loss_type == LossType.CROSS_ENTROPY:
            return nn.CrossEntropyLoss()
        elif loss_type == LossType.BCE:
            return nn.BCELoss()
        elif loss_type == LossType.MSE:
            return nn.MSELoss()
        else:
            raise NotImplementedError("The optimizer type is not implemented!")
