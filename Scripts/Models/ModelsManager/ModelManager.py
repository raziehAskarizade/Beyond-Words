from abc import ABC, abstractmethod
from builtins import *
from typing import Tuple, Dict, Callable

from Scripts.Utils.enums import Optimizer, LossType
import torch
from torch import nn
from Scripts.Models.ModelsManager.Helpers.optimizer_factory import *
from Scripts.Models.ModelsManager.Helpers.loss_function_factory import *


class ModelManager(ABC):

    def __init__(self, optimizer_type: Optimizer, loss_type: LossType,
                 device=torch.device('cpu'), lr=0.01, weight_decay=0.001):
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

        self._optimizer_setters: Dict[Optimizer | int | str, Callable] = {
            Optimizer.ADAM: create_adam_optimizer,
            Optimizer.SGD: create_sgd_optimizer,
            Optimizer.RMS_PROP: create_rms_prop_optimizer
        }

        self._loss_functions: Dict[LossType | int | str, Callable] = {
            LossType.MSE: create_mse_loss,
            LossType.BCE: create_bce_loss,
            LossType.CROSS_ENTROPY: create_ce_loss
        }

        self.model, self.optimizer, self.loss_func = None, None, None
        self._create_model(lr, optimizer_type, loss_type)

        self.history = dict()

    @abstractmethod
    def _create_model(self, lr, optimizer_type, loss_type, **kwargs):
        pass

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

    def set_optimizer(self, lr, optimizer=Optimizer.ADAM, **kwargs):
        if optimizer in self._optimizer_setters:
            self.optimizer = self._optimizer_setters[optimizer](lr, self.model, **kwargs)
        else:
            print('Target optimizer is not implemented or is not added to the dictionary!')

    def set_loss_func(self, loss_type=LossType.MSE, **kwargs):
        if loss_type in self._loss_functions:
            self.loss_func = self._loss_functions[loss_type](**kwargs)
        else:
            print('Target optimizer is not implemented or is not added to the dictionary!')

    @abstractmethod
    def train(self, epoch_num: int = 100, lr: float = None, l2_norm: float = None, optimizer: Optimizer = None):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self, node_x, edge_index):
        pass
