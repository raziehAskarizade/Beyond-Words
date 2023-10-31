from typing import Tuple

import torch

from Scripts.DataManager.GraphLoader.AmazonReviewGraphLoader import AmazonReviewGraphLoader
from Scripts.Models.LightningModels.LightningModels import BinaryLightningModel
from Scripts.Models.ModelsManager.ModelManager import ModelManager
from Scripts.Utils.enums import Optimizer, LossType
from Scripts.Models.BaseModels.GcnGatModel1 import GcnGatModel1


class SimpleGraphClassifierModelManager(ModelManager):

    def __init__(self, graph_handler: AmazonReviewGraphLoader, optimizer_type: Optimizer = Optimizer.ADAM,
                 loss_type: LossType = LossType.CROSS_ENTROPY, device=torch.device('cpu'), lr=0.01, weight_decay=0.001):
        super(SimpleGraphClassifierModelManager, self).__init__(optimizer_type, loss_type, device, lr, weight_decay)
        self.graph_handler: AmazonReviewGraphLoader = graph_handler
        self.lightning_model = BinaryLightningModel(self.model, self.optimizer, self.loss_func, lr=lr)
        
    def _create_model(self, lr, optimizer_type, loss_type, **kwargs):
        self.model = GcnGatModel1(300, 1)
        self.set_optimizer(lr, optimizer_type, **kwargs)
        self.set_loss_func(loss_type, **kwargs)

    def train(self, epoch_num: int = 100, lr: float = None, l2_norm: float = None, optimizer: Optimizer = None):
        pass

    def evaluate(self):
        pass

    def predict(self, node_x, edge_index):
        pass

