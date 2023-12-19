# Fardin Rastakhiz @ 2023

import torch
import torch.nn.functional as F
from torch_geometric.nn import summary
from tqdm import tqdm
from Scripts.Models.LightningModels.LightningModels import BaseLightningModel
from Scripts.Models.ModelsManager.ModelManager import ModelManager
from Scripts.Models.BaseModels.GATGCNClassifierSimple import GNNClassifier
from Scripts.DataManager.GraphLoader.NLabeledGraphLoader import NLabeledGraphLoader
from Scripts.Utils.enums import Optimizer, LossType

import lightning as L


class SimpleNodeClassifierModelManager(ModelManager):

    def __init__(self, graph_handler: NLabeledGraphLoader, device=torch.device('cpu'),
                 lr=0.01, weight_decay=0.001, optimizer_type: Optimizer = Optimizer.ADAM,
                 loss_type: LossType = LossType.CROSS_ENTROPY):
        super(SimpleNodeClassifierModelManager, self).__init__(lr, weight_decay, device)
        self.graph_handler = graph_handler
        self.num_output_classes = self.graph_handler.num_classes
        self.num_input_features = self.graph_handler.num_features
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.model, self.optimizer, self.loss_func = self._create_model(lr, weight_decay, optimizer_type, loss_type)
        self.lightning_model = BaseLightningModel(self.model, self.optimizer, self.loss_func)

    def train(self, epoch_num: int = 100, lr: float = None, l2_norm: float = None, optimizer: Optimizer = None):

        trainer = L.Trainer(max_epochs=100, accelerator='gpu', devices=1)
        trainer.fit(self.lightning_model,
                    train_dataloaders=self.graph_handler.train_dataloader(),
                    val_dataloaders=self.graph_handler.val_dataloader()
                    )



        if lr or l2_norm or optimizer:
            self.set_optimizer(lr, l2_norm, optimizer)

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        train_node_x, train_node_y, train_edges = self.graph_handler.train_dataloader()
        test_node_x, test_node_y, test_edges = self.graph_handler.test_dataloader()
        for i in tqdm(range(epoch_num)):
            self.model.train()
            self.optimizer.zero_grad()
            my_node, my_label, my_edges = self.graph_handler.extract_random_sub_edges_graph()
            y_hat: torch.Tensor = self.model(my_node, my_edges)
            loss = self.loss_func(F.one_hot(my_label, self.num_output_classes).float(), y_hat)
            loss.backward()
            self.optimizer.step()

            self.model.eval()

            loss_value = loss.item()
            train_losses.append(loss_value)
            accuracy = (torch.sum((y_hat.argmax(dim=1) == my_label).float()) / len(my_label)).cpu().numpy()
            train_accuracies.append(accuracy)

            my_node, my_label, my_edges = self.graph_handler.extract_random_sub_edges_graph()
            y_hat: torch.Tensor = self.model(my_node, my_edges)
            loss = self.loss_func(F.one_hot(my_label, self.graph_handler.num_classes).float(), y_hat)
            loss_value = loss.item()
            test_losses.append(loss_value)
            accuracy = (torch.sum((y_hat.argmax(dim=1) == my_label).float()) / len(my_label)).cpu().numpy()
            test_accuracies.append(accuracy)
        self.history['train_losses'] = train_losses
        self.history['train_accuracies'] = train_accuracies
        self.history['test_losses'] = test_losses
        self.history['test_accuracies'] = test_accuracies
        return self.history

    def evaluate(self):
        node_x, node_y, edges = self.graph_handler.val_dataloader()
        self.model.eval()
        y_hat: torch.Tensor = self.model(node_x, edges)
        loss = self.loss_func(F.one_hot(node_y, self.num_output_classes).float(), y_hat)
        return y_hat, loss

    def predict(self, node_x, edge_index):
        self.model.eval()
        y_hat: torch.Tensor = self.model(node_x, edge_index)
        return y_hat

    def draw_summary(self):
        nodes_x, nodes_y, edge_indices_test = self.graph_handler.test_dataloader()
        nodes_x, nodes_y, edge_indices_test = self.graph_handler.extract_random_sub_edges_graph(2)
        print(summary(self.model, nodes_x, edge_indices_test))

    def _create_model(self, lr, l2_norm, optimizer_type, loss_type):
        model = GNNClassifier(input_feature=self.num_input_features, class_counts=self.num_output_classes)
        optimizer = ModelManager._create_optimizer(model, lr, l2_norm, optimizer_type)
        loss_func = ModelManager._create_loss_func(loss_type)
        model.to(self.device)
        return model, optimizer, loss_func

    def set_optimizer(self, lr, l2_norm, optimizer=Optimizer.ADAM):
        if lr:
            self.lr = lr
        if l2_norm:
            self.l2_norm = l2_norm
        if optimizer:
            self.optimizer_type = optimizer
        self.optimizer = ModelManager._create_optimizer(self.model, self.lr, self.l2_norm, self.optimizer_type)
