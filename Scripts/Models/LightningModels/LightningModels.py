from typing import Any
import torch
import torchmetrics
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import lightning as L
from abc import abstractmethod

class BaseLightningModel(L.LightningModule):

    def __init__(self, model, optimizer=None, loss_func=None, learning_rate=0.01, batch_size=64, lr_scheduler=None, user_lr_scheduler=False):
        super(BaseLightningModel, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = model
        # self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters("model", logger=False)
        self.optimizer = self._get_optimizer(optimizer)
        self.lr_scheduler = self._get_lr_scheduler(lr_scheduler) if user_lr_scheduler else None
        self.loss_func = self._get_loss_func(loss_func)

    def forward(self, data_batch, *args, **kwargs):
        return self.model(data_batch)

    def on_train_epoch_start(self) -> None:
        param_groups = next(iter(self.optimizer.param_groups))
        if 'lr' in param_groups and param_groups['lr'] is not None:
            current_learning_rate = float(param_groups['lr'])
            self.log('lr', current_learning_rate, prog_bar=True, batch_size=self.batch_size, on_epoch=True, on_step=False)
    
    def training_step(self, data_batch, *args, **kwargs):
        data, labels = data_batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        out_features = self(data)
        loss = self.loss_func(out_features, labels.view(out_features.shape))
        self.log('train_loss', loss, batch_size=self.batch_size, prog_bar=True, on_epoch=True, on_step=False)
        return loss, out_features

    def validation_step(self, data_batch, *args, **kwargs):
        data, labels = data_batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        out_features = self(data)
        loss = self.loss_func(out_features, labels.view(out_features.shape))
        self.log('val_loss', loss, batch_size=self.batch_size, on_epoch=True, on_step=False)
        return out_features

    def predict_step(self, data_batch, *args: Any, **kwargs: Any) -> Any:
        data, labels = data_batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        return self(data)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.lr_scheduler is None:
            return self.optimizer
        
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def update_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

    def _get_optimizer(self, optimizer):
        return optimizer if optimizer is not None else torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _get_lr_scheduler(self, lr_scheduler):
        return lr_scheduler if lr_scheduler is not None else torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.3, mode='min')
            

        # return [optimier], [lr_scheduler]
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "train_loss",
        #         "interval": "step", #"epoch"
        #         "frequency": 1
        #     }
        # }

    @abstractmethod
    def _get_loss_func(self, loss_func):
        pass

class BinaryLightningModel(BaseLightningModel):

    def __init__(self, model, optimizer=None, loss_func=None, learning_rate=0.01, batch_size=64, lr_scheduler=None, user_lr_scheduler=False):
        super(BinaryLightningModel, self).__init__(model, optimizer, loss_func, learning_rate, batch_size=batch_size, lr_scheduler=lr_scheduler, user_lr_scheduler=user_lr_scheduler)
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")

    def training_step(self, data_batch, *args, **kwargs):        
        loss, out_features = super(BinaryLightningModel, self).training_step(data_batch, *args, **kwargs)
        predicted_labels = out_features if out_features.shape[1] < 2 else torch.argmax(out_features, dim=1)
        self.train_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, data_batch, *args, **kwargs):
        out_features = super(BinaryLightningModel, self).validation_step(data_batch, *args, **kwargs)
        predicted_labels = out_features if out_features.shape[1] < 2 else torch.argmax(out_features, dim=1)
        self.val_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=True, batch_size=self.batch_size)

    def _get_loss_func(self, loss_func):
        return loss_func \
            if loss_func is not None else \
            torch.nn.BCELoss()

class MultiClassLightningModel(BaseLightningModel):

    def __init__(self, model, optimizer=None, loss_func=None, lr=0.01, batch_size=64, user_lr_scheduler=False):
        super(MultiClassLightningModel, self).__init__(model, optimizer, loss_func, lr, batch_size=batch_size, user_lr_scheduler=user_lr_scheduler)
        self.train_acc = torchmetrics.Accuracy(task="multiclass")
        self.val_acc = torchmetrics.Accuracy(task="multiclass")
        self.test_acc = torchmetrics.Accuracy(task="multiclass")

    def training_step(self, data_batch, *args, **kwargs):
        loss, out_features = super(MultiClassLightningModel, self).training_step(data_batch, *args, **kwargs)
        predicted_labels = torch.argmax(out_features, dim=1)
        self.train_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, data_batch, *args, **kwargs):
        out_features = super(MultiClassLightningModel, self).validation_step(data_batch, *args, **kwargs)
        predicted_labels = out_features if out_features.shape[1] < 2 else torch.argmax(out_features, dim=1)
        self.val_acc(predicted_labels, out_features.view(predicted_labels.shape))
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=False)

    def _get_loss_func(self, loss_func):
        return loss_func \
            if loss_func is not None else \
            torch.nn.CrossEntropyLoss()

class MultiLabelLightningModel(BaseLightningModel):

    def __init__(self, model, optimizer=None, loss_func=None, lr=0.01, batch_size=64, user_lr_scheduler=False):
        super(MultiLabelLightningModel, self).__init__(model, optimizer, loss_func, lr, batch_size=batch_size, user_lr_scheduler=user_lr_scheduler)
        self.train_acc = torchmetrics.Accuracy(task="multilabel")
        self.val_acc = torchmetrics.Accuracy(task="multilabel")
        self.test_acc = torchmetrics.Accuracy(task="multilabel")

    def training_step(self, data_batch, *args, **kwargs):
        loss, out_features = super(MultiLabelLightningModel, self).training_step(data_batch, *args, **kwargs)
        predicted_labels = out_features
        self.train_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, data_batch, *args, **kwargs):
        out_features = super(MultiLabelLightningModel, self).validation_step(data_batch, *args, **kwargs)
        predicted_labels = out_features
        self.val_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=False)

    def _get_loss_func(self, loss_func):
        return loss_func \
            if loss_func is not None else \
            torch.nn.CrossEntropyLoss()


# class HeteroBinaryLightningModel(BaseLightningModel):

#     def __init__(self, model, optimizer=None, loss_func=None, learning_rate=0.01, batch_size=64, lr_scheduler=None, user_lr_scheduler=False):
#         super(HeteroBinaryLightningModel, self).__init__(model, optimizer, loss_func, learning_rate, batch_size=batch_size, lr_scheduler=lr_scheduler, user_lr_scheduler=user_lr_scheduler)
#         self.train_acc = torchmetrics.Accuracy(task="binary")
#         self.val_acc = torchmetrics.Accuracy(task="binary")
#         self.test_acc = torchmetrics.Accuracy(task="binary")

#     def training_step(self, data_batch, *args, **kwargs):    
#         data, labels = data_batch
#         data = data.to(self.device)
#         labels = labels.to(self.device)
#         out_features = self(data)
#         h_out_features = HeteroLossArgs(out_features[0], out_features[1])
#         label_features = HeteroLossArgs(labels.view(out_features[0].shape), data.x_dict)
#         loss = self.loss_func(h_out_features, label_features)
#         self.log('train_loss', loss, batch_size=self.batch_size, on_epoch=True, on_step=False)
        
#         predicted_labels = out_features[0] if out_features[0].shape[1] < 2 else torch.argmax(out_features[0], dim=1)
#         self.train_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
#         self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=True, batch_size=self.batch_size)
#         return loss

#     def validation_step(self, data_batch, *args, **kwargs):
#         data, labels = data_batch
#         data = data.to(self.device)
#         labels = labels.to(self.device)
#         out_features = self(data)
#         h_out_features = HeteroLossArgs(out_features[0], out_features[1])
#         label_features = HeteroLossArgs(labels.view(out_features[0].shape), data.x_dict)
#         loss = self.loss_func(h_out_features, label_features)
#         self.log('val_loss', loss, batch_size=self.batch_size, on_epoch=True, on_step=False)
#         predicted_labels = out_features[0] if out_features[0].shape[1] < 2 else torch.argmax(out_features[0], dim=1)
#         self.val_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
#         self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=True, batch_size=self.batch_size)

#     def _get_loss_func(self, loss_func):
#         return loss_func \
#             if loss_func is not None else \
#             torch.nn.BCELoss()
