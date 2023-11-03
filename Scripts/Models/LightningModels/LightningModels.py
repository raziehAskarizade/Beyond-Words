from typing import Any
import torch
import torchmetrics
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import lightning as L
from abc import abstractmethod


class BaseLightningModel(L.LightningModule):

    def __init__(self, model, optimizer=None, loss_func=None, lr=0.01, batch_size=64):
        super(BaseLightningModel, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.model = model
        self.optimizer = self._get_optimizer(optimizer)
        self.loss_func = self._get_loss_func(loss_func)
        self.save_hyperparameters(ignore=['model'])

    def forward(self, data_batch, *args, **kwargs):
        return self.model(data_batch)

    def training_step(self, data_batch, *args, **kwargs):
        data, labels = data_batch
        out_features = self(data)
        loss = self.loss_func(out_features, labels.view(out_features.shape))
        self.log('training_loss', loss)
        return loss, out_features

    def validation_step(self, data_batch, *args, **kwargs):
        data, labels = data_batch
        out_features = self(data)
        loss = self.loss_func(out_features, labels.view(out_features.shape))
        self.log('val_loss', loss)
        return out_features

    def predict_step(self, data_batch, *args: Any, **kwargs: Any) -> Any:
        data, labels = data_batch
        return self(data)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer

    def _get_optimizer(self, optimizer):
        return optimizer \
            if optimizer is not None else \
            torch.optim.Adam(self.model.parameters(), lr=self.lr)

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

    def __init__(self, model, optimizer=None, loss_func=None, lr=0.01, batch_size=64):
        super(BinaryLightningModel, self).__init__(model, optimizer, loss_func, lr, batch_size=batch_size)
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")

    def training_step(self, data_batch, *args, **kwargs):
        loss, out_features = super(BinaryLightningModel, self).training_step(data_batch, *args, **kwargs)
        predicted_labels = out_features if out_features.shape[1] < 2 else torch.argmax(out_features, dim=1)
        self.train_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('training_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)
        return loss

    def validation_step(self, data_batch, *args, **kwargs):
        out_features = super(BinaryLightningModel, self).validation_step(data_batch, *args, **kwargs)
        predicted_labels = out_features if out_features.shape[1] < 2 else torch.argmax(out_features, dim=1)
        self.val_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)

    def _get_loss_func(self, loss_func):
        return loss_func \
            if loss_func is not None else \
            torch.nn.BCELoss()


class MultiClassLightningModel(BaseLightningModel):

    def __init__(self, model, optimizer=None, loss_func=None, lr=0.01, batch_size=64):
        super(MultiClassLightningModel, self).__init__(model, optimizer, loss_func, lr, batch_size=batch_size)
        self.train_acc = torchmetrics.Accuracy(task="multiclass")
        self.val_acc = torchmetrics.Accuracy(task="multiclass")
        self.test_acc = torchmetrics.Accuracy(task="multiclass")

    def training_step(self, data_batch, *args, **kwargs):
        loss, out_features = super(MultiClassLightningModel, self).training_step(data_batch, *args, **kwargs)
        predicted_labels = torch.argmax(out_features, dim=1)
        self.train_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('training_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
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

    def __init__(self, model, optimizer=None, loss_func=None, lr=0.01, batch_size=64):
        super(MultiLabelLightningModel, self).__init__(model, optimizer, loss_func, lr, batch_size=batch_size)
        self.train_acc = torchmetrics.Accuracy(task="multilabel")
        self.val_acc = torchmetrics.Accuracy(task="multilabel")
        self.test_acc = torchmetrics.Accuracy(task="multilabel")

    def training_step(self, data_batch, *args, **kwargs):
        loss, out_features = super(MultiLabelLightningModel, self).training_step(data_batch, *args, **kwargs)
        predicted_labels = out_features
        self.train_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('training_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
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
